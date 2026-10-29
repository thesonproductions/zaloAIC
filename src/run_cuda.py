#!/usr/bin/env python3
"""LifeMate CUDA offline inference entrypoint.

Requires local HF-formatted Qwen3.5-4B and Whisper Turbo model snapshots.
No network/API at runtime. The MLX LoRA checkpoint cannot be loaded here directly.
"""
from __future__ import annotations
import argparse,json,os,random,sys,time,wave
from pathlib import Path
os.environ["HF_HUB_OFFLINE"]="1"
os.environ["TRANSFORMERS_OFFLINE"]="1"
os.environ["HF_DATASETS_OFFLINE"]="1"
os.environ["TOKENIZERS_PARALLELISM"]="false"
os.environ["NO_PROXY"]="*"
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from lifemate_v2.controller import run_episode

def detect_split(data_path:Path,requested:str)->str:
    if requested!="auto":return requested
    p=data_path.resolve()
    matches=[s for s in ("private_test","public_test","train")
             if (p/s/f"{s}.jsonl").is_file() or (p/f"{s}.jsonl").is_file()]
    if len(matches)!=1:
        raise ValueError(f"Expected one mounted split (found {matches}); use --split explicitly")
    return matches[0]

def prepare_data_root(data_path:Path,split:str)->Path:
    """Supports /data/public_test/ or flat /data; no hidden state files inspected."""
    p=data_path.resolve()
    if (p/split/f"{split}.jsonl").is_file():
        return p
    if (p/f"{split}.jsonl").is_file():
        # ToolEnv expects data_root/split/*. Build a legal symlink to the
        # input directory, and let the official SDK access its backing data.
        import tempfile
        base=Path(tempfile.mkdtemp(prefix="lifemate_data_"))
        (base/split).symlink_to(p,target_is_directory=True)
        return base
    raise FileNotFoundError(f"Missing {split}.jsonl under {p}")

class HFLocalModels:
    def __init__(self,llm:Path,asr:Path,max_context:int=16384):
        if not llm.is_dir() or not asr.is_dir():
            raise FileNotFoundError(f"Model directories required, got {llm} and {asr}")
        import numpy as np
        import torch
        from transformers import AutoModelForMultimodalLM,AutoProcessor,pipeline
        if not torch.cuda.is_available():
            raise RuntimeError("This CUDA runner needs NVIDIA GPU")
        self.np=np
        self.torch=torch
        self.max_context=max_context
        self.processor=AutoProcessor.from_pretrained(str(llm),local_files_only=True,trust_remote_code=False)
        self.model=AutoModelForMultimodalLM.from_pretrained(
            str(llm),local_files_only=True,trust_remote_code=False,
            torch_dtype=torch.float16,low_cpu_mem_usage=True
        ).to("cuda")
        self.model.eval()
        self.asr=pipeline("automatic-speech-recognition",model=str(asr),
                          tokenizer=str(asr),feature_extractor=str(asr),
                          device=0,torch_dtype=torch.float16,
                          model_kwargs={"local_files_only":True})
        self.asr_cache={}
        self.llm_params=sum(x.numel() for x in self.model.parameters())
        self.asr_params=sum(x.numel() for x in self.asr.model.parameters())
        if self.llm_params>=5_000_000_000 or self.llm_params+self.asr_params>=6_000_000_000:
            raise RuntimeError("Contest model parameter budget exceeded")
        print(json.dumps({"llm_params":self.llm_params,"asr_params":self.asr_params,
                          "total_params":self.llm_params+self.asr_params}),flush=True)

    def transcribe(self,path:Path)->str:
        key=str(path)
        if key in self.asr_cache:return self.asr_cache[key]
        with wave.open(str(path),'rb') as w:
            if (w.getframerate(),w.getnchannels(),w.getsampwidth())!=(16000,1,2):
                raise ValueError("Audio must be 16kHz mono PCM16 WAV")
            audio=self.np.frombuffer(w.readframes(w.getnframes()),dtype="<i2").astype("float32")/32768.
        result=self.asr({"array":audio,"sampling_rate":16000},
                        generate_kwargs={"language":"vi","task":"transcribe"})
        text=result["text"].strip()
        self.asr_cache[key]=text
        return text

    def plan(self,prompt_content:str,max_tokens:int=220,system:str="")->str:
        import torch
        messages=[{"role":"system","content":system},{"role":"user","content":prompt_content}]
        tokenized=self.processor.apply_chat_template(
            messages,tokenize=True,add_generation_prompt=True,
            return_dict=True,return_tensors="pt",enable_thinking=False
        )
        seq_len=int(tokenized["input_ids"].shape[-1])
        if seq_len+max_tokens>self.max_context:
            raise ValueError(f"Context length {seq_len+max_tokens} exceeds {self.max_context}")
        inputs={k:v.to("cuda") for k,v in tokenized.items() if isinstance(v,torch.Tensor)}
        with torch.inference_mode():
            ids=self.model.generate(**inputs,max_new_tokens=max_tokens,
                do_sample=False,pad_token_id=self.processor.tokenizer.eos_token_id)
        generated=ids[0,seq_len:]
        return self.processor.tokenizer.decode(generated,skip_special_tokens=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-root",type=Path,default=Path("/data"))
    ap.add_argument("--split",default="auto",choices=("auto","train","public_test","private_test"))
    ap.add_argument("--llm",type=Path,default=Path("/models/llm"))
    ap.add_argument("--asr",type=Path,default=Path("/models/asr"))
    ap.add_argument("--out",type=Path,default=Path("/result/submission.jsonl"))
    ap.add_argument("--limit",type=int,default=0)
    ap.add_argument("--per-sample-deadline",type=float,default=27.0)
    args=ap.parse_args()
    import numpy as np,torch
    np.random.seed(42);random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    args.split=detect_split(args.data_root,args.split)
    data=prepare_data_root(args.data_root,args.split)
    sys.path.insert(0,str(data/args.split))
    from zaic_tools import ToolEnv
    catalog=json.loads((data/args.split/f"{args.split}_tools_catalog.json").read_text(encoding="utf-8"))
    model=HFLocalModels(args.llm,args.asr)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    count=0;max_t=0.0;timeouts=0
    with (data/args.split/f"{args.split}.jsonl").open(encoding="utf-8") as f, args.out.open("w",encoding="utf-8") as output:
        for ln in f:
            if not ln.strip():continue
            item=json.loads(ln)
            started=time.perf_counter()
            env=ToolEnv(args.split,item["id"],root=data)
            try:
                submission,diag=run_episode(item,args.split,env,model,catalog,data,
                    oracle=False,deadline_sec=args.per_sample_deadline)
            except Exception as exc:
                # Preserve all actual attempted tool calls and computed state.
                submission={"id":item["id"],"tool_calls":env.get_tool_calls(),"state_hash":env.get_state_hash()}
                print(f"ERROR {item['id']}: {type(exc).__name__}: {exc}",file=sys.stderr)
            t=time.perf_counter()-started
            timeouts+=(t>30.0)
            max_t=max(max_t,t)
            output.write(json.dumps(submission,ensure_ascii=False)+"\n")
            output.flush()
            count+=1
            if count%10==0:print(json.dumps({"completed":count,"last_sec":round(t,2),
               "max_sec":round(max_t,2),"over_30":timeouts}),flush=True)
            if args.limit and count>=args.limit:break
    print(json.dumps({"complete":count,"max_sec":round(max_t,2),"over_30":timeouts,
          "file":str(args.out),"deployment_review":"A30 verification still required"}),flush=True)
if __name__=="__main__":main()
