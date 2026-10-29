#!/usr/bin/env python3
"""LifeMate V2: local offline Mac inference prototype, TRAIN diagnostics / public submission.

This is not yet a CUDA/A30 Docker solution; never access *_environment directly.
"""
from __future__ import annotations
import argparse,json,os,random,sys,time,traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"datasets"/"train+public_test"
sys.path.insert(0,str(ROOT/"src"))
from lifemate_baseline import LocalModels,DEFAULT_LLM,DEFAULT_ASR
from lifemate_v2.controller import run_episode
from lifemate_v2.retrieval import installed_catalog

def records(path:Path):
    with path.open(encoding="utf-8") as f:
        for ln in f:
            if ln.strip():yield json.loads(ln)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--split",choices=["train","public_test"],default="train")
    parser.add_argument("--oracle",action="store_true",help="Only train gold audio_text; planner E1")
    parser.add_argument("--ids",default="",help="Comma-separated sample IDs")
    parser.add_argument("--limit",type=int,default=0)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--llm",default=str(DEFAULT_LLM))
    parser.add_argument("--asr",default=str(DEFAULT_ASR))
    parser.add_argument("--asr-backend",choices=["whisper","funasr"],default="whisper",
                        help="Single ASR backend; FunASR is experimental and opt-in")
    parser.add_argument("--funasr-model",default="",help="Local pre-downloaded FunASR checkpoint")
    parser.add_argument("--funasr-device",choices=["cpu","mps","cuda"],default="mps")
    parser.add_argument("--adapter",default="",help="Optional local MLX LoRA adapter folder")
    parser.add_argument("--compact-plan",action="store_true",help="Match compact SFT prompt format")
    parser.add_argument("--out",default=str(ROOT/"runs"/"v2"/"predictions.jsonl"))
    parser.add_argument("--deadline",type=float,default=27.5)
    parser.add_argument("--log-every",type=int,default=5)
    args=parser.parse_args()
    if args.oracle and args.split!="train":raise SystemExit("--oracle is TRAIN ONLY")
    os.environ["HF_HUB_OFFLINE"]="1"
    os.environ["TRANSFORMERS_OFFLINE"]="1"
    random.seed(42)
    import mlx.core as mx
    mx.random.seed(42)
    sys.path.insert(0,str(DATA/args.split))
    from zaic_tools import ToolEnv
    catalog=json.loads((DATA/args.split/f"{args.split}_tools_catalog.json").read_text())
    out=Path(args.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    wanted=set(args.ids.split(",")) if args.ids else None
    done={r["id"] for r in records(out)} if args.resume and out.exists() else set()
    if args.asr_backend == "funasr":
        if not args.funasr_model:
            parser.error("--funasr-model is required with --asr-backend funasr")
        from lifemate_v2.asr_backend import FunASRLocalModels
        models = FunASRLocalModels(
            Path(args.llm), Path(args.funasr_model), device=args.funasr_device,
            adapter=Path(args.adapter) if args.adapter else None,
            compact_plan=args.compact_plan,
        )
    else:
        models = LocalModels(
            Path(args.llm), Path(args.asr),
            adapter=Path(args.adapter) if args.adapter else None,
            compact_plan=args.compact_plan,
        )
    count=0;errors=0;times=[];start=time.monotonic()
    mode="a" if args.resume else "w"
    with out.open(mode,encoding="utf-8") as fo, Path(str(out)+".diagnostics.jsonl").open(mode,encoding="utf-8") as fd:
        for sample in records(DATA/args.split/f"{args.split}.jsonl"):
            if wanted is not None and sample["id"] not in wanted:continue
            if sample["id"] in done:continue
            t=time.monotonic()
            # ONLY ONE env for this sample's inference execution.
            env=ToolEnv(args.split,sample["id"],root=DATA)
            try:
                record,diag=run_episode(sample,args.split,env,models,catalog,DATA,
                                        oracle=args.oracle,deadline_sec=args.deadline)
            except Exception as exc:
                errors+=1
                record={"id":sample["id"],"tool_calls":env.get_tool_calls(),"state_hash":env.get_state_hash()}
                diag={"id":sample["id"],"error":str(exc),"trace":traceback.format_exc(limit=5),"stop_reason":"exception"}
            dt=time.monotonic()-t
            diag["seconds"]=round(dt,3)
            fo.write(json.dumps(record,ensure_ascii=False)+"\n");fo.flush()
            fd.write(json.dumps(diag,ensure_ascii=False)+"\n");fd.flush()
            count+=1;times.append(dt)
            if count<=3 or count%args.log_every==0:
                print(json.dumps({"done":count,"sample":sample["id"],"calls":len(record["tool_calls"]),
                    "seconds":round(dt,2),"stop":diag.get("stop_reason"),"errors":errors,
                    "elapsed":round(time.monotonic()-start,1)},ensure_ascii=False),flush=True)
            if args.limit and count>=args.limit:break
    print(json.dumps({"processed":count,"errors":errors,
           "avg_seconds":round(sum(times)/len(times),2) if times else 0,
           "over_30":sum(t>30 for t in times),
           "output":str(out)},ensure_ascii=False),flush=True)
if __name__=="__main__":main()
