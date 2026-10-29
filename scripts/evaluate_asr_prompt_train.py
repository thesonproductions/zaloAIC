#!/usr/bin/env python3
"""Offline paired Mac ASR prompt-bias experiment, TRAIN DEV only.

No public/private data, no hidden app state, no model API. Gold transcript
is consumed solely to calculate WER AFTER both decoding alternatives.
Contextual ASR prompt is derived from installed app metadata, never gold text.
"""
from __future__ import annotations
import argparse,json,random,re,unicodedata,wave,os
from pathlib import Path
import numpy as np
os.environ["HF_HUB_OFFLINE"]="1"
os.environ["TRANSFORMERS_OFFLINE"]="1"
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"datasets/train+public_test"
ASR=next((Path.home()/".cache/huggingface/hub/models--mlx-community--whisper-large-v3-turbo/snapshots").iterdir())

def words(s):
    return re.findall(r"\w+",unicodedata.normalize("NFC",s).casefold())

def word_edits(a,b):
    aa,bb=words(a),words(b)
    prev=list(range(len(bb)+1))
    for i,x in enumerate(aa,1):
        row=[i]
        for j,y in enumerate(bb,1):
            row.append(min(row[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
        prev=row
    return prev[-1],len(aa)

def allowed_prompt(item,catalog):
    apps=set(item["context"]["installed_apps"])
    profile=[app.get("name_vi","") for app in catalog["apps"]
             if app["id"] in apps]
    # Restrict vocabulary to actual installed app names. Never gold labels.
    return "Trợ lý tiếng Việt, các ứng dụng: "+", ".join(profile[:16])+"."

def main():
    import mlx_whisper
    ap=argparse.ArgumentParser()
    ap.add_argument("--n",type=int,default=12)
    ap.add_argument("--seed",type=int,default=21477)
    ap.add_argument("--out",type=Path,default=ROOT/"runs/v2/asr_prompt_eval_dev.jsonl")
    args=ap.parse_args()
    manifest=json.loads((ROOT/"runs/v2/train_only/split_manifest.json").read_text())
    dev=set(manifest["train_dev"])
    pool=[]
    for line in (DATA/"train/train.jsonl").open():
        it=json.loads(line)
        if it["id"] in dev:pool.append(it)
    rng=random.Random(args.seed);rng.shuffle(pool)
    selected=pool[:args.n]
    cat=json.loads((DATA/"train/train_tools_catalog.json").read_text())
    args.out.parent.mkdir(exist_ok=True,parents=True)
    totals=[0,0,0];diff=[]
    with args.out.open("w",encoding="utf-8") as out:
        for i,it in enumerate(selected,1):
            path=DATA/"train"/it["audio"]
            with wave.open(str(path),"rb") as f:
                assert (f.getnchannels(),f.getframerate(),f.getsampwidth())==(1,16000,2)
                samples=np.frombuffer(f.readframes(f.getnframes()),dtype="<i2").astype("float32")/32768.
            prompt=allowed_prompt(it,cat)
            opts={"path_or_hf_repo":str(ASR),"language":"vi","temperature":0.,
                  "condition_on_previous_text":False,"fp16":True}
            normal=mlx_whisper.transcribe(samples,**opts)["text"].strip()
            context=mlx_whisper.transcribe(samples,initial_prompt=prompt,**opts)["text"].strip()
            e0,n=word_edits(it["audio_text"],normal)
            e1,_=word_edits(it["audio_text"],context)
            totals[0]+=n;totals[1]+=e0;totals[2]+=e1
            diff.append(e1-e0)
            record={"id":it["id"],"gold":it["audio_text"],
                    "baseline":normal,"contextual":context,
                    "prompt":prompt,"normal_edits":e0,"context_edits":e1,"words":n}
            out.write(json.dumps(record,ensure_ascii=False)+"\n");out.flush()
            print(json.dumps({"done":i,"id":it["id"],"baseline_WER":round(e0/max(1,n),3),
                              "prompt_WER":round(e1/max(1,n),3),
                              "corpus_base":round(totals[1]/totals[0],4),
                              "corpus_prompt":round(totals[2]/totals[0],4)},
                             ensure_ascii=False),flush=True)
    print("RESULT",json.dumps({"n":args.n,"words":totals[0],
                               "baseline_WER":round(totals[1]/totals[0],4),
                               "prompt_WER":round(totals[2]/totals[0],4),
                               "improved":sum(n<0 for n in diff),
                               "regressed":sum(n>0 for n in diff),
                               "unchanged":sum(n==0 for n in diff)},ensure_ascii=False))
if __name__=="__main__":main()
