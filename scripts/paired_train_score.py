#!/usr/bin/env python3
"""Paired train-only LifeMate scores and bootstrap intervals, never public/private.

Gold state hashes come ONLY from a fresh official ToolEnv gold replay,
never directly from environment files. A per-sample aggregate approximates
the organizer's metric and must not be called the official score.
"""
from __future__ import annotations
import argparse,collections,json,random,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from evaluate_train import DATA,ToolEnv,args_match,max_matching,records

def read_predictions(path):
    rows=list(records(path))
    ids=[r["id"] for r in rows]
    if len(ids)!=len(set(ids)):raise ValueError("duplicate IDs in "+str(path))
    return {r["id"]:r for r in rows}

def per_sample(item,pred,schemas,refhash):
    gold=item["label"]["tool_calls"]
    calls=pred["tool_calls"]
    match=max_matching(gold,calls,schemas)
    state=pred["state_hash"]==refhash
    score=float(not calls) if not gold else (2*match/(len(gold)+len(calls)) if state else 0.0)
    return {"score":score,"state":int(state),"gold":len(gold),
            "pred":len(calls),"matched":match,"false_write":int(not gold and bool(calls)),
            "category":"zero" if not gold else "single" if len(gold)==1 else "double" if len(gold)==2 else "multi"}

def summarize(rows):
    if not rows:return {}
    return {"n":len(rows),"approx_score":round(100*statistics.mean(x["score"] for x in rows),3),
            "state_match":sum(x["state"] for x in rows),
            "matched_calls":sum(x["matched"] for x in rows),
            "pred_calls":sum(x["pred"] for x in rows),
            "gold_calls":sum(x["gold"] for x in rows),
            "false_write_on_gold_empty":sum(x["false_write"] for x in rows)}

def paired(path_a,path_b,seed=42,samples=2000):
    a,b=read_predictions(path_a),read_predictions(path_b)
    if set(a)!=set(b):raise ValueError("V1/V2 ID sets differ: rerun exact same frozen slice")
    items={x["id"]:x for x in records(DATA/"train/train.jsonl") if x["id"] in a}
    if len(items)!=len(a):raise ValueError("Unknown IDs or non-train samples")
    catalog=json.loads((DATA/"train/train_tools_catalog.json").read_text())
    specs={t["name"]:t for app in catalog["apps"] for t in app["tools"]}
    cases=[]
    for id in sorted(a):
        # Fresh replay is a separate evaluation run. Never inspect hidden state files.
        env=ToolEnv("train",id,root=DATA)
        for call in items[id]["label"]["tool_calls"]:
            validated=env.validate(call["name"],call["arguments"])
            if not validated.get("ok"):raise RuntimeError(f"Invalid gold {id}: {validated}")
            result=env.call(call["name"],call["arguments"])
            if not result.get("ok"):raise RuntimeError(f"Failed gold replay {id}: {result}")
        h=env.get_state_hash()
        cases.append((id,per_sample(items[id],a[id],specs,h),
                      per_sample(items[id],b[id],specs,h)))
    aa=[x[1] for x in cases];bb=[x[2] for x in cases]
    rng=random.Random(seed)
    differences=[]
    for _ in range(samples):
        index=[rng.randrange(len(cases)) for _ in cases]
        differences.append(100*sum(bb[i]["score"]-aa[i]["score"] for i in index)/len(index))
    differences.sort()
    result={"a":str(path_a),"b":str(path_b),"n":len(cases),
            "model_a":summarize(aa),"model_b":summarize(bb),
            "paired_delta":round(100*statistics.mean(b["score"]-a["score"] for _,a,b in cases),3),
            "bootstrap_95_ci":[round(differences[int(.025*len(differences))],3),
                               round(differences[int(.975*len(differences))],3)],
            "by_stratum":{},
            "notes":"Small nonrandom balanced slices are not unbiased leaderboard estimates. Train oracle transcript only."}
    for k in ("zero","single","double","multi"):
        sub=[(a,b) for _,a,b in cases if a["category"]==k]
        if sub:result["by_stratum"][k]={"a":summarize([x[0] for x in sub]),"b":summarize([x[1] for x in sub])}
    return result

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("a",type=Path);parser.add_argument("b",type=Path)
    args=parser.parse_args()
    print(json.dumps(paired(args.a,args.b),ensure_ascii=False,indent=2))
