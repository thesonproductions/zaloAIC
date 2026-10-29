#!/usr/bin/env python3
"""Train-only, deterministic local evaluation for LifeMate.

Do not use for public/private test labels. Uses official ToolEnv for gold replay,
but never directly inspects environment backing files or state.json.
"""
import argparse
import collections
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"datasets/train+public_test"
SPLIT="train"
sys.path.insert(0,str(DATA/SPLIT))
from zaic_tools import ToolEnv

def records(path):
    with open(path,encoding="utf-8") as f:
        for s in f:
            if s.strip():yield json.loads(s)

def fuzzy_f1(a,b):
    a=re.findall(r"\w+",unicodedata.normalize("NFC",str(a)).lower())
    b=re.findall(r"\w+",unicodedata.normalize("NFC",str(b)).lower())
    if not a and not b:return 1.0
    if not a or not b:return 0.0
    overlap=sum((collections.Counter(a) & collections.Counter(b)).values())
    return 2*overlap/(len(a)+len(b))

def args_match(reference,predicted,schema):
    a=reference.get("arguments",{})
    b=predicted.get("arguments",{})
    if set(a)!=set(b):return False
    props=schema.get("parameters",{}).get("properties",{})
    for key in a:
        if props.get(key,{}).get("x-match")=="fuzzy":
            if fuzzy_f1(a[key],b[key])<=.6:return False
        elif a[key]!=b[key]:return False
    return True

def max_matching(gold,pred,schemas):
    edges=[]
    for pred_call in pred:
        matches=[]
        for gi,gold_call in enumerate(gold):
            if pred_call.get("name")==gold_call.get("name") and args_match(gold_call,pred_call,schemas.get(pred_call["name"],{})):
                matches.append(gi)
        edges.append(matches)
    assigned={}
    def find(pi,seen):
        for gi in edges[pi]:
            if gi in seen:continue
            seen.add(gi)
            if gi not in assigned or find(assigned[gi],seen):
                assigned[gi]=pi
                return True
        return False
    return sum(find(i,set()) for i in range(len(edges)))

def stratified_samples(k=24,seed=42):
    groups=collections.defaultdict(list)
    for x in records(DATA/"train/train.jsonl"):
        if x["id"] in {"train_0004","train_0005","train_0019"}:
            continue
        n=len(x["label"]["tool_calls"])
        group="zero" if n==0 else "single" if n==1 else "double" if n==2 else "multi"
        groups[group].append(x["id"])
    rng=random.Random(seed)
    result=[]
    for key in ["zero","single","double","multi"]:
        candidates=sorted(groups[key]);rng.shuffle(candidates)
        result+=candidates[:max(1,k//4)]
    print(",".join(result))

def score(filename:Path):
    predictions={x["id"]:x for x in records(filename)}
    samples={x["id"]:x for x in records(DATA/"train/train.jsonl") if x["id"] in predictions}
    catalog=json.loads((DATA/"train/train_tools_catalog.json").read_text())
    schemas={tool["name"]:tool for app in catalog["apps"] for tool in app["tools"]}
    stat=collections.Counter()
    failures=[]
    n=len(predictions)
    for sample_id, pred in predictions.items():
        sample=samples[sample_id]
        gold=sample["label"]["tool_calls"]
        env=ToolEnv("train",sample_id,root=DATA)
        for call in gold: env.call(call["name"],call["arguments"])
        refhash=env.get_state_hash()
        pred_calls=pred["tool_calls"]
        matched=max_matching(gold,pred_calls,schemas)
        if not gold:
            score=int(not pred_calls)
        else:
            call_f1=(2*matched/(len(gold)+len(pred_calls))) if (len(gold)+len(pred_calls)) else 0
            score=call_f1 if pred["state_hash"]==refhash else 0
        stat["score_sum"]+=score
        stat["hash_matched"]+=int(pred["state_hash"]==refhash)
        stat["correct_out_of_scope"]+=int(not gold and not pred_calls)
        stat["gold_zero"]+=int(not gold)
        stat["predicted_zero"]+=int(not pred_calls)
        stat["gold_calls"]+=len(gold)
        stat["predicted_calls"]+=len(pred_calls)
        stat["matching_calls"]+=matched
        if score<1:
            failures.append({
                "id":sample_id,"gold":len(gold),"pred":len(pred_calls),
                "matches":matched,"state_correct":pred["state_hash"]==refhash,
                "gold_tools":[x["name"] for x in gold],
                "pred_tools":[x["name"] for x in pred_calls],
            })
    print(json.dumps({
        "samples":n,"score_approx":round(100*stat["score_sum"]/max(n,1),2),
        "state_match_pct":round(100*stat["hash_matched"]/max(n,1),2),
        "gold_empty":stat["gold_zero"],"predicted_empty":stat["predicted_zero"],
        "correct_empty":stat["correct_out_of_scope"],
        "gold_calls":stat["gold_calls"],"predicted_calls":stat["predicted_calls"],"matched_calls":stat["matching_calls"],
        "failing_cases":failures[:20],
    },ensure_ascii=False,indent=2))

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--select",type=int,default=None)
    parser.add_argument("--file",default=None)
    args=parser.parse_args()
    if args.select: stratified_samples(args.select)
    elif args.file:score(Path(args.file))
    else:parser.error("Specify --select or --file")
