#!/usr/bin/env python3
"""Train-only tool-candidate recall diagnostic, separate from execution/scoring.

DO NOT pass gold tool names into the retriever. It must predict from transcript
and the installed-app catalog. Gold labels are accessed ONLY after retrieval.
No private/public labels and no hidden environment files.
"""
from __future__ import annotations
import argparse,collections,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from lifemate_v2.intent import GoalFrame
from lifemate_v2.retrieval import select_tools
DATA=ROOT/"datasets/train+public_test/train"
MANIFEST=ROOT/"runs/v2/train_only/split_manifest.json"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--subset",choices=("train_dev","validation"),default="train_dev")
    ap.add_argument("--k",type=int,default=38)
    args=ap.parse_args()
    manifest=json.loads(MANIFEST.read_text(encoding="utf-8"))
    allowed=set(manifest[args.subset])
    catalog=json.loads((DATA/"train_tools_catalog.json").read_text(encoding="utf-8"))
    scores=collections.Counter()
    missed=collections.Counter()
    with (DATA/"train.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():continue
            item=json.loads(line)
            if item["id"] not in allowed:continue
            # No label is passed to select_tools. A real LLM can improve target selection.
            frame=GoalFrame(summary=item["audio_text"])
            selected={x["name"] for x in select_tools(
                item,catalog,frame,item["audio_text"],limit=args.k)}
            gold=[x["name"] for x in item["label"]["tool_calls"]]
            if not gold:continue
            scores["samples"]+=1
            scores["gold_calls"]+=len(gold)
            hits=sum(n in selected for n in gold)
            scores["retrieved_gold_calls"]+=hits
            scores["complete_gold_traces"]+=hits==len(gold)
            for n in gold:
                if n not in selected:missed[n.split(".")[0]]+=1
    n=scores["samples"];g=scores["gold_calls"]
    print(json.dumps({"subset":args.subset,"k":args.k,"samples_with_calls":n,
                      "gold_calls":g,"retrieved_gold_calls":scores["retrieved_gold_calls"],
                      "call_recall":round(scores["retrieved_gold_calls"]/g,5) if g else 0,
                      "all_gold_tools_recall":round(scores["complete_gold_traces"]/n,5) if n else 0,
                      "missed_by_app":missed.most_common(10),
                      "note":"Retrieval only; not official agent score."},indent=2,ensure_ascii=False))
if __name__=="__main__":main()
