#!/usr/bin/env python3
"""Export verified train-only LifeMate next-action trajectories as MLX-LM chat SFT JSONL.

No actual training, external APIs, or *_environment access. Uses original train labels.
"""
from __future__ import annotations
import json,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from lifemate_v2.controller import PLAN_SYSTEM
FROM=ROOT/"runs/v2/train_only"
DEST=FROM/"mlx_sft"

def data(path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():yield json.loads(line)

def to_chat(row):
    goal=row["gold_transcript"]
    tools=[{"name":t["name"],"kind":t.get("kind"),"description":t.get("description","")[:220],
            "parameters":t.get("parameters",{})}
            for t in row["available_tool_schemas"]]
    observed=[{"name":step["tool"]["name"],"arguments":step["tool"].get("arguments",{}),
               "result":step["result"]} for step in row["prior_observations"]]
    query={
        "latest_request":goal,
        "context":row["context"],
        "history_events_already_applied":row["history"][-12:],
        # Intentionally no gold future goals: only user's utterance and prior observations.
        "goal":{"summary":goal,"target_tools":[],"pending_target_tools":[],
                "semantic_steps":[],"read_only":False,"restrictions":[]},
        "available_tool_schemas":tools,
        "current_request_observations":observed[-6:],
        "previous_invalid_proposals_not_executed":[],
        "instruction":"Predict the single next legal action or __stop__ based only on provided evidence."
    }
    action=row["target_action"]
    tool=action["name"]
    answer={"tool":tool,"arguments":action.get("arguments",{})}
    return {"messages":[
        {"role":"system","content":PLAN_SYSTEM},
        {"role":"user","content":json.dumps(query,ensure_ascii=False,separators=(",",":"))},
        {"role":"assistant","content":json.dumps(answer,ensure_ascii=False,separators=(",",":"))}
    ]}

def main():
    DEST.mkdir(parents=True,exist_ok=True)
    summary={}
    for part,out in (("train_dev","train"),("validation","valid")):
        src=FROM/f"next_action_{part}.jsonl"
        counts=0;chars=[]
        with (DEST/f"{out}.jsonl").open("w",encoding="utf-8") as f:
            for row in data(src):
                record=to_chat(row)
                f.write(json.dumps(record,ensure_ascii=False)+"\n")
                counts+=1;chars.append(sum(len(m["content"]) for m in record["messages"]))
        summary[out]={"samples":counts,"mean_chars":round(sum(chars)/len(chars)) if chars else 0}
    print(json.dumps({"output_dir":str(DEST),"datasets":summary,
                      "training":"NOT RUN; use MLX-LM lora with --mask-prompt"},ensure_ascii=False))
if __name__=="__main__":main()
