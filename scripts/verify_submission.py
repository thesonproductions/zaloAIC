#!/usr/bin/env python3
"""Check generated LifeMate JSONL (one-pass replay only; no direct state file reads)."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "train+public_test"

def lines(path):
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip(): continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON line {i}: {exc}") from exc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",choices=["train","public_test"],default="public_test")
    parser.add_argument("--file",required=True)
    parser.add_argument("--allow-partial",action="store_true")
    args=parser.parse_args()

    source=DATA/args.split/f"{args.split}.jsonl"
    submission=Path(args.file)
    if not submission.is_file():raise SystemExit(f"File not found: {submission}")
    gold_ids=[x["id"] for x in lines(source)]
    outputs=list(lines(submission))
    got_ids=[x.get("id") for x in outputs]
    if len(got_ids)!=len(set(got_ids)):raise SystemExit("FAIL duplicate IDs")
    if not args.allow_partial and got_ids!=gold_ids:
        raise SystemExit(f"FAIL incomplete or wrong order: {len(got_ids)}/{len(gold_ids)}")
    if args.allow_partial and got_ids!=[x for x in gold_ids if x in set(got_ids)]:
        raise SystemExit("FAIL subset IDs not in source order")
    sys.path.insert(0,str(DATA/args.split))
    from zaic_tools import ToolEnv
    mismatch=0
    wrong_schema=0
    for r in outputs:
        if not isinstance(r.get("tool_calls"),list) or not isinstance(r.get("state_hash"),str):
            raise SystemExit(f"FAIL malformed row {r.get('id')}")
        # This verification pass replays, it does not use environment backing files.
        env=ToolEnv(args.split,r["id"],root=DATA)
        for call in r["tool_calls"]:
            res=env.validate(call["name"],call["arguments"])
            if not res.get("ok"):wrong_schema+=1
            env.call(call["name"],call["arguments"])
        if env.get_tool_calls()!=r["tool_calls"] or env.get_state_hash()!=r["state_hash"]:
            mismatch+=1
    print(json.dumps({"samples":len(outputs),"expected_samples":len(gold_ids),"complete":got_ids==gold_ids,"replay_hash_mismatch":mismatch,"schema_errors":wrong_schema},ensure_ascii=False))
    if mismatch or wrong_schema:raise SystemExit("FAIL replay or schema verification")
    if not args.allow_partial and len(outputs)!=len(gold_ids):raise SystemExit("FAIL incomplete")
    print("PASS")

if __name__=="__main__":
    main()
