#!/usr/bin/env python3
"""Prepare official public submission after the inference run has completed.

Reads public_test JSONL and the generated tool-call file. Does not read environment files.
Verification is a separate replay pass mediated only by the organizer's ToolEnv.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"runs/public_v1/submission.jsonl"
DEST=ROOT/"result/submission.jsonl"
DATA=ROOT/"datasets/train+public_test/public_test/public_test.jsonl"

def jsonl(path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def main():
    if not SOURCE.exists():raise SystemExit(f"NOT READY: missing {SOURCE}")
    ids=[r["id"] for r in jsonl(DATA)]
    results=list(jsonl(SOURCE))
    got=[r["id"] for r in results]
    if got!=ids:
        raise SystemExit(f"NOT READY: {len(got)} of {len(ids)} rows, or ID order mismatch. Do not upload yet.")
    cmd=[sys.executable,str(ROOT/"scripts/verify_submission.py"),
         "--split","public_test","--file",str(SOURCE)]
    subprocess.run(cmd,check=True,cwd=ROOT)
    DEST.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(SOURCE,DEST)
    print(f"READY TO UPLOAD: {DEST}; samples={len(got)}")
    print("This file is a Mac-generated public submission; final Docker requires a separate offline CUDA build.")

if __name__=="__main__":
    main()
