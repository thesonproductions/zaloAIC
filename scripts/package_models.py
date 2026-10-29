#!/usr/bin/env python3
"""Stage public, HF-format eligible models for a self-contained offline Docker.

Downloads only in development/build, not inference. Pins last available commit
at/before the contest release cutoff. NEVER touches competition datasets.
"""
from __future__ import annotations
import argparse,datetime as dt,json
from pathlib import Path
from huggingface_hub import HfApi,snapshot_download

ROOT=Path(__file__).resolve().parents[1]
CUTOFF=dt.datetime(2026,10,26,23,59,59,tzinfo=dt.timezone.utc)
MODELS={"llm":"Qwen/Qwen3.5-4B","asr":"openai/whisper-large-v3-turbo"}
def eligible_commit(api,repo):
    commits=api.list_repo_commits(repo)
    valid=[c for c in commits if c.created_at and c.created_at.astimezone(dt.timezone.utc)<=CUTOFF]
    if not valid:raise RuntimeError(f"No eligible revision for {repo} at cutoff")
    return max(valid,key=lambda c:c.created_at)
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dest",type=Path,default=ROOT/"model_bundle")
    parser.add_argument("--metadata-only",action="store_true")
    args=parser.parse_args()
    api=HfApi()
    args.dest.mkdir(exist_ok=True,parents=True)
    report={}
    for key,repo in MODELS.items():
        selected=eligible_commit(api,repo)
        report[key]={"repo":repo,"revision":selected.commit_id,"date":selected.created_at.isoformat()}
        if not args.metadata_only:
            folder=args.dest/key
            snapshot_download(repo_id=repo,revision=selected.commit_id,
                local_dir=str(folder),max_workers=4)
            if not (folder/"config.json").exists():
                raise RuntimeError(f"Incomplete weights for {repo}")
            print(f"READY {repo} {folder}",flush=True)
    (args.dest/"provenance.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2),flush=True)
if __name__=="__main__":main()
