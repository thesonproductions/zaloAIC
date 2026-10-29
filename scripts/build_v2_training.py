#!/usr/bin/env python3
"""Freeze train-only grouped dev/validation IDs and create leakage-safe next-action SFT rows.

Never reads *_environment/ files. Train audio_text and labels only; no test labels.
"""
from __future__ import annotations
import argparse,collections,hashlib,json,re,unicodedata
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"datasets/train+public_test/train"
EXAMINED={"train_0004","train_0005","train_0019",
          "train_0038","train_0225","train_0630","train_0692","train_0866",
          "train_0982","train_0988","train_1098","train_1150","train_1389","train_1561",
          "train_0248"}

def records(path):
    with path.open(encoding="utf-8") as f:
        for ln in f:
            if ln.strip():yield json.loads(ln)

def template(text):
    text=unicodedata.normalize("NFD",text.casefold())
    text="".join(c for c in text if not unicodedata.combining(c)).replace("đ","d")
    # Coarse grouping, NOT a guarantee against semantic paraphrase duplicates.
    text=re.sub(r"\b\d+(?:[.,:]\d+)*\b","<num>",text)
    text=re.sub(r"[^a-z0-9<> ]+"," ",text)
    return " ".join(text.split())

def group_key(item):
    return hashlib.sha256(template(item["audio_text"]).encode()).hexdigest()[:14]

def freeze(rows,seed=42):
    groups=collections.defaultdict(list)
    for x in rows:groups[group_key(x)].append(x)
    train,validation=[],[]
    for key,group in sorted(groups.items()):
        if any(x["id"] in EXAMINED for x in group):
            train.extend(x["id"] for x in group)
            continue
        tag=hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()
        val=int(tag[:8],16)/0xffffffff <.20
        (validation if val else train).extend(x["id"] for x in group)
    ids={x["id"]:i for i,x in enumerate(rows)}
    train.sort(key=ids.__getitem__);validation.sort(key=ids.__getitem__)
    assert set(train).isdisjoint(validation)
    assert len(train)+len(validation)==len(rows)
    return {"seed":seed,"protocol":"hash of accent-normalized transcript with numeric placeholders",
            "warning":"This split is group-heuristic only; manually audit near-duplicate semantic templates",
            "train_dev":train,"validation":validation,"examined_in_dev":sorted(set(train)&EXAMINED)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default=str(ROOT/"runs/v2/train_only"))
    args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    rows=list(records(DATA/"train.jsonl"))
    split_path=out/"split_manifest.json"
    if split_path.exists():
        split=json.loads(split_path.read_text(encoding="utf-8"))
        if set(split["train_dev"])|set(split["validation"]) != {x["id"] for x in rows}:
            raise SystemExit("Frozen split does not match current train IDs")
    else:
        split=freeze(rows)
        split_path.write_text(json.dumps(split,ensure_ascii=False,indent=2),encoding="utf-8")
    catalog=json.loads((DATA/"train_tools_catalog.json").read_text())
    specs={t["name"]:t for a in catalog["apps"] for t in a["tools"]}
    allowed_train=set(split["train_dev"])
    allowed_val=set(split["validation"])
    counts=collections.Counter()
    for partition,ids in [("train_dev",allowed_train),("validation",allowed_val)]:
        path=out/f"next_action_{partition}.jsonl"
        with path.open("w",encoding="utf-8") as f:
            for item in rows:
                if item["id"] not in ids:continue
                allowed={t for a in item["context"]["installed_apps"].values() for t in a["tools"]} | {"sys.ask_missing_info"}
                schemas=[{"name":name,"description":specs[name].get("description",""),
                          "kind":specs[name].get("kind"),
                          "parameters":specs[name].get("parameters",{})}
                         for name in sorted(allowed) if name in specs]
                calls=item["label"]["tool_calls"]
                responses=item["label"]["tool_results"]
                if len(calls)!=len(responses):raise ValueError(f"Gold trace lengths differ: {item['id']}")
                for index in range(len(calls)+1):
                    # Pre-state contains ONLY observations of already-completed gold steps.
                    prior=[{"tool":calls[i],"result":responses[i]} for i in range(index)]
                    target=calls[index] if index<len(calls) else {"name":"__stop__","arguments":{}}
                    example={
                        "source_id":item["id"],
                        "step":index,
                        "gold_transcript":item["audio_text"],
                        "history":item.get("history",[]),
                        "context":item["context"],
                        "available_tool_schemas":schemas,
                        "prior_observations":prior,
                        "target_action":target,
                        "split":partition
                    }
                    f.write(json.dumps(example,ensure_ascii=False)+"\n")
                    counts[partition]+=1
    print(json.dumps({"train_dev_samples":len(allowed_train),"validation_samples":len(allowed_val),
        "train_dev_next_action_examples":counts["train_dev"],
        "validation_next_action_examples":counts["validation"],
        "manifest":str(split_path)},ensure_ascii=False))
if __name__=="__main__":main()
