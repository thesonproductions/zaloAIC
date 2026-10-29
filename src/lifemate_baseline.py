#!/usr/bin/env python3
"""LifeMate v0 local baseline: offline MLX ASR + LLM + official ToolEnv.

Run in the existing Contexta quality-venv on Mac for *public leaderboard only*.
Not yet a Linux/NVIDIA-A30 Docker solution. NEVER reads *_environment files.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import traceback
import wave
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT / "datasets" / "train+public_test"
HF = Path.home() / ".cache/huggingface/hub"
DEFAULT_LLM = next(
    (HF / "models--mlx-community--Qwen3.5-4B-4bit" / "snapshots").glob("*")
)
DEFAULT_ASR = next(
    (HF / "models--mlx-community--whisper-large-v3-turbo" / "snapshots").glob("*")
)
SYSTEM = """You are a tool-using assistant for Vietnamese voice requests.
Return ONLY a compact valid JSON object: {"tool":"exact_tool_name","arguments":{...}}
To finish, return {"tool":"__stop__","arguments":{}}.
You must select ONE next tool action at a time, observe its return value, then decide again.
Available tools are defined in tools_json and restricted by the installed-app allowlist.
No other tools are allowed. Do NOT invent IDs, people, dates, amounts or addresses.
Use READ tools to resolve actual IDs, contact info, current states, events or other needed data.
Resolve relative time using context.current_time and context.timezone.
When the user asks to decrease a setting by half, inspect current state first and set half its original value.
Only execute the LATEST user request. Actions in history were already applied in environment.
If a necessary field remains unknown AFTER using provided context or available READ tools, call sys.ask_missing_info with {"tool_name":"installed_app.tool","params":["missing_field"]}. NEVER ask for a nonexistent/uninstalled tool.
Example: "book a car to the parent meeting" -> calendar.list_events first to get event address, then ride_hailing.book with context.user_profile.work_address and car4.
Example: if a hotel booking app is not installed, stop with no tool calls.
Only ask about fields you cannot resolve by available tools or context.
If user request cannot be handled by installed apps, stop immediately with ZERO calls.
For smart_home.get_status prefer {} (no room filter) unless you are certain the filter matches a canonical device room string.
For "đặt xe đi họp phụ huynh" use calendar.list_events first to find meeting place; do not guess dropoff.
If booking a ride NOW or "ngay bây giờ", omit schedule_time; do not invent payment.
For device actions, return an action to change a device after reading the actual ID.
For ride_hailing.book, get_quotes is NOT necessary when user asks directly to book (unless user explicitly asks about fares).
When user says "đặt ngay bây giờ" or "ngay" omit schedule_time, even if calendar event occurs later.
Do not add read calls for confirmation after successful writes. Stop when done.
Pay attention to conditional instructions, no-op when condition is false, multi-step dependent tools.
Don't speak to user. Don't output explanations, markdown, or reasoning. JSON ONLY."""

def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)

def load_catalog(split: str) -> dict:
    return json.loads((DATA_ROOT / split / f"{split}_tools_catalog.json").read_text(encoding="utf-8"))

def allowed_tools(item: dict, catalog: dict) -> list[dict]:
    approved = {
        t for a in item["context"]["installed_apps"].values()
        for t in a["tools"]
    } | {"sys.ask_missing_info"}
    out = []
    for app in catalog["apps"]:
        for tool in app["tools"]:
            if tool["name"] in approved:
                out.append({
                    "name": tool["name"],
                    "kind": tool.get("kind", "write"),
                    "description": tool.get("description", "")[:240],
                    "params": tool.get("parameters", {}),
                    "returns": str(tool.get("returns", ""))[:150],
                })
    return out

def json_action(raw: str) -> dict | None:
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", s)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    dec = json.JSONDecoder()
    for pos, char in enumerate(s):
        if char == "{":
            try:
                value, _ = dec.raw_decode(s[pos:])
                if isinstance(value, dict):
                    return value
            except json.JSONDecodeError:
                continue
    return None

class LocalModels:
    def __init__(self, llm_dir: Path, asr_dir: Path, adapter: Path | None = None, compact_plan: bool = False):
        from mlx_lm import load
        from mlx_lm.sample_utils import make_sampler
        import mlx_whisper

        self.asr_model = str(asr_dir)
        self.whisper = mlx_whisper
        self.model, self.tokenizer = load(str(llm_dir),adapter_path=str(adapter) if adapter else None)
        self.use_sft = bool(adapter) or compact_plan
        self.sampler = make_sampler(temp=0)
        self.audio_cache: dict[str, str] = {}

    def transcribe(self, path: Path) -> str:
        key = str(path)
        if key in self.audio_cache:
            return self.audio_cache[key]
        with wave.open(str(path), "rb") as audio:
            if (audio.getnchannels(), audio.getframerate(), audio.getsampwidth()) != (1, 16000, 2):
                raise ValueError("Expected WAV PCM16 mono 16kHz")
            pcm = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")
            samples = pcm.astype(np.float32) / 32768.0
        decoded = self.whisper.transcribe(
            samples,
            path_or_hf_repo=self.asr_model,
            language="vi",
            temperature=0.0,
            condition_on_previous_text=False,
            fp16=True,
        )
        result = decoded["text"].strip()
        self.audio_cache[key] = result
        return result

    def plan(self, prompt_content: str, max_tokens: int = 180, system: str = SYSTEM) -> str:
        from mlx_lm import generate
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt_content},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        if len(self.tokenizer.encode(prompt)) > 16384 - max_tokens:
            raise ValueError("LLM context > 16,384 token competition limit")
        return generate(
            self.model, self.tokenizer, prompt, max_tokens=max_tokens,
            sampler=self.sampler, verbose=False
        )

def history_for_prompt(item: dict, split: str, models: LocalModels, oracle: bool) -> list[dict]:
    history = []
    # Bound context; last 10 exchanges are relevant and previous calls are already applied.
    for message in item.get("history", [])[-20:]:
        role = message.get("role", "")
        if role == "user":
            transcript = message.get("audio_text") if oracle else None
            if not transcript and message.get("audio"):
                transcript = models.transcribe(DATA_ROOT / split / message["audio"])
            history.append({"role": "user", "text": transcript or ""})
        elif role == "assistant":
            calls = message.get("tool_calls", [])
            history.append({
                "role": "assistant",
                "text": message.get("text", "")[:450],
                "previous_actions_already_done": [
                    {"name": x.get("name"), "arguments": x.get("arguments", {})}
                    for x in calls[-5:]
                ],
                "results": [
                    str(v)[:350] for v in message.get("tool_results", [])[-3:]
                ],
            })
    return history

def route_apps(item: dict, transcript: str, history: list[dict], catalog: dict, models: LocalModels) -> list[str]:
    installed = item["context"]["installed_apps"]
    summary = []
    for app in catalog["apps"]:
        if app["id"] in installed and app["id"] != "sys":
            tools = [t["name"] for t in app["tools"] if t["name"] in installed[app["id"]]["tools"]]
            summary.append({
                "id": app["id"],
                "description": app.get("description", "")[:120],
                "tools": tools,
            })
    payload = {
        "request": transcript, "user_profile": item["context"].get("user_profile"),
        "latest_history": history[-4:],
        "installed_apps": summary,
    }
    system = (
        "Select ALL app IDs needed to satisfy the newest Vietnamese user request, including any prerequisite read app. "
        "Return JSON ONLY: {\"apps\":[\"first\",\"second\"]}. "
        "Do not select nonexistent apps. At most 5 IDs. "
        "A request can need calendar+ride_hailing, weather+calendar+contacts+sms, etc. "
        "Out of scope if no installed app can perform requested work: return {\"apps\":[]}. "
        "Do not include applications just because they are installed."
    )
    try:
        response = json_action(models.plan(json.dumps(payload, ensure_ascii=False), max_tokens=110, system=system))
        selected = response.get("apps") if isinstance(response, dict) else None
        if not isinstance(selected, list):
            return list(installed)
        clean = list(dict.fromkeys(x for x in selected if isinstance(x, str) and x in installed and x != "sys"))
        return clean[:5]
    except Exception:
        return list(installed)

def prompt_for(item, transcript, schemas, history, observations, issues) -> str:
    relevant = {
        "latest_user_request": transcript,
        "context": {
            "current_time": item["context"].get("current_time"),
            "timezone": item["context"].get("timezone"),
            "location": item["context"].get("location"),
            "user_profile": item["context"].get("user_profile"),
        },
        "earlier_history_already_applied": history,
        "available_tools_json": schemas,
        "current_request_tool_observations": observations,
        "prior_validation_feedback_not_executed": issues[-2:],
        "instruction": "Output the single next tool action as JSON, or __stop__ when completed or out-of-scope.",
    }
    return json.dumps(relevant, ensure_ascii=False, separators=(",", ":"))

def choose_execute(item: dict, split: str, env, models: LocalModels, catalog: dict, oracle=False) -> tuple[dict, dict]:
    transcript = (item.get("audio_text") if oracle else None) or models.transcribe(DATA_ROOT / split / item["audio"])
    previous = history_for_prompt(item, split, models, oracle=oracle)
    routed = route_apps(item, transcript, previous, catalog, models)
    lower = transcript.casefold()
    installed = item["context"]["installed_apps"]
    if "ride_hailing" in installed and "calendar" in installed:
        if any(w in lower for w in ("họp", "hẹn", "lịch", "meeting")) and any(w in lower for w in ("xe", "đón", "đến", "đi ")):
            routed = list(dict.fromkeys(["calendar", "ride_hailing"] + routed))
    if "smart_home" in installed and any(w in lower for w in ("đèn", "điều hòa", "quạt", "thiết bị", "ánh sáng", "độ sáng")):
        routed = list(dict.fromkeys(["smart_home"] + routed))
    if not routed:
        return {
            "id": item["id"], "tool_calls": env.get_tool_calls(),
            "state_hash": env.get_state_hash(),
        }, {"id": item["id"], "transcript": transcript, "calls": 0, "observed": []}
    schemas = [t for t in allowed_tools(item, catalog)
               if t["name"].split(".", 1)[0] in routed or t["name"] == "sys.ask_missing_info"]
    approved = {t["name"] for t in schemas}
    all_approved = {t["name"] for t in allowed_tools(item, catalog)}
    observations = []
    issues = []
    accepted = []
    max_actions = 7
    # Legal, task-grounded prerequisite reads; these calls are made only once.
    if "smart_home" in routed and "smart_home.get_status" in approved and any(w in lower for w in ("đèn", "điều hòa", "quạt", "thiết bị", "ánh sáng", "độ sáng")):
        read_name, read_args = "smart_home.get_status", {}
        result = env.call(read_name, read_args)
        accepted.append({"name": read_name, "arguments": read_args})
        observations.append({"name": read_name, "arguments": read_args, "result": result})
    if "calendar" in routed and "ride_hailing" in routed and "calendar.list_events" in approved and any(w in lower for w in ("họp", "hẹn", "lịch", "meeting")):
        read_name, read_args = "calendar.list_events", {}
        result = env.call(read_name, read_args)
        accepted.append({"name": read_name, "arguments": read_args})
        observations.append({"name": read_name, "arguments": read_args, "result": result})
    for turn in range(max_actions - len(accepted)):
        raw = models.plan(prompt_for(item, transcript, schemas, previous, observations, issues))
        action = json_action(raw)
        if not isinstance(action, dict):
            issues.append({"model_response": raw[:180], "error": "Invalid JSON"})
            if len(issues) >= 3:
                break
            continue
        tool = action.get("tool")
        args = action.get("arguments", {})
        if tool in ("__stop__", "stop", "done", None):
            break
        if not isinstance(args, dict) or tool not in approved:
            issues.append({"tool": str(tool), "error": "Tool unavailable or arguments not object"})
            if len(issues) >= 3:
                break
            continue
        if tool == "sys.ask_missing_info":
            intended = args.get("tool_name") if isinstance(args, dict) else None
            missing = args.get("params", []) if isinstance(args, dict) else []
            if intended not in all_approved or intended == "sys.ask_missing_info" or not missing:
                issues.append({"tool": tool, "error": "Cannot request missing info for unavailable tool; STOP or use legal read tools"})
                if len(issues) >= 3:
                    break
                continue
        if any(a["name"] == tool and a["arguments"] == args for a in accepted):
            issues.append({"tool": tool, "error": "Identical call already executed. Stop or choose next necessary action."})
            if len(issues) >= 3:
                break
            continue
        # Deterministic guardrails on top of probabilistic planning: no extra calls.
        if tool == "ride_hailing.get_quotes" and any(w in lower for w in ("đặt", "gọi", "book")) and not any(w in lower for w in ("giá", "bao nhiêu", "ước tính", "báo giá")):
            issues.append({"tool": tool, "error": "Ride quote is redundant; book now using already known locations"})
            if len(issues) >= 3:
                break
            continue
        if tool == "ride_hailing.book":
            if any(w in lower for w in ("ngay", "bây giờ", "lúc này")):
                args.pop("schedule_time", None)
            if not any(w in lower for w in ("thanh toán", "tiền mặt", "ví", "thẻ", "payment")):
                args.pop("payment", None)
        if tool == "smart_home.light_set":
            if any(w in lower for w in ("độ sáng", "ánh sáng", "sáng lên", "sáng xuống", "một nửa")) and not any(w in lower for w in ("bật", "tắt", "mở đèn")):
                args.pop("power", None)
        status = env.validate(tool, args)
        if not status.get("ok"):
            issues.append({"tool": tool, "arguments": args, "error": status.get("error")})
            if len(issues) >= 3:
                break
            continue
        result = env.call(tool, args)
        accepted.append({"name": tool, "arguments": args})
        observations.append({
            "name": tool,
            "arguments": args,
            "result": result,
        })
        if tool == "sys.ask_missing_info":
            break
        if not result.get("ok"):
            issues.append({"tool": tool, "error": result.get("error")})
            break
    return {
        "id": item["id"],
        "tool_calls": env.get_tool_calls(),
        "state_hash": env.get_state_hash(),
    }, {"id": item["id"], "transcript": transcript, "calls": len(env.get_tool_calls()), "observed": observations}

def run(args):
    random.seed(42)
    np.random.seed(42)
    import mlx.core as mx
    mx.random.seed(42)
    sys.path.insert(0, str(DATA_ROOT / args.split))
    from zaic_tools import ToolEnv

    if args.oracle and args.split != "train":
        raise ValueError("Oracle transcripts permitted only for supervised train")
    input_file = DATA_ROOT / args.split / f"{args.split}.jsonl"
    catalog = load_catalog(args.split)
    records = read_jsonl(input_file)
    wanted = set(args.ids.split(",")) if args.ids else None
    if wanted is not None:
        records = (r for r in records if r["id"] in wanted)
    output = Path(args.output)
    output.parent.mkdir(exist_ok=True, parents=True)
    processed = set()
    if args.resume and output.exists():
        for r in read_jsonl(output):
            processed.add(r["id"])
    models = LocalModels(Path(args.llm), Path(args.asr))
    start = time.monotonic()
    sample_times = []
    count = 0
    errs = 0
    diagnostics = Path(str(output) + ".progress.jsonl")
    file_mode = "a" if args.resume else "w"
    with output.open(file_mode, encoding="utf-8") as writer, diagnostics.open(file_mode, encoding="utf-8") as diag:
        for item in records:
            if item["id"] in processed:
                continue
            t0 = time.monotonic()
            # ONE environment per processed sample. Never replay/reset.
            env = ToolEnv(args.split, item["id"], root=DATA_ROOT)
            try:
                submission, details = choose_execute(item, args.split, env, models, catalog, oracle=args.oracle)
            except Exception as exc:
                # The env's logged calls and actual hash still form a faithful submission.
                errs += 1
                submission = {"id": item["id"], "tool_calls": env.get_tool_calls(), "state_hash": env.get_state_hash()}
                details = {"id": item["id"], "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc(limit=3)}
            seconds = time.monotonic()-t0
            sample_times.append(seconds)
            diag.write(json.dumps({"id": item["id"], "seconds": round(seconds,3), "calls": len(submission["tool_calls"]), "error": details.get("error")}, ensure_ascii=False)+"\n")
            diag.flush()
            writer.write(json.dumps(submission, ensure_ascii=False)+"\n")
            writer.flush()
            count += 1
            if count <= 4 or count % args.log_every == 0:
                print(json.dumps({
                    "completed_this_run": count,
                    "sample_id": item["id"],
                    "seconds": round(seconds,2),
                    "num_calls": len(submission["tool_calls"]),
                    "failures": errs,
                    "elapsed": round(time.monotonic()-start,1),
                }, ensure_ascii=False), flush=True)
            if args.limit and count >= args.limit:
                break
    print(json.dumps({"done": count,"errors":errs,"output":str(output),"mean_seconds":round(sum(sample_times)/len(sample_times),3) if sample_times else None},ensure_ascii=False),flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",choices=["train","public_test"],default="train")
    parser.add_argument("--oracle",action="store_true",help="Use audio_text supervision (TRAIN ONLY)")
    parser.add_argument("--ids",default="")
    parser.add_argument("--output",default=str(PROJECT/"runs/baseline_submission.jsonl"))
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--limit",type=int,default=0)
    parser.add_argument("--log-every",type=int,default=10)
    parser.add_argument("--llm",default=str(DEFAULT_LLM))
    parser.add_argument("--asr",default=str(DEFAULT_ASR))
    run(parser.parse_args())
