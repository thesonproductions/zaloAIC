"""Google ADK adapters for the official per-sample zaic_tools runtime."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from google.adk import Runner
from google.adk.agents import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService, Session
from google.adk.tools import BaseTool
from google.genai import types


AGENT_INSTRUCTION = """You are LifeMate, a Vietnamese voice-assistant benchmark agent.
Act only on the latest user request. Tools exposed to you are the complete allowlist for this sample.
Use read tools to resolve real IDs and current values; never invent IDs, contacts, times, amounts, or addresses.
Use the supplied time, timezone, location, user profile, and conversation history when relevant.
Actions in the history have already been applied: use their results as evidence and do not repeat them.
For a multi-step task, call one tool at a time, inspect its result, then continue until all requested actions are complete.
Do not make optional changes or extra confirmation/read calls. A read-only request must not cause writes.
If a critical required value cannot be found in context, history, or available read tools, use sys.ask_missing_info
when it is available. If no installed tool can perform the task, make no tool calls.
Do not address the user or narrate your reasoning; the submission is determined only by tool calls and final state."""


def _adk_schema(schema: dict[str, Any]) -> types.Schema:
    """Translate the catalog's JSON schema subset into GenAI's supported Schema model."""
    supported = set(types.Schema.model_fields)
    aliases = {
        field.alias: name
        for name, field in types.Schema.model_fields.items()
        if field.alias
    }
    converted: dict[str, Any] = {}
    for key, value in schema.items():
        field = aliases.get(key, key)
        if field not in supported or key.startswith("x-"):
            continue
        if field == "type" and isinstance(value, str):
            value = value.upper()
        elif field == "properties" and isinstance(value, dict):
            value = {name: _adk_schema(prop) for name, prop in value.items()}
        elif field == "items" and isinstance(value, dict):
            value = _adk_schema(value)
        elif field == "additional_properties" and isinstance(value, dict):
            value = _adk_schema(value)
        elif field == "any_of" and isinstance(value, list):
            value = [_adk_schema(option) for option in value]
        elif field == "defs" and isinstance(value, dict):
            value = {name: _adk_schema(definition) for name, definition in value.items()}
        converted[field] = value
    return types.Schema.model_validate(converted)


def _tool_description(spec: dict[str, Any]) -> str:
    description = str(spec.get("description", "")).strip()
    groups = (spec.get("parameters") or {}).get("x-one-of") or []
    if groups:
        choices = [" + ".join(group) for group in groups]
        description += f" At least one complete argument group is required: {' OR '.join(choices)}."
    returns = spec.get("returns")
    if returns:
        description += f" Returns: {returns}."
    return f"Official tool {spec['name']}. {description}".strip()


class SimulatorTool(BaseTool):
    """An ADK function tool bound to one official sample session."""

    def __init__(self, spec: dict[str, Any], env: Any):
        self.tool_name = spec["name"]
        slug = re.sub(r"[^A-Za-z0-9_-]", "_", self.tool_name)
        suffix = hashlib.sha1(self.tool_name.encode("utf-8")).hexdigest()[:8]
        self._spec = spec
        self._env = env
        super().__init__(
            name=f"tool_{slug}_{suffix}",
            description=_tool_description(spec),
        )

    def _get_declaration(self) -> types.FunctionDeclaration:
        parameters = self._spec.get("parameters") or {"type": "object"}
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=_adk_schema(parameters),
        )

    async def run_async(
        self, *, args: dict[str, Any], tool_context: Any
    ) -> dict[str, Any]:
        return self._env.call(self.tool_name, args)


@dataclass
class SampleRuntime:
    """ADK runner plus its one authoritative zaic_tools session."""

    runner: Runner
    env: Any
    adk_session: Session
    tools: tuple[SimulatorTool, ...]
    user_id: str

    async def run_text(self, text: str, *, max_llm_calls: int = 20) -> None:
        """Run one user turn; all tool calls mutate this runtime's same zaic session."""
        message = types.Content(role="user", parts=[types.Part(text=text)])
        async for _ in self.runner.run_async(
            user_id=self.user_id,
            session_id=self.adk_session.id,
            new_message=message,
            run_config=RunConfig(max_llm_calls=max_llm_calls),
        ):
            pass

    def get_tool_calls(self) -> list[dict[str, Any]]:
        return self.env.get_tool_calls()

    def get_state(self) -> dict[str, Any]:
        return self.env.get_state()

    def get_state_hash(self) -> str:
        return self.env.get_state_hash()

    def submission_record(self, sample_id: str) -> dict[str, Any]:
        return {
            "id": sample_id,
            "tool_calls": self.get_tool_calls(),
            "state_hash": self.get_state_hash(),
        }


def _resolve_split(data_root: str | Path, split: str) -> tuple[Path, Path]:
    root = Path(data_root).expanduser()
    for candidate in (root, root / "train+public_test", root.parent):
        split_dir = candidate / split
        if (split_dir / f"{split}_tools_catalog.json").is_file():
            return candidate.resolve(), split_dir.resolve()
    raise FileNotFoundError(
        f"Could not find {split}/{split}_tools_catalog.json under {root}"
    )


def _load_runtime(split_dir: Path):
    package_dir = (split_dir / "zaic_tools").resolve()
    package_file = package_dir / "__init__.py"
    if not package_file.is_file():
        raise FileNotFoundError(f"Missing official tool runtime: {package_file}")
    package_name = f"_zaic_tools_runtime_{hashlib.sha1(str(package_dir).encode()).hexdigest()[:12]}"
    package = sys.modules.get(package_name)
    if package is None:
        spec = importlib.util.spec_from_file_location(
            package_name,
            package_file,
            submodule_search_locations=[str(package_dir)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load official tool runtime from {package_dir}")
        package = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = package
        spec.loader.exec_module(package)
    return package.ToolEnv, package.TOOLS


def _sample_tools(
    item: dict[str, Any],
    catalog: dict[str, Any],
    implemented: set[str],
) -> list[dict[str, Any]]:
    allowed = {
        name
        for app in item["context"]["installed_apps"].values()
        for name in app.get("tools", [])
    } | {"sys.ask_missing_info"}
    specs = [
        tool
        for app in catalog.get("apps", [])
        for tool in app.get("tools", [])
        if tool.get("name") in allowed
    ]
    missing = sorted(spec["name"] for spec in specs if spec["name"] not in implemented)
    if missing:
        raise RuntimeError(
            f"zaic_tools does not implement tools installed for {item['id']}: "
            + ", ".join(missing)
        )
    return specs


async def create_sample_runtime(
    *,
    split: str,
    sample_id: str,
    data_root: str | Path,
    user_id: str | None = None,
    session_id: str | None = None,
    app_name: str = "lifemate_zac_benchmark",
    model: Any | None = None,
    model_name: str | None = None,
    api_base: str | None = None,
    sample: dict[str, Any] | None = None,
) -> SampleRuntime:
    """Open one ToolEnv sample and bind every installed catalog tool to its session.

    The ToolEnv session remains authoritative for app state and the submission
    call log. The ADK session stores conversation turns separately.
    """
    data_root, split_dir = _resolve_split(data_root, split)
    tool_env_factory, implemented = _load_runtime(split_dir)
    catalog_path = split_dir / f"{split}_tools_catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    sample_path = split_dir / f"{split}.jsonl"
    if sample is None:
        if not sample_path.is_file():
            raise FileNotFoundError(f"Missing sample file: {sample_path}")
        with sample_path.open(encoding="utf-8") as samples:
            for line in samples:
                if line.strip():
                    candidate = json.loads(line)
                    if candidate.get("id") == sample_id:
                        sample = candidate
                        break
    if sample is None:
        raise KeyError(f"unknown sample id {sample_id!r} in split {split!r}")
    if sample.get("id") != sample_id:
        raise ValueError(f"sample id {sample.get('id')!r} does not match {sample_id!r}")

    env = tool_env_factory(split, sample_id, root=data_root)
    tools = tuple(
        SimulatorTool(spec, env)
        for spec in _sample_tools(sample, catalog, implemented)
    )
    if model is None:
        selected_model = model_name or os.getenv("MODEL_NAME", "ollama_chat/qwen3:0.6b")
        selected_api_base = api_base or os.getenv(
            "OLLAMA_HOST", "http://localhost:11434"
        )
        model = LiteLlm(model=selected_model, api_base=selected_api_base)

    agent = LlmAgent(
        name="LifeMate",
        model=model,
        instruction=AGENT_INSTRUCTION,
        tools=list(tools),
    )
    user_id = user_id or f"zac-{sample_id}"
    session_id = session_id or sample_id
    session_service = InMemorySessionService()
    adk_session = await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        state={"zaic_split": split, "zaic_sample_id": sample_id},
    )
    runner = Runner(
        agent=agent,
        app_name=app_name,
        session_service=session_service,
    )
    return SampleRuntime(
        runner=runner,
        env=env,
        adk_session=adk_session,
        tools=tools,
        user_id=user_id,
    )
