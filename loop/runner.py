"""A deliberately small, synchronous, deterministic ReAct loop."""

from __future__ import annotations

import copy
import hashlib
import json
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np
import torch

from server.rendering import MODEL_ID, MODEL_REVISION

from .tasks import SYSTEM_PROMPT, Task
from .tools import TOOL_SCHEMAS, ToolError


class ChatClient(Protocol):
    def complete(self, request: Mapping[str, Any]) -> dict[str, Any]: ...


class ChatClientError(RuntimeError):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"chat server returned HTTP {status}: {body}")
        self.status = status
        self.body = body


class OpenAIHTTPClient:
    def __init__(self, base_url: str) -> None:
        self.url = base_url.rstrip("/") + "/v1/chat/completions"

    def complete(self, request: Mapping[str, Any]) -> dict[str, Any]:
        wire = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode()
        http_request = urllib.request.Request(
            self.url,
            data=wire,
            headers={"content-type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(http_request) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            raise ChatClientError(exc.code, detail) from exc


@dataclass(frozen=True)
class LoopConfig:
    seed: int = 0
    model_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION
    direction_revision: str | None = None
    intervention: dict[str, Any] = field(default_factory=lambda: {"enabled": False})
    max_steps: int = 8
    max_tokens: int = 256
    run_kind: str = "primary"

    def __post_init__(self) -> None:
        if self.max_steps < 1 or self.max_tokens < 1:
            raise ValueError("max_steps and max_tokens must be positive")


@dataclass(frozen=True)
class RunResult:
    run_id: str
    success: bool
    stop_reason: str
    events: tuple[dict[str, Any], ...]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _run_id(task: Task, config: LoopConfig) -> str:
    identity = {
        "task_id": task.task_id,
        "seed": config.seed,
        "model_revision": config.model_revision,
        "direction_revision": config.direction_revision,
        "intervention": config.intervention,
        "run_kind": config.run_kind,
    }
    return "run_" + hashlib.sha256(_canonical_json(identity).encode()).hexdigest()[:20]


def seed_everything(seed: int) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    return {
        "python": seed,
        "numpy": seed % (2**32),
        "torch": seed,
        "generation_api": seed,
        "deterministic_algorithms": True,
        "greedy_decoding": True,
        "sampling": False,
        "retries": False,
        "compaction": False,
        "caching": False,
        "batching": False,
        "concurrency": False,
    }


def _tool_call(response: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]] | None:
    try:
        message = response["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("chat response has no choices[0].message") from exc
    calls = message.get("tool_calls") or []
    if not calls:
        return None
    if len(calls) != 1:
        raise ValueError("the minimal loop accepts exactly one tool call per response")
    call = calls[0]
    try:
        function = call["function"]
        arguments = json.loads(function["arguments"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("tool call has invalid function arguments") from exc
    if not isinstance(arguments, dict):
        raise ValueError("tool call arguments must decode to an object")
    return copy.deepcopy(call), arguments


def run_task(client: ChatClient, task: Task, config: LoopConfig) -> RunResult:
    seeds = seed_everything(config.seed)
    run_id = _run_id(task, config)
    environment = task.environment()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task.objective},
    ]
    events: list[dict[str, Any]] = [
        {
            "event": "run_start",
            "task_id": task.task_id,
            "run_id": run_id,
            "seed": config.seed,
            "run_kind": config.run_kind,
            "model": {"id": config.model_id, "revision": config.model_revision},
            "direction_revision": config.direction_revision,
            "intervention": copy.deepcopy(config.intervention),
            "determinism": seeds,
            "excluded_timing_fields": [
                "model_step.latency_ms",
                "model_step.response.created",
                "model_error.latency_ms",
            ],
            "initial_environment": environment.snapshot(),
        }
    ]
    success = False
    stop_reason = "max_steps"
    for step in range(config.max_steps):
        request = {
            "model": config.model_id,
            "messages": copy.deepcopy(messages),
            "tools": copy.deepcopy(TOOL_SCHEMAS),
            "stream": False,
            "temperature": 0.0,
            "max_tokens": config.max_tokens,
            "seed": config.seed,
            "use_cache": False,
            "intervention": copy.deepcopy(config.intervention),
        }
        started = time.perf_counter_ns()
        try:
            response = client.complete(request)
        except Exception as exc:
            latency_ms = (time.perf_counter_ns() - started) / 1_000_000
            error: dict[str, Any] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            if isinstance(exc, ChatClientError):
                error.update({"http_status": exc.status, "response_body": exc.body})
            events.append(
                {
                    "event": "model_error",
                    "step": step,
                    "request": request,
                    "error": error,
                    "latency_ms": latency_ms,
                    "environment": environment.snapshot(),
                }
            )
            stop_reason = "model_error"
            break
        latency_ms = (time.perf_counter_ns() - started) / 1_000_000
        telemetry = response.get("owie", {})
        events.append(
            {
                "event": "model_step",
                "step": step,
                "request": request,
                "response": copy.deepcopy(response),
                "rendered_prompt_hash": telemetry.get("rendered_prompt_hash"),
                "model_revision": telemetry.get("model_revision", config.model_revision),
                "direction_revision": config.direction_revision,
                "intervention": copy.deepcopy(config.intervention),
                "token_counts": copy.deepcopy(response.get("usage", {})),
                "latency_ms": latency_ms,
            }
        )
        message = copy.deepcopy(response["choices"][0]["message"])
        messages.append(message)
        parsed_call = _tool_call(response)
        if parsed_call is None:
            stop_reason = "assistant_stop"
            break
        call, arguments = parsed_call
        name = call["function"]["name"]
        try:
            execution = environment.execute(name, arguments)
            output = execution.output
            error = None
            before, after = execution.state_before, execution.state_after
        except ToolError as exc:
            output = {"ok": False, "error": str(exc)}
            error = str(exc)
            before = after = environment.snapshot()
        events.append(
            {
                "event": "tool_step",
                "step": step,
                "tool_call_id": call.get("id"),
                "tool_name": name,
                "tool_input": arguments,
                "tool_output": output,
                "tool_error": error,
                "environment_before": before,
                "environment_after": after,
            }
        )
        messages.append(
            {
                "role": "tool",
                "name": name,
                "tool_call_id": call.get("id"),
                "content": _canonical_json(output),
            }
        )
        check = task.predicate.evaluate(environment)
        events.append({"event": "success_check", "step": step, **check})
        if check["passed"]:
            success = True
            stop_reason = "success_predicate"
            break
    final_check = task.predicate.evaluate(environment)
    success = success or final_check["passed"]
    events.append(
        {
            "event": "run_end",
            "task_id": task.task_id,
            "run_id": run_id,
            "success": success,
            "stop_reason": stop_reason,
            "success_check": final_check,
            "final_environment": environment.snapshot(),
        }
    )
    return RunResult(run_id, success, stop_reason, tuple(events))


def write_trajectory(path: Path, result: RunResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for event in result.events:
            stream.write(_canonical_json(event) + "\n")


def canonical_trajectory(path: Path) -> bytes:
    """Remove only the timing fields documented in ``run_start``."""
    normalized: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            if event.get("event") in {"model_step", "model_error"}:
                event.pop("latency_ms", None)
            if event.get("event") == "model_step":
                response = event.get("response")
                if isinstance(response, dict):
                    response.pop("created", None)
            normalized.append(event)
    return ("\n".join(_canonical_json(event) for event in normalized) + "\n").encode()


def trajectories_match(first: Path, second: Path) -> bool:
    return canonical_trajectory(first) == canonical_trajectory(second)


def write_determinism_report(path: Path, first: Path, second: Path) -> dict[str, Any]:
    raw_hashes = [hashlib.sha256(item.read_bytes()).hexdigest() for item in (first, second)]
    normalized_hashes = [
        hashlib.sha256(canonical_trajectory(item)).hexdigest() for item in (first, second)
    ]
    report = {
        "schema_version": 1,
        "trajectories": [first.name, second.name],
        "raw_sha256": raw_hashes,
        "normalized_sha256": normalized_hashes,
        "excluded_timing_fields": [
            "model_step.latency_ms",
            "model_step.response.created",
            "model_error.latency_ms",
        ],
        "matched": normalized_hashes[0] == normalized_hashes[1],
    }
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(_canonical_json(report) + "\n")
    return report
