"""Build immutable replay-prefix manifests from loop trajectories."""

from __future__ import annotations

import copy
import hashlib
import json
import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from loop.tasks import TASK_BY_ID, TASK_SET_HASH


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode())


@dataclass(frozen=True)
class ReplayPrefix:
    """One exact model request and the continuation observed after it."""

    prefix_id: str
    task_id: str
    task_hash: str
    task_set_hash: str
    injection: bool
    step: int
    request_hash: str
    request: dict[str, Any]
    rendered_prompt_hash: str
    baseline_message: dict[str, Any]
    environment: dict[str, Any]
    source_path: str
    source_sha256: str


def _read_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(event, dict):
                raise ValueError(f"{path}:{line_number}: event must be an object")
            events.append(event)
    if not events:
        raise ValueError(f"{path}: trajectory is empty")
    return events


def _single_event(events: Iterable[Mapping[str, Any]], kind: str, path: Path) -> Mapping[str, Any]:
    matches = [event for event in events if event.get("event") == kind]
    if len(matches) != 1:
        raise ValueError(f"{path}: expected one {kind!r} event, found {len(matches)}")
    return matches[0]


def _environment_before_step(events: list[dict[str, Any]], step: int) -> dict[str, Any]:
    environment: dict[str, Any] | None = None
    for event in events:
        if event.get("event") == "run_start":
            environment = copy.deepcopy(event["initial_environment"])
        elif event.get("event") == "setup_tool_step":
            environment = copy.deepcopy(event["environment_after"])
        elif event.get("event") == "tool_step" and int(event["step"]) < step:
            environment = copy.deepcopy(event["environment_after"])
    if environment is None:
        raise ValueError("trajectory has no environment before the model step")
    return environment


def load_trajectory_prefixes(
    path: Path,
    *,
    require_candidate_set: bool = True,
    require_neutral: bool = True,
) -> tuple[ReplayPrefix, ...]:
    """Load every model-step boundary from one completed raw trajectory."""

    path = Path(path)
    raw = path.read_bytes()
    source_hash = _sha256_bytes(raw)
    events = _read_events(path)
    start = _single_event(events, "run_start", path)
    _single_event(events, "run_end", path)

    task_id = start.get("task_id")
    task_hash = start.get("task_hash")
    task_set_hash = start.get("candidate_task_set_hash")
    if not all(isinstance(value, str) and value for value in (task_id, task_hash, task_set_hash)):
        raise ValueError(f"{path}: run_start lacks task or task-set identity")
    if require_candidate_set and task_set_hash != TASK_SET_HASH:
        raise ValueError(
            f"{path}: task-set hash {task_set_hash!r} does not match {TASK_SET_HASH!r}"
        )
    if task_id not in TASK_BY_ID:
        raise ValueError(f"{path}: unknown candidate task {task_id!r}")
    if require_neutral:
        if start.get("prompt_defense") is not False:
            raise ValueError(f"{path}: neutral prefixes require prompt_defense=false")
        intervention = start.get("intervention")
        if not isinstance(intervention, dict) or intervention.get("enabled") is not False:
            raise ValueError(f"{path}: neutral prefixes require intervention.enabled=false")

    prefixes: list[ReplayPrefix] = []
    for event in events:
        if event.get("event") != "model_step":
            continue
        step = event.get("step")
        request = event.get("request")
        rendered_hash = event.get("rendered_prompt_hash")
        try:
            baseline_message = event["response"]["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"{path}: model step {step!r} lacks a baseline message") from exc
        if not isinstance(step, int) or step < 0:
            raise ValueError(f"{path}: model step must be a non-negative integer")
        if not isinstance(request, dict):
            raise ValueError(f"{path}: model step {step} request must be an object")
        if not isinstance(rendered_hash, str) or not rendered_hash.startswith("sha256:"):
            raise ValueError(f"{path}: model step {step} lacks a rendered prompt hash")
        request_hash = _sha256_json(request)
        prefix_id = "prefix_" + hashlib.sha256(
            f"{task_hash}\0{step}\0{request_hash}".encode()
        ).hexdigest()[:20]
        prefixes.append(
            ReplayPrefix(
                prefix_id=prefix_id,
                task_id=task_id,
                task_hash=task_hash,
                task_set_hash=task_set_hash,
                injection=TASK_BY_ID[task_id].injection,
                step=step,
                request_hash=request_hash,
                request=copy.deepcopy(request),
                rendered_prompt_hash=rendered_hash,
                baseline_message=copy.deepcopy(baseline_message),
                environment=_environment_before_step(events, step),
                source_path=str(path),
                source_sha256=source_hash,
            )
        )
    if not prefixes:
        raise ValueError(f"{path}: trajectory has no model steps")
    return tuple(prefixes)


def build_prefix_manifest(paths: Iterable[Path]) -> dict[str, Any]:
    """Build a deterministic manifest for a complete neutral baseline set."""

    resolved = sorted({Path(path) for path in paths}, key=lambda path: str(path))
    if not resolved:
        raise ValueError("at least one trajectory is required")
    prefixes = [
        prefix
        for path in resolved
        for prefix in load_trajectory_prefixes(path)
    ]
    task_ids = {prefix.task_id for prefix in prefixes}
    missing = sorted(set(TASK_BY_ID) - task_ids)
    extra = sorted(task_ids - set(TASK_BY_ID))
    if missing or extra:
        raise ValueError(
            f"trajectory set must cover every candidate task; missing={missing}, extra={extra}"
        )
    prefix_ids = [prefix.prefix_id for prefix in prefixes]
    if len(prefix_ids) != len(set(prefix_ids)):
        raise ValueError("trajectory set contains duplicate replay prefixes")

    model_pairs = {
        (prefix.request.get("model"), prefix.request.get("seed"))
        for prefix in prefixes
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "task_set_hash": TASK_SET_HASH,
        "models_and_seeds": [
            {"model": model, "seed": seed}
            for model, seed in sorted(model_pairs, key=lambda item: (str(item[0]), str(item[1])))
        ],
        "trajectory_count": len(resolved),
        "prefix_count": len(prefixes),
        "injection_prefix_count": sum(prefix.injection for prefix in prefixes),
        "prefixes": [asdict(prefix) for prefix in prefixes],
    }
    manifest["manifest_hash"] = _sha256_json(manifest)
    return manifest


def write_prefix_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    """Write a new manifest without replacing an existing artifact."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("trajectories", nargs="+", type=Path)
    args = parser.parse_args(argv)
    manifest = build_prefix_manifest(args.trajectories)
    write_prefix_manifest(args.output, manifest)
    print(
        f"{args.output}: trajectories={manifest['trajectory_count']} "
        f"prefixes={manifest['prefix_count']} hash={manifest['manifest_hash']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
