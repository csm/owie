"""Measure Checkpoint 5 retain and safety baselines at tool-result prefixes."""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Iterable, Mapping

from directions.bundle import current_git_revision
from directions.model import ModelHandle, load_model
from evals import metrics
from evals.schema import DATA_ROOT, hash_file, load_retain_set, load_safety_set
from loop.runner import seed_everything
from replay.prefixes import verify_prefix_manifest


@dataclass(frozen=True)
class CalibrationConfig:
    seed: int = 0
    max_new_tokens_safety: int = 96
    checkpoint_budget_hours: float = 144.0
    prior_checkpoint_hours: float = 0.5

    def __post_init__(self) -> None:
        if self.max_new_tokens_safety < 1:
            raise ValueError("max_new_tokens_safety must be positive")
        if self.checkpoint_budget_hours <= 0.0:
            raise ValueError("checkpoint_budget_hours must be positive")
        if not 0.0 <= self.prior_checkpoint_hours < self.checkpoint_budget_hours:
            raise ValueError("prior_checkpoint_hours must be within the checkpoint budget")


def _dependencies() -> dict[str, str]:
    packages: dict[str, str] = {}
    for name in ("torch", "transformers", "tokenizers", "safetensors", "numpy"):
        try:
            packages[name] = version(name)
        except Exception:  # pragma: no cover - absence is recorded
            packages[name] = "unavailable"
    return packages


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _completed(path: Path) -> set[tuple[str, str]]:
    if not path.is_file():
        return set()
    completed: set[tuple[str, str]] = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
                completed.add((row["metric"], row["item_id"]))
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    return completed


def _append(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        stream.write("\n")


def _write_json(path: Path, value: Mapping[str, Any], *, replace: bool) -> None:
    mode = "w" if replace else "x"
    with path.open(mode, encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def _rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def summarize_records(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    records = list(rows)
    retain = [row for row in records if row.get("metric") == "retain"]
    safety = [row for row in records if row.get("metric") == "safety"]

    def perplexity(items: list[Mapping[str, Any]]) -> float:
        tokens = sum(int(item["tokens"]) for item in items)
        if tokens == 0:
            raise ValueError("retain summary needs at least one scored token")
        return math.exp(-sum(float(item["logprob"]) for item in items) / tokens)

    dependent = [row for row in retain if row["tool_dependent"]]
    independent = [row for row in retain if not row["tool_dependent"]]
    user = [row for row in safety if row["channel"] == "user"]
    tool = [row for row in safety if row["channel"] == "tool"]
    if not retain or not safety or not dependent or not independent or not user or not tool:
        raise ValueError("calibration summary requires complete retain and safety groups")

    def rate(items: list[Mapping[str, Any]], key: str) -> float:
        return sum(bool(item[key]) for item in items) / len(items)

    return {
        "retain": {
            "items": len(retain),
            "tokens": sum(int(row["tokens"]) for row in retain),
            "perplexity": perplexity(retain),
            "tool_dependent_perplexity": perplexity(dependent),
            "tool_independent_perplexity": perplexity(independent),
        },
        "safety": {
            "items": len(safety),
            "compliance_rate": rate(safety, "complied"),
            "user_compliance_rate": rate(user, "complied"),
            "tool_compliance_rate": rate(tool, "complied"),
            "complied_in_text_rate": rate(safety, "complied_in_text"),
            "tool_call_rate": rate(safety, "emitted_tool_call"),
        },
    }


def run_calibration(
    run_dir: Path,
    prefix_manifest_path: Path,
    config: CalibrationConfig,
    *,
    handle: ModelHandle | None = None,
    data_root: Path = DATA_ROOT,
    local_files_only: bool = False,
) -> Path:
    """Run or resume the complete no-intervention calibration cell."""

    started = time.monotonic()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"
    manifest_path = run_dir / "manifest.json"
    status_path = run_dir / "status.json"
    summary_path = run_dir / "summary.json"

    prefix_manifest_path = Path(prefix_manifest_path)
    prefix_manifest = _read_json(prefix_manifest_path)
    verify_prefix_manifest(prefix_manifest)
    data_root = Path(data_root)
    dataset_hashes = {
        "retain": hash_file(data_root / "retain.jsonl"),
        "safety": hash_file(data_root / "safety.jsonl"),
    }
    retain_items = load_retain_set(root=data_root)
    safety_items = load_safety_set(root=data_root)

    if handle is None:
        handle = load_model(local_files_only=local_files_only)
    seeds = seed_everything(config.seed)
    manifest = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "git_revision": current_git_revision(),
        "model": handle.describe(),
        "config": asdict(config),
        "determinism": seeds,
        "prefix_manifest": {
            "path": str(prefix_manifest_path),
            "manifest_hash": prefix_manifest["manifest_hash"],
            "task_set_hash": prefix_manifest["task_set_hash"],
        },
        "dataset_hashes": dataset_hashes,
        "dependencies": _dependencies(),
        "python": sys.version,
        "platform": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "intervention": {"enabled": False},
    }
    if manifest_path.is_file():
        if _read_json(manifest_path) != manifest:
            raise ValueError("existing calibration manifest does not match this run")
    else:
        _write_json(manifest_path, manifest, replace=False)

    previous_session_hours = 0.0
    if status_path.is_file():
        previous_session_hours = float(
            _read_json(status_path).get("calibration_hours", 0.0)
        )

    def cumulative_hours() -> float:
        return (
            config.prior_checkpoint_hours
            + previous_session_hours
            + (time.monotonic() - started) / 3600.0
        )

    completed = _completed(results_path)
    stopped_for_budget = False
    work = [
        ("retain", item.item_id, item)
        for item in retain_items
    ] + [
        ("safety", item.item_id, item)
        for item in safety_items
    ]
    for metric_name, item_id, item in work:
        if (metric_name, item_id) in completed:
            continue
        if cumulative_hours() >= config.checkpoint_budget_hours:
            stopped_for_budget = True
            break
        item_started = time.monotonic()
        if metric_name == "retain":
            record = metrics.score_retain(handle, [item])[0]
        else:
            record = metrics.score_safety(
                handle,
                [item],
                max_new_tokens=config.max_new_tokens_safety,
            )[0]
        _append(
            results_path,
            {
                "metric": metric_name,
                **record,
                "seconds": time.monotonic() - item_started,
            },
        )
        completed.add((metric_name, item_id))
        _write_json(
            status_path,
            {
                "complete": False,
                "stopped_for_budget": False,
                "completed_items": len(completed),
                "total_items": len(work),
                "prior_checkpoint_hours": config.prior_checkpoint_hours,
                "calibration_hours": previous_session_hours
                + (time.monotonic() - started) / 3600.0,
                "cumulative_checkpoint_hours": cumulative_hours(),
                "checkpoint_budget_hours": config.checkpoint_budget_hours,
            },
            replace=True,
        )

    complete = len(completed) == len(work)
    summary: dict[str, Any] | None = None
    if complete:
        summary = summarize_records(_rows(results_path))
        if summary_path.is_file():
            if _read_json(summary_path) != summary:
                raise ValueError("existing calibration summary does not match raw results")
        else:
            _write_json(summary_path, summary, replace=False)
    _write_json(
        status_path,
        {
            "complete": complete,
            "stopped_for_budget": stopped_for_budget,
            "completed_items": len(completed),
            "total_items": len(work),
            "prior_checkpoint_hours": config.prior_checkpoint_hours,
            "calibration_hours": previous_session_hours
            + (time.monotonic() - started) / 3600.0,
            "cumulative_checkpoint_hours": cumulative_hours(),
            "checkpoint_budget_hours": config.checkpoint_budget_hours,
        },
        replace=True,
    )
    if summary is not None:
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return results_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--prefix-manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens-safety", type=int, default=96)
    parser.add_argument("--budget-hours", type=float, default=144.0)
    parser.add_argument("--prior-hours", type=float, default=0.5)
    parser.add_argument("--device")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)
    handle = load_model(device=args.device, local_files_only=args.local_files_only)
    config = CalibrationConfig(
        seed=args.seed,
        max_new_tokens_safety=args.max_new_tokens_safety,
        checkpoint_budget_hours=args.budget_hours,
        prior_checkpoint_hours=args.prior_hours,
    )
    run_calibration(
        args.run_dir,
        args.prefix_manifest,
        config,
        handle=handle,
        local_files_only=args.local_files_only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
