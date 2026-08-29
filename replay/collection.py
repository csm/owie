"""Resumable primary-prefix collection with a cumulative budget stop."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from directions.bundle import current_git_revision
from loop.runner import ChatClient, OpenAIHTTPClient
from loop.tasks import TASK_BY_ID
from replay.arms import FROZEN_ARMS, FROZEN_SEEDS, frozen_arm_hash
from replay.prefixes import load_trajectory_prefixes
from replay.runner import Continuation, ReplayArm, ReplayConfig, continuation_id, resume


@dataclass(frozen=True)
class CollectionConfig:
    checkpoint_budget_hours: float = 144.0
    prior_checkpoint_hours: float = 0.5116092620369616
    max_steps: int = 8

    def __post_init__(self) -> None:
        if self.checkpoint_budget_hours <= 0.0:
            raise ValueError("checkpoint_budget_hours must be positive")
        if not 0.0 <= self.prior_checkpoint_hours < self.checkpoint_budget_hours:
            raise ValueError("prior_checkpoint_hours must be within the checkpoint budget")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")


def _dependencies() -> dict[str, str]:
    packages: dict[str, str] = {}
    for name in ("torch", "transformers", "tokenizers", "safetensors", "numpy"):
        try:
            packages[name] = version(name)
        except Exception:  # pragma: no cover - absence is recorded
            packages[name] = "unavailable"
    return packages


def _write_json(path: Path, value: Mapping[str, Any], *, replace: bool) -> None:
    with path.open("w" if replace else "x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def _append(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        stream.write("\n")


def _completed(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    values: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                values.add(json.loads(line)["continuation_id"])
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    return values


def _record(continuation: Continuation, source_path: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source_path": str(source_path),
        **asdict(continuation),
    }


def collect_primary(
    run_dir: Path,
    trajectories: Iterable[Path],
    client: ChatClient,
    config: CollectionConfig = CollectionConfig(),
    *,
    arms: Sequence[ReplayArm] = FROZEN_ARMS,
    seeds: Sequence[int] = FROZEN_SEEDS,
) -> Path:
    """Collect every step-zero task, arm, and seed without duplicate records."""

    started = time.monotonic()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"
    manifest_path = run_dir / "manifest.json"
    status_path = run_dir / "status.json"
    sources = tuple(sorted({Path(path) for path in trajectories}, key=str))
    if not sources:
        raise ValueError("at least one trajectory is required")
    if not arms or not seeds:
        raise ValueError("collection requires at least one arm and seed")
    if len({arm.arm_id for arm in arms}) != len(arms):
        raise ValueError("arm ids must be unique")
    if len(set(seeds)) != len(seeds):
        raise ValueError("generation seeds must be unique")

    prefixes = []
    for source in sources:
        step_zero = [prefix for prefix in load_trajectory_prefixes(source) if prefix.step == 0]
        if len(step_zero) != 1:
            raise ValueError(f"{source}: expected one step-zero prefix")
        prefixes.append(step_zero[0])
    if len({prefix.task_id for prefix in prefixes}) != len(prefixes):
        raise ValueError("primary trajectories must contain unique tasks")
    if tuple(arms) == FROZEN_ARMS and {prefix.task_id for prefix in prefixes} != set(TASK_BY_ID):
        raise ValueError("frozen primary collection requires every candidate task")

    manifest = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "git_revision": current_git_revision(),
        "collection": "primary_step_zero",
        "config": asdict(config),
        "task_set_hash": prefixes[0].task_set_hash,
        "sources": [
            {
                "path": str(source),
                "task_id": prefix.task_id,
                "task_hash": prefix.task_hash,
                "source_sha256": prefix.source_sha256,
                "prefix_id": prefix.prefix_id,
                "request_hash": prefix.request_hash,
                "rendered_prompt_hash": prefix.rendered_prompt_hash,
            }
            for source, prefix in zip(sources, prefixes)
        ],
        "arms": [asdict(arm) for arm in arms],
        "seeds": list(seeds),
        "frozen_arm_hash": frozen_arm_hash() if tuple(arms) == FROZEN_ARMS else None,
        "frozen_artifacts": {
            "prefix_manifest_hash": "sha256:b38dee512862caa80bfd7f6525b516e4f801bae1adf7fc02f705878681fe5c18",
            "c1_vector_hash": "sha256:b9484622f01d453daba92ff92e2901e24f9695e0b880297413253e0171a2be72",
            "c3_vector_hash": "sha256:c418b98cb21498b5e0cc3bac4c1189bd97df014b97db930afb0c2e4f43817eb7",
            "sae_selection_hash": "sha256:2de8241291dc5504f71abb7926ca2e83f9a04ae9a8bc6618b31f3d6e1493eab2",
            "sae_weights_hash": "sha256:5223dd47c15704c036fef4cbec5feb45355e4b60db7676a4e4e80f1d62cec66d",
        },
        "dependencies": _dependencies(),
        "python": sys.version,
        "platform": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
    }
    if manifest_path.is_file():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("existing collection manifest does not match this run")
    else:
        _write_json(manifest_path, manifest, replace=False)

    prior_collection_hours = 0.0
    if status_path.is_file():
        prior_collection_hours = float(
            json.loads(status_path.read_text(encoding="utf-8")).get(
                "collection_hours", 0.0
            )
        )

    def collection_hours() -> float:
        return prior_collection_hours + (time.monotonic() - started) / 3600.0

    def cumulative_hours() -> float:
        return config.prior_checkpoint_hours + collection_hours()

    completed = _completed(results_path)
    total = len(sources) * len(arms) * len(seeds)
    stopped_for_budget = False
    for source in sources:
        for seed in seeds:
            for arm in arms:
                prefix = next(item for item in prefixes if item.source_path == str(source))
                expected_id = continuation_id(prefix, arm, seed, config.max_steps)
                if expected_id in completed:
                    continue
                if cumulative_hours() >= config.checkpoint_budget_hours:
                    stopped_for_budget = True
                    break
                identity = resume(
                    source,
                    0,
                    ReplayConfig(client=client, arm=arm, max_steps=config.max_steps),
                    [seed],
                )[0]
                _append(results_path, _record(identity, source))
                completed.add(identity.continuation_id)
                _write_json(
                    status_path,
                    {
                        "complete": False,
                        "stopped_for_budget": False,
                        "completed_continuations": len(completed),
                        "total_continuations": total,
                        "prior_checkpoint_hours": config.prior_checkpoint_hours,
                        "collection_hours": collection_hours(),
                        "cumulative_checkpoint_hours": cumulative_hours(),
                        "checkpoint_budget_hours": config.checkpoint_budget_hours,
                    },
                    replace=True,
                )
            if stopped_for_budget:
                break
        if stopped_for_budget:
            break
    complete = len(completed) == total
    _write_json(
        status_path,
        {
            "complete": complete,
            "stopped_for_budget": stopped_for_budget,
            "completed_continuations": len(completed),
            "total_continuations": total,
            "prior_checkpoint_hours": config.prior_checkpoint_hours,
            "collection_hours": collection_hours(),
            "cumulative_checkpoint_hours": cumulative_hours(),
            "checkpoint_budget_hours": config.checkpoint_budget_hours,
        },
        replace=True,
    )
    return results_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--budget-hours", type=float, default=144.0)
    parser.add_argument(
        "--prior-hours",
        type=float,
        default=CollectionConfig.prior_checkpoint_hours,
    )
    parser.add_argument("trajectories", nargs="+", type=Path)
    args = parser.parse_args(argv)
    collect_primary(
        args.run_dir,
        args.trajectories,
        OpenAIHTTPClient(args.base_url),
        CollectionConfig(
            checkpoint_budget_hours=args.budget_hours,
            prior_checkpoint_hours=args.prior_hours,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
