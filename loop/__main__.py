"""Run or repeat one frozen Checkpoint 4 task against the local shim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from server.rendering import (
    MODEL_ID,
    MODEL_REVISION,
    PILOT_MODEL_ID,
    PILOT_MODEL_REVISION,
)

from .runner import (
    LoopConfig,
    OpenAIHTTPClient,
    run_task,
    write_determinism_report,
    write_trajectory,
)
from .tasks import TASK_BY_ID

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--task", choices=sorted(TASK_BY_ID), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeat", type=int, choices=(1, 2), default=1)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--pilot-3b", action="store_true")
    parser.add_argument("--direction-revision")
    parser.add_argument("--intervention-json", default='{"enabled":false}')
    args = parser.parse_args(argv)
    try:
        intervention = json.loads(args.intervention_json)
    except json.JSONDecodeError as exc:
        parser.error(f"--intervention-json is not valid JSON: {exc}")
    if not isinstance(intervention, dict):
        parser.error("--intervention-json must decode to an object")

    model_id = PILOT_MODEL_ID if args.pilot_3b else MODEL_ID
    revision = PILOT_MODEL_REVISION if args.pilot_3b else MODEL_REVISION
    config = LoopConfig(
        seed=args.seed,
        model_id=model_id,
        model_revision=revision,
        direction_revision=args.direction_revision,
        intervention=intervention,
        max_steps=args.max_steps,
        max_tokens=args.max_tokens,
        run_kind="pilot" if args.pilot_3b else "primary",
    )
    client = OpenAIHTTPClient(args.base_url)
    paths = []
    had_error = False
    for repetition in range(1, args.repeat + 1):
        result = run_task(client, TASK_BY_ID[args.task], config)
        path = args.run_dir / f"trajectory-{repetition}.jsonl"
        write_trajectory(path, result)
        paths.append(path)
        print(f"{path}: success={result.success} stop={result.stop_reason}")
        had_error = had_error or result.stop_reason == "model_error"
    if had_error:
        return 2
    if len(paths) == 2:
        report = write_determinism_report(args.run_dir / "determinism.json", *paths)
        matched = report["matched"]
        print(f"determinism_after_documented_timing_exclusions={str(matched).lower()}")
        return 0 if matched else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
