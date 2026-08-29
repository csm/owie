"""Run retain and safety evaluation for every frozen replay arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from directions import read_bundle
from directions.bundle import current_git_revision
from directions.model import ModelHandle, load_model
from evals import metrics
from evals.schema import (
    DATA_ROOT,
    RetainItem,
    SafetyItem,
    hash_file,
    load_retain_set,
    load_safety_set,
)
from loop.runner import seed_everything
from loop.tasks import PROMPT_DEFENSE
from replay.arms import FROZEN_ARMS, FROZEN_SEEDS, frozen_arm_hash
from replay.calibration import summarize_records
from replay.prefixes import verify_prefix_manifest
from replay.runner import ReplayArm, reserved_token_guard
from server.backend import (
    RegisteredDirection,
    RegisteredSAEFeature,
    hash_direction_bundle,
    registered_sham,
)
from server.runtime import InterventionConfig, RequestState, SAEFeature, installed_intervention_hook


C1_VECTOR_HASH = "sha256:b9484622f01d453daba92ff92e2901e24f9695e0b880297413253e0171a2be72"
C3_VECTOR_HASH = "sha256:c418b98cb21498b5e0cc3bac4c1189bd97df014b97db930afb0c2e4f43817eb7"
SAE_SELECTION_HASH = "sha256:2de8241291dc5504f71abb7926ca2e83f9a04ae9a8bc6618b31f3d6e1493eab2"
SAE_WEIGHTS_HASH = "sha256:5223dd47c15704c036fef4cbec5feb45355e4b60db7676a4e4e80f1d62cec66d"
SAE_FEATURE_ID = "sae-c1-rank0-feature-1584"


@dataclass(frozen=True)
class ArmEvaluationConfig:
    seed: int = 0
    max_new_tokens_safety: int = 96
    checkpoint_budget_hours: float = 144.0
    prior_checkpoint_hours: float = 0.5116092620369616

    def __post_init__(self) -> None:
        if self.max_new_tokens_safety < 1:
            raise ValueError("max_new_tokens_safety must be positive")
        if self.checkpoint_budget_hours <= 0.0:
            raise ValueError("checkpoint_budget_hours must be positive")
        if not 0.0 <= self.prior_checkpoint_hours < self.checkpoint_budget_hours:
            raise ValueError("prior_checkpoint_hours must be within the checkpoint budget")


@dataclass(frozen=True)
class EvaluationWork:
    arm: ReplayArm
    metric: str
    item: RetainItem | SafetyItem

    @property
    def identity(self) -> tuple[str, str, str]:
        return self.arm.arm_id, self.metric, self.item.item_id


def enumerate_arm_work(
    arms: Sequence[ReplayArm],
    retain_items: Sequence[RetainItem],
    safety_items: Sequence[SafetyItem],
) -> tuple[EvaluationWork, ...]:
    """Interleave metrics by item and keep every arm in each paired block."""

    if not arms or len({arm.arm_id for arm in arms}) != len(arms):
        raise ValueError("evaluation arms must be non-empty with unique ids")
    if not retain_items or not safety_items:
        raise ValueError("retain and safety datasets must both be non-empty")
    work: list[EvaluationWork] = []
    for retain_item, safety_item in zip(retain_items, safety_items, strict=True):
        work.extend(EvaluationWork(arm, "retain", retain_item) for arm in arms)
        work.extend(EvaluationWork(arm, "safety", safety_item) for arm in arms)
    return tuple(work)


def prepare_item(
    item: RetainItem | SafetyItem, arm: ReplayArm
) -> RetainItem | SafetyItem:
    """Apply the exact prompt-changing comparator before metric rendering."""

    if arm.prompt_defense:
        return replace(item, system=f"{item.system}\n{PROMPT_DEFENSE}")
    if arm.reserved_token_guard:
        return replace(item, tool_output=reserved_token_guard(item.tool_output))
    return item


def arm_hook_factory(
    handle: ModelHandle,
    arm: ReplayArm,
    directions: Mapping[str, RegisteredDirection],
    sae_features: Mapping[str, RegisteredSAEFeature],
):
    """Build one metrics hook factory from a replay arm registration."""

    config = InterventionConfig(**arm.intervention)
    if not config.enabled:
        return lambda _rendered: nullcontext()

    def factory(rendered: Any):
        state = RequestState(
            config,
            rendered.primary_mask,
            rendered.whole_tool_block_mask,
        )
        if config.mode == "clamp_sae":
            try:
                registered_sae = sae_features[config.direction_id or ""]
            except KeyError as exc:
                raise ValueError(f"unknown SAE feature {config.direction_id!r}") from exc
            if config.layer != registered_sae.layer:
                raise ValueError(
                    f"SAE feature {config.direction_id!r} belongs at layer "
                    f"{registered_sae.layer}, not requested layer {config.layer}"
                )
            return installed_intervention_hook(
                handle.model, state, None, registered_sae.feature
            )
        try:
            registered = directions[config.direction_id or ""]
        except KeyError as exc:
            raise ValueError(f"unknown direction_id {config.direction_id!r}") from exc
        if config.layer != registered.layer:
            raise ValueError(
                f"direction {config.direction_id!r} was fitted at layer "
                f"{registered.layer}, not requested layer {config.layer}"
            )
        if config.direction_norm != registered.normalization:
            raise ValueError(
                f"direction {config.direction_id!r} uses normalization "
                f"{registered.normalization!r}, not {config.direction_norm!r}"
            )
        return installed_intervention_hook(handle.model, state, registered.vector)

    return factory


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen_registrations(
    handle: ModelHandle,
    *,
    direction_root: Path,
    sae_selection_path: Path,
    sae_cache_dir: Path,
    local_files_only: bool,
) -> tuple[dict[str, RegisteredDirection], dict[str, RegisteredSAEFeature]]:
    """Load and verify all direction, sham, and SAE artifacts in the arm grid."""

    directions: dict[str, RegisteredDirection] = {}
    expected_hashes = {"c1-l10-dim": C1_VECTOR_HASH, "c3-l10-dim": C3_VECTOR_HASH}
    for direction_id, expected_hash in expected_hashes.items():
        bundle = read_bundle(direction_id, root=direction_root)
        if (
            bundle.manifest.model_id != handle.model_id
            or bundle.manifest.model_revision != handle.model_revision
        ):
            raise ValueError(f"direction {direction_id!r} does not target the loaded model")
        if _sha256(bundle.path / "vector.safetensors") != expected_hash:
            raise ValueError(f"direction {direction_id!r} vector hash does not match protocol")
        directions[direction_id] = RegisteredDirection(
            bundle.vector,
            layer=bundle.manifest.layer,
            normalization=bundle.manifest.normalization,
            hook_point=bundle.manifest.hook_point,
            bundle_hash=hash_direction_bundle(bundle.path),
        )
    for seed in FROZEN_SEEDS:
        directions[f"sham-{seed}"] = registered_sham(seed, handle.d_model, layer=10)

    selection = json.loads(sae_selection_path.read_text(encoding="utf-8"))["c1"]
    if selection.get("selection_hash") != SAE_SELECTION_HASH:
        raise ValueError("the C1 SAE selection hash does not match the frozen protocol")
    feature_index = int(selection["features"][0]["feature_index"])
    if feature_index != 1584:
        raise ValueError("the frozen C1 rank-zero SAE feature is not 1584")
    from directions.sae import load_sae

    sae = load_sae(cache_dir=sae_cache_dir, local_files_only=local_files_only)
    if sae.safetensors_hash != SAE_WEIGHTS_HASH:
        raise ValueError("the SAE weights hash does not match the frozen protocol")
    encoder_row, encoder_bias, decoder_column = sae.feature(feature_index)
    artifact_hash = "sha256:" + hashlib.sha256(
        f"{sae.safetensors_hash}\0{selection['selection_hash']}\0{feature_index}".encode()
    ).hexdigest()
    sae_features = {
        SAE_FEATURE_ID: RegisteredSAEFeature(
            SAEFeature(
                feature_index=feature_index,
                encoder_row=encoder_row,
                encoder_bias=encoder_bias,
                decoder_column=decoder_column,
                sae_hash=sae.safetensors_hash,
            ),
            layer=sae.layer,
            artifact_hash=artifact_hash,
        )
    }
    return directions, sae_features


def _dependencies() -> dict[str, str]:
    packages: dict[str, str] = {}
    for name in ("torch", "transformers", "tokenizers", "safetensors", "numpy"):
        try:
            packages[name] = version(name)
        except Exception:  # pragma: no cover
            packages[name] = "unavailable"
    return packages


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any], *, replace_existing: bool) -> None:
    with path.open("w" if replace_existing else "x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def _append(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        stream.write("\n")


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _summary(rows: Iterable[Mapping[str, Any]], arms: Sequence[ReplayArm]) -> dict[str, Any]:
    records = list(rows)
    summaries = {
        arm.arm_id: summarize_records(
            row for row in records if row.get("arm_id") == arm.arm_id
        )
        for arm in arms
    }
    baseline = summaries["none"]
    for arm_id, summary in summaries.items():
        summary["retain"]["perplexity_ratio_to_none"] = (
            summary["retain"]["perplexity"] / baseline["retain"]["perplexity"]
        )
        summary["retain"]["tool_dependent_ratio_to_none"] = (
            summary["retain"]["tool_dependent_perplexity"]
            / baseline["retain"]["tool_dependent_perplexity"]
        )
        summary["safety"]["additional_user_compliance"] = (
            summary["safety"]["user_compliance_rate"]
            - baseline["safety"]["user_compliance_rate"]
        )
        summary["safety"]["additional_tool_compliance"] = (
            summary["safety"]["tool_compliance_rate"]
            - baseline["safety"]["tool_compliance_rate"]
        )
    return {"baseline_arm": "none", "arms": summaries}


def evaluate_arms(
    run_dir: Path,
    prefix_manifest_path: Path,
    handle: ModelHandle,
    directions: Mapping[str, RegisteredDirection],
    sae_features: Mapping[str, RegisteredSAEFeature],
    config: ArmEvaluationConfig = ArmEvaluationConfig(),
    *,
    arms: Sequence[ReplayArm] = FROZEN_ARMS,
    data_root: Path = DATA_ROOT,
) -> Path:
    """Run or resume every retain and safety item for every arm."""

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
    retain_items = load_retain_set(root=data_root)
    safety_items = load_safety_set(root=data_root)
    work = enumerate_arm_work(arms, retain_items, safety_items)
    seeds = seed_everything(config.seed)
    manifest = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "git_revision": current_git_revision(),
        "collection": "arm_retain_and_safety",
        "model": handle.describe(),
        "config": asdict(config),
        "determinism": seeds,
        "prefix_manifest": {
            "path": str(prefix_manifest_path),
            "manifest_hash": prefix_manifest["manifest_hash"],
            "task_set_hash": prefix_manifest["task_set_hash"],
        },
        "dataset_hashes": {
            "retain": hash_file(data_root / "retain.jsonl"),
            "safety": hash_file(data_root / "safety.jsonl"),
        },
        "arms": [asdict(arm) for arm in arms],
        "frozen_arm_hash": frozen_arm_hash() if tuple(arms) == FROZEN_ARMS else None,
        "registered_directions": {
            key: value.bundle_hash for key, value in directions.items()
        },
        "registered_sae_features": {
            key: value.artifact_hash for key, value in sae_features.items()
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
        if _read_json(manifest_path) != manifest:
            raise ValueError("existing arm-evaluation manifest does not match this run")
    else:
        _write_json(manifest_path, manifest, replace_existing=False)

    previous_hours = 0.0
    if status_path.is_file():
        previous_hours = float(_read_json(status_path).get("evaluation_hours", 0.0))

    def evaluation_hours() -> float:
        return previous_hours + (time.monotonic() - started) / 3600.0

    def cumulative_hours() -> float:
        return config.prior_checkpoint_hours + evaluation_hours()

    completed = {
        (row["arm_id"], row["metric"], row["item_id"])
        for row in _rows(results_path)
        if all(key in row for key in ("arm_id", "metric", "item_id"))
    }
    stopped_for_budget = False
    for cell in work:
        if cell.identity in completed:
            continue
        if cumulative_hours() >= config.checkpoint_budget_hours:
            stopped_for_budget = True
            break
        item_started = time.monotonic()
        item = prepare_item(cell.item, cell.arm)
        hook_factory = arm_hook_factory(handle, cell.arm, directions, sae_features)
        if cell.metric == "retain":
            record = metrics.score_retain(handle, [item], hook_factory)[0]
        else:
            record = metrics.score_safety(
                handle,
                [item],
                hook_factory,
                max_new_tokens=config.max_new_tokens_safety,
            )[0]
        _append(
            results_path,
            {
                "schema_version": 1,
                "arm_id": cell.arm.arm_id,
                "prompt_changing": cell.arm.prompt_changing,
                "intervention": cell.arm.intervention,
                "metric": cell.metric,
                **record,
                "seconds": time.monotonic() - item_started,
            },
        )
        completed.add(cell.identity)
        _write_json(
            status_path,
            {
                "complete": False,
                "stopped_for_budget": False,
                "completed_items": len(completed),
                "total_items": len(work),
                "prior_checkpoint_hours": config.prior_checkpoint_hours,
                "evaluation_hours": evaluation_hours(),
                "cumulative_checkpoint_hours": cumulative_hours(),
                "checkpoint_budget_hours": config.checkpoint_budget_hours,
            },
            replace_existing=True,
        )

    complete = len(completed) == len(work)
    if complete:
        summary = _summary(_rows(results_path), arms)
        if summary_path.is_file():
            if _read_json(summary_path) != summary:
                raise ValueError("existing arm-evaluation summary does not match results")
        else:
            _write_json(summary_path, summary, replace_existing=False)
    _write_json(
        status_path,
        {
            "complete": complete,
            "stopped_for_budget": stopped_for_budget,
            "completed_items": len(completed),
            "total_items": len(work),
            "prior_checkpoint_hours": config.prior_checkpoint_hours,
            "evaluation_hours": evaluation_hours(),
            "cumulative_checkpoint_hours": cumulative_hours(),
            "checkpoint_budget_hours": config.checkpoint_budget_hours,
        },
        replace_existing=True,
    )
    return results_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--prefix-manifest", type=Path, required=True)
    parser.add_argument("--direction-root", type=Path, required=True)
    parser.add_argument("--sae-selection", type=Path, required=True)
    parser.add_argument("--sae-cache-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens-safety", type=int, default=96)
    parser.add_argument("--budget-hours", type=float, default=144.0)
    parser.add_argument("--prior-hours", type=float, default=ArmEvaluationConfig.prior_checkpoint_hours)
    parser.add_argument("--device")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)
    handle = load_model(device=args.device, local_files_only=args.local_files_only)
    directions, sae_features = load_frozen_registrations(
        handle,
        direction_root=args.direction_root,
        sae_selection_path=args.sae_selection,
        sae_cache_dir=args.sae_cache_dir,
        local_files_only=args.local_files_only,
    )
    evaluate_arms(
        args.run_dir,
        args.prefix_manifest,
        handle,
        directions,
        sae_features,
        ArmEvaluationConfig(
            seed=args.seed,
            max_new_tokens_safety=args.max_new_tokens_safety,
            checkpoint_budget_hours=args.budget_hours,
            prior_checkpoint_hours=args.prior_hours,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
