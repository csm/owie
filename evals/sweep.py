"""The Phase 0 layer sweep: cells, arms, resumption, and the budget stop.

One *cell* is one (arm, concept, layer, parameter) combination, evaluated on
every metric. A cell is written to ``results.jsonl`` as a single line the
moment it completes and is never rewritten. That is what makes the run
resumable and what makes a wall-clock stop lose at most one cell's work
(PREFLIGHT.md §6).

The tranche order is load-bearing, not cosmetic. Tranche A is the complete
coarse pass over every arm; tranche B refines around the best coarse layers.
Running A to completion first means a stop at the budget yields a complete
coarse sweep rather than a dense band with arms missing — an honest partial
result rather than a complete-looking dishonest one.

Three things this module deliberately does not do:

*It does not reduce work under time pressure.* The budget check stops the run;
it never drops an arm, drops the safety evaluation, or shortens a generation
cap. Those are prohibited mid-run by DECISIONS.md B5 and are not reachable
from any flag here.

*It does not choose the winning layer.* Selection happens in ``analysis``,
under a rule frozen in ``EXPERIMENT_PROTOCOL.md`` before collection.

*It does not aggregate.* Cells carry per-item records; point estimates and
intervals are derived later and can be recomputed without the model.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch

from directions.bundle import current_git_revision, read_bundle
from directions.extract import extract_contrast_means, write_extraction_summary
from directions.fit import direction_id, fit_directions, write_diagnostic_directions
from directions.model import HOOK_POINT, ModelHandle, load_model
from evals import metrics
from evals.schema import (
    DATA_ROOT,
    hash_file,
    load_capability_set,
    load_contrast_set,
    load_injection_set,
    load_retain_set,
    load_safety_set,
    load_structured_set,
    rendered_tool_content,
    validate_contrast_set,
)

__all__ = ["SweepConfig", "Cell", "run_sweep", "main"]

CONCEPTS = ("c1", "c2", "c3")

# Tranche A, fixed before collection: even layers 10-26, plus layer 19
# unconditionally because the SAE arm is locked to it (DECISIONS.md B4).
COARSE_LAYERS = (10, 12, 14, 16, 18, 19, 20, 22, 24, 26)

# Additive steering coefficients, as multiples of the layer's mean residual
# norm at tool-content positions. Expressed relatively so that one grid is
# comparable across layers whose activation scales differ by an order of
# magnitude. Both signs are swept: the direction's useful polarity is an
# empirical question, and sweeping one sign would presuppose the answer.
ALPHA_MULTIPLIERS = (-1.0, -0.5, 0.5, 1.0)

# Sham seeds. Three random directions at matched (unit) norm, projected out
# exactly as the fitted direction is. Rogue Scalpel found random directions are
# not inert, so if these move the metrics the primary effect must be reported
# against sham rather than against no-intervention.
SHAM_SEEDS = (11, 22, 33)

# Features clamped in the SAE arm, taken in rank order from the frozen
# selection. Clamped individually, each to zero.
SAE_CLAMP_FEATURES = 3
SAE_CLAMP_VALUE = 0.0


@dataclass(frozen=True)
class SweepConfig:
    """Everything that must be fixed before collection starts."""

    max_new_tokens_structured: int = 96
    max_new_tokens_capability: int = 48
    max_new_tokens_safety: int = 96
    budget_hours: float = 72.0
    layers: tuple[int, ...] = COARSE_LAYERS
    alpha_multipliers: tuple[float, ...] = ALPHA_MULTIPLIERS
    sham_seeds: tuple[int, ...] = SHAM_SEEDS
    sae_clamp_features: int = SAE_CLAMP_FEATURES
    tranche: str = "A"
    include_sae: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Cell:
    """One evaluated condition. ``key`` identifies it for resumption."""

    arm: str
    concept: str | None
    layer: int | None
    parameter: str

    @property
    def key(self) -> str:
        return f"{self.arm}|{self.concept or '-'}|{self.layer if self.layer is not None else '-'}|{self.parameter}"


@dataclass
class SweepState:
    """Loaded artifacts shared by every cell."""

    handle: ModelHandle
    bundles: dict[tuple[str, int], Any]
    residual_norms: dict[int, float]
    sae: Any = None
    sae_features: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


def _hook_factory(state: SweepState, config: Any, direction: torch.Tensor | None,
                  sae_feature: Any = None) -> Any:
    """Build the per-cell hook installer used by every metric."""
    from server.runtime import RequestState, installed_intervention_hook

    def factory(rendered: Any) -> Any:
        request_state = RequestState(
            config=config,
            primary_mask=tuple(rendered.primary_mask),
            whole_tool_block_mask=tuple(rendered.whole_tool_block_mask),
        )
        return installed_intervention_hook(
            state.handle.model, request_state, direction, sae_feature
        )

    return factory


def _evaluate(
    state: SweepState,
    config: SweepConfig,
    hook_factory: Any,
    datasets: dict[str, Any],
) -> dict[str, Any]:
    """Run every metric for one cell. The safety evaluation is not optional."""
    handle = state.handle
    return {
        "injection": metrics.score_injection(handle, datasets["injection"], hook_factory),
        "retain": metrics.score_retain(handle, datasets["retain"], hook_factory),
        "structured": metrics.score_structured(
            handle,
            datasets["structured"],
            hook_factory,
            max_new_tokens=config.max_new_tokens_structured,
        ),
        "capability": metrics.score_capability(
            handle,
            datasets["capability"],
            hook_factory,
            max_new_tokens=config.max_new_tokens_capability,
        ),
        "safety": metrics.score_safety(
            handle,
            datasets["safety"],
            hook_factory,
            max_new_tokens=config.max_new_tokens_safety,
        ),
    }


def _sham_direction(d_model: int, seed: int) -> torch.Tensor:
    """A random unit direction, reproducible from its recorded seed alone."""
    generator = torch.Generator().manual_seed(seed)
    vector = torch.randn(d_model, generator=generator, dtype=torch.float32)
    return vector / vector.norm()


def enumerate_cells(config: SweepConfig, state: SweepState) -> Iterator[Cell]:
    """Cell order within a tranche. Baseline first, so a stop always has it."""
    yield Cell("none", None, None, "-")
    for layer in config.layers:
        for concept in CONCEPTS:
            yield Cell("projection", concept, layer, "-")
    for layer in config.layers:
        for concept in CONCEPTS:
            for multiplier in config.alpha_multipliers:
                yield Cell("additive", concept, layer, f"c={multiplier:+.2f}")
    for layer in config.layers:
        for seed in config.sham_seeds:
            yield Cell("sham", None, layer, f"seed={seed}")
    if config.include_sae and state.sae is not None:
        for concept in CONCEPTS:
            for rank in range(config.sae_clamp_features):
                yield Cell("sae_clamp", concept, state.sae.layer, f"rank={rank}")


def _cell_hook(state: SweepState, cell: Cell, config: SweepConfig) -> tuple[Any, dict[str, Any]]:
    """The hook factory and the provenance block for one cell."""
    from server.runtime import InterventionConfig, SAEFeature

    if cell.arm == "none":
        return (lambda _rendered: nullcontext()), {"mode": "none"}

    if cell.arm == "projection":
        bundle = state.bundles[(cell.concept, cell.layer)]
        intervention = InterventionConfig(
            enabled=True,
            direction_id=bundle.manifest.direction_id,
            layer=cell.layer,
            mode="project",
            scope="tool_content",
            direction_norm="unit",
        )
        return _hook_factory(state, intervention, bundle.vector), {
            "mode": "project",
            "direction_id": bundle.manifest.direction_id,
            "contrast_set_hash": bundle.manifest.contrast_set_hash,
        }

    if cell.arm == "additive":
        bundle = state.bundles[(cell.concept, cell.layer)]
        multiplier = float(cell.parameter.split("=")[1])
        alpha = multiplier * state.residual_norms[cell.layer]
        intervention = InterventionConfig(
            enabled=True,
            direction_id=bundle.manifest.direction_id,
            layer=cell.layer,
            mode="add",
            scope="tool_content",
            direction_norm="unit",
            alpha=alpha,
        )
        return _hook_factory(state, intervention, bundle.vector), {
            "mode": "add",
            "direction_id": bundle.manifest.direction_id,
            "alpha_multiplier": multiplier,
            "alpha": alpha,
            "mean_residual_norm": state.residual_norms[cell.layer],
        }

    if cell.arm == "sham":
        seed = int(cell.parameter.split("=")[1])
        vector = _sham_direction(state.handle.d_model, seed)
        intervention = InterventionConfig(
            enabled=True,
            direction_id=f"sham-{seed}",
            layer=cell.layer,
            mode="project",
            scope="tool_content",
            direction_norm="unit",
        )
        return _hook_factory(state, intervention, vector), {
            "mode": "project",
            "direction_id": f"sham-{seed}",
            "sham_seed": seed,
        }

    if cell.arm == "sae_clamp":
        rank = int(cell.parameter.split("=")[1])
        selection = state.sae_features[cell.concept]["features"][rank]
        index = int(selection["feature_index"])
        encoder_row, encoder_bias, decoder_column = state.sae.feature(index)
        intervention = InterventionConfig(
            enabled=True,
            direction_id=None,
            layer=state.sae.layer,
            mode="clamp_sae",
            scope="tool_content",
        )
        feature = SAEFeature(
            feature_index=index,
            encoder_row=encoder_row,
            encoder_bias=encoder_bias,
            decoder_column=decoder_column,
            clamp_value=SAE_CLAMP_VALUE,
            sae_hash=state.sae.safetensors_hash,
        )
        return _hook_factory(state, intervention, None, feature), {
            "mode": "clamp_sae",
            "feature_index": index,
            "feature_rank": rank,
            "clamp_value": SAE_CLAMP_VALUE,
            "selection_hash": state.sae_features[cell.concept]["selection_hash"],
            "sae_hash": state.sae.safetensors_hash,
        }

    raise ValueError(f"unknown arm {cell.arm!r}")


def _completed_keys(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    keys: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                keys.add(json.loads(line)["cell_key"])
            except (json.JSONDecodeError, KeyError):
                # A truncated final line is exactly what an interrupted run
                # leaves behind. Ignoring it means that cell is simply redone.
                continue
    return keys


def _write_manifest(
    path: Path,
    handle: ModelHandle,
    config: SweepConfig,
    dataset_hashes: dict[str, str],
    extra: dict[str, Any],
) -> None:
    from importlib.metadata import version

    packages = {}
    for name in ("torch", "transformers", "tokenizers", "safetensors", "numpy"):
        try:
            packages[name] = version(name)
        except Exception:  # pragma: no cover - a missing package is reportable
            packages[name] = "unavailable"
    payload = {
        "run_id": path.parent.name,
        "git_revision": current_git_revision(),
        "model": handle.describe(),
        "hook_point": HOOK_POINT,
        "sweep_config": config.to_dict(),
        "dataset_hashes": dataset_hashes,
        "dependencies": packages,
        "python": sys.version,
        "platform": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "torch_device": str(handle.device),
        **extra,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_sweep(
    run_dir: Path,
    config: SweepConfig,
    *,
    model_id: str | None = None,
    model_revision: str | None = None,
    device: str | None = None,
    data_root: Path | None = None,
    refit: bool = True,
    handle: ModelHandle | None = None,
) -> Path:
    """Fit directions if needed, then evaluate every cell of the tranche."""
    started = time.monotonic()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"
    bundles_root = run_dir / "directions"
    data_root = Path(data_root) if data_root is not None else DATA_ROOT

    contrasts = {}
    for concept in CONCEPTS:
        pairs = load_contrast_set(concept, root=data_root)
        validate_contrast_set(pairs, concept=concept, renderer=rendered_tool_content)
        contrasts[concept] = pairs
    datasets = {
        "injection": load_injection_set(root=data_root),
        "retain": load_retain_set(root=data_root),
        "structured": load_structured_set(root=data_root),
        "capability": load_capability_set(root=data_root),
        "safety": load_safety_set(root=data_root),
    }
    dataset_hashes = {
        name: hash_file(data_root / f"{name}.jsonl") for name in datasets
    }
    dataset_hashes.update(
        {
            f"contrasts/{concept}": hash_file(data_root / "contrasts" / f"{concept}.jsonl")
            for concept in CONCEPTS
        }
    )

    if handle is None:
        load_kwargs: dict[str, Any] = {"device": device}
        if model_id:
            load_kwargs["model_id"] = model_id
        if model_revision:
            load_kwargs["revision"] = model_revision
        handle = load_model(**load_kwargs)
    print(f"loaded {handle.model_id} on {handle.device}", flush=True)

    residual_norms: dict[int, float] = {}
    if refit and not bundles_root.exists():
        for concept in CONCEPTS:
            print(f"extracting activations for {concept}", flush=True)
            result = extract_contrast_means(
                handle, contrasts[concept], concept=concept, split="train"
            )
            fit_directions(
                handle,
                result,
                contrasts[concept],
                root=bundles_root,
                extraction_config={
                    "concept": concept,
                    "contrast_file": f"contrasts/{concept}.jsonl",
                    "system_prompt_frozen": True,
                    "split_by": "scenario_family",
                },
                layers=None,
            )
            write_diagnostic_directions(
                run_dir / "diagnostics" / f"{concept}.safetensors", result
            )
            write_extraction_summary(run_dir / "extraction" / f"{concept}.json", [result])
            norms = result.mean_residual_norm()
            for layer in range(handle.n_layers):
                residual_norms[layer] = float(norms[layer])

    bundles: dict[tuple[str, int], Any] = {}
    for concept in CONCEPTS:
        for layer in config.layers:
            bundle = read_bundle(
                direction_id(concept, layer, pilot=handle.is_pilot), root=bundles_root
            )
            bundles[(concept, layer)] = bundle
            residual_norms.setdefault(
                layer, float(bundle.manifest.extra["mean_residual_norm"])
            )

    state = SweepState(handle=handle, bundles=bundles, residual_norms=residual_norms)

    sae_error: str | None = None
    if config.include_sae:
        try:
            from directions.sae import load_sae, select_features

            state.sae = load_sae(cache_dir=run_dir.parent / "sae")
            selection_path = run_dir / "sae_features.json"
            if selection_path.is_file():
                state.sae_features = json.loads(selection_path.read_text(encoding="utf-8"))
            else:
                state.sae_features = {
                    concept: select_features(
                        handle,
                        state.sae,
                        contrasts[concept],
                        top_k=max(5, config.sae_clamp_features),
                    )
                    for concept in CONCEPTS
                }
                selection_path.write_text(
                    json.dumps(state.sae_features, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
        except Exception as exc:  # the arm is recorded as excluded, not dropped
            sae_error = f"{type(exc).__name__}: {exc}"
            state.sae = None
            print(f"SAE arm unavailable: {sae_error}", flush=True)

    _write_manifest(
        run_dir / "manifest.json",
        handle,
        config,
        dataset_hashes,
        {
            "sae": state.sae.describe() if state.sae is not None else None,
            "sae_exclusion_reason": sae_error,
            "residual_norms_by_layer": {
                str(layer): round(value, 4) for layer, value in sorted(residual_norms.items())
            },
        },
    )

    completed = _completed_keys(results_path)
    cells = list(enumerate_cells(config, state))
    budget_seconds = config.budget_hours * 3600.0
    stopped_early = False

    for index, cell in enumerate(cells, start=1):
        if cell.key in completed:
            continue
        elapsed = time.monotonic() - started
        if elapsed >= budget_seconds:
            stopped_early = True
            print(
                f"budget of {config.budget_hours} h reached after {index - 1} cells; "
                "stopping as pre-registered (DECISIONS.md B5)",
                flush=True,
            )
            break
        hook_factory, provenance = _cell_hook(state, cell, config)
        cell_started = time.monotonic()
        records = _evaluate(state, config, hook_factory, datasets)
        duration = time.monotonic() - cell_started
        row = {
            "cell_key": cell.key,
            "tranche": config.tranche,
            "arm": cell.arm,
            "concept": cell.concept,
            "layer": cell.layer,
            "parameter": cell.parameter,
            "intervention": provenance,
            "seconds": round(duration, 3),
            "records": records,
        }
        with results_path.open("a", encoding="utf-8") as handle_out:
            handle_out.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle_out.write("\n")
            handle_out.flush()
        print(
            f"[{index}/{len(cells)}] {cell.key} in {duration / 60:.1f} min "
            f"(elapsed {(time.monotonic() - started) / 3600:.2f} h)",
            flush=True,
        )

    (run_dir / "status.json").write_text(
        json.dumps(
            {
                "tranche": config.tranche,
                "cells_total": len(cells),
                "cells_completed": len(_completed_keys(results_path)),
                "stopped_on_budget": stopped_early,
                "elapsed_hours": round((time.monotonic() - started) / 3600.0, 4),
                "sae_exclusion_reason": sae_error,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return results_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--tranche", default="A", choices=("A", "B"))
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="override the swept layers; tranche B supplies the refinement band",
    )
    parser.add_argument("--budget-hours", type=float, default=72.0)
    parser.add_argument("--model-id", default=None, help="pilot runs only (DECISIONS.md B6)")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--no-sae", action="store_true")
    args = parser.parse_args(argv)

    config = SweepConfig(
        budget_hours=args.budget_hours,
        layers=tuple(args.layers) if args.layers else COARSE_LAYERS,
        tranche=args.tranche,
        include_sae=not args.no_sae,
    )
    path = run_sweep(
        args.run_dir,
        config,
        model_id=args.model_id,
        model_revision=args.model_revision,
        device=args.device,
        data_root=args.data_root,
    )
    print(f"raw results: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
