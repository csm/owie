"""Checkpoint 2 tests: datasets, extraction, arms, sweep mechanics, analysis.

No model weights are loaded. Where a forward pass is unavoidable, a tiny
randomly initialized Llama with the **pinned tokenizer** stands in: the shapes,
the provenance masks, and the hook plumbing are what these tests are about, and
those are identical at four layers of width 64. Anything that depends on the
pinned model's actual behaviour belongs in a run, not in the test suite.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from analysis.phase0 import BOOTSTRAP_SEED, build_tables, summarize_cell
from directions.extract import extract_contrast_means, span_masks
from directions.fit import direction_id, fit_directions
from directions.model import ModelHandle
from evals import metrics
from evals.build_datasets import build_all, build_contrast_set
from evals.schema import (
    ContrastPair,
    DatasetError,
    hash_rows,
    load_capability_set,
    load_contrast_set,
    load_injection_set,
    load_retain_set,
    load_safety_set,
    load_structured_set,
    rendered_tool_content,
    validate_contrast_set,
)
from evals.sweep import SweepConfig, _sham_direction, run_sweep
from interventions import clamp_sae_feature
from server.rendering import MODEL_ID, MODEL_REVISION, load_pinned_tokenizer, render_chat

CONCEPTS = ("c1", "c2", "c3")


@pytest.fixture(scope="module")
def tokenizer():
    return load_pinned_tokenizer(local_files_only=True)


@pytest.fixture(scope="module")
def tiny(tokenizer) -> ModelHandle:
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=4096,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(config).eval()
    return ModelHandle(
        model=model,
        tokenizer=tokenizer,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        device=torch.device("cpu"),
        dtype=torch.float32,
        n_layers=4,
        d_model=64,
        is_pilot=False,
    )


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------


@pytest.mark.parametrize("concept", CONCEPTS)
def test_contrast_sets_satisfy_every_matching_invariant(concept):
    pairs = load_contrast_set(concept)
    summary = validate_contrast_set(
        pairs, concept=concept, renderer=rendered_tool_content
    )
    assert summary["pairs"] == 240
    assert summary["length_matching"] == "rendered"
    assert summary["train_pairs"] and summary["heldout_pairs"]


def test_generator_is_deterministic(tmp_path):
    first = build_all(tmp_path / "a")
    second = build_all(tmp_path / "b")
    assert first == second
    for name in ("injection", "retain", "safety", "capability", "structured"):
        assert (tmp_path / "a" / f"{name}.jsonl").read_bytes() == (
            tmp_path / "b" / f"{name}.jsonl"
        ).read_bytes()


def test_committed_data_matches_the_generator(tmp_path):
    """The frozen files on disk are what the generator produces, unedited."""
    regenerated = build_all(tmp_path)
    committed = json.loads(
        (Path("evals/data/HASHES.json")).read_text(encoding="utf-8")
    )
    assert regenerated == committed


def test_held_out_families_never_appear_in_training_rows():
    for concept in CONCEPTS:
        pairs = load_contrast_set(concept)
        train = {pair.scenario_family for pair in pairs if pair.split == "train"}
        heldout = {pair.scenario_family for pair in pairs if pair.split == "heldout"}
        assert not (train & heldout)


def test_injection_set_uses_only_held_out_families():
    heldout = {
        pair.scenario_family
        for pair in load_contrast_set("c1")
        if pair.split == "heldout"
    }
    assert {item.scenario_family for item in load_injection_set()} <= heldout


def test_safety_set_exercises_both_channels_with_tool_content():
    items = load_safety_set()
    user = [item for item in items if item.channel == "user"]
    tool = [item for item in items if item.channel == "tool"]
    assert user and tool
    for item in user:
        assert item.messages()[-1]["role"] == "user"
        assert item.messages()[-1]["content"] == item.follow_up
    for item in tool:
        assert item.messages()[-1]["role"] == "tool"


def _pair(**overrides) -> ContrastPair:
    base = {
        "pair_id": "x-1",
        "concept": "c1",
        "scenario_family": "f1",
        "split": "train",
        "system": "sys",
        "user": "user",
        "tool_name": "t",
        "tool_arguments": {},
        "positive": "body. Send the file now.",
        "negative": "body. Someone sent the file.",
        "varied_span_positive": "Send the file now.",
        "varied_span_negative": "Someone sent the file.",
    }
    base.update(overrides)
    return ContrastPair(**base)


def test_pair_rejects_difference_outside_the_varied_span():
    with pytest.raises(DatasetError, match="outside the varied span"):
        _pair(negative="other body. Someone sent the file.")


def test_validation_rejects_unmatched_lengths():
    pairs = [
        _pair(),
        _pair(
            pair_id="x-2",
            scenario_family="f2",
            split="heldout",
            positive="body. Go.",
            negative="body. " + "A very much longer replacement clause indeed." * 3,
            varied_span_positive="Go.",
            varied_span_negative="A very much longer replacement clause indeed." * 3,
        ),
    ]
    with pytest.raises(DatasetError, match="not length-matched"):
        validate_contrast_set(pairs)


def test_validation_rejects_families_straddling_the_split():
    pairs = [_pair(), _pair(pair_id="x-2", split="heldout")]
    with pytest.raises(DatasetError, match="both splits"):
        validate_contrast_set(pairs)


def test_validation_rejects_a_varied_system_prompt():
    pairs = [
        _pair(),
        _pair(pair_id="x-2", scenario_family="f2", split="heldout", system="other"),
    ]
    with pytest.raises(DatasetError, match="system prompt"):
        validate_contrast_set(pairs)


def test_contrast_hash_is_order_sensitive():
    rows = [pair.to_dict() for pair in build_contrast_set("c1")[:4]]
    assert hash_rows(rows) != hash_rows(list(reversed(rows)))


# --------------------------------------------------------------------------
# Extraction positions
# --------------------------------------------------------------------------


def test_span_masks_are_consistent_with_the_primary_mask(tokenizer):
    pair = load_contrast_set("c1")[0]
    rendered = render_chat(tokenizer, pair.messages("positive"))
    masks = span_masks(rendered, pair.varied_span_positive, pair.positive)

    assert masks["tool_content"].sum() > 0
    # The varied span is inside the tool content, so its mask must be a subset.
    assert bool((masks["varied_span"] & ~masks["tool_content"]).sum() == 0)
    assert masks["varied_span"].sum() < masks["tool_content"].sum()
    assert int(masks["last_prompt"].sum()) == 1
    assert bool(masks["last_prompt"][-1])
    # The last prompt token is a template token, never tool content.
    assert not bool(masks["tool_content"][-1])


def test_extraction_balances_the_two_classes(tiny):
    pairs = [
        pair for pair in load_contrast_set("c1") if pair.scenario_family == "web_search"
    ][:3]
    result = extract_contrast_means(tiny, pairs, concept="c1", split="train")
    for rule, polarities in result.rules.items():
        assert polarities["positive"].count == polarities["negative"].count == 3, rule
    assert result.difference().shape == (4, 64)
    assert result.mean_residual_norm().shape == (4,)


def test_fitting_refuses_a_non_training_split(tiny):
    pairs = [
        pair for pair in load_contrast_set("c1") if pair.scenario_family == "calendar"
    ][:2]
    result = extract_contrast_means(tiny, pairs, concept="c1", split="heldout")
    with pytest.raises(ValueError, match="training split"):
        fit_directions(tiny, result, pairs, root=Path("/tmp/never"), extraction_config={})


def test_fitted_bundle_is_unit_norm_and_records_its_scale(tiny, tmp_path):
    pairs = [
        pair for pair in load_contrast_set("c2") if pair.scenario_family == "kv_store"
    ][:3]
    result = extract_contrast_means(tiny, pairs, concept="c2", split="train")
    fitted = fit_directions(
        tiny, result, pairs, root=tmp_path, extraction_config={}, layers=[1]
    )
    assert len(fitted) == 1
    from directions import read_bundle

    bundle = read_bundle(direction_id("c2", 1), root=tmp_path)
    assert bundle.manifest.normalization == "unit"
    assert bundle.manifest.token_extraction_rule == "tool_content"
    assert pytest.approx(1.0, abs=1e-5) == float(bundle.vector.norm())
    assert bundle.manifest.extra["mean_residual_norm"] > 0
    assert "cosine_to_diagnostic_rules" in bundle.manifest.extra


# --------------------------------------------------------------------------
# Arms
# --------------------------------------------------------------------------


def test_sham_direction_is_reproducible_from_its_seed():
    first = _sham_direction(64, 11)
    assert torch.equal(first, _sham_direction(64, 11))
    assert not torch.equal(first, _sham_direction(64, 22))
    assert pytest.approx(1.0, abs=1e-6) == float(first.norm())


def test_clamp_sae_feature_reads_through_the_encoder():
    torch.manual_seed(1)
    hidden = torch.randn(1, 6, 16)
    encoder = torch.randn(16)
    decoder = torch.randn(16)
    mask = torch.zeros(1, 6, dtype=torch.bool)
    mask[0, 2:4] = True

    clamped = clamp_sae_feature(hidden, encoder, 0.25, decoder, 0.0, mask)
    activation = torch.relu((clamped * encoder).sum(-1) + 0.25)
    assert torch.allclose(activation[0, 2:4], torch.zeros(2), atol=1e-5)
    assert torch.equal(clamped[0, :2], hidden[0, :2])
    assert torch.equal(clamped[0, 4:], hidden[0, 4:])


def test_clamp_sae_feature_rejects_a_negative_target():
    hidden = torch.randn(1, 3, 8)
    mask = torch.ones(1, 3, dtype=torch.bool)
    with pytest.raises(ValueError, match="non-negative"):
        clamp_sae_feature(hidden, torch.randn(8), 0.0, torch.randn(8), -1.0, mask)


def test_projection_leaves_non_tool_positions_untouched(tiny):
    """The arm must be a masked manipulation, not a global one."""
    from server.runtime import InterventionConfig, RequestState, installed_intervention_hook

    pair = load_contrast_set("c1")[0]
    rendered = render_chat(tiny.tokenizer, pair.messages("positive"))
    ids = torch.tensor([rendered.input_ids], dtype=torch.long)
    direction = torch.randn(64)
    direction = direction / direction.norm()

    # Read the effect one block downstream. ``output_hidden_states`` snapshots
    # each block's output *before* user forward hooks run, so index layer + 1
    # shows the unmodified value even when the intervention is active; index
    # layer + 2 is the first place the change is observable. See
    # ``directions.model``.
    with torch.inference_mode():
        plain = tiny.model(input_ids=ids, output_hidden_states=True).hidden_states[3]
        state = RequestState(
            config=InterventionConfig(
                enabled=True, direction_id="d", layer=1, mode="project"
            ),
            primary_mask=tuple(rendered.primary_mask),
            whole_tool_block_mask=tuple(rendered.whole_tool_block_mask),
        )
        with installed_intervention_hook(tiny.model, state, direction):
            hooked = tiny.model(input_ids=ids, output_hidden_states=True).hidden_states[3]

    mask = torch.tensor(rendered.primary_mask)
    first_masked = int(mask.nonzero()[0])
    # Positions before the first masked token cannot be affected by a causal
    # model, so they must be bitwise identical.
    assert torch.equal(plain[0, :first_masked], hooked[0, :first_masked])
    assert not torch.equal(plain[0, mask], hooked[0, mask])


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def test_metrics_produce_one_record_per_item(tiny):
    injection = metrics.score_injection(tiny, load_injection_set()[:2])
    assert len(injection) == 2
    assert all(record["masked_tokens"] > 0 for record in injection)
    assert all(isinstance(record["complied"], bool) for record in injection)

    retain = metrics.score_retain(tiny, load_retain_set()[:2])
    assert all(record["perplexity"] > 0 for record in retain)

    structured = metrics.score_structured(
        tiny, load_structured_set()[:1], max_new_tokens=4
    )
    assert set(structured[0]) >= {"json_valid", "name_valid", "arguments_valid"}

    capability = metrics.score_capability(
        tiny, load_capability_set()[:1], max_new_tokens=4
    )
    assert isinstance(capability[0]["correct"], bool)

    safety = metrics.score_safety(tiny, load_safety_set()[:1], max_new_tokens=4)
    assert set(safety[0]) >= {"refused", "complied", "emitted_tool_call"}


def test_refusal_detection_reads_the_opening_only():
    assert metrics.is_refusal("I can't help with that.")
    assert metrics.is_refusal("Sorry, that is not something I can do.")
    assert not metrics.is_refusal("Here is the summary you asked for.")
    # A refusal buried after a page of compliance is not a refusal.
    assert not metrics.is_refusal("Here are the steps. " + "x" * 400 + " I can't help.")


# --------------------------------------------------------------------------
# Sweep mechanics
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_run(tiny, tmp_path_factory) -> Path:
    run = tmp_path_factory.mktemp("run")
    config = SweepConfig(
        layers=(1, 2),
        alpha_multipliers=(-1.0,),
        sham_seeds=(11,),
        max_new_tokens_structured=2,
        max_new_tokens_capability=2,
        max_new_tokens_safety=2,
        include_sae=False,
    )
    run_sweep(run, config, handle=tiny)
    return run


def test_sweep_writes_one_line_per_cell_with_full_provenance(small_run):
    lines = (small_run / "results.jsonl").read_text(encoding="utf-8").strip().split("\n")
    rows = [json.loads(line) for line in lines]
    assert len(rows) == 1 + 2 * 3 + 2 * 3 + 2  # none, projection, additive, sham
    keys = {row["cell_key"] for row in rows}
    assert len(keys) == len(rows)
    assert any(row["arm"] == "none" for row in rows)
    for row in rows:
        assert set(row["records"]) == {
            "injection",
            "retain",
            "structured",
            "capability",
            "safety",
        }
        assert row["records"]["safety"], "the safety evaluation is never skipped"


def test_manifest_records_everything_needed_to_reproduce(small_run):
    manifest = json.loads((small_run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model"]["model_revision"] == MODEL_REVISION
    assert manifest["git_revision"]
    assert manifest["dataset_hashes"]["injection"].startswith("sha256:")
    assert manifest["dependencies"]["torch"]
    assert manifest["sweep_config"]["max_new_tokens_safety"] == 2


def test_sweep_resumes_without_repeating_completed_cells(tiny, small_run):
    before = (small_run / "results.jsonl").read_text(encoding="utf-8")
    config = SweepConfig(
        layers=(1, 2),
        alpha_multipliers=(-1.0,),
        sham_seeds=(11,),
        max_new_tokens_structured=2,
        max_new_tokens_capability=2,
        max_new_tokens_safety=2,
        include_sae=False,
    )
    run_sweep(small_run, config, handle=tiny, refit=False)
    assert (small_run / "results.jsonl").read_text(encoding="utf-8") == before


def test_budget_stop_leaves_completed_cells_intact(tiny, tmp_path):
    config = SweepConfig(
        layers=(1,),
        alpha_multipliers=(-1.0,),
        sham_seeds=(11,),
        max_new_tokens_structured=2,
        max_new_tokens_capability=2,
        max_new_tokens_safety=2,
        include_sae=False,
        budget_hours=0.0,
    )
    run_sweep(tmp_path, config, handle=tiny)
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["stopped_on_budget"] is True
    assert status["cells_completed"] == 0
    assert not (tmp_path / "results.jsonl").exists()


def test_bundles_are_never_overwritten(tiny, small_run):
    from directions import BundleExistsError, read_bundle

    bundle = read_bundle(direction_id("c1", 1), root=small_run / "directions")
    assert bundle.manifest.layer == 1
    with pytest.raises(BundleExistsError):
        fit_directions(
            tiny,
            extract_contrast_means(
                tiny,
                [p for p in load_contrast_set("c1") if p.scenario_family == "kv_store"][:2],
                concept="c1",
                split="train",
            ),
            load_contrast_set("c1"),
            root=small_run / "directions",
            extraction_config={},
            layers=[1],
        )


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------


def _synthetic_cell(key: str, arm: str, comply: float, safety_comply: float) -> dict:
    injection = [
        {"item_id": f"i{i}", "complied": i < comply * 10, "margin": 1.0 if i < comply * 10 else -1.0}
        for i in range(10)
    ]
    retain = [
        {"item_id": f"r{i}", "tool_dependent": i % 2 == 0, "logprob": -10.0, "tokens": 10,
         "nll_per_token": 1.0, "perplexity": 2.718}
        for i in range(4)
    ]
    structured = [{"item_id": f"s{i}", "fully_valid": True, "json_valid": True,
                   "name_valid": True, "arguments_valid": True} for i in range(4)]
    capability = [{"item_id": f"c{i}", "tool_dependent": i % 2 == 0, "correct": True}
                  for i in range(4)]
    safety = [
        {"item_id": f"f{i}", "channel": "user" if i % 2 else "tool", "category": "x",
         "refused": i >= safety_comply * 10, "complied": i < safety_comply * 10,
         "emitted_tool_call": False, "complied_in_text": i < safety_comply * 10}
        for i in range(10)
    ]
    return {
        "cell_key": key,
        "arm": arm,
        "concept": "c1",
        "layer": 14,
        "parameter": "-",
        "seconds": 1.0,
        "records": {
            "injection": injection,
            "retain": retain,
            "structured": structured,
            "capability": capability,
            "safety": safety,
        },
    }


def _write_cells(path: Path, cells: list[dict]) -> Path:
    results = path / "results.jsonl"
    results.write_text(
        "\n".join(json.dumps(cell, sort_keys=True) for cell in cells) + "\n",
        encoding="utf-8",
    )
    return results


def test_analysis_selects_an_eligible_arm(tmp_path):
    cells = [
        _synthetic_cell("none|-|-|-", "none", comply=0.8, safety_comply=0.0),
        _synthetic_cell("projection|c1|14|-", "projection", comply=0.3, safety_comply=0.0),
    ]
    report = build_tables(_write_cells(tmp_path, cells))
    assert not report["kill_gate_triggered"]
    assert report["selected"]["cell_key"] == "projection|c1|14|-"
    assert report["selected"]["gates"]["injection_reduction"] == pytest.approx(0.5)
    assert report["tranche_b_layers"] == [13, 14, 15]


def test_analysis_triggers_the_kill_gate_when_safety_regresses(tmp_path):
    cells = [
        _synthetic_cell("none|-|-|-", "none", comply=0.8, safety_comply=0.0),
        _synthetic_cell("projection|c1|14|-", "projection", comply=0.2, safety_comply=0.5),
    ]
    report = build_tables(_write_cells(tmp_path, cells))
    assert report["kill_gate_triggered"]
    row = [cell for cell in report["cells"] if cell["arm"] == "projection"][0]
    assert row["gates"]["checks"]["safety"]["pass"] is False
    assert row["gates"]["effective"] is True  # the effect is real; the cost is not acceptable


def test_analysis_requires_a_baseline(tmp_path):
    cells = [_synthetic_cell("projection|c1|14|-", "projection", 0.2, 0.0)]
    with pytest.raises(SystemExit, match="baseline"):
        build_tables(_write_cells(tmp_path, cells))


def test_analysis_tolerates_a_truncated_final_line(tmp_path):
    cells = [
        _synthetic_cell("none|-|-|-", "none", 0.8, 0.0),
        _synthetic_cell("projection|c1|14|-", "projection", 0.3, 0.0),
    ]
    results = _write_cells(tmp_path, cells)
    with results.open("a", encoding="utf-8") as handle:
        handle.write('{"cell_key": "truncat')
    report = build_tables(results)
    assert len(report["cells"]) == 2


def test_bootstrap_intervals_are_reproducible(tmp_path):
    cells = [
        _synthetic_cell("none|-|-|-", "none", 0.8, 0.0),
        _synthetic_cell("projection|c1|14|-", "projection", 0.3, 0.0),
    ]
    results = _write_cells(tmp_path, cells)
    first = build_tables(results)
    second = build_tables(results)
    assert first["cells"][1]["contrast"] == second["cells"][1]["contrast"]
    assert BOOTSTRAP_SEED == 20260824


def test_cell_summary_splits_metrics_by_tool_dependence():
    summary = summarize_cell(_synthetic_cell("k", "none", 0.5, 0.0))
    assert summary.point["capability_tool_dependent"] == 1.0
    assert summary.point["retain_perplexity_tool_dependent"] > 0
    assert summary.point["safety_compliance_user_channel"] >= 0
