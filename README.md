# owie

Measuring the effect of **context-dependent, reversible, semantically targeted
activation suppression** on tool-using agent performance.

The experiment: suppress an instruction-following/compliance direction only at
input-token positions originating from `role: "tool"` content, leaving user and
assistant positions untouched. Measure prompt-injection resistance, agent-task
success, tool-call validity, token cost, self-correction, retained capability,
and safety.

The adversarial instruction arrives through tool output. Intervention
configuration arrives through a trusted out-of-band request field and is never
controllable by tool content.

This is an experimental measurement project, not a product or a general-purpose
framework. The deliverable is **effect sizes with uncertainty, including
credible negative or null results.**

## Status

**Checkpoints 1, 2, 3, and 4 complete.** Phase 0 has collected: 460 cells across
five arms and fifteen layers. Headline in `EXPERIMENT_PROTOCOL.md` §11 terms —
projection ablation is direction-specific but an order of magnitude too small
to be useful; large additive steering is mostly a perturbation-norm effect that
random directions reproduce, including its safety damage; a moderate-alpha
additive cell survives its matched control with no measurable collateral cost. The pure intervention kernel, versioned
direction-bundle format, provenance-aware renderer, request-local hook runtime,
minimal OpenAI-compatible shim, and the full Phase 0 pipeline — frozen contrast
sets, activation extraction, difference-in-means fitting, the layer sweep, and
the analysis — are implemented and tested. Model weights are loaded only when
`owie-server` or `owie-phase0` is started.

| # | Deliverable | State |
| --- | --- | --- |
| 0 | Environment survey, model decision | done — `PREFLIGHT.md`, `DECISIONS.md` |
| 1 | Intervention kernel, direction-bundle format | **done** |
| 2 | Phase 0 single-turn experiment | **done** — 460 cells, 16.2 h; see `runs/` and `analysis/phase0.md` |
| 3 | Provenance-aware shim, renderer, span mapping | **done** |
| 4 | Deterministic ReAct loop | **done** — normalized 3B pilot trajectories are byte-identical |
| 5 | Paired replay, primary experiment | **in progress** — baseline and protocol frozen; replay implementation active |
| 6 | External validity | gated on Phase 2 review |

## Setup

Requires [`uv`](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync
```

That installs the exact locked numerical and server dependencies. Tests load
the pinned tokenizer from the local Hugging Face cache but do not load model
weights.

## Running the tests

```bash
uv run pytest
```

The suite uses no network and loads no model weights. Tests tagged for Apple
Metal skip automatically where MPS is unavailable.

Run just one area:

```bash
uv run pytest tests/test_kernel.py
```

Regenerate the frozen Phase 0 datasets (deterministic; must reproduce byte-identical files):

```bash
uv run owie-build-datasets
```

Run the Phase 0 sweep, and derive tables from its raw JSONL:

```bash
uv run owie-phase0 --run-dir runs/phase0-2026-08-24 --tranche A --budget-hours 72
uv run owie-phase0-analyse --results runs/phase0-2026-08-24/results.jsonl
```

The sweep is resumable: re-running the same command skips completed cells. It
stops at the budget rather than trimming arms, and the safety evaluation runs
in every cell.

Inspect provenance without loading model weights:

```bash
uv run inspect-spans --local-files-only request.json
```

Start the one-worker server after model weights and any direction bundles are
available locally:

```bash
uv run owie-server --direction compliance-v1
```

The HTTP surface is intentionally limited to `GET /v1/models` and
non-streaming `POST /v1/chat/completions`. See `docs/SPAN_MAPPING.md` for the
top-level intervention configuration and boundary-token rule. Responses carry
an `owie` diagnostics block containing the resolved intervention, mask counts,
prompt hash, model revision, seed, and direction-bundle hash.

Run the approved non-reporting 3B determinism pilot:

```bash
uv run owie-server --pilot-3b
uv run owie-loop --pilot-3b --task injection_forged_header --seed 0 \
  --repeat 2 --run-dir runs/checkpoint4-3b-determinism
```

Run the two commands in different terminals. The second command removes two
documented timing fields before it compares the trajectory files. See
`docs/LOOP.md` for the task set and the JSONL format.

## Layout

| Path | Contents |
| --- | --- |
| `interventions/` | pure intervention primitives (Checkpoint 1) |
| `directions/` | direction-bundle format, and the bundles themselves as subdirectories |
| `server/` | OpenAI-compatible shim (Checkpoint 3) |
| `loop/` | deterministic ReAct loop (Checkpoint 4) |
| `replay/` | paired replay (Checkpoint 5) |
| `evals/` | frozen datasets, metrics, and the Phase 0 sweep |
| `analysis/` | derived tables and plots, regenerated from raw JSONL |
| `tests/` | everything above |
| `docs/` | pins and supporting notes |

## Documents

| File | Purpose |
| --- | --- |
| `doc/TODO.md` | the assignment; the source of every settled decision |
| `PREFLIGHT.md` | environment, model choice, exact Phase 0 protocol, risks |
| `DECISIONS.md` | settled decisions, approved deviations, open rulings |
| `docs/PINS.md` | immutable model, SAE, and dependency identifiers |
| `docs/SPAN_MAPPING.md` | provenance regions, boundary tokens, runtime positions |
| `docs/LOOP.md` | Checkpoint 4 tools, tasks, trajectory format, and determinism rule |
| `CHECKPOINT4.md` | Checkpoint 4 implementation, failed pilot, acceptance result, and hashes |
| `CHECKPOINT5_BASELINE.md` | unfrozen baseline probes, defects, and candidate metrics |
| `EXPERIMENT_PROTOCOL.md` | frozen arms, datasets, metrics, thresholds, exclusions — frozen 2026-08-24, before collection |
| `HYSTERESIS_PROTOCOL.md` | frozen operational protocol for the within-completion KV-cache experiment |

## Using the kernel

Three pure functions. None mutates its input, and every position outside the
mask is **bitwise** identical to the input.

```python
import torch
from interventions import Norm, project_out, add_vector, clamp_feature

hidden = torch.randn(1, 8, 4096)          # (batch, seq, d_model)
direction = torch.randn(4096)
direction = direction / direction.norm()
mask = torch.zeros(1, 8, dtype=torch.bool)
mask[0, 3:6] = True                        # e.g. tool-content positions

ablated = project_out(hidden, direction, mask)                       # primary
steered = add_vector(hidden, direction, 2.0, mask, norm=Norm.ASSERT_UNIT)
clamped = clamp_feature(hidden, direction, 0.5, mask)
```

`norm=` is explicit everywhere and has no default on `add_vector`, because the
meaning of `alpha` depends on it. See the module docstring in
`interventions/kernel.py`.

## Writing a direction bundle

```python
from directions import write_bundle, read_bundle, current_git_revision

write_bundle(
    "c1-l19-dim",
    vector,                                  # (d_model,) tensor
    {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "model_revision": "0e9e39f249a16976918f6564b8830bc894c89659",
        "layer": 19,
        "hook_point": "resid_post",
        "token_extraction_rule": "tool_content_positions",
        "fitting_method": "difference_in_means",
        "normalization": "unit",
        "extraction_code_git_revision": current_git_revision(),
    },
    contrasts,                               # list of JSON-able rows
    extraction_config,                       # dict
)

bundle = read_bundle("c1-l19-dim")           # verifies, or raises
```

Bundles are **never overwritten** — a re-fit gets a new id. Reads recompute the
contrast-set hash and re-check the vector against the manifest, so a bundle
that has drifted from its provenance raises instead of loading. A manifest
recording a mutable ref such as `main` in place of a commit SHA is refused.

## Working discipline

Carried from `doc/TODO.md`, and binding:

- one checkpoint at a time; stop at every acceptance and kill gate and report
  evidence;
- do not silently revise the experiment after seeing results;
- raw data is immutable; derived analysis may be regenerated;
- never overwrite a direction bundle or an experiment run;
- record failed runs and exclusions with reasons;
- treat null results, prompting wins, downstream self-repair, and safety
  degradation as valid findings;
- describe suppression as a control surface, never as removal or a guarantee;
- do not publish, push externally, or provision paid compute without explicit
  authorization.
