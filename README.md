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

**Checkpoint 1 complete.** The pure intervention kernel and the versioned
direction-bundle format are implemented and tested. No model weights have been
downloaded and nothing loads a model.

| # | Deliverable | State |
| --- | --- | --- |
| 0 | Environment survey, model decision | done — `PREFLIGHT.md`, `DECISIONS.md` |
| 1 | Intervention kernel, direction-bundle format | **done** |
| 2 | Phase 0 single-turn experiment | not started |
| 3 | Provenance-aware shim, renderer, span mapping | not started |
| 4 | Deterministic ReAct loop | not started |
| 5 | Paired replay, primary experiment | not started |
| 6 | External validity | gated on Phase 2 review |

## Setup

Requires [`uv`](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync
```

That installs `torch`, `safetensors`, and `numpy` plus the dev group. It
deliberately does **not** install `transformers` or `huggingface_hub`:
Checkpoint 1's acceptance criterion is that its tests pass without loading a
model, and the cheapest way to keep that true is for the dependency not to
exist yet.

## Running the tests

```bash
uv run pytest
```

161 tests, about a second, no network and no weights. Tests tagged for Apple
Metal skip automatically where MPS is unavailable.

Run just one area:

```bash
uv run pytest tests/test_kernel.py
```

## Layout

| Path | Contents |
| --- | --- |
| `interventions/` | pure intervention primitives (Checkpoint 1) |
| `directions/` | direction-bundle format, and the bundles themselves as subdirectories |
| `server/` | OpenAI-compatible shim (Checkpoint 3) |
| `loop/` | deterministic ReAct loop (Checkpoint 4) |
| `replay/` | paired replay (Checkpoint 5) |
| `evals/` | task sets and success predicates |
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
| `EXPERIMENT_PROTOCOL.md` | frozen arms, datasets, metrics, exclusions — written before Checkpoint 2 collection |
| `HYSTERESIS_PROTOCOL.md` | the KV-cache experiment — written before Checkpoint 5 |

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
