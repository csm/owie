# DECISIONS

Settled decisions and approved deviations. Append-only in spirit: superseded
entries are marked, not deleted.

---

## A. Inherited from the project plan

Carried verbatim from `doc/TODO.md`. Not to be reopened without explicit human
approval.

1. Do not fork OpenCode, LangGraph, Codex CLI, Claude Code, or another agent harness.
2. Apply interventions inside a local inference server with access to the model forward pass.
3. Serve an OpenAI-compatible HTTP API; existing harnesses connect later by changing `baseURL`.
4. Open-weight Hugging Face models only.
5. Projection ablation is the primary primitive: `x' = x - r̂(r̂ᵀx)`.
6. Additive steering is a comparison arm, not the default.
7. SAE feature clamping only if suitable features already exist for the chosen model.
8. Exact orchestration state — message provenance and explicit phase flags — is the initial gate. No learned probe first.
9. Single-turn layer sweep completes before any agent loop is built.
10. A small deterministic ReAct loop precedes any production harness integration.
11. Hugging Face Transformers and ordinary PyTorch hooks through Phase 2.
12. No vLLM work until Phase 2 demonstrates an effect worth scaling.
13. The deliverable is effect sizes with uncertainty, including credible negative or null results.

Out of scope: UI, generalized steering framework, multiple model families,
distributed serving, agent-harness fork.

---

## B. Checkpoint 0 decisions

Decided by Casey Marshall, 2026-08-23.

### B1. Compute — SUPERSEDED 2026-08-23 by B1′

*Original decision:* the session container had no GPU and no route to Hugging
Face, so model-bearing phases were routed to hardware the human operates, and
the work was split between two environments.

*Why superseded:* development moved to the human's own laptop. There is now one
machine that can run every phase. The split no longer exists and its
consequences — CPU-only development here, vendored tokenizer files, a separate
GPU host — are all void.

### B1′. Compute — one machine, Apple M3 Max, MPS

Decided 2026-08-23 (revision), pending re-approval of the budget.

All checkpoints run on a single MacBook Pro (`Mac15,9`, M3 Max, 48 GiB unified
memory, 37.4 GiB addressable by Metal, 270 GiB free disk). Accelerator is
Metal/MPS; there is no CUDA. Hugging Face, arXiv, and PyPI are all reachable.

Capacity is ample — an 8B model in bf16 is ~16 GB. **Throughput is the binding
constraint.** Measured 239 GB/s memory bandwidth caps autoregressive decode near
15 tok/s and realistically 5–10 tok/s under HF Transformers, roughly 5× slower
than the 24 GB CUDA card the original cost model assumed.

Consequence: the Phase 0 sweep is restructured — forward-only metrics wherever
the metric permits, coarse-then-refine layer band, pre-registered generation
caps. Details in `PREFLIGHT.md` §6 and §7. Arms and the safety eval are **not**
reduced.

Sub-item resolved by B5 below. No paid compute is provisioned.

### B2. Model — `meta-llama/Llama-3.1-8B-Instruct`

Chosen for first-class `ipython`-role tool-output provenance and for the
existence of a layer-matched Instruct SAE. Decision stands.

Both caveats attached to it on 2026-08-23 are now **resolved by direct
measurement**:

- **Revision SHA — resolved.** `0e9e39f249a16976918f6564b8830bc894c89659`
  (last modified 2024-09-25). To be recorded in `docs/PINS.md`. The requirement
  that extraction fail loudly against a placeholder revision stands.
- **Gating — resolved.** The repository is `gated: manual`; the token in the
  local credential store has access, confirmed by authenticated 200 responses at
  the pinned SHA and a successful tokenizer download. The token stays out of the
  repository, run manifests, and logs.

Pre-registered replication target if Llama access ever becomes an obstacle:
`Qwen2.5-7B-Instruct` (`a09a35458c702b33eeacc393d103063234e8bc28`), accepting
the weaker provenance structure described in `PREFLIGHT.md` §3.

Re-confirmed 2026-08-23 after the throughput picture changed: **8B stands.**
Capacity is ample at 37.4 GiB addressable and the SAE arm depends on this
model. A 3B pilot is approved separately as a non-reporting substrate — see B6.

### B3. First target concept — fit all three, select on held-out data

Three contrast sets are built and three directions fitted per layer:

- **C1** source-agnostic imperative compliance;
- **C2** general instruction-following;
- **C3** refusal/harmlessness at reversed polarity.

Selection happens on held-out data under a rule frozen in
`EXPERIMENT_PROTOCOL.md` **before** the sweep runs. This roughly triples Phase 0
contrast-set and sweep cost relative to a single concept; that cost is accepted.

Selection rule as proposed in `PREFLIGHT.md` §5: maximise held-out
injection-resistance gain subject to hard retain-set, structured-output, and
safety constraints, ties broken toward lower collateral cost. Numeric
thresholds to be fixed before collection.

### B4. SAE arm — include if suitable features exist; they do

`Goodfire/Llama-3.1-8B-Instruct-SAE-l19`
(`f6775a221e47b44233af4bac2c7b65189265519a`, ungated) is trained on the pinned
model itself (not a base-model proxy) and is suitable. The clamping arm is
therefore approved for Phase 0 and for the Checkpoint 5 primary arms.

Note added 2026-08-23: it ships as a `.pth` torch pickle, not safetensors — see
D12.

Accepted constraint: that SAE covers **layer 19 only**, so the arm is
layer-locked. Handling, pre-registered: always report the SAE arm at layer 19
alongside layer-19 projection ablation as its layer-matched companion, and if
the swept optimum is a different layer, state the depth difference explicitly
rather than absorbing it into the comparison.

`OpenMOSS-Team/Llama-Scope` (all 32 layers; the previously recorded `fnlp/`
identifier is stale) is **not** a primary arm — it is trained on
Llama-3.1-8B **base**, and applying a base dictionary to Instruct activations
introduces an unquantified mismatch on exactly the tuning under study. Optional
exploratory check only, reported separately.

### B5. Compute budget — 3 days wall clock for the Phase 0 sweep

Decided by Casey Marshall, 2026-08-23. Resolves the open sub-item in B1′ and
D9.

The Phase 0 sweep gets **3 days (~72 h)** of otherwise-idle laptop time. This
is a ceiling, not a target. It sits mid-range in the 2–5 day estimate, so it may
bind, and the consequences of binding are pre-registered in `PREFLIGHT.md` §6
**before any result is seen**:

- the sweep writes raw JSONL incrementally, one record per completed cell, and
  is resumable; a wall-clock stop destroys no completed work;
- tranche A — the full coarse pass at **every arm**, including layer 19 and its
  projection companion — completes before refinement begins, so a stop yields a
  complete coarse sweep rather than a dense band with missing arms;
- if tranche A has not completed at 72 h, **stop and report it as far as it
  got**. Recovering time by dropping an arm, dropping the safety eval, or
  shortening the generation cap mid-run is **prohibited** — each trades an
  honest partial result for a dishonest complete-looking one;
- the alpha grid is the largest single multiplier in the cell count and is the
  first thing to shrink, as a number fixed in `EXPERIMENT_PROTOCOL.md` before
  collection, never as a mid-run adjustment;
- **overrunning 72 h requires a fresh ruling recorded here.** It is not a
  judgement call for whoever is watching the run.

**Not covered:** Checkpoint 5's paired replay, costed separately at 4–10 days.
It needs its own budget ruling before it starts.

### B6. 3B pilot — approved, placed at the checkpoints that need it

Decided by Casey Marshall, 2026-08-23. Resolves D10.

`meta-llama/Llama-3.2-3B-Instruct`
(`0cb88a4f764b7a12671c53f0838cd831a0843b95`, access confirmed) is approved as a
**pilot substrate**. It shares the Llama 3 template family, so every
chat-template finding in `PREFLIGHT.md` §3 transfers unchanged, and it runs
roughly 3× faster.

Planned, not run now. Two placements:

- **Checkpoint 4 — the determinism acceptance test.** Primary placement. MPS
  determinism is unproven and a failure invalidates Checkpoint 5's
  paired-replay design; the test does not care about model quality.
- **Checkpoint 2 — a dry run** of extraction → fit → a two-layer sweep slice,
  to exercise the pipeline end to end before spending any of the B5 budget on
  8B. Charged to development time, not to the 72 h.

**Hard constraint:** 3B has no matched SAE and is not the pinned model. It
never carries a reported result, never appears in an effect-size table, and
never substitutes for an 8B arm. Runs against it are tagged as pilot runs in
their manifests so they cannot later be mistaken for primary data.

---

## C. Approved deviations

**C1. Phase 0 sweep restructured for MPS throughput** — 2026-08-23, approved
alongside B1′ and B5. Coarse-then-refine layer band (even layers 10–26, then
±1 around the best, with layer 19 always swept for the SAE arm) in place of a
dense 8–28 sweep; metrics split into forward-only and length-capped-decode
classes. Motivation and figures in `PREFLIGHT.md` §6 and §7. **No arm is
dropped and the safety eval is unreduced.** Recorded as a deviation, not folded
into the protocol silently, so that any further trimming under time pressure
remains visible. B5 fixes the time box at 72 h and pre-registers exactly what
may and may not be cut if it binds.

---

## D. Open items requiring a ruling

Tracked here so they are not settled silently by whoever writes the code first.

| # | Item | Status | Proposal / resolution |
| --- | --- | --- | --- |
| D1 | Model revision SHA pin | **RESOLVED** | `0e9e39f249a16976918f6564b8830bc894c89659`; record in `docs/PINS.md` |
| D2 | Activation extraction position — tool-content positions (a), last prompt token (b), or first generated token (c) | open, before Checkpoint 2 | **(a)**; store (b) and (c) as diagnostics. Now cheaply retirable by reading CAST in full — arXiv is reachable. `PREFLIGHT.md` §5 |
| D3 | Single-layer vs multi-layer band application | open, before Checkpoint 2 | single-layer first, matching the Checkpoint 1 signatures; a band is a pre-registered addition, not a post-hoc knob |
| D4 | Numeric kill-gate thresholds for retain-set, structured-output, safety | open, before Checkpoint 2 | fix in `EXPERIMENT_PROTOCOL.md` before the sweep |
| D5 | Boundary-token disposition rule | open, Checkpoint 3 | **now demonstrated to be unavoidable** — a single token spans template and content characters in the most trivial example, and no tokenization separates them. Define one disposition, log every ambiguous token. `PREFLIGHT.md` §3 finding 2 |
| D6 | Whether tokenizer files may be vendored into this repo | **MOOT** | the tokenizer is fetched directly at the pinned revision; there is nothing to vendor |
| D7 | `transformers` major version — v5 is current | open, before Checkpoint 3 | **evidence added**: the same template renders different bytes under bare Jinja2 vs `transformers` 5.15.1 (`ensure_ascii` policy). 5.15.1 is verified installable and renders the pinned template; pin only after byte-equality is asserted in a test. `PREFLIGHT.md` §3 finding 6 |
| D8 | How code reaches the GPU host | **MOOT** | there is no separate host |
| D9 | Wall-clock budget for the Phase 0 sweep on this machine | **RESOLVED** | 3 days / ~72 h, with pre-registered truncation rules. See B5 |
| D10 | Whether to add a 3B pilot model | **RESOLVED** | approved as a non-reporting pilot; placed at Checkpoint 4 (determinism) and as a Checkpoint 2 dry run. See B6 |
| D11 | Tool content can forge role headers — the template does not escape `<\|eot_id\|>` etc. | open, Checkpoint 3/4 | span provenance must come from the renderer's bookkeeping, never from recognising delimiters in output; add to the Checkpoint 4 injection task set. `PREFLIGHT.md` §3 finding 4 |
| D12 | SAE ships as `.pth` (torch pickle), off the `safetensors` baseline | open, before Checkpoint 2 | load once under `weights_only=True`, convert to safetensors, hash both, record the conversion in the manifest |
| D13 | Wall-clock budget for Checkpoint 5 paired replay | open, before Checkpoint 5 | costed at 4–10 days; **not** covered by B5 and needs its own ruling |
| D14 | `clamp_feature` reads the current coefficient by projection, not through the SAE encoder | open, before the Checkpoint 2 SAE arm | the Checkpoint 1 signature is `clamp_feature(hidden, decoder_direction, value, span_mask)` and was implemented faithfully. A true SAE clamp needs the **encoder** (with bias and nonlinearity) to read the feature activation; projection onto the decoder direction coincides with it only for an orthonormal dictionary, which SAE dictionaries are not. Widening the signature is a pre-registered change, not a quiet fix. Documented in `interventions/kernel.py` |
