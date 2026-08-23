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

### B1. Compute — experiment runs on the human's own GPU

The session container has no GPU and no route to Hugging Face (see
`PREFLIGHT.md` §1). Model-bearing phases run on hardware the human operates.

Consequence: Checkpoint 1 and the model-free parts of Checkpoints 3 and 4 are
built and tested in this repository on CPU; Checkpoint 2 and the weight-bearing
parts of 3 and 5 execute on the GPU host. No cloud GPU is provisioned and no
egress-policy change is requested.

### B2. Model — `meta-llama/Llama-3.1-8B-Instruct`

Chosen for first-class `ipython`-role tool-output provenance and for the
existence of a layer-matched Instruct SAE. Gated repo; license acceptance and
token handling are the GPU host's responsibility and no token enters this
repository.

**Blocking sub-item:** the immutable revision SHA is unresolved — Hugging Face
is unreachable from the container where this was decided. It must be captured
on first download and recorded in `docs/PINS.md` before any extraction run.
Extraction must fail loudly against a placeholder revision.

Pre-registered replication target if Llama access becomes an obstacle:
`Qwen2.5-7B-Instruct`, accepting the weaker provenance structure described in
`PREFLIGHT.md` §3.

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

`Goodfire/Llama-3.1-8B-Instruct-SAE-l19` is trained on the pinned model itself
(not a base-model proxy) and is suitable. The clamping arm is therefore
approved for Phase 0 and for the Checkpoint 5 primary arms.

Accepted constraint: that SAE covers **layer 19 only**, so the arm is
layer-locked. Handling, pre-registered: always report the SAE arm at layer 19
alongside layer-19 projection ablation as its layer-matched companion, and if
the swept optimum is a different layer, state the depth difference explicitly
rather than absorbing it into the comparison.

`fnlp/Llama-Scope` (all 32 layers) is **not** a primary arm — it is trained on
Llama-3.1-8B **base**, and applying a base dictionary to Instruct activations
introduces an unquantified mismatch on exactly the tuning under study. Optional
exploratory check only, reported separately.

---

## C. Approved deviations

None yet.

---

## D. Open items requiring a ruling

Tracked here so they are not settled silently by whoever writes the code first.

| # | Item | Needed by | Proposal |
| --- | --- | --- | --- |
| D1 | Model revision SHA pin | before Checkpoint 2 | capture on GPU host, record in `docs/PINS.md` |
| D2 | Activation extraction position — tool-content positions (a), last prompt token (b), or first generated token (c) | before Checkpoint 2 | **(a)**; store (b) and (c) as diagnostics. `PREFLIGHT.md` §5 |
| D3 | Single-layer vs multi-layer band application | before Checkpoint 2 | single-layer first, matching the Checkpoint 1 signatures; a band is a pre-registered addition, not a post-hoc knob |
| D4 | Numeric kill-gate thresholds for retain-set, structured-output, safety | before Checkpoint 2 | fix in `EXPERIMENT_PROTOCOL.md` before the sweep |
| D5 | Boundary-token disposition rule | Checkpoint 3 | define, document, log every ambiguous token |
| D6 | Whether tokenizer files may be vendored into this repo, enabling span-mapping tests to run and be CI-checked here | Checkpoint 3 | requested; a few MB, no weights |
| D7 | `transformers` major version — v5 is current on PyPI | before Checkpoint 3 | pin only after byte-equality is verified on the GPU host |
| D8 | How code reaches the GPU host | before Checkpoint 2 | clone this branch, unless the human prefers otherwise |
