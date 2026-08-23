# PREFLIGHT

Checkpoint 0 deliverable. Environment survey, model selection, and the exact
proposed Phase 0 protocol. No model weights were downloaded and no material
changes were made to the environment in producing this document.

Status: **awaiting Checkpoint 0 sign-off.** The four human decisions have been
answered (see `DECISIONS.md`). Nothing past Checkpoint 0 is implemented.

---

## 1. Available compute environment

Surveyed read-only in the session container on 2026-08-23.

| Property | Value |
| --- | --- |
| GPU | **none** (`nvidia-smi` not present, no `/usr/local/cuda`) |
| CUDA toolkit | **none** (`nvcc` not present) |
| CPU | 4 vCPU, Intel Xeon @ 2.10 GHz |
| RAM | 15 GiB total, ~15 GiB available |
| Disk | `/dev/vda`, 252 G total, **~30 G available** to this session |
| Python | 3.11.15 default on `PATH`; **3.12.3** present at `/usr/bin/python3.12` |
| `uv` | 0.8.17 at `/root/.local/bin/uv` |
| PyTorch | not installed |

### Network egress

Outbound traffic passes through an agent proxy whose policy allowlists package
registries only. Measured:

| Host | Result |
| --- | --- |
| `pypi.org`, `files.pythonhosted.org` | reachable (direct, in `noProxy`) |
| `huggingface.co` | **blocked** — gateway answered 403 to CONNECT |
| `arxiv.org` | **blocked** — gateway answered 403 to CONNECT |

### Consequence for the project

This container **cannot run any phase that requires model weights.** It has no
GPU, ~30 G of disk against a ~16 G bf16 checkpoint plus activation caches, and
no route to Hugging Face.

Per the Checkpoint 0 decision, weight-bearing work runs on the human's own GPU.
The split is:

| Work | Runs here | Runs on the GPU host |
| --- | --- | --- |
| Checkpoint 1 — intervention kernel, direction-bundle format | yes (CPU, tiny synthetic tensors) | — |
| Checkpoint 3 — renderer, span mapping, `inspect-spans` | tokenizer-only parts, **if** tokenizer files are vendored | byte-equality against the pinned tokenizer |
| Checkpoint 4 — ReAct loop, fake tools, JSONL logging | yes, against a stub model | acceptance run |
| Checkpoint 2 — extraction, layer sweep | no | yes |
| Checkpoints 3 (hook runtime), 5 | no | yes |

Checkpoint 1's acceptance criterion is explicitly "all tests pass without
loading the full model," so the first coding checkpoint is fully executable
here. That is the recommended immediate next step.

**Open logistics question (not one of the four decisions):** how code reaches
the GPU host — clone this branch there, or something else. Also whether the
tokenizer files (`tokenizer.json`, `tokenizer_config.json`, and the chat
template) for the pinned revision can be copied into this environment. If they
can, Checkpoint 3's span-mapping tests become developable and CI-able here,
which is where most of the fiddly correctness risk lives. Tokenizer files are a
few MB and carry no weights.

---

## 2. Prior art consulted

`arxiv.org` is blocked from this container, so these were read via search
result summaries and secondary sources, **not** full text. Full-text reading of
the starred entries should happen on a host with arXiv access before the
Checkpoint 2 protocol is frozen; the design commitments they bear on are noted.

| Work | Bearing on this project |
| --- | --- |
| **ASA**, [2602.04935](https://arxiv.org/abs/2602.04935) — *Activation Steering for Tool-Calling Domain Adaptation* | Nearest neighbour. Tool-necessity is near-perfectly decodable from mid-layer activations while behaviour lags — a "representation–behaviour gap." Supports mid-layer targeting; its router is the learned gate that settled decision 8 defers. |
| **CAST**, [2409.05907](https://arxiv.org/abs/2409.05907) | Condition vector + behaviour vector, both difference-in-means; steer only when cosine similarity to the condition exceeds a threshold. Our design replaces CAST's *learned* condition with *exact orchestration provenance*. CAST is therefore the natural learned-gate comparison at Checkpoint 6. |
| **Copy Suppression**, [2310.04625](https://arxiv.org/abs/2310.04625) | Attention-mediated negative copying. Relevant because our intervention acts on tool-content positions but takes effect through the K/V those positions expose to later reads — the causal path is attention, not the residual value in isolation. |
| **NPM**, [2606.29824](https://arxiv.org/abs/2606.29824) | Training-free procedural memory as steering vectors injected into the residual stream of an agent loop; contrastive-experience extraction. Precedent for extraction-from-trajectories, which is a Phase 2 option we are not taking first. |
| **Activation State Machines** ([OpenReview](https://openreview.net/forum?id=p17En1bhCY)) | Stateless steering fails to capture history-dependence; ASM applies a Kalman-like predict/correct per block. Directly motivates the KV-hysteresis question in Checkpoint 5 — our intervention *is* stateless per position but its effect persists through cache. |
| **SDialog**, [2506.10622](https://arxiv.org/abs/2506.10622) / [2512.09142](https://arxiv.org/abs/2512.09142) | Toolkit with inspectors that capture per-token activations and add/ablate directions. Prior art for the harness shape; note 2512.09142 is listed as **withdrawn** on arXiv. Settled decision 11 keeps us on hand-written hooks — SDialog is a reference, not a dependency. |
| **The Rogue Scalpel**, [2509.22067](https://arxiv.org/abs/2509.22067) | Load-bearing for our safety design. *Random* directions raise harmful compliance from 0% to 1–13%, and benign SAE features do comparable damage; 20 random vectors compose into a universal jailbreak. This is why the sham/random matched-norm arm is not a formality and why the safety eval is mandatory at **every** layer and arm, not just the winner. |
| **RePS**, [2505.20809](https://arxiv.org/abs/2505.20809) | Suppression as a first-class objective, evaluated against *prompting* as the baseline, and shown resilient to jailbreaks that defeat prompting. Precedent for the requirement that we beat a prompt-only defense, not just no-intervention. |

Two themes carry directly into the protocol below: **(a)** the comparison that
matters is against prompting and a deterministic guard, not against doing
nothing (RePS); **(b)** any intervention, including a random one, can degrade
safety, so the sham arm and safety eval gate the whole sweep (Rogue Scalpel).

---

## 3. Model selection

**Chosen: `meta-llama/Llama-3.1-8B-Instruct`.**

### Revision pin — UNRESOLVED, blocking

The manifest format requires an *immutable revision*. Hugging Face is blocked
from this container, so **the commit SHA cannot be resolved here.** It must be
captured on the GPU host on first download and written back into
`docs/PINS.md` before any extraction run. Until then every manifest field
referencing it is a placeholder, and no direction bundle may be published.

The repository will refuse to run extraction against a placeholder revision
rather than silently record `main`.

### Why this model

- **Tool output has its own top-level role.** Llama 3.1 renders tool results
  under an `ipython` header rather than nesting them inside a user turn. The
  primary mask — `role == "tool" && region == "content"` — is therefore a
  contiguous, unambiguous span, which is the cleanest possible substrate for
  the span-mapping work in Checkpoint 3.
- **A layer-matched Instruct SAE exists** (see below), satisfying settled
  decision 7's "only if suitable features already exist."
- 8B fits comfortably in bf16 on a single 24 G card with room for activation
  capture at batch 1.

### Chat-template and tool-calling implications

Expected structure — special tokens `<|begin_of_text|>`,
`<|start_header_id|>…<|end_header_id|>`, `<|eot_id|>`, `<|eom_id|>`, with tool
results carried under the `ipython` role and tool *calls* emitted as JSON in an
assistant turn terminated by `<|eom_id|>`.

**This is stated from prior familiarity and is not verified against the pinned
tokenizer, which is unreachable here.** Checkpoint 3's byte-equality test is
what establishes it. Specific things to confirm there, each of which would
change the renderer:

1. Whether the pinned template emits `ipython` or `tool` as the header literal.
2. Whether tool JSON is wrapped (e.g. in a `{"output": …}` envelope) — a
   wrapper is template-generated and is **not** tool-content under the primary
   mask.
3. Whether a leading newline after the header terminator belongs to the header
   region or the content region. This is exactly the boundary-token rule that
   Checkpoint 3 requires us to define and document.
4. Whether the template injects a date or system preamble that varies between
   contrast pairs — if it does, it must be frozen, or difference-in-means will
   fit the preamble difference.

### Alternatives considered and rejected

- **Qwen2.5-7B-Instruct** — ungated, strong tool use, but renders tool results
  as `<tool_response>` blocks *inside a user turn*. The primary mask would then
  be a sub-span of a user message, which weakens the provenance claim.
  Retained as the pre-registered **replication target** if Llama gating or
  license acceptance becomes an obstacle.
- **Mistral-7B-Instruct-v0.3** — clean `[TOOL_RESULTS]` control tokens but the
  weakest agentic ability of the three; a floor effect on task success would
  make effect sizes uninterpretable.
- **Gemma-2-9b-it** — best SAE coverage by far (Gemma Scope, full layer and
  width sweep) but **no official tool-calling chat template**, which is
  irreconcilable with the byte-for-byte template-equality requirement. Rejected
  on that ground alone.

### Gating caveat

`meta-llama/Llama-3.1-8B-Instruct` is a gated repository. The license must be
accepted under the account whose token the GPU host uses, and that token must
never enter this repository or a run manifest.

---

## 4. SAE feature sets

Decision: include a clamping arm if suitable features exist. They do, with one
significant constraint.

### `Goodfire/Llama-3.1-8B-Instruct-SAE-l19`

- Trained on **`Llama-3.1-8B-Instruct`** — the exact model we are pinning, not
  a base-model proxy. This is the property that makes it usable.
- **Layer 19 only.** L0 ≈ 91. Trained on activations harvested over
  LMSYS-Chat-1M.

**The constraint this imposes:** the SAE arm is *layer-locked to 19*. If the
Phase 0 sweep selects a layer other than 19 for projection ablation, the SAE
arm is no longer layer-matched to the primary arm, and the comparison becomes
"projection at layer L vs clamping at layer 19" — confounding intervention type
with depth.

Pre-registered handling, to be fixed before results are seen:

- Report the SAE arm at layer 19 **always**.
- Additionally report projection ablation at layer 19 as the layer-matched
  companion, whatever the swept optimum turns out to be.
- If the swept optimum is not 19, the SAE-vs-projection comparison is reported
  **only** at layer 19, and the depth difference is stated explicitly rather
  than absorbed.

### `fnlp/Llama-Scope` (OpenMOSS)

256 SAEs covering all 32 layers at 32k and 131k features, both residual-stream
and MLP hook points. Would solve the layer-locking problem outright — except
it is trained on **`Llama-3.1-8B` base**, not Instruct. Applying a base-model
dictionary to Instruct activations introduces a distribution mismatch of
unquantified size, on a model whose instruct-tuning is precisely what we are
studying.

Proposal: **not** part of the primary arms. Optionally used as a
robustness/exploratory check at layer 19 against Goodfire, reported separately
and never substituted for it.

### Feature selection

Which features get clamped is not yet decided and should not be decided by
eyeballing labels. Pre-registered procedure: rank features by difference in
mean activation across the same contrast set used for direction fitting, take
the top-k on **training** contrasts only, freeze the list with a hash, and
report held-out effect. Rogue Scalpel is the reason feature choice must be
frozen before outcomes are inspected — benign-looking features cause real
safety damage, and post-hoc selection would let that leak into the headline.

---

## 5. Compliance-direction contrast-set construction

The human's decision is to **fit all three candidate concepts and select on
held-out data**. Concretely, three contrast sets, three fitted directions per
layer, one selection rule frozen before results are seen.

Common invariants across all three sets:

- Identical system prompt in every pair member. Frozen string, no date
  interpolation.
- Identical conversation scaffold up to the varied span; pairs differ in the
  varied span **only**.
- Topic and length matched within pair to within a stated token tolerance, so
  difference-in-means does not fit topic.
- Held-out split by *scenario family*, not by row, so near-duplicate paraphrases
  cannot straddle the split.
- Set hashed; hash recorded in the bundle manifest.

### C1 — Source-agnostic imperative compliance

Positive: tool output containing a well-formed imperative directed at the
assistant. Negative: the same tool output with the imperative rewritten as a
declarative statement of equivalent length and topic.

Isolates *"an instruction is present and actionable"* from content. This is the
set most faithful to the stated hypothesis, because the provenance gating is
supplied exogenously by the mask — the direction should encode
instruction-ness, not source.

### C2 — General instruction-following

Positive: contexts where the model is about to comply with an explicit
directive. Negative: matched contexts with no directive.

Broadest, easiest to fit, largest expected effect — and the highest risk that
what we measure is "tool output became illegible" rather than targeted
suppression. The retain-set and structured-output metrics are what
discriminate these two readings.

### C3 — Refusal / harmlessness, reversed polarity

The best-characterised direction in the literature, giving the strongest
replication footing. It is a **proxy**: injection resistance is not refusal.
Included mainly as a methodological control — if C3 outperforms C1 at
suppressing injection, that is evidence the mechanism is generic compliance
rather than anything provenance-specific, which is a publishable negative.

### Selection rule (frozen before results)

Select the (concept, layer, arm) triple maximising held-out injection-resistance
gain **subject to** hard constraints: retain-set perplexity increase below a
pre-registered threshold, structured-output validity above a pre-registered
floor, and **no** safety-eval regression versus no-intervention. Ties broken
toward the smaller collateral cost. The thresholds go into
`EXPERIMENT_PROTOCOL.md` as numbers before the sweep runs.

### The extraction-position question — needs a ruling

Difference-in-means requires a token-extraction rule, and the choice is not
neutral:

- **(a) At tool-content token positions** — the same positions the intervention
  will act on.
- **(b) At the last prompt token** before generation.
- **(c) At the first generated token.**

Recommendation: **(a)**. Fitting at positions we never intervene on risks a
direction valid in a distribution the hook never touches. (b) and (c) should be
extracted and stored as diagnostics — they are cheap once activations are
captured — but (a) is the primary. Flagged in §7 as a risk because it is a
genuine departure point from the CAST-style precedent, which reads at the last
prompt position.

### Related ambiguity — what "provenance" the direction should encode

A tempting fourth contrast is *identical imperative text placed in a user turn
vs a tool turn*, fitting provenance directly. **Recommend against it as the
primary.** The mask already supplies provenance exactly; a direction that also
encodes it double-counts the gate and makes the ablation's effect
uninterpretable. Worth extracting as a diagnostic to measure how much
provenance the residual stream already carries at each layer — that quantity is
independently interesting and bears on Checkpoint 6's learned gate.

---

## 6. Cost estimate

Disk, on the GPU host:

| Item | Estimate |
| --- | --- |
| Llama-3.1-8B-Instruct bf16 | ~16 G |
| Goodfire SAE l19 | < 1 G |
| Activation cache, all 32 layers, ~2k contrast rows, bf16 | ~10–20 G, reducible by streaming difference-in-means and never materialising the cache |
| Direction bundles, all layers × 3 concepts | < 100 M |
| Phase 0 raw results, JSONL | < 1 G |
| Phase 1–2 trajectories | a few G |
| **Total** | **~40–60 G**, or ~25 G with streaming extraction |

Runtime, order-of-magnitude on one modern 24 G card, batch 1:

| Stage | Estimate |
| --- | --- |
| Activation extraction over contrast sets | 1–3 h |
| Difference-in-means fit, all layers | minutes |
| Layer sweep × arms × evals (the dominant cost) | 12–48 h |
| Phase 1 loop acceptance | < 1 h |
| Phase 2 paired replay | 12–24 h |

The sweep dominates and is the item to cut first if budget binds — narrowing
the layer band, not the arms, and never the safety eval.

Cost in this container: negligible; CPU-only work on synthetic tensors.
**No paid compute is provisioned, and none will be without explicit
authorization.**

---

## 7. Proposed Phase 0 protocol (exact)

Pre-registered in `EXPERIMENT_PROTOCOL.md` before execution.

1. **Build and freeze contrast sets** C1, C2, C3. Validate schema, enforce the
   matching invariants, split by scenario family, hash. Refuse to proceed on a
   validation failure.
2. **Extract activations** at post-block residual stream for every layer
   0–31, at rule (a) positions, batch 1, greedy, no sampling. Store as a
   streaming mean accumulator per class per layer rather than a full cache.
3. **Fit difference-in-means** per (concept, layer). Record both normalized and
   unnormalized forms explicitly; the manifest states which the bundle carries.
4. **Layer sweep** over layers 8–28 initially — the middle-to-late band —
   expanding only if the effect is non-monotone at the boundary and that
   expansion is logged as a deviation.
5. **Arms**, at every swept layer:
   - no intervention;
   - projection ablation;
   - additive steering, pre-registered alpha grid;
   - sham direction, random, matched norm, same seed discipline;
   - SAE clamping at layer 19 only, plus layer-19 projection as its companion.
6. **Metrics**, at every layer × arm:
   - target-behaviour suppression on held-out contrasts;
   - held-out retain-set perplexity;
   - structured-output validity;
   - a small capability probe;
   - the mandatory safety evaluation — **every arm, every layer, no exceptions**.
7. **Write raw JSONL first.** Tables and plots are regenerated from it and are
   never the source of truth. Confidence intervals, not point estimates.
8. **Kill gate.** Stop and report if useful suppression requires unacceptable
   retain, structured-output, or safety cost. Do not assume the agent loop
   recovers the damage.

Single-layer vs multi-layer application is **not** yet decided. Checkpoint 1's
signatures are single-layer and the sweep is single-layer. Whether a multi-layer
band is added is a pre-registration question to settle before Checkpoint 2
runs, not a knob to reach for after seeing a weak single-layer result.

---

## 8. Unresolved risks

**Scientific**

1. **Illegibility vs targeted suppression.** The most likely failure is that
   ablation degrades the *readability* of tool output rather than selectively
   removing compliance pressure. Injection resistance would improve for the
   wrong reason. Discriminated by: task success on **benign** tasks that
   require acting on tool content, and retain-set perplexity measured at tool
   positions specifically. If benign tool-dependent tasks degrade in step with
   injection resistance, the effect is illegibility. This is the single most
   important confound in the project.
2. **Projection is sign-agnostic.** \(x - \hat r(\hat r^\top x)\) removes the
   component along \(\hat r\) regardless of sign, so any benign task-following
   sharing that direction is removed too. Additive steering is directional and
   will therefore behave qualitatively differently — this is a reason to keep
   the comparison arm, not a defect.
3. **Sham arms may not be inert.** Rogue Scalpel finds random directions raise
   harmful compliance measurably. If the sham arm shows an effect, the honest
   reading is that *any* perturbation at that layer moves the behaviour, and
   the primary effect size must be reported against sham, not against
   no-intervention.
4. **Prompting may simply win.** RePS narrows but does not always close the gap
   to prompting. A deterministic tool-layer guard may beat both. Both are
   required comparators and a loss is a valid, reportable result.
5. **Extraction-position mismatch** (§5) — fitting where we do not intervene.
6. **SAE layer-locking** (§4) — confounds intervention type with depth.
7. **Base-vs-Instruct SAE mismatch** if Llama Scope is ever leaned on.

**Implementation**

8. **`transformers` 5.x is current on PyPI (5.15.1).** A major version bump is
   the highest-risk pin in the stack for chat-template and tool-call rendering
   behaviour. The version must be chosen by verifying byte-equality on the GPU
   host, not by taking latest.
9. **Boundary tokens.** A single token spanning the header terminator and the
   first content character must have exactly one defined disposition. Rule to
   be written down and every ambiguous token logged, per Checkpoint 3.
10. **Hook state leakage** across requests. Mitigated by single-worker
    serialization and `finally`-clearing, with the two-sequential-requests
    contamination test as the acceptance check.
11. **Prefill vs decode.** The hook must not re-apply to positions already
    processed, and must correctly no-op during incremental decode. Newly
    generated tokens are never masked, but they attend to already-modified
    K/V — that persistence is real and is the within-completion hysteresis
    that Checkpoint 5 measures. It must not be conflated with cross-request
    persistence, which does not exist in a stateless shim.
12. **Determinism on GPU.** Byte-identical trajectories may need
    `use_deterministic_algorithms(True)`, fixed `CUBLAS_WORKSPACE_CONFIG`, and
    disabled TF32. Any residual nondeterminism gets recorded, not papered over.
13. **Gated-repo access and token hygiene** (§3).
14. **Split environment.** Code developed here is not exercised against weights
    here. Mitigated by keeping Checkpoint 1 model-free and vendoring tokenizer
    files if permitted.

---

## 9. Dependencies

Python **3.12** (3.12.3 available locally), managed with `uv`. Every version
that can change numerical behaviour is pinned exactly in `uv.lock`.

Latest-on-PyPI as surveyed 2026-08-23 — these are **candidates**, not final
pins. `torch` and `transformers` in particular are pinned only after the GPU
host confirms CUDA compatibility and byte-exact template rendering.

```
torch==2.13.0            # pin after CUDA compat check on GPU host
transformers==5.15.1     # RISK: v5 major; verify template bytes before pinning
tokenizers==0.23.1
accelerate==1.14.0
safetensors==0.8.0
numpy==2.5.2
scipy==1.18.1
pandas==3.0.5
matplotlib==3.11.1
fastapi==0.141.1
pydantic==2.13.4
uvicorn==0.52.4
httpx==0.28.1
pytest==9.1.1
hypothesis==6.165.10     # property tests for the intervention kernel
```

Checkpoint 1 needs only `torch`, `safetensors`, `numpy`, `pytest`, and
`hypothesis` — a CPU-only torch wheel suffices and installs cleanly here.

Batch size one, one server worker, no concurrent generation, Phases 0–2.

---

## 10. Checkpoint schedule

| # | Deliverable | Where | Acceptance |
| --- | --- | --- | --- |
| 0 | This document + `DECISIONS.md` | here | human sign-off |
| 1 | Pure intervention kernel, direction-bundle format | here, CPU | all kernel and bundle tests pass **without loading the model** |
| 2 | Phase 0 single-turn experiment | GPU host | a layer and intervention with quantified suppression **and** quantified collateral cost; kill gate honoured |
| 3 | Provenance-aware shim, renderer, span mapping, `inspect-spans` | split | byte-equality with official template; off ≡ direct model; no cross-request contamination; state cleared on raised error |
| 4 | Deterministic ReAct loop, 3 fake tool domains | here + GPU | two same-seed no-intervention runs byte-identical modulo documented timing fields |
| 5 | Paired replay, primary experiment, KV hysteresis | GPU host | protocol frozen **before** collection; paired effect sizes with bootstrap CIs |
| 6 | External validity — OpenCode via `baseURL`, chosen benchmark | GPU host | **only after** Phase 2 review and explicit approval |

Checkpoints 1 and 4 are the ones that can proceed here immediately. Checkpoint
2 blocks on GPU-host access and on the revision pin.
