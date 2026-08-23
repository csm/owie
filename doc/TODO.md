# Bootstrap Assignment: Contextual Activation Suppression in Agent Loops

Read this entire assignment before modifying files. This is an experimental measurement project, not a product or general-purpose framework. Preserve the causal validity of the experiment over convenience or extensibility.

## Objective

Build a minimal, reproducible harness that measures the effect of context-dependent, reversible, semantically targeted activation suppression on tool-using agent performance.

The primary experiment is:

> Suppress an instruction-following/compliance direction only at input-token positions originating from `role: "tool"` content, while leaving user and assistant positions untouched. Measure prompt-injection resistance, agent-task success, tool-call validity, token cost, self-correction, retained capability, and safety.

The adversarial instruction arrives through tool output. Intervention configuration arrives through a trusted out-of-band request field and must never be controllable by tool content.

## Settled decisions

Do not reopen these without explicit human approval:

1. Do not fork OpenCode, LangGraph, Codex CLI, Claude Code, or another agent harness.
2. Apply interventions inside a local inference server with access to the model forward pass.
3. Serve an OpenAI-compatible HTTP API. Existing harnesses connect later by changing `baseURL`.
4. Use open-weight Hugging Face models only.
5. Projection ablation is the primary primitive:

   \[
   x' = x - \hat r(\hat r^\top x)
   \]

6. Implement additive steering as a comparison arm, not the default.
7. Add SAE feature clamping only if suitable features already exist for the chosen model.
8. Use exact orchestration state—message provenance and explicit phase flags—as the initial gate. Do not begin with a learned probe.
9. Complete a single-turn layer sweep before building an agent loop.
10. Build a small deterministic ReAct loop before integrating a production harness.
11. Use Hugging Face Transformers and ordinary PyTorch hooks through Phase 2.
12. Do not begin vLLM work until Phase 2 demonstrates an effect worth scaling.
13. The deliverable is effect sizes with uncertainty, including credible negative or null results.

Do not build a UI, generalized steering framework, multiple-model-family support, distributed serving, or an agent-harness fork.

## Technology baseline

Unless the human overrides it:

- Python 3.12
- `uv` for dependency and environment management
- PyTorch and Hugging Face Transformers
- FastAPI and Pydantic
- `safetensors` for direction vectors
- pytest, including property tests
- NumPy, SciPy, pandas, and a minimal plotting library
- JSONL trajectory/event storage
- Batch size one, one server worker, and no concurrent generation in Phases 0–2
- Hand-written PyTorch forward hooks rather than adopting a large interpretability framework

Pin every dependency that can change numerical behavior. Pin the model identifier and immutable revision.

## Checkpoint 0: preflight—stop before coding

Before downloading a model or implementing the harness:

1. Read the supplied project plan and these works:

   - ASA, arXiv 2602.04935
   - CAST, arXiv 2409.05907
   - Copy Suppression, arXiv 2310.04625
   - NPM, arXiv 2606.29824
   - Activation State Machines
   - SDialog, arXiv 2506.10622 and 2512.09142
   - The Rogue Scalpel, arXiv 2509.22067
   - RePS, arXiv 2505.20809

2. Inspect the available compute environment without making material changes.
3. Produce `PREFLIGHT.md` containing:

   - available GPU, VRAM, RAM, disk, CUDA, and PyTorch compatibility;
   - two or three viable model/revision choices;
   - likely chat-template and tool-calling implications of each;
   - available SAE feature sets, if any;
   - proposed compliance-direction contrast-set construction based on the prior art;
   - estimated disk, runtime, and experiment cost;
   - exact proposed Phase 0 protocol;
   - unresolved scientific or implementation risks;
   - a dependency list and checkpoint schedule.

4. Ask the human to decide:

   - model and size;
   - first target concept;
   - whether an available SAE feature should be included;
   - compute budget.

Do not infer these choices and do not continue past Checkpoint 0 until they are answered.

## Repository structure

After preflight approval, create:

```text
directions/
interventions/
server/
loop/
replay/
evals/
analysis/
tests/
docs/
```

Use normal Python package structure beneath these directories where helpful, but do not introduce an elaborate monorepo or plugin system.

Maintain:

- `README.md`: setup and exact commands;
- `DECISIONS.md`: settled decisions and approved deviations;
- `EXPERIMENT_PROTOCOL.md`: frozen experimental arms, datasets, metrics, and exclusions;
- `PREFLIGHT.md`: environment and model decision;
- `HYSTERESIS_PROTOCOL.md`: exact KV-cache experiment;
- machine-readable run manifests recording git revision, model revision, direction revision, seeds, dependencies, hardware, and intervention configuration.

## Checkpoint 1: pure intervention kernel

Implement interventions as pure functions:

```python
project_out(hidden, direction, span_mask)
add_vector(hidden, vector, alpha, span_mask)
clamp_feature(hidden, decoder_direction, value, span_mask)
```

Requirements:

- no in-place mutation;
- explicit shape checks;
- device and dtype preservation;
- batch-size-one support first;
- suppression outside the mask must be exact identity;
- normalized and unnormalized direction handling must be explicit;
- projection must be numerically orthogonal to the direction;
- projection must be idempotent within tolerance;
- all-zero and all-one masks must be tested;
- malformed masks must fail loudly.

Create a versioned direction-bundle format:

```text
directions/<direction-id>/
  vector.safetensors
  contrasts.jsonl
  manifest.json
  extraction_config.json
```

The manifest must include model ID, immutable revision, layer, hook point, token-extraction rule, fitting method, normalization, contrast-set hash, extraction-code git revision, dtype, and creation timestamp.

Acceptance: all intervention and direction-bundle tests pass without loading the full model.

## Checkpoint 2: Phase 0 single-turn experiment

Do not build the HTTP server or agent loop yet.

Implement:

1. Contrast-set validation and hashing.
2. Activation extraction at the agreed token positions.
3. Difference-in-means direction fitting at every candidate layer.
4. A layer sweep over the model’s middle-to-late region, expanding if results justify it.
5. For every layer and intervention arm:

   - target-behavior suppression;
   - held-out retain-set perplexity;
   - structured-output validity;
   - a small capability probe;
   - the mandatory safety evaluation.

Include at least:

- no intervention;
- projection ablation;
- additive steering with a preregistered alpha sweep;
- sham/random matched-norm direction;
- SAE clamping if approved.

Write raw results to machine-readable files before generating tables or plots. Report confidence intervals, not just point estimates.

Acceptance: identify a layer and intervention with quantified target suppression and quantified collateral cost.

Kill gate: stop for human review if useful target suppression requires an unacceptable retain-set, structured-output, or safety cost. Do not assume an agent loop will recover the damage.

## Checkpoint 3: provenance-aware shim

Build only the OpenAI-compatible surface needed by the experiment:

- `GET /v1/models`
- non-streaming `POST /v1/chat/completions`
- tool-call response encoding required by the minimal loop

Streaming, batching, authentication, production concurrency, and Responses API compatibility are out of scope.

### Chat rendering and span mapping

Implement a model-family-specific renderer that:

1. produces output byte-for-byte identical to the pinned tokenizer’s official chat template;
2. simultaneously records character spans for:

   - message content;
   - template-generated role markers;
   - separators and special tokens;
   - tool-call envelopes;
   - tool-result content;

3. tokenizes the final unmarked text with offset mappings;
4. converts character provenance into token masks.

The primary mask is `role == "tool" && region == "content"`. Template role markers are not tool-output content.

Also retain a secondary `whole_tool_block` mask for analysis, but do not silently substitute it for the primary condition.

Tests must include:

- one and multiple tool messages;
- empty content;
- Unicode and multibyte characters;
- adjacent messages;
- JSON tool output;
- tool output containing template-like delimiters;
- long tool output;
- boundary tokens overlapping two provenance regions;
- byte equality with the official template;
- an assertion that every content character belongs to exactly one message;
- an assertion that only intended tokens are selected.

Define and document the boundary-token rule. Log every ambiguous token in debug output.

Provide an `inspect-spans` CLI that prints rendered text, character regions, token IDs, decoded tokens, and mask membership. This is diagnostic infrastructure, not a UI.

### Hook runtime

The request-local intervention state must not live in an unsafe mutable global. In the first version, serialize requests with a single-worker/single-generation lock and ensure state is cleared in `finally`.

The hook must correctly distinguish:

- full-prefix prefill;
- incremental decode calls;
- token/cache positions;
- masked source positions;
- newly generated assistant tokens.

Default behavior applies projection only to the requested provenance positions. Unknown request fields should otherwise pass through or be ignored according to the OpenAI-compatible contract.

Acceptance:

- intervention-off responses match direct-model responses;
- span inspection is correct on every fixture;
- two sequential requests with different configs cannot contaminate one another;
- an intentionally raised generation error still clears hook state.

## Checkpoint 4: deterministic minimal loop

Implement an approximately 300-line ReAct loop with exactly three deterministic tool domains:

- fake filesystem;
- fake key-value store;
- fake HTTP fetch returning canned documents.

Each task must have a programmatic success predicate. Include benign tasks and injection tasks where an HTTP document contains instructions conflicting with the system objective.

Log every run to JSONL:

- task and run IDs;
- seed;
- complete request and response bodies;
- rendered prompt hash;
- model and direction revisions;
- intervention config;
- tool inputs and outputs;
- success checks;
- token counts and latency;
- environment state transitions.

For reproducibility:

- use greedy decoding for the Phase 1 acceptance test;
- seed Python, NumPy, PyTorch, and the generation API;
- request deterministic PyTorch algorithms where supported;
- disable sampling, retries, compaction, caching, batching, and concurrency;
- record unavoidable nondeterminism.

Acceptance: two no-intervention runs of the same task and seed produce byte-identical trajectory files after excluding explicitly documented timing fields.

Do not proceed if this fails.

## Checkpoint 5: paired replay and primary experiment

Implement:

```python
resume(trajectory, step_k, config, seeds) -> list[Continuation]
```

Replay must teacher-force a byte-identical clean prefix and vary only the intervention condition. Free-running rollouts are secondary external-validity measurements, not the primary causal estimate.

Freeze tasks, arms, metrics, and exclusions in `EXPERIMENT_PROTOCOL.md` before collecting primary results.

Required arms:

1. no intervention;
2. projection ablation on tool-content tokens;
3. additive steering on the same tokens;
4. prompt-only defense;
5. deterministic tool-layer guard;
6. sham-direction intervention;
7. whole-tool-block projection as a secondary scope comparison;
8. SAE clamping if approved.

Pair intervention-on and intervention-off continuations by prefix and seed. Preserve common random numbers where possible.

Report:

- injection attack success;
- tool-call schema validity;
- tool-name validity;
- argument-schema validity;
- steps to completion;
- token and tool-call cost;
- task success;
- retain-set perplexity sampled at trajectory prefixes;
- self-correction rate;
- safety-eval deltas;
- paired effect sizes and bootstrap confidence intervals.

The intervention must be compared against both prompting and a deterministic tool-layer guard. Do not frame an inferior result as success.

### KV-cache hysteresis

Write the operational protocol before implementing it.

Define release at a token boundary within one completion:

- persistent: intervention remains active after the boundary;
- dirty-cache release: hook is removed, but previously modified KV state remains;
- clean recomputation: teacher-force the exact same generated prefix and rebuild KV state without intervention before continuing.

The dirty-versus-clean difference estimates cache hysteresis.

Note explicitly that the initial stateless HF HTTP shim recomputes conversation prefixes between agent HTTP requests. Therefore cross-request hysteresis does not exist unless session-level cache reuse is deliberately introduced. Do not conflate within-completion KV hysteresis with across-agent-step persistence.

## Checkpoint 6: external validity, only after approval

After Phase 2 review:

1. Configure unmodified OpenCode to use the shim as an OpenAI-compatible custom provider through `baseURL`.
2. Do not fork OpenCode.
3. Verify that OpenCode’s rendered requests preserve the message provenance required by the shim.
4. Run an established benchmark selected by the human:

   - τ-bench or τ²-bench;
   - MTU-Bench;
   - BFCL for isolated tool-call validity.

5. Add a learned gate only after oracle provenance gating is measured.
6. Report the oracle-gate versus learned-gate delta as the gate-error budget.
7. Consider vLLM only if throughput is now the limiting factor.

If moving to vLLM:

- disable automatic prefix caching;
- use eager execution;
- prove arm isolation with contamination tests;
- do not trust cached-prefix behavior without an explicit test;
- document any monkeypatch against the exact vLLM revision.

## Working discipline

- Work one checkpoint at a time.
- Run relevant tests before moving on.
- Stop at every acceptance or kill gate and report evidence.
- Do not silently revise the experiment after seeing results.
- Keep raw data immutable; derived analysis may be regenerated.
- Never overwrite a direction bundle or experiment run.
- Record failed runs and exclusions with reasons.
- Prefer small, inspectable implementations.
- Treat null results, prompting wins, downstream self-repair, and safety degradation as valid findings.
- Describe suppression as a control surface, never removal or a guarantee.
- Do not publish, push externally, or provision paid compute without explicit authorization.

## Immediate response expected

For your first response, do not write code. Report:

1. your understanding of the causal question;
2. the proposed preflight investigation;
3. any ambiguity you see in the compliance-direction extraction protocol;
4. the four decisions needed from the human;
5. the files you will create after approval;
6. the first acceptance test you will target.
