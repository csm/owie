# HYSTERESIS_PROTOCOL

Status: operational protocol frozen on 2026-08-28, before cache experiment code.

This protocol measures within-completion KV-cache hysteresis. It does not measure persistence across agent HTTP requests.

## 1. Scope

The stateless HTTP server recomputes the full conversation for each request. Thus, it has no cross-request cache state.

This experiment uses a direct, single-process generation function. The function keeps one KV cache during one assistant completion.

The experiment uses every injection prefix in the frozen paired-replay task set. The paired-replay manifest supplies the task-set hash.

Do not run this experiment before the paired-replay manifest contains that hash.

## 2. Intervention arms

Use each direction-bearing arm in the frozen paired-replay protocol. Use its exact bundle hash, layer, mode, scope, normalization, and alpha.

Run the matched sham for each directional arm. Do not substitute a new direction or layer after results are visible.

The primary mask remains `role == "tool" && region == "content"`. Generated assistant tokens are outside this mask.

## 3. Common prefix

Render the complete conversation through the injected tool result. Record the rendered bytes, token IDs, provenance masks, and SHA-256 hash.

Run an intervened prefill with `use_cache=True`. Then greedily generate eight assistant tokens.

The release boundary is after generated token 8. This boundary is `k = 8` in every eligible item.

Exclude an item if EOS occurs before token 9. Record the exclusion as `eos_before_release`.

Use a maximum of 64 generated tokens. Stop at EOS or the limit.

## 4. Conditions

Run four conditions for each prefix and intervention arm:

1. `no_intervention`: Build a clean cache and generate normally.
2. `persistent`: Apply the intervention during prefill. Keep the hook installed after token 8.
3. `dirty_release`: Apply the intervention during prefill. Remove the hook after token 8. Keep the modified cache.
4. `clean_recompute`: Teacher-force the exact prompt and first eight tokens from `persistent` without intervention. Continue from the clean cache.

Each condition uses the same greedy decoding rule. Sampling is disabled.

The `persistent`, `dirty_release`, and `clean_recompute` conditions must have byte-identical tokens through the release boundary.

Generated tokens are outside the provenance mask. Thus, `persistent` and `dirty_release` must remain identical after release.

A difference between these two conditions is an implementation failure, not a scientific result.

## 5. Cache construction

Build each condition in a separate process-local run. Do not clone or serialize a live Transformers cache object.

For `dirty_release`, repeat the intervened prefill and teacher-force the eight persistent tokens. Then remove the hook.

For `clean_recompute`, teacher-force the same complete token prefix with intervention disabled.

Hash each cache tensor after a CPU copy. Record tensor names, shapes, dtypes, and SHA-256 hashes.

Do not compare raw pointer values or object representations.

## 6. Seeds and numerical controls

Use seeds 11, 22, and 33 for every item and condition. Record Python, NumPy, PyTorch, MPS, and generation seeds.

Request deterministic PyTorch algorithms. Use batch size one, one process, and no concurrent generation.

Greedy decoding makes the seeds nominal. Keep them to expose hidden random operations.

## 7. Raw records

Write one JSONL record for each item, arm, seed, and condition. Never overwrite a completed record.

Each record contains:

- the task, prefix, model, direction, code, and dependency revisions.
- the rendered prompt hash and token IDs.
- the release boundary and teacher-forced token IDs.
- the intervention configuration and selected-token count.
- the cache tensor metadata and hashes.
- the logits at token 9 and all generated token IDs.
- the completion text, token counts, and latency.
- the stop reason, exclusions, and any numerical error.

The run status records cumulative time against the 144 h Checkpoint 5 ceiling.

## 8. Metrics

Compute these paired metrics at token 9:

- KL divergence from `dirty_release` to `clean_recompute`.
- Jensen-Shannon divergence.
- the change in the selected-token log probability.
- whether the greedy next token matches.
- the rank of the dirty-release token under clean logits.

Compute these continuation metrics after token 8:

- exact continuation match.
- common-prefix length.
- token edit distance.
- tool-call envelope, name, and argument validity.
- task success and attack success.

Move logits to CPU float64 before probability calculations. Keep raw model logits in their original dtype.

## 9. Interpretation

The paired `dirty_release` versus `clean_recompute` difference estimates within-completion cache hysteresis.

The `persistent` versus `dirty_release` comparison is an implementation control. It does not estimate a separate mechanism in this mask design.

Do not call a difference cross-request persistence. The HTTP server has no session-level cache reuse.

Report point estimates and paired bootstrap intervals. Report every exclusion and both matched-sham comparisons.

## 10. Acceptance before collection

The implementation must pass these tests before a model run:

- all conditions have identical tokens through the release boundary.
- `persistent` and `dirty_release` have identical continuations on a tiny model.
- clean recomputation matches direct no-intervention teacher forcing.
- an error removes the hook and clears all request state.
- repeated same-seed tiny-model runs produce byte-identical raw records.
- cache tensor hashes change when a selected prefill position changes.
- cache tensor hashes do not change when the provenance mask is all zero.

Stop before collection if any acceptance test fails.
