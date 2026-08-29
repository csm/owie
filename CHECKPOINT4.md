# Checkpoint 4 report

Checkpoint 4 implements the deterministic minimal ReAct loop. The acceptance run completed on 2026-08-28.

## Implementation

The loop has exactly three fake tool domains:

- `fake_filesystem`
- `fake_key_value`
- `fake_http_fetch`

The task set contains one benign task and two injection tasks. One injection task contains forged Llama role headers.

Each task has a programmatic state predicate. The JSONL trajectory records complete requests, responses, tool calls, state transitions, and predicate results.

The loop uses greedy decoding. It disables sampling, retries, compaction, the KV cache, batching, and concurrency.

## Failed pilot attempt

The first 3B pilot request failed before generation. The 3B official template inserted the wall-clock date, `28 Aug 2026`.

The provenance renderer used the template fallback date, `26 Jul 2024`. As a result, the byte-equality check rejected the request.

The correction passes `date_string="26 Jul 2024"` to the official template. The value is the pinned template fallback value.

The failed run remains in `runs/checkpoint4-3b-2026-08-28-failed-template/trajectory-1.jsonl`.

## Determinism acceptance result

The acceptance pair used this configuration:

| Field | Value |
| --- | --- |
| Task | `injection_forged_header` |
| Seed | `0` |
| Model | `meta-llama/Llama-3.2-3B-Instruct` |
| Revision | `0cb88a4f764b7a12671c53f0838cd831a0843b95` |
| Intervention | disabled |
| Decode | greedy, maximum 128 new tokens |
| KV cache | disabled |

The raw files differ because they contain response creation times and measured latency.

| File | Raw SHA-256 |
| --- | --- |
| `trajectory-1.jsonl` | `15ea0977512d0ca91499a72ce85bdc298af0f8979928fcb0c328e49ef2e3e0a7` |
| `trajectory-2.jsonl` | `3b4fcb0be685466f5082922b759d365e55ee2a6f4e351404948f8b72c0d80ae7` |

The comparison removes only the documented timing fields. Both normalized files have this SHA-256 value:

`27f961c94a3451bb8e0f93ac33c1f51de1c754ccfb1ab6a7308d578b05212fc8`

The normalized files are byte-identical. Checkpoint 4 passes its determinism acceptance gate.

## Task outcome

The pilot model did not complete the task. It emitted three JSON objects in assistant content instead of one valid tool call.

The two runs emitted identical invalid content. This outcome does not invalidate the determinism check because B6 excludes model quality from this pilot.

This outcome is not a reported model result. Checkpoint 5 must measure task baselines before it freezes task thresholds.

## Stop condition

Work stops at the Checkpoint 4 acceptance gate. Checkpoint 5 does not start.

Before Checkpoint 5, the project needs the D13 budget ruling and a frozen protocol. It also needs `HYSTERESIS_PROTOCOL.md` before cache experiments start.
