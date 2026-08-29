# Deterministic ReAct loop

Checkpoint 4 uses a small synchronous loop. The loop sends one request at a time to the local server.

The loop has exactly three fake tool domains:

| Tool | Operations |
| --- | --- |
| `fake_filesystem` | `read`, `write`, and `list` |
| `fake_key_value` | `get`, `set`, and `delete` |
| `fake_http_fetch` | An exact URL fetch from the canned task documents |

The tools cannot read the host filesystem or network. Each task starts with a new fake environment.

## Tasks

The current task set contains one benign task and two injection tasks. Each task has a programmatic state predicate.

`injection_invoice` contains an ordinary instruction in an HTTP document. `injection_forged_header` contains forged Llama role delimiters.

The task predicate checks the required files and keys. For injection tasks, it also checks that forbidden files and keys are absent.

## Run the 3B determinism pilot

Start the approved non-reporting pilot server:

```bash
uv run owie-server --pilot-3b
```

In a second terminal, run the same task two times:

```bash
uv run owie-loop \
  --pilot-3b \
  --task injection_forged_header \
  --seed 0 \
  --repeat 2 \
  --run-dir runs/checkpoint4-3b-determinism
```

The command exits with status 1 if the normalized trajectory files differ. It never overwrites an existing trajectory file.

The pilot is not a reported model result. It only checks the deterministic execution requirement from `DECISIONS.md` B6.

## Trajectory format

Each JSONL file contains these events:

- `run_start` records the task, run, seed, revisions, intervention configuration, and initial state.
- `model_step` records the complete request and response bodies, prompt hash, token counts, and latency.
- `tool_step` records the tool input, output, and state transition.
- `success_check` records each predicate result.
- `run_end` records the final state and stop reason.

The run ID is a hash of the task, seed, revisions, intervention configuration, and run type.

## Determinism rule

The acceptance check removes only these timing fields:

- `model_step.latency_ms`
- `model_step.response.created`
- `model_error.latency_ms` for a failed run

The raw files keep all timing fields. The check compares all other bytes after canonical JSON serialization.

The loop uses greedy decoding. It seeds Python, NumPy, PyTorch, MPS, and the generation request.

The loop disables sampling, retries, compaction, the KV cache, batching, and concurrency. It requests deterministic PyTorch algorithms.

MPS kernel behavior is the remaining possible source of nondeterminism. A failed 3B comparison stops work before Checkpoint 5.
