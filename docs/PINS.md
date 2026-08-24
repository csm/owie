# PINS

Immutable identifiers. Every value here is a commit SHA or an exact version —
never a mutable ref. Code refuses to record a placeholder in a direction-bundle
manifest (`DECISIONS.md` B2, `directions/manifest.py`).

Resolved 2026-08-23 from the Hugging Face API.

## Model

| Field | Value |
| --- | --- |
| Model | `meta-llama/Llama-3.1-8B-Instruct` |
| Revision | `0e9e39f249a16976918f6564b8830bc894c89659` |
| Last modified | 2024-09-25T17:00:57Z |
| Gating | `manual`; access confirmed under the local credential store |

Tool-output provenance renders under the `ipython` role. See `PREFLIGHT.md` §3
for the verified chat-template behaviour, which is **not** what a reading of
the template alone suggests.

## Pilot model (non-reporting)

| Field | Value |
| --- | --- |
| Model | `meta-llama/Llama-3.2-3B-Instruct` |
| Revision | `0cb88a4f764b7a12671c53f0838cd831a0843b95` |
| Gating | `manual`; access confirmed |

Approved for pipeline debugging and the Checkpoint 4 determinism test only
(`DECISIONS.md` B6). Never carries a reported result.

## Replication target (pre-registered)

| Field | Value |
| --- | --- |
| Model | `Qwen/Qwen2.5-7B-Instruct` |
| Revision | `a09a35458c702b33eeacc393d103063234e8bc28` |
| Gating | ungated |

## SAE

| Field | Value |
| --- | --- |
| Repo | `Goodfire/Llama-3.1-8B-Instruct-SAE-l19` |
| Revision | `f6775a221e47b44233af4bac2c7b65189265519a` |
| Layer | 19 only |
| Format | `.pth` torch pickle — convert to safetensors on first load (D12) |

`OpenMOSS-Team/Llama-Scope` is **not** a primary arm: it is trained on the base
model, not Instruct. Exploratory use only, reported separately.

## Environment

| Field | Value |
| --- | --- |
| Python | 3.12.12 |
| Accelerator | Apple M3 Max, Metal/MPS. **No CUDA. No float64.** |
| torch | 2.13.0 |
| safetensors | 0.8.0 |
| numpy | 2.5.2 |

Exact dependency versions live in `uv.lock`. `transformers` is **not yet
pinned**: it is not a Checkpoint 1 dependency, and its version changes rendered
prompt bytes (`PREFLIGHT.md` §3 finding 6, D7). It gets pinned at Checkpoint 3,
behind a byte-equality test, not before.
