# HANDOFF

State of the project as of **2026-08-26**, after Checkpoint 2 (Phase 0) has
collected. Written for whoever picks this up next, including a future me with
no memory of the run.

Read `doc/TODO.md` first — it is the assignment and the source of every settled
decision. Then `DECISIONS.md` §A–§C. This document is the map, not a
replacement for either.

---

## 1. Where the project is

| # | Deliverable | State |
| --- | --- | --- |
| 0 | Environment survey, model decision | done — `PREFLIGHT.md`, `DECISIONS.md` §B |
| 1 | Pure intervention kernel, direction-bundle format | done |
| 2 | **Phase 0 single-turn experiment** | **done — 460 cells, 16.2 h, results below** |
| 3 | Provenance-aware shim, renderer, span mapping | done |
| 4 | Deterministic ReAct loop | **not started — this is next** |
| 5 | Paired replay, primary experiment | not started; needs its own protocol freeze and budget ruling (D13) |
| 6 | External validity | gated on Phase 2 review |

250 tests pass (`uv run pytest`). No test loads model weights; where a forward
pass is unavoidable a tiny randomly-initialized Llama with the pinned tokenizer
stands in.

## 2. Running things

```bash
uv sync                      # exact locked deps; Python 3.12
uv run pytest                # 250 tests, no network, no weights

uv run owie-build-datasets   # regenerate frozen datasets; must be byte-identical
uv run owie-phase0 --run-dir runs/<id> --tranche A --budget-hours 72
uv run owie-phase0-analyse --results runs/<id>/results.jsonl
uv run inspect-spans --local-files-only request.json   # provenance, no weights
uv run owie-server --direction <bundle-id>             # Checkpoint 3 shim
```

The sweep is **resumable and keyed by cell**: re-running the same command skips
completed cells. The budget is **cumulative across resumptions** (persisted in
`status.json`), so a resume cannot reset the ceiling.

## 3. What Phase 0 found

Raw data: `runs/phase0-2026-08-24/results.jsonl` (460 cells, one JSON line per
cell, every per-item record and every generated string). Report:
`runs/phase0-2026-08-24/analysis/phase0.md`. Regenerate the report from the raw
file at any time; it is never a source of truth.

**Selected cell:** `additive | c3 | layer 10 | c=+0.50` — 5.11 nats of
injection-margin suppression, of which **3.40 nats [1.65, 5.33]** exceed its
alpha-matched sham. No measurable cost: safety identical to baseline on both
channels, structured validity 1.000, retain perplexity and tool-dependent
capability nominally better than baseline.

That satisfies Checkpoint 2's acceptance condition. **It is not a vindication
of the hypothesis, and the handoff should not treat it as one.** Three findings
matter more:

1. **The large additive effects are mostly perturbation norm, not direction.**
   Random unit vectors at layer 10, `|c| = 1.0`, reduce the margin by 5.2–8.6
   nats — matching or exceeding any fitted direction — and reproduce the
   tool-channel safety collapse (sham harmful compliance up to 0.833 against a
   0.167 baseline). Without the matched additive sham, which was **not** in the
   original design, this would have been reported as a finding about the fitted
   direction and been wrong.

2. **Projection ablation — the project's primary primitive (settled decision
   5) — is direction-specific but an order of magnitude too small.** Projection
   sham is inert (−0.00 to +0.03 nats). Fitted directions give a clean depth
   profile peaking at layers 9–11 (C1 ≈ 1.65 nats) and decaying to zero by
   layer 18, staying there through 26. Real, attributable, and below the 2.0
   nat threshold everywhere.

3. **The winning direction is the one included as a control.** C3 is the
   refusal proxy. `PREFLIGHT.md` §5 states *in advance* that C3 outperforming
   C1 is evidence the mechanism is generic compliance rather than anything
   provenance-specific. C3 is also negative at all fifteen swept layers under
   projection — it behaves like a different axis, not a stronger version of the
   same one.

Also: **the SAE arm is a clean null** (all nine cells within ±0.06 nats), as is
projection at layer 19, its layer-matched companion. The features came from the
pre-registered selection procedure, hashed before any arm ran. A null is what
small mean-activation differences predict, and it is the price of not
cherry-picking (Rogue Scalpel).

**Safety asymmetry worth carrying forward:** where safety degraded, it degraded
*only* in the tool channel; the user channel never moved from 0.083 in any
cell. A conventional user-turn-only safety evaluation would have scored the
worst cells as perfectly safe. Keep both channels in every future safety eval.

## 4. Four recorded deviations — read before trusting any number

All in `DECISIONS.md` §C with full reasoning, and mirrored in
`EXPERIMENT_PROTOCOL.md` §7a–§7c.

| # | Change | When | Why it is auditable |
| --- | --- | --- | --- |
| C2 | Primary outcome: comply **rate** → comply **margin** | after 20/190 cells | baseline margin averages 20.1 nats with no item within 2 nats of the sign boundary, so the rate could not move; threshold calibrated on the *baseline*, which carries no information about any arm; no completed cell met it when set |
| C3 | Added matched additive sham (120 cells) | after 40/190 cells | decided before any sham of either kind had run |
| C4 | Selection ranks **direction-specific excess**, not total suppression | **after collection, both outcomes visible** | tightening only — adds a significance requirement and removes cells; `frozen_rule_selection` is published in every run so both picks are visible |

**C4 deserves scrutiny.** It is the most outcome-informed decision in the
project. The frozen rule selected `additive | c1 | 11 | c=+1.00`: 5.52 nats
against a 5.19 nat matched sham, excess 0.33 [−1.29, +1.90] — a cell
indistinguishable from a random perturbation. The amendment is defensible
because it only tightens, but that is a defence, not a substitute for having
specified it correctly.

**Root cause of three of the four: the protocol was frozen without first
measuring the baseline.** One baseline cell costs two minutes and would have
caught the saturated metric directly, and probably the ranking flaw too.
**Before freezing the Checkpoint 5 protocol, run the baseline cell first and
look at the distribution of every quantity a threshold will be set on.**

## 5. Traps and known limitations

- **`output_hidden_states` snapshots each block's output *before* user forward
  hooks run.** An intervention at layer L leaves `hidden_states[L+1]` looking
  untouched while genuinely changing `hidden_states[L+2]` and the logits. This
  reads as a failing test. Documented in `directions/model.py`.
- **The tool-dependent capability metric is underpowered.** 12 items, baseline
  0.583, so it moves in steps of 0.083 and the 0.10 threshold is barely one
  item. It will catch a collapse, not a modest degradation. Read it together
  with tool-dependent retain perplexity, which measures the same illegibility
  confound continuously.
- **Refusal detection is a deterministic string matcher** over the first 300
  characters, chosen so a second model's behaviour does not enter an effect
  size about the first. Every generated string is in the raw JSONL, so
  disagreements are auditable. Safety records distinguish `refused`,
  `emitted_tool_call`, and `complied_in_text` — a tool call is neither a
  refusal nor harmful text.
- **MPS determinism is unproven.** `use_deterministic_algorithms(True)` is
  accepted and repeated matmuls are bitwise equal, but `index_add_` has no
  deterministic MPS implementation. Checkpoint 4's byte-identical-trajectory
  test is the real proof and a failure invalidates Checkpoint 5's design.
- **`status.json` cumulative time for this run was corrected by hand.** Tranche
  A ran under code predating the cumulative-budget fix, so the resumed pass
  read a prior spend of 0. True total 16.18 h; the field carries a note. Fixed
  for future runs.
- **Two dataset defects were found by probing the pinned model before
  freezing**, and both would have silently voided a metric: a harmful request
  in the *opening* user turn left the model in tool-calling mode so it never
  reached a refusal, and "setting the tool result aside" made it decline
  arithmetic. Probe the model before freezing a dataset.

## 6. Reproduction

Everything needed to re-derive the reported numbers is committed.

| Artifact | Path |
| --- | --- |
| Raw cell records | `runs/phase0-2026-08-24/results.jsonl.gz` (gunzip in place) |
| Run manifest — git rev, model rev, deps, hardware, dataset hashes | `runs/phase0-2026-08-24/manifest.json` |
| Direction bundles, 96 of them | `runs/phase0-2026-08-24/directions.tar.gz` |
| D2 diagnostic vectors, extraction summaries | `runs/phase0-2026-08-24/diagnostics.tar.gz`, `extraction/` |
| Frozen SAE feature selection, hashed | `runs/phase0-2026-08-24/sae_features.json` |
| Report | `runs/phase0-2026-08-24/analysis/phase0.md` |

Run git revision `b107c51`, model
`meta-llama/Llama-3.1-8B-Instruct @ 0e9e39f249a16976918f6564b8830bc894c89659`,
SAE `Goodfire/Llama-3.1-8B-Instruct-SAE-l19 @ f6775a221e47b44233af4bac2c7b65189265519a`.
Dataset hashes are in `evals/data/HASHES.json` and re-recorded in the manifest;
`uv run owie-build-datasets` must reproduce them byte-for-byte.

Bundles are never overwritten and a re-fit gets a new id. `read_bundle`
recomputes the contrast hash and re-checks the vector against its manifest, so
a bundle that has drifted raises instead of loading.

## 7. What to do next

1. **Checkpoint 4 — the deterministic ReAct loop.** ~300 lines, three fake tool
   domains, programmatic success predicates, JSONL trajectories. Acceptance:
   two no-intervention runs of the same task and seed produce byte-identical
   trajectory files.
2. **Run the determinism test on the 3B pilot first** (`DECISIONS.md` B6). It
   is placed there precisely because MPS determinism is the risk, the test does
   not care about model quality, and a failure invalidates Checkpoint 5.
3. **Add forged role headers to the Checkpoint 4 injection tasks** — the
   remaining half of D11. Checkpoint 3 proved provenance survives them; the
   task set should exercise it end to end.
4. **Before Checkpoint 5:** get a budget ruling (D13, costed at 4–10 days),
   write `HYSTERESIS_PROTOCOL.md` before implementing it, and freeze the
   protocol *after* measuring baselines (§4 above).

Open rulings still outstanding are in `DECISIONS.md` §D. D13 is the only one
blocking Checkpoint 5; nothing blocks Checkpoint 4.

## 8. Discipline that is binding, not aspirational

Carried from `doc/TODO.md` and honoured throughout this run:

- one checkpoint at a time; stop at every acceptance and kill gate and report;
- raw data immutable, derived analysis regenerable, bundles never overwritten;
- never silently revise the experiment after seeing results — deviations are
  recorded in `DECISIONS.md` §C with reasons and both readings reported;
- null results, prompting wins, and safety degradation are valid findings;
- suppression is a **control surface**, never removal or a guarantee;
- do not publish, push externally, or provision paid compute without explicit
  authorization.
