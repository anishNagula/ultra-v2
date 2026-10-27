# Real transfer: preregistration

Registered before any confirmatory run. No target test metric had been computed by this study when
this file was written. The diagnosis (`DIAGNOSIS.md`), design and pilot (`DESIGN.md`) used validation
only. The SHA-256 of this file, the code, the run scripts and the data are in
`results/preregistration_hashes.txt`, committed before `run_confirmatory.sh` was started.

## Questions

1. **Primary (the mission's criterion, on the registered C-time protocol).** Does transferring weights
   from the source client give the data-limited target a gain of at least +0.03 target-test MRR over
   fine-tuning on its own data alone (the registered local arm)?
2. **Secondary (the project's frozen question, `../../CYBER_TRANSFER_PROTOCOL.md`, applied to this
   split).** Does source training transfer to the target with no target training, compared with the
   unchanged pretrained ULTRA checkpoint?

The secondary question has its own PASS/FAIL. It cannot substitute for the primary one.

## Data and evaluation (unchanged from `../enterprise_split/PREREGISTRATION.md`)

- C-time split of Enterprise ATT&CK v19.2, `../enterprise_split/data/v1/C-time`, hashes in its
  `manifest.json`. Every run file records the SHA-256 of each target file and of `source/train.txt`.
- Target task: actor `uses` facts. Validation (314) and test (314) are identical at every f.
- Metric: filtered MRR over all target entities, head and tail queries. A triple's MRR is the mean of
  its two reciprocal ranks. The filters include `hidden.txt`.

## Arms (`src/run_method.py`, protocol label `real-transfer v1`)

- **Shared target stage.** The registered local arm, unchanged
  (`transfer_conditions/src/conditions_runner.py::run_local`):
  - AdamW lr 5e-4, weight decay 0;
  - 20 rounds × 10 steps, batch 8, 64 negatives;
  - sampling stream seed·1000003 + 7919 + round·97;
  - round chosen on target validation, 0 = the starting weights;
  - test scored once for that round (fine-tuned) and once for the starting weights (zero-shot).
- **local.** Starts from `ckpts/ultra_4g.pth`. This is the registered local arm; seed 20261211's
  validation curve reproduces bit for bit (`results/smoke/`).
- **sparse_source.** Starts from source-only weights trained on sparse per-head views of the source
  graph, with f equal to the target's fraction (`DESIGN.md`, method A).
- **dense_source.** The same on the complete source graph. At f = 1.00 this is the regime-matched arm.
- **Source stage.** 30 rounds × 10 steps, batch 8, 64 negatives, AdamW lr 5e-4, weight decay 0.
  - Source validation: a fixed 10% of source actor `uses` facts, held out, scored every round and
    recorded.
  - Checkpoint: the final source round (`--source-checkpoint final`). This was fixed from the
    validation-only pilot (`DESIGN.md`, decision 1). The source stage reads no target data.
  - Only these weights cross to the target, once. Raw facts never do.

## Hypotheses

Δ is always candidate − baseline target-test MRR, paired by seed and test triple. "test" is the
target stage's validation-selected round. "zero_shot" is the weights the target stage started from,
scored with no target training. For local, these are the pretrained checkpoint.

**H1, primary** (the mission's criterion):
- P1: at f = 0.10, sparse_source[test] − local[test] meets the full success criterion.
- Prediction, from `DIAGNOSIS.md` E1 and the pilot: FAIL.
  - Pilot validation Δ was +0.002 to +0.005 with rule a starting weights.
  - With the registered final-round weights, the zero-shot start alone is +0.0001 to +0.0093 above
    the same seeds' local.
  - The upper end of the 95% t-CI should lie below +0.03.

**Secondary**, with the same statistics. They cannot substitute for H1.
- P2 (f = 0.25): sparse_source[test] − local[test]. Predicted FAIL; pilot Δ +0.001 to +0.007.
- P3 (f = 1.00): dense_source[test] − local[test]. Predicted positive but below +0.03; pilot Δ
  +0.008 to +0.016 with rule a starting weights.
- **Z1, the zero-shot question** (f = 0.10): sparse_source[zero_shot] − local[zero_shot], i.e. the
  pretrained checkpoint. This is the decision for secondary question 2. Prediction: positive. Pilot
  validation Δ +0.0385, +0.0404, +0.0422 with the final-round rule. Whether it clears +0.03 on test is
  uncertain.
- Z2 (f = 0.25): the same. Predicted positive but below +0.03; pilot +0.011 to +0.018.
- Z3 (f = 1.00): dense_source[zero_shot] − local[zero_shot]. Predicted positive and probably below
  +0.03; pilot +0.017 to +0.027 with rule a.

**Mechanism tests** (`DIAGNOSIS.md` E2). These are directional predictions, judged by the 95% t-CI,
with no +0.03 threshold:
- M2 (f = 0.10) and M4 (f = 0.25): sparse_source[zero_shot] − dense_source[zero_shot] > 0. The t-CI
  should lie above 0.
- M1 (f = 0.10): dense_source[zero_shot] − local[zero_shot] < +0.01. The mean should be below +0.01;
  the sign is not predicted.
- M3 (f = 0.10): dense_source[test] − local[test] within ±0.01. The mean should lie in that range.

**Reference, descriptive only:** R1 (f = 0.10) and R2 (f = 0.25), sparse_source[zero_shot] −
local[test]. Does zero-shot transfer reach what fine-tuning on the target's own data reaches?

All comparisons and their order are in `run_analysis.sh`. Each one writes
`results/confirmatory/analysis/<name>.json`, with its paired bootstrap and comparison file next to it.

**What the outcomes mean.**
- P1 alone answers the mission's criterion. PASS needs all three parts of the full success criterion.
- Z1 answers the secondary, zero-shot question. A Z1 PASS with a P1 FAIL would mean: source training
  transfers to a target that does no training of its own, but it adds less than +0.03 once the target
  fine-tunes on its own data.
- No multiplicity correction is applied, because P1 is the only primary test. Every other result is
  labelled secondary, mechanism or reference.

## Seeds

- Confirmatory: 20261501–20261510, 10 per arm per fraction, never used before.
- Pilot: 20261401–20261403 (validation only).
- Smoke tests: 20261309, plus registered seed 20261211 for the reproduction check (validation only;
  `results/smoke/`, `results/smoke_final/`).

## Statistics

`src/analyze_transfer.py` pairs runs by seed and test triple. It reports:
- the per-seed Δ;
- the mean Δ;
- the seed-level 95% t-CI (`transfer_runner.summarize_direction`);
- the hierarchical paired bootstrap 95% CI (`fedultra_fresh/src/bootstrap_cyber_transfer.py`,
  20,000 samples, seed 20261026).

**Full success criterion (unchanged):** mean Δ ≥ +0.03, with both CIs entirely above 0.

## Stopping rules

1. The grid (`run_confirmatory.sh`) runs to completion with no interim look at test metrics.
   - Progress logs print `test=scored`, never a test value.
   - A crashed job is resumed with the same arguments; the runner skips completed seeds.
   - No seed is dropped, added or replaced.
   - A cell that cannot complete for a technical reason is reported as missing.
2. After registration, nothing in the data, budgets, hyperparameters or code changes, except bug fixes.
   Each fix is logged with its time in `results/README.md` before the affected results are read.
3. The analysis (`run_analysis.sh`) runs once, after the grid finishes. Every comparison is reported.
4. After the analysis, no further method is evaluated on this test set within this study. If the
   primary hypothesis fails, the study ends with that negative primary result.

## Registration

`results/preregistration_hashes.txt` lists the SHA-256 of:
- this file, `DESIGN.md` and `DIAGNOSIS.md`;
- `run_confirmatory.sh` and `run_analysis.sh`;
- every file in `src/`;
- the reused code (`fedultra_fresh/src/{fedavg_runner,transfer_runner,evaluation_protocol,bootstrap_cyber_transfer}.py`,
  `transfer_conditions/src/{common,conditions_runner}.py`) and `ultra/`'s model and task code;
- the checkpoint;
- every C-time data file.

Before the analysis runs, they are recomputed and must match.
