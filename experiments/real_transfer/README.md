# Real transfer: can source training help a data-limited client on real cybersecurity data?

This study asks the question `enterprise_split` left open. There, on the registered C-time split of
**real Enterprise ATT&CK v19.2**, FedAvg scored 0.0047 test MRR *below* target-only fine-tuning at
f = 0.10. Here the target is the same split under the same protocol and criterion. The study:
1. diagnoses that failure on validation;
2. designs one regime-matched transfer method;
3. preregisters it and tests it once on the untouched test set.

No fact is planted, invented or added to either graph. Raw source facts never leave the source. Only
model weights cross, once.

**Result: the mission's criterion is not met.**
- **Primary hypothesis (P1).** At f = 0.10, regime-matched source training (`sparse_source`) beats
  the registered local arm on all 10 seeds, with both 95% CIs above 0. The gain is **+0.0062 test
  MRR, far below the required +0.03**, so the full success criterion fails.
- **Zero-shot question (Z1, secondary).** With no target training on either side, the same source
  weights beat the unchanged pretrained ULTRA checkpoint by **+0.0545**, with both CIs above 0. This
  is a full PASS. It answers a different question and cannot substitute for P1.

See [Confirmatory results](#confirmatory-results).

Files, in the order they were fixed:
1. [`DIAGNOSIS.md`](DIAGNOSIS.md): why the registered FedAvg failed, from validation data.
2. [`DESIGN.md`](DESIGN.md): constraints, candidates, the method, the pilot and two decisions.
3. [`PREREGISTRATION.md`](PREREGISTRATION.md): committed in `f317866` at 10:03Z, before the grid
   started, with the hashes in `results/preregistration_hashes.txt`.
4. [`results/README.md`](results/README.md): the timed run log, deviations and disclosures.
5. `results/confirmatory/analysis/`: the registered analysis, run once at 12:49Z.
6. `results/confirmatory/report.md`: the tables below, generated from the run and analysis files.

**Provenance.** Branch `final/real-transfer`. Baseline: commit `5bd8d0b`, the registered
`enterprise_split` study and its results; `main` is unchanged at `427966a`. Preregistration:
`f317866`. Results: the commit that adds this file.

## Diagnosis (`DIAGNOSIS.md`)

What each explanation rests on. "Validation" means the evidence used target validation data only.
"Test" means a registered confirmatory prediction was checked on the untouched test set.

| | Explanation | Status |
|---|---|---|
| E1 | **Graph-limited ceiling.** At f = 0.10, the target graph carries too little evidence for any change to ULTRA's weights to add +0.03. 87% of validation answers are ≥4 hops away or disconnected. Training on the oracle validation labels themselves plateaus at 0.077, local's level. | Supported on validation. It is consistent with test: local fine-tuned, dense_source fine-tuned and sparse_source (zero-shot or fine-tuned) all land within 0.087–0.093 at f = 0.10, against 0.17–0.18 at f = 1.00. It is not proven as the cause; no registered test isolates it. |
| E2 | **Regime mismatch.** The registered source trained with its complete graph as context, while the target predicts from a sparse one. | The intervention effect is **confirmed on test** by four registered directional predictions (M1–M4, all held). Sparse-view source training beats complete-graph source training zero-shot by +0.058 at f = 0.10 and +0.027 at f = 0.25 (t-CIs above 0). That this mismatch caused the registered FedAvg's loss rests on validation only: `fedavg_sparse`, 3 seeds. |
| E3 | Task mix in the source loss | Not supported (validation). |
| E4 | Equal-weight averaging dilutes the target's updates | Plausible contributor only (validation). |
| E5 | Entity-level knowledge that ULTRA cannot carry | Real but small. The best popularity ranker reaches +0.009 over local at f = 0.10 (validation). |

## Method (`DESIGN.md`, method A; `src/sparse.py`, `src/run_method.py`)

1. **Source stage.** This runs on the source client, on its own facts only:
   - Each of 30 rounds × 10 steps draws a fresh **per-head view** of the source graph using the
     target's own scarcity rule (keep max(1, round(f·n_h)) facts per head), with f equal to the
     target's fraction.
   - The kept facts are the message graph. The held-out actor `uses` facts are the positives.
   - The optimiser and loss are the registered ones.
   - A fixed 10% of source actor `uses` facts is held out as source validation and recorded.
   - The final round is kept (decision 1).
2. **One transfer.** The source weights go to the target once. No fact or gradient crosses.
3. **Target stage.** The registered local arm runs unchanged from those weights:
   - 20 rounds × 10 steps, AdamW lr 5e-4;
   - round chosen on target validation, with 0 = the starting weights;
   - test scored once for that round (fine-tuned) and once for the starting weights (zero-shot).

**Arms.**
- `local`: the registered local arm, from the pretrained checkpoint.
- `sparse_source`: the method.
- `dense_source`: the same source stage on the complete source graph. This is the control; at
  f = 1.00 it is the regime-matched arm.

**Rationale.** E2 predicts that training the source in the regime the target predicts in removes the
harm and transfers ULTRA's relational reasoning, not entity facts. The design (candidates D–G) keeps
the method inside the constraints: real facts only, weights only, and the registered target budget
and evaluation. Because the target stage is the local arm unchanged, `sparse_source − local` isolates
the effect of the starting weights.

## Confirmatory results

**Run.**
- The grid (`run_confirmatory.sh`) ran from 10:05:25Z to 12:08:57Z, in 2 h 04 min.
- It produced 8 run files × 10 seeds (20261501–20261510), with no failures and no seed dropped,
  added or replaced.

**Checks before the analysis** (`report/report_confirmatory.py --check`, 12:48Z; it reads no metric):
- every file has the registered protocol, configuration, checkpoint and data hashes;
- 314 aligned test triples, fine-tuned and zero-shot;
- final-round source weights;
- identical dense-source weights across fractions for each seed.

**The analysis.** `run_analysis.sh` re-verified all 46 registered hashes and ran each comparison
once, from 12:49:00Z to 12:49:49Z (`logs/analysis.log`).

Δ is candidate − baseline target-test MRR, paired by seed and test triple.
- **Full criterion:** mean Δ ≥ +0.03, with the seed-level t-CI and the hierarchical paired
  bootstrap CI (20,000 samples) both entirely above 0.
- **"pretrained"** is the local arm's zero-shot score: the unchanged checkpoint, identical across
  seeds.

| Comparison | Role | Δ | Baseline MRR | Candidate MRR | Mean Δ | Seed-level t-CI 95% | Bootstrap CI 95% | Seeds Δ > 0 | Full criterion |
|---|---|---|---|---|---|---|---|---|---|
| **P1, f = 0.10** | **primary** | sparse_source[test] − local[test] | 0.0872 | 0.0933 | **+0.0062** | [+0.0034, +0.0090] | [+0.0019, +0.0107] | 10/10 | **FAIL** |
| P2, f = 0.25 | secondary | sparse_source[test] − local[test] | 0.0906 | 0.0959 | +0.0053 | [+0.0019, +0.0087] | [−0.0010, +0.0120] | 9/10 | fail |
| P3, f = 1.00 | secondary | dense_source[test] − local[test] | 0.1706 | 0.1839 | +0.0133 | [+0.0084, +0.0182] | [+0.0036, +0.0234] | 10/10 | fail |
| **Z1, f = 0.10** | **secondary: zero-shot question** | sparse_source[zero_shot] − pretrained | 0.0383 | 0.0929 | **+0.0545** | [+0.0529, +0.0562] | [+0.0373, +0.0730] | 10/10 | **PASS** |
| Z2, f = 0.25 | secondary | sparse_source[zero_shot] − pretrained | 0.0717 | 0.0972 | +0.0255 | [+0.0239, +0.0272] | [+0.0130, +0.0381] | 10/10 | fail |
| Z3, f = 1.00 | secondary | dense_source[zero_shot] − pretrained | 0.1704 | 0.1839 | +0.0135 | [+0.0086, +0.0184] | [+0.0018, +0.0253] | 10/10 | fail |
| M1, f = 0.10 | mechanism | dense_source[zero_shot] − pretrained | 0.0383 | 0.0344 | −0.0039 | [−0.0107, +0.0028] | [−0.0123, +0.0043] | 2/10 | – |
| M2, f = 0.10 | mechanism | sparse_source[zero_shot] − dense_source[zero_shot] | 0.0344 | 0.0929 | +0.0584 | [+0.0513, +0.0656] | [+0.0409, +0.0774] | 10/10 | – |
| M3, f = 0.10 | mechanism | dense_source[test] − local[test] | 0.0872 | 0.0881 | +0.0009 | [−0.0009, +0.0027] | [−0.0029, +0.0047] | 8/10 | – |
| M4, f = 0.25 | mechanism | sparse_source[zero_shot] − dense_source[zero_shot] | 0.0701 | 0.0972 | +0.0272 | [+0.0219, +0.0324] | [+0.0116, +0.0431] | 10/10 | – |
| R1, f = 0.10 | reference | sparse_source[zero_shot] − local[test] | 0.0872 | 0.0929 | +0.0057 | [+0.0024, +0.0089] | [+0.0009, +0.0104] | 9/10 | – |
| R2, f = 0.25 | reference | sparse_source[zero_shot] − local[test] | 0.0906 | 0.0972 | +0.0066 | [+0.0042, +0.0090] | [−0.0015, +0.0149] | 10/10 | – |

The mechanism tests have no +0.03 threshold. Each one has a registered direction:

| Test | Registered prediction | Observed | Holds |
|---|---|---|---|
| P1 | FAIL, with the t-CI's upper end below +0.03 | FAIL; upper end +0.0090 | yes |
| M1 | mean below +0.01 | −0.0039 | yes |
| M2 | t-CI above 0 | [+0.0513, +0.0656] | yes |
| M3 | mean within ±0.01 | +0.0009 | yes |
| M4 | t-CI above 0 | [+0.0219, +0.0324] | yes |

Per seed: Δ, with the selected target round (baseline / candidate) in brackets. The Z rows compare
starting weights, so no round is selected.

| Seed | P1, f = 0.10 | P2, f = 0.25 | P3, f = 1.00 | Z1, f = 0.10 | Z2, f = 0.25 | Z3, f = 1.00 |
|---|---|---|---|---|---|---|
| 20261501 | +0.0022 (5 / 11) | +0.0049 (6 / 6) | +0.0075 (4 / 0) | +0.0546 | +0.0265 | +0.0046 |
| 20261502 | +0.0013 (3 / 4) | +0.0097 (5 / 0) | +0.0146 (3 / 0) | +0.0521 | +0.0266 | +0.0225 |
| 20261503 | +0.0029 (11 / 12) | +0.0079 (4 / 2) | +0.0088 (7 / 0) | +0.0517 | +0.0254 | +0.0066 |
| 20261504 | +0.0052 (16 / 12) | +0.0121 (6 / 0) | +0.0291 (17 / 0) | +0.0542 | +0.0254 | +0.0177 |
| 20261505 | +0.0041 (15 / 15) | +0.0039 (6 / 16) | +0.0207 (2 / 0) | +0.0531 | +0.0193 | +0.0238 |
| 20261506 | +0.0071 (7 / 0) | +0.0003 (3 / 5) | +0.0069 (6 / 0) | +0.0581 | +0.0259 | +0.0077 |
| 20261507 | +0.0142 (15 / 0) | +0.0018 (17 / 17) | +0.0086 (19 / 0) | +0.0586 | +0.0250 | +0.0168 |
| 20261508 | +0.0095 (5 / 7) | +0.0075 (7 / 0) | +0.0128 (10 / 0) | +0.0535 | +0.0266 | +0.0156 |
| 20261509 | +0.0068 (3 / 4) | +0.0085 (4 / 7) | +0.0122 (1 / 0) | +0.0555 | +0.0266 | +0.0120 |
| 20261510 | +0.0085 (7 / 6) | −0.0033 (16 / 12) | +0.0116 (6 / 0) | +0.0538 | +0.0279 | +0.0077 |

The source stage always hands over its final round (30). Per-seed test MRR for every arm is in
`results/confirmatory/analysis/*.json` (`per_seed`) and the run files.

### Descriptive (not registered; computed from the saved runs, selects nothing)

Means over the 10 seeds:

| f | Arm | Test MRR | Hits@1 | Hits@3 | Hits@10 | Zero-shot test MRR | Best validation MRR | Mean selected round | Seconds per seed (job) | Source stage, seconds per seed |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.10 | local | 0.0872 | 0.0441 | 0.0811 | 0.1495 | 0.0383 | 0.0774 | 8.7 | 102 | – |
| 0.10 | dense_source | 0.0881 | 0.0432 | 0.0839 | 0.1511 | 0.0344 | 0.0792 | 11.5 | 123 | 667 |
| 0.10 | sparse_source | 0.0933 | 0.0455 | 0.0924 | 0.1597 | 0.0929 | 0.0839 | 7.1 | 345 | 219 |
| 0.25 | local | 0.0906 | 0.0347 | 0.0876 | 0.1783 | 0.0717 | 0.0935 | 7.4 | 120 | – |
| 0.25 | dense_source | 0.0892 | 0.0331 | 0.0839 | 0.1861 | 0.0701 | 0.0947 | 7.4 | 156 | 667 |
| 0.25 | sparse_source | 0.0959 | 0.0385 | 0.0912 | 0.1849 | 0.0972 | 0.0963 | 6.5 | 429 | 273 |
| 1.00 | local | 0.1706 | 0.0863 | 0.1842 | 0.3328 | 0.1704 | 0.1532 | 7.5 | 285 | – |
| 1.00 | dense_source | 0.1839 | 0.0903 | 0.2099 | 0.3575 | 0.1839 | 0.1692 | 0.0 | 286 | 667 |

How to read the runtime columns:
- `sparse_source`'s job time includes its own source stage.
- `dense_source` loads its source stage from a cache, trained once per seed in parallel warm-up
  jobs, so its job time excludes it.
- Times are wall-clock with 2 threads per job and up to 4 jobs in parallel on an Apple M4 CPU.

**Two checks on unusual-looking rows.**
- At f = 1.00, dense_source's target stage selected round 0 on all 10 seeds, so its fine-tuned and
  zero-shot test scores are the same numbers.
- local's zero-shot score is the pretrained model on every seed. Its zero-shot validation scores
  (0.0407, 0.0786, 0.1457) equal the registered pretrained values. At f = 1.00 the pretrained test
  MRR (0.1704) is close to the fine-tuned mean (0.1706) by coincidence; per seed, the fine-tuned
  score ranges from 0.159 to 0.179.

### Interpretation

**Against the mission's criterion: FAIL.** Regime-matched transfer is a real improvement. At
f = 0.10:
- it beats local on 10 of 10 seeds, with both CIs above 0;
- it turns the registered method's loss (FedAvg − local = −0.0047, on different seeds, so the
  comparison is descriptive) into a gain.

But the gain is about a fifth of the required +0.03. The upper end of its 95% t-CI is +0.0090. The
registered prediction was FAIL, from E1, and it held.

**What the source contributes, and where it stops.**
- At f = 0.10, the source weights alone, with no target training, score 0.0929. That is above
  local's fine-tuned 0.0872 (R1: +0.0057, both CIs above 0). Fine-tuning them on the target adds
  only 0.0004 more.
- At f = 0.10, every route that trains on target-shaped data ends between 0.087 and 0.093:
  - local fine-tuned: 0.0872;
  - dense_source fine-tuned: 0.0881;
  - sparse_source zero-shot: 0.0929;
  - sparse_source fine-tuned: 0.0933.

  With the target's full graph, the same model reaches 0.17–0.18.
- This pattern is what E1 predicts. The target's deficit is missing evidence in its own graph, and
  a weight transfer that keeps raw facts local cannot supply that evidence.

**Zero-shot transfer: PASS (Z1).** Training the source in the regime the target predicts in carries
a large, reproducible gain to an unseen client of the same domain: +0.0545 over pretrained ULTRA,
from 0.0383 to 0.0929.
- The seed-level t-CI is very narrow because the baseline is the same deterministic model on every
  seed.
- The hierarchical bootstrap, which also resamples test triples, gives the more informative
  [+0.037, +0.073].
- The same comparison at f = 0.25 (Z2, +0.0255) and f = 1.00 (Z3, +0.0135) is positive with CIs
  above 0, but below +0.03.

**Mechanism (E2): confirmed for the intervention.** At f = 0.10:
- complete-graph source training gives no zero-shot gain (M1, −0.0039);
- it adds nothing once fine-tuned (M3, +0.0009);
- sparse-view source training on the same facts gives +0.058 over it zero-shot (M2).

The regime the source trains in, not the source data, decides whether its weights transfer to a
sparse target.

## Limitations

- **One split, one release, one checkpoint.** The study covers the C-time split of Enterprise
  ATT&CK v19.2, the `ultra_4g` checkpoint, and one target budget (20 rounds × 10 steps). The
  bootstrap resamples seeds and the 314 test triples; it does not cover alternate snapshots,
  partitions or domains.
- **Regime matching needs one scalar of target metadata**, the per-head retention fraction f.
  - Here f is the split's design value; a real target would have to estimate it from its own
    degrees and share it.
  - The arms used f = 0.10, 0.25 and 1.00 exactly. How well the method tolerates a misestimated f
    was not tested.
- **Not federated averaging.** The method is a one-shot source-to-target weight transfer. It does
  not test FedAvg with sparse source updates; that was a validation-only diagnostic (`DESIGN.md`,
  candidate C).
- **The test set is now spent for this study.** Following the stopping rule, no further method was
  evaluated on it. A new method, or an independent confirmation of Z1, needs a new held-out set,
  for example actors first documented in a later ATT&CK release.
- **Pre-registration deviations and disclosures** are listed in `results/README.md`:
  - the analysis module was renamed before any analysis ran;
  - analysis pairing compares target-stage settings only;
  - the source checkpoint rule was changed on validation evidence before registration;
  - this README's report script was added after registration, before the analysis ran;
  - one registration time in the run log was corrected.

## Privacy

- **What stays local.** Raw source facts never leave the source client. Target facts never leave
  the target, and the source stage reads no target data (`test_method.py` checks this).
- **What crosses.** One set of model weights, once, plus the scalar f.
- **What is not guaranteed.** The weights are not differentially private. Nothing here bounds what
  an attacker could infer about source facts from them, for example by membership inference. This
  was not evaluated.

## Uncompleted work

- **No method reached +0.03 over local on real data.** The diagnosis (E1) says why one is unlikely
  under these constraints: weight-only transfer cannot add evidence missing from the target's graph.
  Routes that could add it were rejected because they break the constraints (`DESIGN.md`):
  - sharing source facts, or edges derived from them;
  - entity-level priors;
  - a larger target budget.
- **No replication on fresh data.** The zero-shot result (Z1) has not been replicated on a second
  real split or a later ATT&CK release.
- **No privacy analysis** of the transferred weights.

## Reproduce

From the repository root, with the project's `.venv`:

```bash
cd experiments/real_transfer/src
python test_method.py                                                   # 12 unit tests
python reachability.py --output ../results/diagnostics/reachability.json
python popularity.py --output ../results/diagnostics/popularity.json
cd .. && bash run_diagnostics.sh && bash run_diagnostics_f025.sh        # Phase-2 diagnostics (validation only)
bash run_diagnostics_fedavg.sh && bash run_diagnostics_select.sh
(cd src && python summarize_diagnostics.py)                             # results/diagnostics/summary.json
bash run_smoke.sh && bash run_pilot.sh && bash run_smoke_final.sh       # smoke tests and pilot (validation only)
bash run_confirmatory.sh                                                # registered grid, about 2 h on 4 parallel jobs
(cd report && python report_confirmatory.py --check)                    # structure and hashes only
bash run_analysis.sh                                                    # registered analysis (run once)
(cd report && python report_confirmatory.py)                            # results/confirmatory/report.md
```

`run_analysis.sh` refuses to run if any file in `results/preregistration_hashes.txt` has changed.
Cached source weights (`results/*/source_cache/`) are not committed (`../../EXCLUDED_DATA.md`).
`run_confirmatory.sh` regenerates them deterministically, and each run file records their SHA-256.

## Files (all new; outside this folder, only a section added to the root `EXCLUDED_DATA.md`)

- `README.md`, `DIAGNOSIS.md`, `DESIGN.md`, `PREREGISTRATION.md`, `.gitignore`
- `src/`:
  - `env.py` (paths; the reused code from `fedultra_fresh/src` and `transfer_conditions/src` is
    imported unmodified);
  - `sparse.py` (views and the source stage);
  - `run_method.py` (arms, cache, scoring);
  - `analyze_transfer.py` (pairing and the criterion);
  - `test_method.py`;
  - `diagnose.py`, `reachability.py`, `popularity.py`, `summarize_diagnostics.py`.
- `report/report_confirmatory.py`: checks and tables, added after registration.
- `run_*.sh`: the diagnostics, smoke tests, pilot, confirmatory grid and analysis.
- `results/`:
  - `diagnostics/`, `smoke/`, `smoke_final/`, `pilot/` (validation only);
  - `preregistration_hashes.txt`, `README.md`;
  - `confirmatory/`: one run file per arm and fraction, `analysis/` (for each comparison: summary,
    paired comparison and bootstrap), `report.md`.
- `logs/`: one log per job, plus `run_confirmatory.out` (the grid) and `analysis.log`.
