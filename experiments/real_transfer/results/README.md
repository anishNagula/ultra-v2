# Results log: runs, decisions and deviations

Times are UTC, 2026-10-27, from file modification times. Each decision was written before the results
it affects were read. Before registration, no target test metric was computed anywhere in this study.
The only test numbers used are the registered C-time results, published before this study began.

## Before registration (validation only)

| Time | What | Output |
|---|---|---|
| 08:41–08:53 | Phase-2 diagnostics: reachability, `oracle_labels`, `dense_weights`, `source_sparse`, `source_dense` (seeds 20261401–02) | `diagnostics/*.json`, `../logs/diagnostic_*.log` |
| 08:59 | `fedavg_replay`: the rebuilt FedAvg equals the registered curve (seed 20261211, rounds 0–3) | `diagnostics/fedavg_replay.json` |
| 09:03–09:05 | Smoke test, rule a. All four runs completed. Local seed 20261211 reproduces its registered 21-value validation curve and selected round. Cache reuse is checked. | `smoke/` |
| 09:04–09:06 | Diagnostics at f = 0.25: `oracle_labels`, `source_sparse` | `diagnostics/*_f0.25.json` |
| 09:09 | `fedavg_sparse` (registered seeds 20261211–13, paired with the registered validation curves) | `diagnostics/fedavg_sparse.json` |
| 09:14 | Popularity rankers | `diagnostics/popularity.json` |
| 09:26–09:55 | Pilot (seeds 20261401–03, `--skip-test`, rule a), every arm at every fraction | `pilot/` |
| 09:39–09:43 | `source_select`: the pilot's source trajectories, verified identical, with target validation every round | `diagnostics/source_select_f*.json` |
| 09:44–09:51 | Decisions 1 (final-round source checkpoint) and 2 (P1 candidate), written to `DESIGN.md`; diagnosis and draft preregistration updated | `../DESIGN.md`, `../DIAGNOSIS.md` |
| 09:53–09:54 | Smoke test of the registered configuration: final-round checkpoint, `--source-only` caching, cache hit | `smoke_final/` |

### Fixes and disclosures before registration

1. **Name collision, no effect on results.** `src/analyze.py` was shadowed on import by
   `transfer_conditions/src/analyze.py`, because `env.py` puts the reused code first on `sys.path`.
   It was renamed `src/analyze_transfer.py` before any analysis ran.
2. **Analysis pairing.** `analyze_transfer.pair` compares only the target-stage settings that every
   arm shares, so two source arms with different source settings can be compared. A unit test covers
   both the refusal and the permitted case.
3. **Source checkpoint rule changed from the plan, on validation evidence only** (`DESIGN.md`,
   decision 1). The pilot ran with rule a, the best source-validation round. At f = 0.10 that rule
   stops within 7 rounds on every seed. Source validation moves against target validation
   (ρ = −0.39, −0.80, −0.57), while the final round is better by +0.006 to +0.008. The registered rule
   is the final round. `run_pilot.sh` and `run_smoke.sh` now pin `--source-checkpoint source_valid`,
   so rerunning them reproduces what was run.
4. **Source-stage weights are not committed** (`../.gitignore`, `../../../EXCLUDED_DATA.md`). They are
   deterministic and each run file records their SHA-256.

## Registration and confirmatory run

| Time | What | Output |
|---|---|---|
| 10:02 | `PREREGISTRATION.md` finalised. `preregistration_hashes.txt` written: 46 SHA-256 hashes, shasum format, paths from the repository root. All 46 verify. | `preregistration_hashes.txt` |
| 10:03 | Commit `f317866` on branch `final/real-transfer`: the registered files, plus the code, diagnostics, pilot and smoke results. No target test metric existed. | |
| 10:05 | `run_confirmatory.sh` started | `../logs/run_confirmatory.out`, `../logs/confirmatory_*.log` |
| 11:21 | Stage 2 started. Stage 1's 6 jobs all reported `done`. | |
| 12:08 | Grid finished. All 10 jobs reported `done`, and no log has a `Traceback`, `Error` or `Killed` line. | `confirmatory/*.json` |
| 12:48 | `report/report_confirmatory.py --check` passed. It reads no metric value; it checks structure and hashes only. | |
| 12:49 | `run_analysis.sh` ran once: 46 hashes re-verified, 12 comparisons | `../logs/analysis.log`, `confirmatory/analysis/` |
| 12:50 | `report/report_confirmatory.py` wrote the tables | `confirmatory/report.md` |

### Disclosures after registration

5. **`report/report_confirmatory.py` was added after registration, before the analysis ran.** It is
   not a registered file and lives outside `src/`, so the registered set is unchanged. Its `--check`
   mode reads only structure, configuration and hashes. Its report mode formats the registered
   analysis outputs and computes descriptive means from the run files. It selects nothing and
   computes no new test statistic. Its "registered directional predictions" table encodes the
   predictions as written in `PREREGISTRATION.md`.
6. **The stopping rule was followed.** After the analysis, no further method was evaluated on the
   test set. P1 failed, so the study ends with that negative primary result.

The version of this file committed in `f317866` gave 10:05 as the finalisation time. That was an
estimate written before the commit. The times above come from the file modification times and the
commit time.
