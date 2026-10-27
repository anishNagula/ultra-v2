# Results log

Deviations from [`../PREREGISTRATION.md`](../PREREGISTRATION.md) and clarifications of it, each
written before the affected results were read. Times are UTC, 2026-10-26.

## Deviations

**None to the protocol.** One operational restart:

- **10:50, launcher restart.** The first launch of `run_all.sh` wrote every job's log to
  `logs/.log` because of a shell-expansion bug in the log name. All jobs were stopped before
  any seed finished and no result file existed. The log name was fixed (`${@: -1}`) and the
  grid was relaunched from scratch with unchanged job arguments.

## Clarification: how H2 and H3 are judged (written 13:48, before `analyze.py` was run)

`PREREGISTRATION.md` states H2 and H3 as inequalities but does not say how they are tested.
They are judged with the same two 95% CIs as the full success criterion: a seed-level t-CI and
a hierarchical paired bootstrap that resamples seeds and test triples jointly (20,000 samples,
seed 20261026). The +0.03 floor is not applied, because it is a floor on a transfer gain, not on
a difference between gains.

- **H2** is supported if mean[Δ_shared(13) − Δ_shared(408)] > 0 and both CIs lie entirely above 0.
- **H3(a)** is supported at a given n if mean[Δ_shared − Δ_disjoint] > 0 and both CIs lie entirely above 0.
- **H3(b)**, Δ_disjoint(408) ≤ 0, holds if its mean is ≤ 0. Negative transfer is called *shown*
  only if both of its CIs lie entirely below 0.
- **H3** is supported if (a) holds at all four n and (b) holds.

The contrast bootstrap is `contrast()` in `src/analyze.py`. That file was last modified at
10:51:15, before the first result file was written (11:23:40), and is unchanged since
(sha256 `8d12d8004a90f58e8628f472cb27fb0d08077d1739b94b43876bbd3cd1902a5c`).

**Disclosure.** During the run, status checks printed the last line of each job's log, which
gives one seed's test MRR for that arm. No comparison or aggregate had been computed when
this clarification was written.

## Analysis run

Checks made before the run, all of which matched:
- every registered file and data manifest against `preregistration_hashes.txt`;
- every result file's recorded data hashes against the data on disk.

`src/analyze.py` was then run once, from 13:48:20 to 13:49:04. Outputs:
- `analysis_summary.json`;
- per-cell `*_comparison.json` and `*_bootstrap.json` files in `synthetic/` and `real/`;
- the console output, in `../logs/analyze.log`.

The results are in the study [`README.md`](../README.md#results).
