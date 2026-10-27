# Results log: amendments, deviations and disclosures

Times are UTC, 2026-10-26. Each entry was written before the results it affects were read.

## Before registration (17:25:34)

1. **17:11:37 — design fixed.** `DESIGN.md` was hashed before any candidate was built.
2. **Calibration script, no effect on data.** `scipy` is not installed, so `calibrate_proxy.py`
   computes Spearman's ρ with NumPy ranks. No package was installed. The rerun added source-only
   rule MRR, which Amendment 1 needed.
3. **17:19:23 — DESIGN Amendment 1.**
   - **What changed.** The proxy-based go rules (ii) and (iii), and the proxy-based choices of
     candidate and primary fraction, were replaced by:
     - an availability rule (A ≥ 0.9);
     - an ULTRA-need gate from the validation-only smoke test;
     - a primary fraction fixed at f = 0.10.
   - **Reason.** The calibration on earlier studies showed the proxy is about 0 where federated
     ULTRA gained most. Study S shared had proxy −0.019 against an actual Δ of +0.100 at n = 13.
   - **Disclosure.** The viability job (`viability.py`) finished while the amendment was being
     written. Its output (`logs/viability.log`, `results/viability.json`) was first read after
     the amended `DESIGN.md` was hashed.
4. **Viability bug fix, no effect on selection.** In the plain-text actor-mention check,
   actors whose names are all shorter than 4 characters (for example "RTM") produced an empty
   regex that matched every description. Such actors are now skipped. Only the residual
   semantic-overlap count changed (C-time: 130 → 1; C-rand: 112 → 2). The selection rule does not
   use it, and the decision was the same before and after the fix.

## After registration

No deviations from the protocol.

- **17:33: smoke test passed, decision GO.** See `smoke/smoke_summary.json` (seed 20261209, validation only).
  - (a) All four runs completed.
  - (b) Round-0 validation MRR is identical in both arms at f = 0.10 and f = 1.00.
  - (c) Local's best validation MRR is 0.0781 at f = 0.10 and 0.1508 at f = 1.00, a gap of
    0.0727 ≥ 0.03.

  Together with V1–V4 (pass), availability A = 1.046 ≥ 0.9, and `test_pipeline.py` (10/10), every
  go condition holds. The grid was then started, before any target test metric existed.
- **17:33:07–18:56:52: confirmatory grid** (`logs/confirmatory.log`).
  - All 6 jobs (60 runs) completed with no failures and no resumes.
  - No seed was dropped, added or replaced, and test metrics were not looked at before the
    analysis.
- **18:57:57: registered analysis, run once** (`logs/analysis.log`,
  `confirmatory/analysis_summary.json`). Checks made just before it ran:
  - every run file holds the 10 registered seeds and complete test results for the 314 test
    triples;
  - protocol label, client weights, checkpoint and data hashes match `data/v1/C-time/manifest.json`;
  - all 25 hashes in `preregistration_hashes.txt` were recomputed and match.

  **Result: H1 not supported.** At f = 0.10, Δ = −0.0047, t-CI [−0.0071, −0.0024] and bootstrap
  CI [−0.0090, −0.0009]. The secondary cells also fail. Full results are in `../README.md`.
- **After the analysis:** descriptive summaries (Hits@k, validation curves, selected rounds,
  run times) were computed from the saved run files for `../README.md`. They are labelled as not
  registered and change no result or decision.
