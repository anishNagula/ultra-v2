# Enterprise split: can federated ULTRA help a data-limited client of the same real domain?

Two clients are cut from **real Enterprise ATT&CK v19.2** facts, by threat-actor portfolio. No fact
is planted or invented. The test is whether a data-limited target gains at least +0.03 test MRR,
with 95% CIs above 0, from federating with the source rather than training alone. The criterion
and protocol are those of `fedultra_fresh`. Test metrics are never used to choose data,
partitions, hyperparameters or checkpoints.

**Result: H1 is not supported.** At f = 0.10, FedAvg scores 0.0047 test MRR *below* target-only
fine-tuning, with both 95% CIs below 0. A +0.03 gain is excluded at every fraction. See
[Confirmatory results](#confirmatory-results).

Files, in the order they were fixed:
1. [`DESIGN.md`](DESIGN.md): partition algorithm, candidates, metrics and go/no-go, plus
   Amendment 1.
2. `results/viability.json`: the Phase 2 checks.
3. [`PREREGISTRATION.md`](PREREGISTRATION.md): registered 17:25:34Z, hashes in
   `results/preregistration_hashes.txt`.
4. [`results/README.md`](results/README.md): amendments, deviations and disclosures.
5. `results/confirmatory/analysis_summary.json`: the registered analysis, run once at 18:57:57Z.

## Design report: why this partition is a plausible transfer scenario

**Scenario.** Two threat-intelligence teams model the same ATT&CK technique space but track
different actors.
- The selected partition, **C-time**, gives the **source** the 47 earliest-documented actor units
  and the **target** the 113 later-documented ones. It models an established team and a newer team
  covering recently documented actors.
- The target is data-limited. It keeps 10% (primary), 25% or 100% of its training facts.
- Each team trains on its own graph and shares only model weights.

This removes the schema mismatch that sank Enterprise→ICS and Enterprise→Mobile. Both clients have
the same five relations and identical ULTRA relation graphs, and same-name pretrained relation
cosine is 1.000. The clients still differ in what each one knows.

**Partition algorithm v1** (`src/build_split.py`, details in DESIGN.md):
1. **Actor units.** Groups and campaigns are joined into units through:
   - STIX `attributed-to` relationships;
   - description links to each other's ATT&CK pages;
   - shared aliases.

   This gives 160 units. Each unit goes **wholly** to one client, so overlapping real-world actors
   with cross-references stay together.
2. **Actor facts follow their unit.** These are actor uses technique, actor uses software, and
   campaign attributed-to group.
3. **Software and reference facts go to the client whose actors use them most**, counting
   training facts only. Software facts are software uses technique. Reference facts are
   mitigates, detects and subtechnique-of, anchored on their technique. Ties go to a seeded coin
   per entity.
4. **Every fact lands in exactly one client.** Shared entities, such as techniques, software and
   mitigations, appear in both graphs, but no fact does.
5. **Target evaluation** is 10% validation and 10% test from the target's actor `uses` facts.
   The answer to every query is a real fact about one of the target's own actors.
6. **Scarcity.** For each head entity, the target keeps max(1, round(f·n)) of its facts, nested
   across f. Removed facts are hidden: filtered at evaluation, never trained on.

**What the checks found.** These were measured, not assumed.
- **Valid and leak-free.** Both candidates have zero leakage of every kind:
  - no target fact, or its reverse, is in the source;
  - no target validation or test fact is in target training or hidden;
  - no source fact joins the two entities of a target validation or test fact;
  - no actor is in both clients.
- **The source holds the target's patterns.** On the target's complete graph, the source's path
  rules rank the target's held-out facts as well as the target's own rules do (A = 1.046). Every
  relation composition the target uses also has source support.
- **The patterns are weak in absolute terms.** Validation rule MRR is about 0.07, against 0.38
  for the planted synthetic pattern.
- **At 10% the target's evidence is thin.** Only 12.7% of validation facts are joined by a
  ≤3-hop path in its training graph, against 65% at 25%. Federation shares weights, not facts,
  so it cannot restore missing evidence paths.

## Candidate statistics and leakage audit (`results/viability.json`)

| | C-rand | C-time (selected) |
|---|---|---|
| Actor units (target / source) | 80 / 80 | 113 / 47 |
| Actor facts (target / source) | 3,556 / 3,565 | 3,499 / 3,622 |
| Source: training facts / entities | 10,612 / 1,619 | 11,149 / 1,643 |
| Target: entities | 1,533 | 1,480 |
| Target train / hidden at f = 0.10 | 1,566 / 8,314 | 1,492 / 7,834 |
| Target train / hidden at f = 0.25 | 2,889 / 6,991 | 2,760 / 6,566 |
| Target train / hidden at f = 1.00 | 9,880 / 0 | 9,326 / 0 |
| Target validation / test | 308 / 303 | 314 / 314 |
| Shared entities (techniques / software / mitigations) | 534 / 81 / 42 | 506 / 79 / 43 |
| Leakage L1–L4, every fraction | 0 | 0 |
| Software / techniques used by actors of both clients | 81 / 360 | 79 / 344 |
| Residual semantic overlap (plain-text cross-client actor mentions) | 2 | 1 (WIRTE → Molerats) |

The query-support and rule columns below use validation facts only. "Both endpoints in train"
means the share of validation and test facts whose head and tail both keep a training edge.

| Candidate | f | Train facts: uses / mitigates / detects / subtechnique-of / attributed-to | Relation paths with instances, length 2 / 3 | `uses` rules (support ≥ 2), length 2 / 3 | Validation facts joined by a ≤3-hop path | … by a source-supported rule | Rule MRR own / source / pooled | Proxy (pooled − own) | Both endpoints in train (valid / test) |
|---|---|---|---|---|---|---|---|---|---|
| C-rand source | – | 9,103 / 839 / 394 / 262 / 14 | 31 / 86 | 7 / 23 | – | – | – | – | – |
| C-rand target | 0.10 | 978 / 71 / 303 / 213 / 1 | 29 / 68 | 3 / 3 | 14.3% | 14.3% | 0.0264 / 0.0218 / 0.0218 | −0.0046 | 87.3% / 83.2% |
| C-rand target | 0.25 | 2,214 / 156 / 303 / 213 / 3 | 29 / 74 | 5 / 8 | 71.8% | 71.8% | 0.0334 / 0.0320 / 0.0320 | −0.0015 | 96.1% / 95.7% |
| C-rand target | 1.00 | 8,743 / 609 / 303 / 213 / 12 | 30 / 84 | 7 / 22 | 100.0% | 100.0% | 0.0818 / 0.0855 / 0.0871 | +0.0053 | 100% / 100% |
| C-time source | – | 9,737 / 759 / 373 / 262 / 18 | 31 / 86 | 7 / 23 | – | – | – | – | – |
| C-time target | 0.10 | 874 / 80 / 324 / 213 / 1 | 29 / 72 | 0 / 2 | 12.7% | 12.7% | 0.0232 / 0.0247 / 0.0247 | +0.0015 | 88.2% / 89.8% |
| C-time target | 0.25 | 2,043 / 177 / 324 / 213 / 3 | 29 / 82 | 5 / 5 | 65.3% | 65.3% | 0.0323 / 0.0295 / 0.0293 | −0.0030 | 94.3% / 95.5% |
| C-time target | 1.00 | 8,092 / 689 / 324 / 213 / 8 | 30 / 84 | 7 / 22 | 99.4% | 99.4% | 0.0651 / 0.0681 / 0.0681 | +0.0030 | 100% / 100% |

| Candidate | V1 leakage | V2 relation coverage | V3 ≥ 250 validation and test | V4 ≥ 80% coverage | Availability A |
|---|---|---|---|---|---|
| C-rand | pass | pass | pass | pass | 1.0450 |
| C-time | pass | pass | pass | pass | **1.0457** (selected by the pre-fixed rule; the margin is negligible) |

**Proxy calibration (`results/proxy_calibration.json`), which led to Amendment 1.** The original
proxy (pooled − own rule MRR) was about 0 exactly where federated ULTRA gained most. Study S shared
had a proxy of −0.019 against an actual Δ of +0.100 at n = 13, and +0.000 against +0.118 at
n = 41. Across 14 earlier cells, Spearman's ρ is 0.45.

Availability separated the cell families. It was 1.01 for Study S shared, the only family with
transfer; 0.73 for Mobile; 0.32 for ICS; and 0.20 for Study S disjoint.

## Smoke test and go/no-go

Smoke test: seed 20261209, validation only (`results/smoke/smoke_summary.json`).

| Run | Seconds per seed | Best validation MRR (round) |
|---|---|---|
| local, f = 0.10 | 87 | 0.0781 (4) |
| local, f = 1.00 | 239 | 0.1508 (16) |
| FedAvg, f = 0.10 | 286 | 0.0735 (2), descriptive |
| FedAvg, f = 1.00 | 397 | 0.1562 (7), descriptive |

| Go condition (DESIGN.md, Amendment 1) | Result |
|---|---|
| (i) V1–V4: leakage, coverage, evaluation size and coverage | pass |
| (ii) Availability A ≥ 0.9 | pass (1.046) |
| (iii) ULTRA need: local best validation MRR at f = 0.10 at least 0.03 below f = 1.00 | pass (gap 0.073) |
| (iv) Automated tests | pass (10/10) |
| Smoke-test execution and round-0 parity | pass |

**Recommendation: GO**, with the risks stated before training. The shared patterns are weak
(rule MRR about 0.07). At f = 0.10 only 13% of validation facts have a ≤3-hop evidence path in
the target graph. The one FedAvg smoke seed is not ahead of local at f = 0.10 on validation.
These lower the odds of a +0.03 gain, but none of them invalidates the test, so the registered
experiment is run as planned.

## Confirmatory results

The registered grid ran from 17:33:07Z to 18:56:52Z. It had 60 runs (2 arms × 3 fractions × 10
seeds), with no failures and no seed dropped, added or replaced. Before the analysis ran:
- every run file holds the 10 registered seeds and test results for all 314 test triples;
- every file carries the registered protocol label, client weights [1, 1], checkpoint and the
  manifest's data hashes;
- all 25 hashes in `results/preregistration_hashes.txt` still match.

The registered analysis (`src/analyze_confirmatory.py`) ran once, at 18:57:57Z. Its output is in
`logs/analysis.log` and `results/confirmatory/analysis_summary.json`.

**H1 is not supported.** At f = 0.10, FedAvg scores slightly *below* target-only fine-tuning, and
both 95% CIs lie entirely below zero. Neither secondary cell gains either. Every cell's upper 95%
bound is below +0.006, so a +0.03 gain is excluded at every fraction. The result is negative, not
inconclusive.

| Cell | Local test MRR | FedAvg test MRR | Δ (FedAvg − local) | Seed-level t-CI 95% | Hierarchical bootstrap CI 95% | Seeds with Δ > 0 | Full criterion |
|---|---|---|---|---|---|---|---|
| **f = 0.10 (H1)** | 0.0852 | 0.0805 | **−0.0047** | [−0.0071, −0.0024] | [−0.0090, −0.0009] | 1/10 | fail |
| f = 0.25 (secondary) | 0.0915 | 0.0872 | **−0.0043** | [−0.0067, −0.0019] | [−0.0105, +0.0015] | 1/10 | fail |
| f = 1.00 (secondary) | 0.1745 | 0.1730 | **−0.0015** | [−0.0086, +0.0056] | [−0.0092, +0.0057] | 5/10 | fail |

Per seed, paired by seed and test triple:

| Seed | Δ at f = 0.10 | Δ at f = 0.25 | Δ at f = 1.00 | Selected round, local / FedAvg (0.10, 0.25, 1.00) |
|---|---|---|---|---|
| 20261211 | −0.0023 | −0.0041 | +0.0001 | 3 / 18, 3 / 5, 10 / 8 |
| 20261212 | −0.0066 | −0.0086 | −0.0020 | 8 / 11, 6 / 3, 5 / 17 |
| 20261213 | −0.0096 | −0.00005 | −0.0196 | 3 / 17, 12 / 9, 3 / 16 |
| 20261214 | −0.0074 | −0.0050 | +0.0029 | 2 / 20, 20 / 14, 8 / 18 |
| 20261215 | −0.0063 | −0.0104 | +0.0029 | 12 / 15, 2 / 11, 4 / 3 |
| 20261216 | −0.0017 | +0.0006 | −0.0112 | 18 / 11, 2 / 17, 4 / 18 |
| 20261217 | −0.0013 | −0.0025 | −0.0021 | 6 / 12, 14 / 4, 3 / 3 |
| 20261218 | −0.0069 | −0.0051 | +0.0172 | 3 / 4, 17 / 15, 9 / 13 |
| 20261219 | +0.0003 | −0.0035 | −0.0074 | 8 / 16, 2 / 8, 11 / 13 |
| 20261220 | −0.0056 | −0.0044 | +0.0043 | 4 / 8, 8 / 17, 10 / 5 |

### Descriptive (not registered; computed from the saved runs, selects nothing)

Test ranking metrics, mean over seeds:

| f | Arm | MRR | Hits@1 | Hits@3 | Hits@10 |
|---|---|---|---|---|---|
| 0.10 | local | 0.0852 | 0.0417 | 0.0791 | 0.1455 |
| 0.10 | FedAvg | 0.0805 | 0.0366 | 0.0742 | 0.1411 |
| 0.25 | local | 0.0915 | 0.0350 | 0.0903 | 0.1782 |
| 0.25 | FedAvg | 0.0872 | 0.0318 | 0.0826 | 0.1895 |
| 1.00 | local | 0.1745 | 0.0917 | 0.1906 | 0.3311 |
| 1.00 | FedAvg | 0.1730 | 0.0898 | 0.1882 | 0.3314 |

Validation over training:

| f | Round-0 validation MRR (both arms) | Best validation MRR, local / FedAvg | Seeds where FedAvg's best is higher | Mean selected round, local / FedAvg | Rounds 1–20 where FedAvg's mean validation MRR is above local's | Seconds per seed, local / FedAvg |
|---|---|---|---|---|---|---|
| 0.10 | 0.0407 | 0.0773 / 0.0727 | 0/10 | 6.7 / 13.2 | 0/20 | 104 / 373 |
| 0.25 | 0.0786 | 0.0944 / 0.0917 | 2/10 | 8.6 / 10.3 | 1/20 | 116 / 349 |
| 1.00 | 0.1457 | 0.1539 / 0.1564 | 7/10 | 6.7 / 11.4 | 19/20 | 307 / 502 |

### Interpretation

**The test was fair and had room to succeed.**
- There is no leakage of any kind, both clients have the same schema, and the source's patterns
  cover the target's (A = 1.05).
- The target needed help. Its local test MRR is 0.085 at f = 0.10 against 0.175 with all its data,
  a gap of 0.089.

**Federation closed none of that gap.** At f = 0.10 the loss is small but consistent:
- 9 of 10 seeds are negative on test;
- FedAvg's best validation MRR is lower in 10 of 10 seeds;
- its mean validation MRR is below local's at all 20 rounds.

**Why, as far as the saved runs show.** These explanations fit the data, but no registered control
tested them.
1. *Most of the target's deficit is missing evidence, which weights cannot carry.* ULTRA predicts
   by message passing over the client's own graph.
   - Before any training, the same pretrained model scores 0.041 validation MRR at f = 0.10 and
     0.146 at f = 1.00.
   - Only 12.7% of f = 0.10 validation facts are joined by a ≤3-hop path in the target graph.
   - FedAvg shares weights, and raw triples stay local by design, so it cannot add the missing
     paths.
2. *The source has little that the target cannot already learn.* The clients share one schema and
   relation graph. The source's rules rank the target's facts only about as well as the target's
   own rules do (A ≈ 1.05). With little new to transfer, equal-weight averaging mainly dilutes the
   target's own updates. At f = 0.10, FedAvg peaks later (mean round 13.2 against 6.7) and lower.
3. *At full data the averaging is roughly neutral.* FedAvg's mean validation MRR is slightly higher
   at 19 of 20 rounds (+0.0025 at the best round). This does not carry over to test (Δ −0.0015,
   with a CI spanning 0).

Study S shared (Δ +0.100) is the contrast. There the source carried a strong pattern (rule MRR
0.38) that a data-limited target could not learn alone. Here the real Enterprise patterns are weak
(about 0.07), and the target's shortfall is in its graph, not in its weights.

**Scope.** The result covers:
- this partition (C-time);
- this federation protocol: FedAvg with equal weights, a fresh AdamW each round, and 20 rounds ×
  10 steps;
- this checkpoint.

It does not test other federation methods, such as personalised fine-tuning after FedAvg. Those
would be new hypotheses, each needing its own pre-registration. As the spec requires, the data
were not re-cut after this result.

**Answer to the study question.** On real Enterprise ATT&CK data, under the registered protocol,
federated ULTRA gives a data-limited cybersecurity client no meaningful gain. At 10% of its data
the client does slightly worse with federation than training alone.

## Reproduce

```bash
cd experiments/enterprise_split/src
python calibrate_proxy.py                                              # proxy vs earlier FedAvg results
python build_split.py --candidates C-rand C-time --output-root ../data/candidates
python viability.py --root ../data/candidates                          # Phase 2 checks + selection
python build_split.py --candidates C-time --output-root ../data/v1     # versioned splits + manifest
python test_pipeline.py                                                # leakage, coverage and federation tests
cd .. && bash run_confirmatory.sh --smoke                              # validation-only smoke test (seed 20261209)
cd src && python analyze_confirmatory.py --smoke                       # smoke-test gates
cd .. && bash run_confirmatory.sh                                      # registered grid: 2 arms x 3 fractions x 10 seeds
cd src && python analyze_confirmatory.py                               # registered analysis
```

## Files (all new; nothing outside this folder was modified)

- `DESIGN.md`, `PREREGISTRATION.md`, `README.md`, `run_confirmatory.sh`
- `src/env.py`: paths. The reused code is imported unmodified from `fedultra_fresh/src` and
  `transfer_conditions/src`; its hashes are in `results/preregistration_hashes.txt`.
- `src/build_split.py`, `src/viability.py`, `src/calibrate_proxy.py`, `src/test_pipeline.py`,
  `src/run_arm.py`, `src/analyze_confirmatory.py`
- `data/candidates/{C-rand,C-time}/`: candidate builds.
- `data/v1/C-time/`: the versioned splits, with `manifest.json` (file hashes, seeds, bundle hash,
  actor lists).
- `results/`: `design_hash.txt`, `proxy_calibration.json`, `viability.json`,
  `preregistration_hashes.txt`, `README.md`, `smoke/`
- `results/confirmatory/`: one file per arm and fraction (`{local,fedavg}_f{0.10,0.25,1.00}.json`),
  paired comparisons and bootstraps (`f*_comparison.json`, `f*_bootstrap.json`) and
  `analysis_summary.json`.
- `logs/`: one log per script and job, plus `confirmatory.log` (the grid) and `analysis.log`
  (the registered analysis).
