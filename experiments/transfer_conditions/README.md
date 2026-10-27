# Transfer conditions: when does federated ULTRA transfer multi-hop attack patterns?

The fresh FedULTRA experiments found no federated gain on Enterprise↔ICS at full data
(FedAvg − local = −0.004). The audit traced this to the data, not the model:
- ICS has enough data of its own.
- Its dominant relation (`targets`) has no Enterprise counterpart.
- The clients' shared updates are mostly score calibration.

This study changes the data to create the conditions transfer needs, then tests whether
federated ULTRA delivers it. The design, hypotheses and pass/fail rules were fixed in
[`PREREGISTRATION.md`](PREREGISTRATION.md) before any model was trained on this data.
Hashes are in `results/preregistration_hashes.txt`.

| Study | Data | What it can show |
|---|---|---|
| **S** | Synthetic ATT&CK-like clients with a planted multi-hop pattern; the target is short of `uses` facts | *When* federated ULTRA transfers patterns between clients that share no entities and no relation IDs. Not evidence about ATT&CK. |
| **R** | Real ATT&CK facts only: Enterprise source, Mobile (confirmatory) or ICS (validation-only development) as a data-poor target | Whether a data-poor real ATT&CK client gains from federating with Enterprise. |
| Exploratory | Real STIX tactic and platform edges added to each training graph | Whether restoring the dropped structure gives the clients anything to exchange (model-agnostic rule test, validation only). |

## Results

The registered analysis (`src/analyze.py`) was run once, after all 24 jobs had finished.
Δ is FedAvg − local target test MRR, paired by seed. A cell meets the full success criterion
if mean Δ ≥ +0.03 and both 95% CIs lie above 0. The two CIs are the seed-level t-CI and the
hierarchical paired bootstrap over seeds and test triples.

There were no protocol deviations. The decision rule for H2 and H3 was written into
[`results/README.md`](results/README.md) before the analysis ran. All numbers are in
`results/analysis_summary.json`.

| Hypothesis | Outcome |
|---|---|
| **H1** (primary, synthetic): shared source, n = 13, meets the full success criterion | **Supported.** Δ = +0.100, CIs [+0.080, +0.119] and [+0.078, +0.122] |
| **H2**: the shared-source gain is larger at n = 13 than at n = 408 | **Supported.** Difference +0.091 |
| **H3**: shared beats disjoint at every n, and Δ_disjoint(408) ≤ 0 | **Supported.** Shared − disjoint is +0.10 to +0.12 at every n. Δ_disjoint(408) = −0.094: negative transfer shown |
| **H4** (primary, real data): Mobile at 25% meets the full success criterion | **Not supported.** Δ = +0.005, CIs include 0 |

### Study S: synthetic clients (not evidence about ATT&CK)

Target test MRR, 5 seeds per cell. The *shared* source follows the target's toolkit pattern
and the *disjoint* source follows the campaign pattern. Both sources share the same local runs.

| Source | Target facts n | Local | FedAvg | Δ | Seed t-CI | Bootstrap CI | Seeds with Δ > 0 | Full success |
|---|---|---|---|---|---|---|---|---|
| shared | 13 (H1) | 0.308 | 0.408 | **+0.100** | [+0.080, +0.119] | [+0.078, +0.122] | 5/5 | **pass** |
| shared | 41 | 0.292 | 0.411 | +0.118 | [+0.070, +0.167] | [+0.084, +0.159] | 5/5 | pass |
| shared | 136 | 0.395 | 0.419 | +0.024 | [−0.008, +0.056] | [+0.0005, +0.047] | 3/5 | fail |
| shared | 408 | 0.474 | 0.482 | +0.008 | [−0.017, +0.034] | [−0.014, +0.029] | 4/5 | fail |
| disjoint | 13 | 0.308 | 0.304 | −0.004 | [−0.012, +0.003] | [−0.015, +0.006] | 1/5 | fail |
| disjoint | 41 | 0.292 | 0.295 | +0.003 | [−0.012, +0.018] | [−0.014, +0.021] | 3/5 | fail |
| disjoint | 136 | 0.395 | 0.312 | −0.082 | [−0.152, −0.013] | [−0.132, −0.039] | 0/5 | fail |
| disjoint | 408 | 0.474 | 0.381 | −0.094 | [−0.123, −0.064] | [−0.120, −0.067] | 0/5 | fail |

| Registered contrast | Mean | Seed t-CI | Bootstrap CI |
|---|---|---|---|
| H2: Δ_shared(13) − Δ_shared(408) | +0.091 | [+0.063, +0.120] | [+0.063, +0.123] |
| H3(a), n = 13: Δ_shared − Δ_disjoint | +0.104 | [+0.078, +0.129] | [+0.082, +0.126] |
| H3(a), n = 41: Δ_shared − Δ_disjoint | +0.115 | [+0.066, +0.164] | [+0.083, +0.152] |
| H3(a), n = 136: Δ_shared − Δ_disjoint | +0.106 | [+0.058, +0.155] | [+0.075, +0.143] |
| H3(a), n = 408: Δ_shared − Δ_disjoint | +0.102 | [+0.068, +0.136] | [+0.076, +0.129] |
| H3(b): Δ_disjoint(408) ≤ 0 | −0.094 | [−0.123, −0.064] | [−0.120, −0.067] |

### Study R: real ATT&CK facts, Enterprise as the source

Mobile, the confirmatory target: test MRR, 10 seeds.

| Mobile training facts | Local | FedAvg | Δ | Seed t-CI | Bootstrap CI | Seeds with Δ > 0 | Full success |
|---|---|---|---|---|---|---|---|
| 10% (149), secondary | 0.166 | 0.168 | +0.001 | [−0.003, +0.006] | [−0.010, +0.013] | 6/10 | fail |
| 25% (374), H4 | 0.210 | 0.216 | **+0.005** | [−0.002, +0.013] | [−0.007, +0.019] | 7/10 | **fail** |
| 100% (1,495), secondary | 0.304 | 0.304 | +0.001 | [−0.008, +0.010] | [−0.009, +0.011] | 6/10 | fail |

ICS, the development target: validation MRR only, 3 seeds, descriptive. Validation MRR at the
validation-selected round is optimistic for both arms.

| ICS training facts | Pretrained (round 0) | Local | FedAvg | Δ |
|---|---|---|---|---|
| 10% (134) | 0.235 | 0.279 | 0.274 | −0.004 |
| 25% (334) | 0.357 | 0.357 | 0.370 | +0.013 |
| 100% (1,333) | 0.459 | 0.498 | 0.487 | −0.011 |

### Reading the runs (descriptive)

- **Pairing.** Round 0 (the pretrained model) scores identically in both arms for every pair
  and seed, so the arms are correctly paired.
- **Synthetic, n ≤ 41.**
  - Local fine-tuning cannot improve on the pretrained model; it selects round 0 or 1 in every seed.
  - FedAvg with the shared source reaches its best validation score at rounds 12–20.
  - With 13 facts plus federation, the target scores 0.408, above local training on 136 facts (0.395).
- **Synthetic, n ≥ 136.**
  - The target learns the pattern from its own data (local selects rounds 12–20), so the shared
    source adds little.
  - With the disjoint source, validation selection mostly stops FedAvg at rounds 1–9, and it
    still ends 0.08–0.09 below local.
- **Mobile.**
  - Scarcity hurts: local test MRR is 0.166 at 10% and 0.304 at 100%.
  - Local training improves on the pretrained model at 10% and 25% (validation 0.095 → 0.144 and
    0.138 → 0.187), but FedAvg adds nothing on top.
  - The upper ends of the CIs bound the gain from Enterprise at about +0.013 (10%) and +0.019 (25%).

### Exploratory: restoring real STIX structure (not registered, validation only)

`src/explore_enrichment.py` takes edges from the v19.2 bundles and adds them to each training graph:
- technique→tactic edges, for every client;
- technique→platform edges, for Enterprise and Mobile only (ICS lists no platforms).

It then mines path rules on the target alone, on Enterprise alone, and on both pooled, and
reports validation rule MRR.

| Target | Graph | Own | Enterprise | Pooled | Pooled − own |
|---|---|---|---|---|---|
| ICS | as built | 0.3159 | 0.1014 | 0.3140 | −0.0019 |
| ICS | enriched | 0.3297 | 0.1160 | 0.3277 | −0.0020 |
| ICS, 25% | as built | 0.2488 | 0.0842 | 0.2612 | +0.0124 |
| ICS, 25% | enriched | 0.2638 | 0.0833 | 0.2580 | −0.0058 |
| Mobile | as built | 0.1408 | 0.1024 | 0.1024 | −0.0385 |
| Mobile | enriched | 0.1651 | 0.1236 | 0.1236 | −0.0415 |
| Mobile, 25% | as built | 0.0848 | 0.0859 | 0.0859 | +0.0010 |
| Mobile, 25% | enriched | 0.1258 | 0.1199 | 0.1199 | −0.0059 |

The added structure raises each client's own rule MRR. On the enriched graphs, pooling with
Enterprise never beats the target's own statistics. The added structure makes each client more
self-sufficient without giving the clients anything to exchange.

### What it means

- **The method works when the data allows it.**
  - Federated ULTRA transfers a multi-hop attack pattern between clients that share no entities:
    +0.10 MRR for a target with 13 facts, over three times the +0.03 bar.
  - The gain fades as the target gets its own data.
  - It turns into a loss of about 0.09 when the source's pattern differs.
- **The real ATT&CK domains do not meet the conditions.**
  - Making Mobile data-poor creates the need, but Enterprise's updates give Mobile at most about
    +0.01 to +0.02 MRR. The ICS development runs look the same.
  - The original full-data Enterprise↔ICS result (−0.004) stands.

## Reproduce

```bash
cd experiments/transfer_conditions/src
python synth_attack_kg.py                  # Study S data (seed 20261101) -> data/synthetic
python build_real_targets.py               # Study R data (Mobile split + 10%/25%/100% targets) -> data/real
python relation_roles.py --clients ../data/synthetic/target_n13 ../data/synthetic/source_toolkit   # sanity check S3
cd .. && bash run_all.sh                   # all 24 registered jobs, 4 in parallel (~3 h on a 10-core laptop)
cd src && python analyze.py                # pairing, bootstrap, registered contrasts -> results/analysis_summary.json
python explore_enrichment.py               # exploratory enrichment check (validation only)
```

`run_all.sh --dry-run` lists the jobs. Re-running `run_all.sh` resumes: seeds already saved in an output file are skipped.

## Files

- `src/common.py` loads a client folder. It is like `load_client`, plus `hidden.txt`: true facts the
  client does not observe, which are filtered at evaluation but never trained on.
- `src/conditions_runner.py` runs the local or FedAvg arm (equal client weights, best round by target validation).
- `src/compare_arms.py` pairs the arms by seed, in the input format of `fedultra_fresh/src/bootstrap_cyber_transfer.py`.
- `src/analyze.py` runs the registered analysis; `src/explore_enrichment.py` runs the exploratory enrichment check.
- `results/README.md`: deviations log and the H2/H3 decision rule.
- `results/analysis_summary.json`: every registered cell and contrast.
- `results/synthetic`, `results/real`: raw runs, comparisons and bootstraps.
- `logs/`: per-job logs, plus `analyze.log` and `explore_enrichment.log`.
