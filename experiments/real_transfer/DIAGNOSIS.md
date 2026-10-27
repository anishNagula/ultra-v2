# Diagnosis: why ULTRA transfer failed on the real C-time split

All new numbers here are target **validation** MRR. The only test numbers are the registered C-time
results, already published in `../enterprise_split/README.md`. No new test metric was computed for this
diagnosis. Every number is recomputed from the saved run files by `src/summarize_diagnostics.py`,
which writes `results/diagnostics/summary.json`.

## The result to explain

Registered C-time study (`../enterprise_split`): FedAvg (source + target, weights [1, 1]) against
local target fine-tuning, 10 seeds, the same 200 target updates.

| Fraction | Local test | FedAvg test | Δ | t-CI | Local best validation | FedAvg best validation |
|---|---|---|---|---|---|---|
| f0.10 (primary) | 0.0852 | 0.0805 | −0.0047 | [−0.0071, −0.0024] | 0.0773 | 0.0727 |
| f0.25 | 0.0915 | 0.0872 | −0.0043 | [−0.0067, −0.0019] | 0.0944 | 0.0917 |
| f1.00 | 0.1745 | 0.1730 | −0.0015 | includes 0 | 0.1539 | 0.1564 |

Two things need explaining. Federation **hurt** at f0.10 and f0.25. And a **+0.03 gain over local**
was never in reach.

## Evidence collected (validation only)

| Probe | What it isolates | Seeds | File |
|---|---|---|---|
| `reachability` | How far answers are from the query entity in the target graph; pretrained and popularity MRR by hop band | none (deterministic) | `results/diagnostics/reachability.json` |
| `oracle_labels` | Same sparse graph, 6.3× (f0.10) / 3.4× (f0.25) more training positives: the target's hidden facts. Not a method. | 20261401–02 | `oracle_labels*.json` |
| `dense_weights` | Weights trained on the target's own complete graph (f1.00), scored on its sparse f0.10 graph | 20261401–02 | `dense_weights.json` |
| `source_sparse` / `source_dense` | Source-only training on actor `uses` facts, scored zero-shot on the target. Only the source's training regime differs: sparse per-head views vs the complete graph. | 20261401–02 | `source_sparse*.json`, `source_dense.json` |
| `fedavg_replay` | The registered FedAvg, rebuilt; must equal the registered curve | 20261211 | `fedavg_replay.json` |
| `fedavg_sparse` | The registered FedAvg with one change: the source's 10 steps per round use a fresh per-head view of its graph | 20261211–13 (registered seeds, paired) | `fedavg_sparse.json` |
| `popularity` | Rankers that count how often each entity is an answer, from target and/or source facts | none | `popularity.json` |

"Best" below is the maximum over 21–41 noisy evaluations, so it overstates what is reliably reached.
"Plateau" is the mean of the last 10 rounds.

## Explanations

Each explanation is rated: **supported** (predictions checked on validation, including by intervention
where stated), **plausible** (consistent but not isolated), or **not supported**. None is proven: these
are validation results with 2–3 seeds. The confirmatory study (`PREREGISTRATION.md`) tests the predictions
marked *(tested)* on the untouched test set.

### E1. The sparse target graph caps what any weights can reach

**Claim.** At f0.10 and f0.25, local fine-tuning already sits about where every training route plateaus
on the same graph. ULTRA has no entity embeddings, so its weights cannot add evidence the target graph
does not contain. A +0.03 gain over local therefore cannot come from better weights.

**Supporting evidence.**
- *Reachability (f0.10).* Only 0.6% of validation answers are within 2 hops of the query entity and 12.1%
  at 3 hops. The rest are 4–6 hops away (43.0%) or farther or disconnected (44.3%). Pretrained MRR is
  0.136 at 3 hops but 0.021 and 0.025 in the far bands. At f0.25, 65.3% are within 3 hops.
- *More labels, same graph.* `oracle_labels` adds every hidden target fact as a positive (also filtered
  as a known fact). This gives 6.3× more positives at f0.10, yet the plateau is 0.077, best 0.083 / 0.081.
  At f0.25 (3.4× more positives) the plateau is 0.091, best 0.0945 / 0.0954. Local's registered mean
  best is 0.0773 and 0.0944.
- *A different route, same level.* `source_sparse` uses no target label at all. Its plateau is
  0.078 / 0.081 at f0.10 (best 0.084) and 0.095 at f0.25 (best 0.099 / 0.096).
- *Entity-level knowledge does not close the gap either.* Even per-entity counts, which weights cannot
  carry, are no route. Source and target counts pooled give 0.0808 at f0.10, +0.009 over target counts
  alone (0.0722).

**Counterevidence and limits.**
- This is a probe, not a bound. Each route used ULTRA, AdamW at 5e-4 and the same batch size. A different
  architecture (entity parameters, deeper message passing) or objective could exceed it.
- E1 does not explain f1.00. There 99% of answers are within 3 hops, local improves on pretrained, and
  FedAvg still did not help (−0.0015).

**Prediction *(tested)*.** On test, no fine-tuned candidate beats local by +0.03 at f0.10. The upper
95% t-CI bound of the paired difference is below +0.03.

**Status: supported** as the reason +0.03 over local is out of reach at f0.10 and f0.25 for any
weight-transfer method.

### E2. The FedAvg source trained in the wrong regime

**Claim.** Updates computed with a complete graph as context fit a model that relies on evidence the
sparse target graph lacks. They are at best unhelpful for the sparse target. Averaging them into the
global model every round dilutes, or reverses, the target's own progress. That explains the negative
sign at f0.10 and f0.25.

**Supporting evidence.**
- *Same data, only density changes.* `dense_weights` trains on the target's own f1.00 graph. That raises
  f1.00 validation from 0.146 to a best of 0.153 / 0.156. The same weights on the f0.10 graph fall from
  0.041 to a plateau of 0.019 / 0.024, with lows of 0.007 / 0.012.
- *Same source, only the regime changes.* `source_dense` and `source_sparse` share source facts, task,
  budget and seeds. Zero-shot on f0.10, `source_dense` stays at 0.011–0.037 after round 1, below the
  pretrained 0.041 in every round. `source_sparse` stays at 0.072–0.084, above it in every round.
- *The registered curves match the pattern.* FedAvg's best validation is below local where the target
  is sparse: f0.10 0.0727 vs 0.0773, f0.25 0.0917 vs 0.0944. It is above local where the target is as
  dense as the source: f1.00 0.1564 vs 0.1539.
- *Intervention.* `fedavg_replay` reproduces the registered FedAvg validation curve bit for bit (seed
  20261211, rounds 0–3). `fedavg_sparse` changes only the source's message graph, to a fresh per-head
  view each round, with that round's held-out facts as positives. The registered sampling stream,
  relations, client weights and target updates are unchanged. On the paired seeds 20261211–13:
  - best validation 0.0800, against 0.0731 (registered FedAvg) and 0.0767 (local), higher than both on
    every seed;
  - mean over rounds 1–20 0.0740, against 0.0585 and 0.0643.

**Counterevidence and limits.**
- Three seeds, validation only.
- "Regime" bundles two differences the probes do not separate: the message graph's density, and how
  positives relate to it (held out of the view vs removed one edge at a time).
- Matching is not sufficient for a gain. At f1.00 the regimes already match, yet the registered
  FedAvg test Δ was −0.0015, because the target there has abundant labels of its own.
- The registered harm (−0.005) is far smaller than `source_dense`'s zero-shot harm, because the target's
  own updates each round undo most of it.
- *Harmful, or only unhelpful?* The pilot's dense source stage holds out 10% of the source's actor
  `uses` facts. Its weights, from source rounds 27, 19 and 23, score 0.0434, 0.0439 and 0.0389 zero-shot
  on f0.10, against 0.0407 pretrained. At f0.25 the first seed scores 0.0761 against 0.0786. So dense-regime
  weights are reliably not helpful, but whether they are worse than pretrained depends on the run.
  `source_dense` (no holdout) was below pretrained in every round. The pilot runs were not.
- *Fine-tuning recovers.* Starting the target stage from dense-source weights gives 0.0847, 0.0737 and
  0.0793 against local's 0.0791, 0.0790 and 0.0736 (f0.10 pilot). The harm, where there is any, comes
  from averaging the dense updates in every round, as FedAvg does, not from a dense starting point.

**Predictions *(tested)*.** On test:
- M2 (f0.10) and M4 (f0.25): sparse-source zero-shot − dense-source zero-shot > 0. This is the regime
  effect itself.
- M1 (f0.10): dense-source zero-shot − pretrained < +0.01, i.e. not helpful. `source_dense`
  predicted a negative difference, the pilot about zero, so the sign is left open.
- M3 (f0.10): dense-source fine-tuned − local within ±0.01.
- Z1 (f0.10): sparse-source zero-shot − pretrained > 0.

**Status: supported**, including by intervention, as the cause of the negative FedAvg effect at f0.10.
Not yet tested on test data. Whether dense updates are harmful in themselves, or only unhelpful and
then diluting, is not settled.

### E3. The FedAvg source learned the wrong task mix

**Claim.** The registered source trained on all 11,149 of its facts, of which only 3,604 are actor
`uses`. Its updates went to tasks the target never evaluates.

**Evidence.**
- Against, as a necessary cause: `source_dense` trained on actor `uses` facts only and still hurt.
- Against, as a sufficient cause: `fedavg_sparse` kept every relation as positives and helped.

**Status: not supported** as the cause of the harm.

### E4. FedAvg averaging slows target progress

**Claim.** Averaging two client updates halves each round's effective target step, so FedAvg peaks later
and lower.

**Evidence.**
- For: the registered FedAvg selected later rounds (mean 13.2 vs 6.7 at f0.10).
- Against: `fedavg_sparse` averages in exactly the same way and still beats local's best.

**Status: plausible** as an influence on the curve's shape. **Not supported** as the cause of the
deficit.

### E5. What the source knows about specific shared entities cannot cross as weights

**Claim.** The clients share 344 techniques and 79 software as entities (not as facts). The source's
evidence about them, such as how often actors use a technique, is entity-level knowledge, which ULTRA's
entity-agnostic weights cannot carry.

**Evidence.**
- For: 93% of target validation answers appear in source actor `uses` facts, against 68% in the target's
  f0.10 graph. Source counts rank them better than target counts (0.0780 vs 0.0722).
- Against, as a route to +0.03: the pooled gain is +0.009, under a third of the margin.

**Status: supported** that the knowledge exists, but it is small. A method using it would share
per-entity source statistics, which is outside this study's weights-only design (`DESIGN.md`).

## Summary

| | Status | What would change it |
|---|---|---|
| E1 graph-limited ceiling | supported (validation) | a weight-transfer method beating local by ≥ 0.03 on validation |
| E2 regime mismatch | supported, including an intervention (validation) | M2, M4 or Z1 failing on test |
| E3 task mix | not supported | — |
| E4 averaging | plausible contributor, not the cause | — |
| E5 entity-level knowledge | real but small | — |

**Measured directly:**
- the regime intervention reverses FedAvg's deficit on all three paired seeds;
- dense-regime weights do not help on the sparse graph. They are clearly worse than pretrained in
  two of five f0.10 runs (the two without a holdout) and about level in the other three;
- three different training routes plateau at about the same level on each sparse graph.

**Not established:**
- that the plateau is the true maximum for every possible method;
- *why* dense-regime weights hurt inside the model; one candidate is reliance on multi-hop relational
  evidence that sparse graphs lack, but no probe isolates it;
- that any of this generalises beyond this split.
