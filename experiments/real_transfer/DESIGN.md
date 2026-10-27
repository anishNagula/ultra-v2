# Real transfer: design of the solution search

The diagnosis is in [`DIAGNOSIS.md`](DIAGNOSIS.md), the registered plan in
[`PREREGISTRATION.md`](PREREGISTRATION.md), and the run log in [`results/README.md`](results/README.md).

## Fixed constraints

- **Data.** Real data only: the registered C-time split of Enterprise ATT&CK v19.2
  (`../enterprise_split/data/v1/C-time`). It is read and never written; its hashes are checked in every
  run file. No synthetic edges, fabricated labels or planted rules.
- **Privacy.** Raw source facts never leave the source client. Only model weights cross, once, from
  source to target.
- **Target stage.** It matches the registered local arm exactly
  (`transfer_conditions/src/conditions_runner.py::run_local`):
  - checkpoint `ckpts/ultra_4g.pth`;
  - AdamW lr 5e-4, weight decay 0;
  - 20 rounds × 10 steps, batch 8, 64 negatives;
  - the same sampling stream;
  - round chosen on target validation (round 0 = the starting weights);
  - test scored once.
- **Evaluation.** Filtered MRR over all target entities, both directions, hidden facts in the filters
  (`fedavg_runner.evaluate`, `transfer_runner.evaluate_with_query_mrr`), unchanged.
- **Criterion.** Unchanged: mean paired Δ ≥ +0.03, with the seed-level 95% t-CI and the hierarchical
  paired-bootstrap 95% CI both entirely above 0.
- **Selection.** Methods, hyperparameters and checkpoints are chosen on validation only, before any test
  metric exists.

## What the diagnosis implies for the search

1. **E1 (ceiling).** At f0.10, local fine-tuning is already at the level every training route reaches
   on the target's graph. No change to ULTRA's weights is expected to add +0.03 over local. The
   preregistration still tests it, as the primary hypothesis, so this expectation can be refuted.
2. **E2 (regime).** The registered FedAvg was harmful because the source trained with a complete graph
   as context, while the target predicts from a sparse one. Matching the source's training regime to
   the target's should remove the harm. On validation it also gives large **zero-shot** transfer:
   without a single target update, source-only training in the matched regime reaches about local's
   fine-tuned level.
3. **The baseline.** The question the regime result can answer is the project's frozen one
   (`../../CYBER_TRANSFER_PROTOCOL.md`): does source training transfer to an unseen cybersecurity graph,
   compared with the unchanged pretrained model, with no target gradient updates? It is
   preregistered as a **secondary** question with its own PASS/FAIL. It does not replace the primary
   hypothesis against local.

## Candidates

| | Candidate | Hypothesis it tests | Allowed? | Decision |
|---|---|---|---|---|
| A | **sparse_source**: source-only training on sparse per-head views of its own graph, then (i) zero-shot on the target and (ii) the registered local fine-tuning from those weights | E2, regime matching | yes: weights only, real facts only | **chosen** |
| B | **dense_source**: A with the complete source graph | control for A: only the regime differs | yes | **control** |
| C | **fedavg_sparse**: the registered FedAvg with sparse-view source updates | E2 for the registered method itself | yes | diagnostic only (validation) |
| D | Entity-level transfer: per-technique priors or embeddings shared from the source | E5 | shares per-entity source statistics; not ULTRA's entity-agnostic transfer | rejected; its ceiling is measured by `popularity.py` (+0.009 at f0.10) |
| E | Adding source facts, or edges derived from them, to the target graph | E1 | no: raw facts would leave the source, and derived edges are synthetic | rejected |
| F | Tuning FedAvg's learning rate, rounds or client weights | none | yes, but no mechanism to test | rejected: hyperparameter hunting, bounded by E1 |
| G | A larger target budget | none | no: changes the registered target budget | rejected |

**Why A rather than C as the candidate.**
- A produces a zero-shot model, which the secondary question needs; C never does.
- A shares weights once rather than every round.
- A's target stage is the registered local arm unchanged, so A − local isolates the starting weights.
- C's validation result (`DIAGNOSIS.md`, E2) answers the mechanism question for the registered method,
  but C is not carried into the confirmatory study.

## Method A in detail (`src/sparse.py::train_source`, `src/run_method.py`)

1. **Source validation.** A fixed 10% of the source's actor `uses` facts (360 facts, key
   `real-transfer source validation v1`), the same for every seed. They are never trained on and never
   enter a message graph. They are scored every round in the regime trained for: on one fixed per-head
   view of the remaining facts (f < 1), or on all of them (f = 1).
2. **Views.** Each round r draws a fresh per-head view of the remaining source facts with the
   target's own construction rule (`enterprise_split/src/build_split.py::scarce`). Per head entity,
   it keeps max(1, round(f · n_h)) facts after a shuffle keyed by (seed, r, head).
   - The kept facts, with inverses, are the message graph.
   - The held-out actor `uses` facts are the positives.
   - Negatives are filtered against every source fact.
   - f equals the target's fraction: 0.10 or 0.25. At f = 1.00 the regime-matched source is the
     complete graph, which is arm B.
3. **Training.** 30 rounds × 10 steps, batch 8, 64 negatives, AdamW lr 5e-4, weight decay 0: the
   registered optimiser settings. The loss is the registered one, and `train_steps` equals
   `train_updates` bit for bit (`src/test_method.py`).
4. **Checkpoint.** Chosen by the rule fixed after the pilot (see below), without test data.
5. **Target stage.** The registered local arm from those weights. The record scores two weight sets
   on test, each once:
   - the validation-selected round (fine-tuned);
   - the starting weights (zero-shot).

**Assumption stated.** Regime matching needs one scalar of target metadata, the per-head retention
fraction f. Here it is the split's design value. In deployment it would be estimated from the target's
observed degrees, which the target can share without sharing facts.

## Pilot and decisions (validation only)

**Setup.**
- Smoke test (`run_smoke.sh`, `results/smoke/`): every arm ran end to end. The local arm reproduces
  registered seed 20261211's validation curve bit for bit (21 values) and its selected round.
- Pilot (`run_pilot.sh`, `results/pilot/`): seeds 20261401–03, `--skip-test`, every arm at every
  fraction. It used the source checkpoint rule first planned: the round with the best source
  validation (rule a).
- Probe (`source_select`, `results/diagnostics/source_select_f*.json`): it reruns the pilot's source
  stage of the same seeds, verified identical, and scores target validation after every source round.
  This compares checkpoint rules on one trajectory.

### Decision 1: the source checkpoint is the final round

Zero-shot target validation of the source weights under each rule, per seed:

| Fraction | Source-validation argmax (rule a) | Final round 30 (rule b) | Target-validation argmax | Spearman ρ, source vs target, rounds 1–30 |
|---|---|---|---|---|
| f0.10 | 0.0733 (r1), 0.0740 (r7), 0.0748 (r7) | 0.0792, 0.0811, 0.0829 | 0.0834, 0.0859, 0.0848 | −0.39, −0.80, −0.57 |
| f0.25 | 0.0961 (r29), 0.0963 (r23), 0.0950 (r24) | 0.0963, 0.0900, 0.0961 | 0.0971, 0.0991, 0.0981 | +0.58, +0.47, +0.26 |

- At f0.10, source validation peaks within 7 rounds and then falls, while target validation rises to
  its plateau (last-10 mean 0.081–0.084). The two move in opposite directions (ρ < 0 on every seed).
  Rule a stops before the plateau on every seed; rule b beats it by +0.006, +0.007 and +0.008.
- At f0.25 both rules pick weights from the plateau (last-10 mean 0.094–0.096). Rule a is ahead
  by 0.0017 on average, all of it from one seed's final-round dip (0.0900).
- Target-validation selection would use target labels. That would weaken the zero-shot question, and
  part of its lead is selection on noise.

**Decision.** Every source arm keeps its final source round. Source validation is still held out,
scored and recorded. This choice uses validation only and was fixed before any test metric existed.
The rule a pilot runs are kept and reported. Their source trajectories are identical to rule b's;
only the starting weights handed to the target differ.

### Decision 2: the primary candidate is sparse_source

Fine-tuned validation (the target stage's selected round), rule a starting weights, paired seeds:

| Fraction | local | sparse_source | Δ | dense_source | Δ |
|---|---|---|---|---|---|
| f0.10 | 0.0791, 0.0790, 0.0736 | 0.0813, 0.0823, 0.0788 | +0.0022, +0.0033, +0.0052 | 0.0847, 0.0737, 0.0793 | +0.0056, −0.0053, +0.0057 |
| f0.25 | 0.0953, 0.0896, 0.0931 | 0.0964, 0.0963, 0.0950 | +0.0011, +0.0067, +0.0019 | 0.0947, 0.0940, 0.0945 | −0.0006, +0.0044, +0.0014 |
| f1.00 | 0.1533, 0.1555, 0.1563 | (regime-matched source is dense) | | 0.1650, 0.1636, 0.1727 | +0.0117, +0.0081, +0.0164 |

- sparse_source is the candidate the diagnosis motivates (E2), and it beats local on every pilot
  seed at f0.10. dense_source's differences at f0.10 change sign.
- At f1.00, where the dense source is the regime-matched one, it beats local on every seed.
- Every gain is well below +0.03, as E1 predicts for the sparse fractions.

**Zero-shot (rule b) against the pretrained model** (0.0407 at f0.10, 0.0786 at f0.25):
- f0.10: +0.0385, +0.0404, +0.0422 (mean +0.040).
- f0.25: +0.0177, +0.0114, +0.0175 (mean +0.016).

Dense-source zero-shot (rule a, source rounds 27, 19, 23) against pretrained:
- f0.10: 0.0434, 0.0439, 0.0389, i.e. +0.003, +0.003 and −0.002;
- f0.25: 0.0761, 0.0870, 0.0727, i.e. −0.003, +0.008 and −0.006;
- f1.00 (regime-matched): 0.1631, 0.1636, 0.1727 against 0.1457, i.e. +0.017, +0.018 and +0.027.

At the sparse fractions the dense regime is not helpful zero-shot, but not clearly harmful either
(`DIAGNOSIS.md`, E2). At the dense fraction it transfers.

### Unchanged

Source rounds (30), view fractions (matched to the target), optimiser settings, the target stage,
the evaluation and the criterion. No setting was tuned beyond the two decisions above.
