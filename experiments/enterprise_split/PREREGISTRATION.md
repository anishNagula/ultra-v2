# Enterprise split: pre-registration

Registered before any model was trained or evaluated on this data. Hashes are in
`results/preregistration_hashes.txt`.

The design and its Amendment 1 are in [`DESIGN.md`](DESIGN.md). The viability results are in
`results/viability.json`. Nothing in `experiments/fedultra_fresh/`, `experiments/transfer_conditions/`
or `ultra/` is modified.

## Question

Can federated ULTRA give a data-limited cybersecurity client a meaningful gain over training on
its own data? Both clients come from one real domain, Enterprise ATT&CK v19.2, split by threat-actor
portfolio. No fact is invented.

## Data (`data/v1/C-time`, hashes in its `manifest.json`)

C-time was selected by the DESIGN rule: availability A = 1.0457, against 1.0450 for C-rand. Both
candidates passed V1–V4. In C-time the source holds the 47 earliest-documented actor units and the
target the 113 later ones.

| Client | Train | Validation | Test | Hidden |
|---|---|---|---|---|
| Source | 11,149 | – | – | – |
| Target, f = 0.10 (primary) | 1,492 | 314 | 314 | 7,834 |
| Target, f = 0.25 | 2,760 | 314 | 314 | 6,566 |
| Target, f = 1.00 | 9,326 | 314 | 314 | 0 |

- The source is never evaluated.
- Target validation and test facts are target actor `uses` facts: actor → technique or software.
  They are identical at every f.
- The score is filtered MRR over all target entities, for both head and tail queries. A triple's
  MRR is the mean of its two reciprocal ranks.
- The filters include `hidden.txt` (true facts the client does not observe). Hidden facts never
  enter training or message passing.

**Checks passed before registration:**
- `test_pipeline.py`, 10/10, on `data/v1/C-time`:
  - leakage L1–L4, exact, inverse-equivalent and entity-pair: all zero;
  - every fact is in exactly one client and none is invented;
  - actor units are not split;
  - relation coverage, nested fractions, deterministic rebuild;
  - training-only message passing;
  - FedAvg exchanges only weights, and each client trains on its own graph.
- Relation roles align: both clients have identical relation graphs, and same-name pretrained
  relation cosine is 1.000.

**Residual semantic overlap.** The target actor WIRTE's description mentions the source actor
Molerats. 79 software and 344 techniques are used by actors in both clients. These are shared
entities, not shared facts.

**Known risk, stated before any training.** At f = 0.10 only 12.7% of target validation facts
are joined by a path of three hops or fewer in the target's training graph (65.3% at f = 0.25).
Path-rule MRR is low in both clients, 0.065–0.068 at full data. The planted Study S pattern reached
about 0.38.

## Arms

The two arms are identical except for federation.

- **Shared:**
  - Initial weights: `ckpts/ultra_4g.pth`.
  - Optimiser: AdamW, lr 5e-4, weight decay 0.
  - Budget: 20 rounds × 10 steps (200 target updates), batch 8, 64 negatives.
  - The target uses the same sampling stream in both arms.
  - Round selection: the round (0 = pretrained) with the best target validation MRR. Target test is
    scored once, for that round.
- **local:** target-only fine-tuning. AdamW state is kept across rounds.
- **fedavg:** target + source, equal client weights [1, 1].
  - Each round, each client starts from the global weights with a fresh AdamW and takes 10 steps
    on its own graph. The source samples from stream seed + 1.
  - The two weight sets are averaged. Raw triples never leave a client.
- **Runner:** `src/run_arm.py`. This is `transfer_conditions/src/conditions_runner.py` unchanged,
  except for the protocol label.

## Hypotheses

- **H1 (primary).** At f = 0.10, Δ = FedAvg − local target test MRR meets the **full success
  criterion**:
  - mean Δ ≥ +0.03;
  - the seed-level 95% t-CI lies entirely above 0;
  - the hierarchical paired bootstrap 95% CI lies entirely above 0.

  The bootstrap is `fedultra_fresh/src/bootstrap_cyber_transfer.py`: 20,000 samples, seed
  20261026, resampling seeds and test triples jointly. Deltas are paired by seed and test triple
  (`transfer_conditions/src/compare_arms.py`).
- **Secondary**, with the same statistics, labelled secondary. They cannot substitute for H1:
  f = 0.25 and f = 1.00.

## Seeds

- Confirmatory: 20261211–20261220 (10 per arm per fraction).
- Smoke test: 20261209 (validation only; not part of any result).

## Gates and stopping rules

1. **Smoke test** (validation only, `--skip-test`, seed 20261209): local and FedAvg at
   f = 0.10 and f = 1.00. It passes if:
   - (a) all four runs complete without error;
   - (b) round-0 validation MRR is identical in the local and FedAvg runs at each f;
   - (c) DESIGN go condition (iii) holds: local's best validation MRR at f = 0.10 is at least
     0.03 below its best at f = 1.00.

   If (a) or (b) fails, the code is fixed, a deviation is logged, and the smoke test is rerun.
   If (c) fails, the study stops (no-go) and no test metric is computed. FedAvg smoke-test numbers
   are reported but do not enter any decision.
2. **The confirmatory grid** (6 jobs: 2 arms × 3 fractions × 10 seeds) runs to completion, with
   no interim look at test metrics.
   - A crashed job is resumed with the same arguments; the runner skips seeds already saved.
   - No seed is dropped, added or replaced.
   - A cell that cannot complete for a technical reason is reported as missing.
3. After registration, nothing changes in the data, budget, hyperparameters or code, except bug
   fixes. Those are logged in `results/README.md`, with their time, before the affected results
   are read.
4. The analysis (`src/analyze_confirmatory.py`) runs once, after the grid finishes. Every cell is
   reported, whatever its outcome.
