# Frozen ULTRA cybersecurity transfer protocol (v2: unbridged target)

## Research question

**To what extent can a pretrained knowledge-graph foundation model transfer relational reasoning to a previously unseen cybersecurity knowledge graph?**

## Fixed experiment

- **Source:** the existing MITRE Enterprise ATT&CK train/validation/test split.
- **Target:** a deterministic sample of at most 20,000 CVEs from the existing CAPEC/CWE/NVD graph, including their CPE context and the CAPEC/CWE catalog relationships.
- **No cross-graph links:** do not add ATT&CK–CAPEC edges or infer identity links. The source and target entity IDs must be disjoint.
- **Target task:** predict held-out `(CVE, has_weakness, CWE)` facts.
- **Target message-passing graph:** target training triples only. All non-`has_weakness` triples are retained as training context. Only `has_weakness` triples are split into train/validation/test.
- **Comparison:** the unchanged pretrained ULTRA checkpoint vs. a copy fine-tuned on Enterprise source training data. Choose the source checkpoint using Enterprise validation MRR only. No target gradient updates. Target validation metrics are not scored.
- **Primary metric:** filtered target-test MRR with both head and tail corruption and all target graph entities as candidates.
- **Acceptance rule:** mean paired ΔMRR >= +0.03, with both the two-sided seed-level t 95% confidence interval and the hierarchical paired-bootstrap 95% confidence interval entirely above zero. This does not change the +0.03 threshold; it makes the uncertainty check explicit.
- **Seeds:** `20261027` through `20261036` (10 confirmatory runs).
- **Training budget:** 5 rounds × 10 source updates per seed, batch size 8, 64 negatives, learning rate 0.0005, weight decay 0.

## Why the earlier bridge attempt was rejected

The 36 ATT&CK objects carrying CAPEC references were not present in the active source split. They were legacy/deprecated ATT&CK objects, so adding their UUIDs as bridge nodes would either fail the source split membership check or require changing the source graph. We do not change the source split or fabricate a bridge to make the pipeline pass. This version instead tests unbridged cross-KG relational transfer, which directly matches the unseen-graph question.

## Deterministic target sampling

The full CAPEC/CWE/NVD graph is large enough that exhaustive all-entity ranking may be unnecessarily expensive on a laptop. The builder therefore selects at most 20,000 CVE IDs by sorting them on SHA-256 of the string `20261026`, a NUL separator, and the CVE ID. The link split then uses seed `20261026` with requested 10% validation and 10% test holdouts. Selection happens before link splitting and does not use target labels, model outputs, or target performance. All catalog-only CAPEC/CWE relationships are retained; CVE and CPE edges are retained only for selected CVEs. Do not change `--max-cves`, seed, split fractions, or graph construction after examining model results.

## Run order

From the repository root:

```bash
python scripts/cybersecurity/prepare_attack_to_cve_transfer.py
```

Review `data/processed/cyber_transfer/audit.json`. The builder fails if source and target IDs overlap, split invariants fail, or valid/test are empty.

Then run a plumbing-only diagnostic:

```bash
python experiments/fedultra_fresh/src/cyber_transfer_runner.py \
  --seeds 20261027 --rounds 1 --local-steps 1 --eval-limit 16 \
  --output experiments/fedultra_fresh/results/cyber_transfer_smoke.json
```

The smoke result is not evidence for or against the hypothesis. It only checks checkpoint loading, source training, target inference, and metrics. The runner validates audit and artifact hashes before training.

Only if the graph audit and smoke run pass, run the confirmatory experiment:

```bash
python experiments/fedultra_fresh/src/cyber_transfer_runner.py \
  --output experiments/fedultra_fresh/results/cyber_transfer_confirmatory.json
```

Do not use `--eval-limit` in the confirmatory run. Do not overwrite existing results unless you have saved them elsewhere.

After all ten full seed runs finish, calculate the hierarchical paired-bootstrap interval:

```bash
python experiments/fedultra_fresh/src/bootstrap_cyber_transfer.py \
  --input experiments/fedultra_fresh/results/cyber_transfer_confirmatory.json \
  --output experiments/fedultra_fresh/results/cyber_transfer_bootstrap.json
```

The bootstrap refuses diagnostic runs and verifies that every seed evaluated the same target test triples in the same order. The final `meets_full_success_criterion` field is the decision gate.

## Interpretation limits

This evaluates link prediction on one sampled CAPEC/CWE/CVE graph. It does not directly establish operational attack detection, privacy guarantees, or broad generalization to every cybersecurity domain. A result below the frozen acceptance rule is a failure to meet the project's criterion; do not tune against target test metrics or change the threshold afterward.
