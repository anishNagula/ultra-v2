# ULTRA attack-pattern path-transfer extension (Phase 1)

This overlay adds a new experiment without overwriting the existing transfer results.

## What it changes

- Adds `ultra/cybersecurity.py`, which selects STIX attack-pattern-related triples and detects alternate 2–4-hop paths using only the graph context available to the model.
- Adds `experiments/fedultra_fresh/src/attack_pattern_transfer_runner.py`.
- Fine-tunes the existing pretrained ULTRA checkpoint with a task-aware objective: one pass through path-supported attack-pattern triples plus an equally sized background sample per epoch.
- Selects checkpoints using **source validation path-supported MRR**, not aggregate MRR.
- Reports zero-shot vs source-fine-tuned performance on target attack-pattern triples, including the primary path-supported subset.
- Uses STIX entity types only to select/stratify the task. They are not passed as model features.
- Does not change or overwrite the previous confirmatory JSON, split files, checkpoint, or upstream `ultra/models.py` weights.

This is deliberately the first controlled modification: it tests whether task alignment and path-focused learning help before adding new model parameters. It is not yet a claim that ULTRA has learned full attack-campaign reconstruction.

## Install

From the repository root (`~/code/ultra-v2`):

```bash
unzip -o attack-pattern-transfer-overlay.zip -d .
python -m py_compile ultra/cybersecurity.py experiments/fedultra_fresh/src/attack_pattern_transfer_runner.py
```

The zip contains paths relative to the repository root.

## 1. Audit query counts first — no training

Default source files expected:

- `datasets/cybersecurity/mitre/enterprise/v19.2/enterprise-attack-19.2.json`
- `datasets/cybersecurity/mitre/ics/v19.2/ics-attack-19.2.json`

If yours are elsewhere, pass the paths explicitly:

```bash
python experiments/fedultra_fresh/src/attack_pattern_transfer_runner.py \
  --enterprise-json /absolute/path/enterprise-attack-19.2.json \
  --ics-json /absolute/path/ics-attack-19.2.json \
  --source enterprise --target ics --audit-only
```

Read the counts for train/valid/test. Do **not** start training if the target has fewer than 5 path-supported test queries or the source has fewer than 16 path-supported training triples / 5 path-supported validation queries. The runner enforces these minimums.

## 2. Run Enterprise -> ICS

```bash
python experiments/fedultra_fresh/src/attack_pattern_transfer_runner.py \
  --source enterprise --target ics \
  --seeds 20261027,20261028,20261029,20261030,20261031 \
  --epochs 5 \
  --output experiments/fedultra_fresh/results/attack_pattern_enterprise_to_ics.json
```

## 3. Run ICS -> Enterprise separately

```bash
python experiments/fedultra_fresh/src/attack_pattern_transfer_runner.py \
  --source ics --target enterprise \
  --seeds 20261027,20261028,20261029,20261030,20261031 \
  --epochs 5 \
  --output experiments/fedultra_fresh/results/attack_pattern_ics_to_enterprise.json
```

## Primary result

`delta_path_supported_mrr = transferred MRR - zero-shot MRR` on held-out target triples whose endpoints have an alternate 2–4-hop path in the target training graph. All ranking queries use the same target graph context and filtered-ranking protocol. Source validation alone chooses the checkpoint.

The existing practical threshold remains +0.03 mean MRR, with a 95% interval entirely above zero. The current runner reports a seed-level t interval; it does not yet compute a paired query bootstrap. Do not call this a final acceptance result until a paired query-level uncertainty analysis is added and the path-query counts are adequate.

## Interpretation limits

- This evaluates path-supported attack-pattern-related link prediction between two ATT&CK graphs, not full attack-campaign reconstruction.
- It uses graph topology and STIX type metadata for task selection, but does not add text/semantic embeddings to ULTRA.
- It keeps ULTRA's architecture and pretrained checkpoint unchanged. A later architectural experiment can add explicit relation-composition learning, but should be compared against this controlled task-aware baseline on identical splits.
- The `--epochs` budget is one pass through the selected path examples plus matched background examples per epoch; update counts are recorded. Do not compare it to the old 50-update run as if the budgets were equal.

## 0. Audit path quality before training (recommended)

The first query-count audit reports high path-support rates, which can mean the criterion is too broad. Run this diagnostic first:

```bash
python experiments/fedultra_fresh/src/audit_attack_paths.py \
  --enterprise-json "$HOME/code/ULTRA/datasets/cybersecurity/mitre/enterprise/v19.2/enterprise-attack-19.2.json" \
  --ics-json "$HOME/code/ULTRA/datasets/cybersecurity/mitre/ics/v19.2/ics-attack-19.2.json" \
  --max-hops 4 \
  --output experiments/fedultra_fresh/results/attack_path_audit.json
```

It reports shortest alternate-path length, representative relation/direction sequences, endpoint types, unsupported examples, and exact train/test fact overlap. It uses only the target TRAIN graph as context for VALID/TEST queries. This is diagnostic only; do not start training until the signatures are reviewed and any train/test overlap is resolved.

## Phase 1b: strict semantic-path eligibility audit (run before training)

The generic alternate-path audit found that almost every held-out query has some
undirected path, commonly through shared `uses`/`mitigates` neighbors. That is too
broad to justify calling all such queries multi-hop attack-pattern reasoning.

Run the stricter audit first:

```bash
python -m py_compile experiments/fedultra_fresh/src/audit_semantic_attack_paths.py
python experiments/fedultra_fresh/src/audit_semantic_attack_paths.py \
  --enterprise-json "$HOME/code/ULTRA/datasets/cybersecurity/mitre/enterprise/v19.2/enterprise-attack-19.2.json" \
  --ics-json "$HOME/code/ULTRA/datasets/cybersecurity/mitre/ics/v19.2/ics-attack-19.2.json" \
  --output experiments/fedultra_fresh/results/semantic_attack_path_audit.json
```

This audit uses a small, explicit catalog of directed meta-path hypotheses:

- `uses`: malware attributed to an intrusion set that uses a technique; or actor uses a child technique whose parent is the candidate.
- `mitigates`: a course of action mitigates a child technique of the candidate parent.
- `targets`: a child technique's parent targets the candidate asset.
- `subtechnique-of`: a two-edge hierarchy path supports a transitive ancestor candidate.

The catalog is deliberately conservative and must be reviewed against the exact
STIX relation semantics. It is not a model and does not train. It reports per-relation
counts, examples and unsupported queries. Do not proceed to training until we have
reviewed these counts; if support is too sparse, revise the task definition rather
than loosen the path criterion solely to obtain more queries.
