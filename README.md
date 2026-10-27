# ULTRA relation-composition transfer patch

Adds a separate experiment runner. It does not modify `ultra/models.py`, the pretrained checkpoint, or existing runners/results.

## What changes

- Keeps stock ULTRA scoring and its ordinary link-prediction loss.
- Adds an auxiliary relation-composition classifier trained from two-hop paths derived **only from source TRAIN triples**. The classifier predicts the direct relation on the endpoints from the two path relation representations. Its gradients flow through ULTRA's relation reasoner; the classifier is training-only and is not used at inference.
- Compares pretrained zero-shot ULTRA, ordinary source fine-tuning, and relation-composition fine-tuning on the same target test attack-pattern-related triples.
- Selects each fine-tuned checkpoint using source validation attack-pattern MRR, never target validation/test metrics.
- Reports both composition-vs-zero-shot and composition-vs-standard-fine-tuning gains. The latter is necessary to attribute any improvement to the new objective.

This is structural relation composition, not a claim that arbitrary two-hop paths are causal attack sequences. The method is experimental and should be accepted only if the confirmatory results meet the pre-registered threshold.

## Install

From `~/code/ultra-v2`:

```bash
unzip -o ~/Downloads/ultra-relation-composition-transfer-patch.zip -d .
python -m py_compile experiments/fedultra_fresh/src/relation_composition_transfer_runner.py
```

## Smoke test (one optimizer step; no checkpoint/result files written)

```bash
python experiments/fedultra_fresh/src/relation_composition_transfer_runner.py \
  --source enterprise --target ics --smoke-test
```

## Full confirmatory run

Run the smoke test first. Then, for the preselected direction Enterprise -> ICS:

```bash
python experiments/fedultra_fresh/src/relation_composition_transfer_runner.py \
  --source enterprise --target ics \
  --seeds 20261027,20261028,20261029,20261030,20261031 \
  --epochs 5 --batch-size 8 --num-negative 64 \
  --composition-batch-size 32 --composition-weight 0.1 \
  --output experiments/fedultra_fresh/results/relation_composition_transfer.json
```

Do not use the target test set to tune `composition-weight`, epochs, or other settings. If tuning is needed, do it on source validation data and freeze the protocol before confirmatory runs.
