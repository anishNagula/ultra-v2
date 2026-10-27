#!/usr/bin/env python3
"""Controlled ULTRA transfer experiment with an auxiliary relation-composition loss.

Three target evaluations are compared on identical held-out attack-pattern-related
queries: pretrained zero-shot ULTRA, ordinary source fine-tuning, and source
fine-tuning with a relation-composition auxiliary objective. Composition examples
are derived from source TRAIN triples only. No target test facts are used for
training or checkpoint selection.

This objective learns structural relation composition, not a causal attack path.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
import time
from datetime import datetime
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

import fedavg_runner as base
from evaluation_protocol import CLIENTS, load_client
from ultra.cybersecurity import attack_pattern_indices, load_stix_type_map

DEFAULT_JSON = {
    "enterprise": ROOT / "datasets/cybersecurity/mitre/enterprise/v19.2/enterprise-attack-19.2.json",
    "ics": ROOT / "datasets/cybersecurity/mitre/ics/v19.2/ics-attack-19.2.json",
}
DEFAULT_OUT = ROOT / "experiments/fedultra_fresh/results/relation_composition_transfer.json"
def log(message):
    """Print timestamped progress immediately, even when stdout is buffered."""
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"[{stamp}] {message}", flush=True)


T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
       7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
       13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
       19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
       25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def paired_ci(values):
    n = len(values)
    if n < 2:
        return None
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    se = math.sqrt(variance / n)
    crit = T95.get(n - 1, 1.96)
    return [mean - crit * se, mean + crit * se]


class RelationCompositionHead(nn.Module):
    """Predict a direct relation from two contextual relation representations."""
    def __init__(self, relation_dim: int, num_direct_relations: int):
        super().__init__()
        hidden = max(32, relation_dim)
        self.network = nn.Sequential(
            nn.Linear(2 * relation_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_direct_relations),
        )

    def forward(self, first, second):
        return self.network(torch.cat((first, second), dim=-1))


def build_composition_examples(raw_triples, entity_to_id, relation_to_id):
    """Return unique (r1, r2, target_relation) examples from TRAIN-only 2-hop paths.

    Each source triple is also a direct target relation label. Paths are traversed
    in the directed message graph, including explicit inverse relations. The target
    fact itself cannot be a 2-hop path edge because intermediate nodes equal to either
    endpoint are excluded. Counts are retained for audit but examples are deduplicated
    so highly connected entity pairs do not dominate the auxiliary objective.
    """
    num_direct = len(relation_to_id)
    outgoing = defaultdict(list)
    for h, r, t in raw_triples:
        hi, ti, ri = entity_to_id[h], entity_to_id[t], relation_to_id[r]
        outgoing[hi].append((ri, ti))
        outgoing[ti].append((ri + num_direct, hi))

    counts = Counter()
    path_instances = 0
    for h, relation, t in raw_triples:
        hi, ti, target_r = entity_to_id[h], entity_to_id[t], relation_to_id[relation]
        for r1, middle in outgoing[hi]:
            if middle == hi or middle == ti:
                continue
            for r2, end in outgoing[middle]:
                if end == ti and end != hi:
                    counts[(r1, r2, target_r)] += 1
                    path_instances += 1

    examples = sorted(counts)
    tensor = torch.tensor(examples, dtype=torch.long) if examples else torch.empty((0, 3), dtype=torch.long)
    return tensor, {
        "path_instances": int(path_instances),
        "unique_relation_compositions": int(len(examples)),
        "relation_composition_counts": [
            {"r1": int(a), "r2": int(b), "target_relation": int(c), "frequency": int(n)}
            for (a, b, c), n in sorted(counts.items())
        ],
    }


def encode_triples(triples, bundle):
    e2i, r2i = bundle["entity_to_id"], bundle["relation_to_id"]
    unknown = [x for x in triples if x[0] not in e2i or x[2] not in e2i or x[1] not in r2i]
    if unknown:
        raise ValueError(f"Triples contain entities/relations absent from train: {unknown[:3]}")
    return torch.tensor([[e2i[h], e2i[t], r2i[r]] for h, r, t in triples], dtype=torch.long)


def attack_pattern_queries(bundle, split, type_map):
    triples = bundle["raw_triples"][split]
    indices = attack_pattern_indices(triples, type_map)
    return [triples[i] for i in indices]


@torch.no_grad()
def evaluate_queries(model, bundle, split, triples, eval_batch_size=4):
    if not triples:
        return {"mrr": None, "hits@1": None, "hits@3": None, "hits@10": None,
                "triples": 0, "ranking_queries": 0}
    filter_graph = bundle[f"{split}_filter"]
    train_graph = bundle["train_data"]
    positives = encode_triples(triples, bundle)
    model.eval()
    ranks = []
    total_eval_batches = (len(positives) + eval_batch_size - 1) // eval_batch_size
    eval_started = time.monotonic()
    for start in range(0, len(positives), eval_batch_size):
        batch = positives[start:start + eval_batch_size]
        batch_number = start // eval_batch_size + 1
        if batch_number == 1 or batch_number % 25 == 0 or batch_number == total_eval_batches:
            log(f"EVAL {split}: batch {batch_number}/{total_eval_batches} "
                f"queries={min(start + len(batch), len(positives))}/{len(positives)}")
        tail_batch, head_batch = base.tasks.all_negative(train_graph, batch)
        tail_scores = model(train_graph, tail_batch)
        head_scores = model(train_graph, head_batch)
        tail_mask, head_mask = base.tasks.strict_negative_mask(filter_graph, batch)
        tail_rank = base.tasks.compute_ranking(tail_scores, batch[:, 1], tail_mask).float()
        head_rank = base.tasks.compute_ranking(head_scores, batch[:, 0], head_mask).float()
        ranks.extend((tail_rank, head_rank))
    all_ranks = torch.cat(ranks)
    result = {
        "mrr": float((1.0 / all_ranks).mean()),
        "hits@1": float((all_ranks <= 1).float().mean()),
        "hits@3": float((all_ranks <= 3).float().mean()),
        "hits@10": float((all_ranks <= 10).float().mean()),
        "triples": len(triples), "ranking_queries": int(all_ranks.numel()),
    }
    if not all(math.isfinite(result[k]) for k in ("mrr", "hits@1", "hits@3", "hits@10")):
        raise FloatingPointError(f"Non-finite metrics: {result}")
    log(f"EVAL {split} complete: mrr={result['mrr']:.6f} "
        f"queries={len(triples)} elapsed={time.monotonic() - eval_started:.1f}s")
    return result


def link_prediction_loss(model, train_data, positives, args):
    batch = base.tasks.negative_sampling(
        train_data, positives, num_negative=args.num_negative, strict=True
    )
    pred = model(train_data, batch)
    target = torch.zeros_like(pred)
    target[:, 0] = 1
    loss_each = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    weights = torch.ones_like(pred)
    if args.num_negative > 0:
        with torch.no_grad():
            weights[:, 1:] = F.softmax(pred[:, 1:], dim=-1)
    loss = ((loss_each * weights).sum(dim=-1) / weights.sum(dim=-1)).mean()
    return loss


def relation_features(model, train_data):
    """Get one contextual representation per directed relation node."""
    num_relations = int(train_data.relation_graph.num_nodes)
    ids = torch.arange(num_relations, dtype=torch.long, device=train_data.edge_index.device)
    contextual = model.relation_model(train_data.relation_graph, query=ids)
    if contextual.ndim != 3 or contextual.shape[0] != num_relations or contextual.shape[1] != num_relations:
        raise RuntimeError(f"Unexpected relation-model output shape: {tuple(contextual.shape)}")
    row = torch.arange(num_relations, device=contextual.device)
    return contextual[row, row]


def make_composer(model, train_data, num_direct_relations):
    was_training = model.training
    model.eval()
    with torch.no_grad():
        features = relation_features(model, train_data)
    model.train(was_training)
    return RelationCompositionHead(features.shape[-1], num_direct_relations)


def train_one_variant(source, seed, bundle, type_map, checkpoint_state, args, use_composition):
    torch.manual_seed(seed)
    random.seed(seed)
    model = base.new_model(checkpoint_state).cpu()
    raw_train = bundle["raw_triples"]["train"]
    positives = base.triples_tensor(bundle["train_data"])
    comp_examples, comp_audit = build_composition_examples(
        raw_train, bundle["entity_to_id"], bundle["relation_to_id"]
    )
    if use_composition and len(comp_examples) == 0:
        raise RuntimeError(f"{source}: no 2-hop relation-composition examples from TRAIN triples")

    composer = None
    if use_composition:
        composer = make_composer(model, bundle["train_data"], len(bundle["relation_to_id"]))
        parameters = list(model.parameters()) + list(composer.parameters())
    else:
        parameters = list(model.parameters())
    optimizer = optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)

    valid_queries = attack_pattern_queries(bundle, "valid", type_map)
    if len(valid_queries) < args.min_valid_queries:
        raise RuntimeError(f"{source}: only {len(valid_queries)} attack-pattern validation queries")
    log(f"TRAIN {source} seed={seed} variant={'composition' if use_composition else 'standard'} "
        f"starting: train_triples={len(positives)} epochs={args.epochs} "
        f"batch_size={args.batch_size} negatives={args.num_negative}")
    initial = evaluate_queries(model, bundle, "valid", valid_queries, args.eval_batch_size)["mrr"]
    best_mrr, best_state, best_epoch = initial, base.clone_state(model), 0
    history = [{"epoch": 0, "source_valid_attack_pattern_mrr": initial}]
    generator = torch.Generator(device="cpu")
    comp_generator = torch.Generator(device="cpu")
    comp_examples = comp_examples.cpu()
    total_updates = 0

    for epoch in range(1, args.epochs + 1):
        epoch_seed = seed * 1000003 + epoch * 7919
        generator.manual_seed(epoch_seed)
        comp_generator.manual_seed(epoch_seed + 104729)
        order = torch.randperm(len(positives), generator=generator)
        model.train()
        if composer is not None:
            composer.train()
        epoch_link, epoch_comp = [], []
        epoch_started = time.monotonic()
        total_batches = (len(order) + args.batch_size - 1) // args.batch_size
        log(f"TRAIN {source} seed={seed} variant={'composition' if use_composition else 'standard'} "
            f"epoch={epoch}/{args.epochs} started: batches={total_batches}")
        for batch_index, start in enumerate(range(0, len(order), args.batch_size)):
            ids = order[start:start + args.batch_size]
            if len(ids) % 2:
                ids = ids[:-1]
            if len(ids) < 2:
                continue
            batch_pos = positives[ids]
            batch_started = time.monotonic()
            log(f"TRAIN {source} seed={seed} variant={'composition' if use_composition else 'standard'} "
                f"epoch={epoch}/{args.epochs} batch={batch_index + 1}/{total_batches} "
                f"phase=forward_backward_start positives={len(batch_pos)}")
            # Reset the stochastic stream per matched batch so standard and
            # composition variants use identical negative-sampling/dropout seeds.
            torch.manual_seed(seed * 1000003 + epoch * 7919 + batch_index * 97)
            primary_loss = link_prediction_loss(model, bundle["train_data"], batch_pos, args)
            total_loss = primary_loss
            comp_loss_value = None
            if use_composition:
                comp_ids = torch.randint(len(comp_examples), (args.composition_batch_size,), generator=comp_generator)
                comp_batch = comp_examples[comp_ids]
                feats = relation_features(model, bundle["train_data"])
                r1 = feats[comp_batch[:, 0]]
                r2 = feats[comp_batch[:, 1]]
                logits = composer(r1, r2)
                comp_loss = F.cross_entropy(logits, comp_batch[:, 2])
                total_loss = total_loss + args.composition_weight * comp_loss
                comp_loss_value = float(comp_loss.detach())
            if not torch.isfinite(total_loss):
                raise FloatingPointError(f"{source} seed={seed}: non-finite total loss")
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            grads = [p.grad for p in parameters if p.grad is not None]
            if not grads or any(not torch.isfinite(g).all().item() for g in grads):
                raise FloatingPointError(f"{source} seed={seed}: missing/non-finite gradients")
            optimizer.step()
            total_updates += 1
            epoch_link.append(float(primary_loss.detach()))
            if comp_loss_value is not None:
                epoch_comp.append(comp_loss_value)
            log(f"TRAIN {source} seed={seed} variant={'composition' if use_composition else 'standard'} "
                f"epoch={epoch}/{args.epochs} batch={batch_index + 1}/{total_batches} "
                f"phase=optimizer_step_complete elapsed={time.monotonic() - batch_started:.1f}s "
                f"link_loss={float(primary_loss.detach()):.5f} "
                f"total_loss={float(total_loss.detach()):.5f}")

        log(f"TRAIN {source} seed={seed} epoch={epoch} training_complete "
            f"elapsed={time.monotonic() - epoch_started:.1f}s updates={total_updates}; validating")
        val_mrr = evaluate_queries(model, bundle, "valid", valid_queries, args.eval_batch_size)["mrr"]
        row = {"epoch": epoch, "updates_total": total_updates,
               "mean_link_loss": sum(epoch_link) / max(1, len(epoch_link)),
               "source_valid_attack_pattern_mrr": val_mrr}
        if epoch_comp:
            row["mean_relation_composition_loss"] = sum(epoch_comp) / len(epoch_comp)
        history.append(row)
        log(f"{source} seed={seed} variant={'composition' if use_composition else 'standard'} "
            f"epoch={epoch} updates={total_updates} source_valid_attack_pattern_mrr={val_mrr:.6f}")
        if val_mrr > best_mrr:
            best_mrr, best_state, best_epoch = val_mrr, base.clone_state(model), epoch

    model.load_state_dict(best_state, strict=True)
    return model, {
        "variant": "relation_composition" if use_composition else "standard_finetuning",
        "selected_source_validation_epoch": best_epoch,
        "best_source_validation_attack_pattern_mrr": best_mrr,
        "updates_total": total_updates,
        "composition_audit": comp_audit,
        "history": history,
    }


def run_smoke_test(source, bundles, type_maps, checkpoint_state, args):
    bundle = bundles[source]
    model = base.new_model(checkpoint_state).cpu()
    examples, audit = build_composition_examples(
        bundle["raw_triples"]["train"], bundle["entity_to_id"], bundle["relation_to_id"]
    )
    if not len(examples):
        raise RuntimeError("Smoke test failed: no relation-composition examples")
    composer = make_composer(model, bundle["train_data"], len(bundle["relation_to_id"]))
    params = list(model.parameters()) + list(composer.parameters())
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    positives = base.triples_tensor(bundle["train_data"])
    positives = positives[:args.batch_size]
    if len(positives) % 2:
        positives = positives[:-1]
    model.train(); composer.train()
    primary = link_prediction_loss(model, bundle["train_data"], positives, args)
    sample = examples[:min(len(examples), args.composition_batch_size)]
    features = relation_features(model, bundle["train_data"])
    logits = composer(features[sample[:, 0]], features[sample[:, 1]])
    aux = F.cross_entropy(logits, sample[:, 2])
    loss = primary + args.composition_weight * aux
    optimizer.zero_grad(set_to_none=True); loss.backward()
    grads = [p.grad for p in params if p.grad is not None]
    if not torch.isfinite(loss) or not grads or any(not torch.isfinite(g).all().item() for g in grads):
        raise FloatingPointError("Smoke test failed: non-finite loss/gradient")
    optimizer.step()
    log(json.dumps({"smoke_test": "PASS", "source": source,
                      "primary_link_loss": float(primary.detach()),
                      "relation_composition_loss": float(aux.detach()),
                      "total_loss": float(loss.detach()),
                      "composition_audit": audit,
                      "checkpoint_modified": False, "results_written": False}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", choices=CLIENTS, default="enterprise")
    p.add_argument("--target", choices=CLIENTS, default="ics")
    p.add_argument("--enterprise-json", type=Path, default=DEFAULT_JSON["enterprise"])
    p.add_argument("--ics-json", type=Path, default=DEFAULT_JSON["ics"])
    p.add_argument("--seeds", default="20261027,20261028,20261029,20261030,20261031")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-negative", type=int, default=64)
    p.add_argument("--composition-batch-size", type=int, default=32)
    p.add_argument("--composition-weight", type=float, default=0.1)
    p.add_argument("--eval-batch-size", type=int, default=4)
    p.add_argument("--min-valid-queries", type=int, default=20)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    log(f"Runner started: source={args.source} target={args.target} "
        f"seeds={args.seeds} epochs={args.epochs} batch_size={args.batch_size} "
        f"num_negative={args.num_negative} composition_weight={args.composition_weight}")

    if args.source == args.target:
        raise ValueError("source and target must differ")
    if args.epochs < 1 or args.batch_size < 2 or args.batch_size % 2:
        raise ValueError("epochs must be positive; batch-size must be even and >= 2")
    if args.composition_batch_size < 1 or args.composition_weight < 0:
        raise ValueError("composition batch size must be positive and composition weight non-negative")
    for name, path in (("enterprise", args.enterprise_json), ("ics", args.ics_json)):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name} STIX JSON: {path}")

    log("Loading STIX entity type maps")
    type_maps = {"enterprise": load_stix_type_map(args.enterprise_json),
                 "ics": load_stix_type_map(args.ics_json)}
    log("Loading prepared Enterprise and ICS datasets")
    bundles = {"enterprise": load_client("enterprise"), "ics": load_client("ics")}
    if not base.CHECKPOINT.is_file():
        raise FileNotFoundError(f"Missing pretrained ULTRA checkpoint: {base.CHECKPOINT}")
    log(f"Loading pretrained checkpoint: {base.CHECKPOINT}")
    checkpoint = torch.load(base.CHECKPOINT, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"Unexpected checkpoint format: {base.CHECKPOINT}")
    checkpoint_state = checkpoint["model"]

    if args.smoke_test:
        run_smoke_test(args.source, bundles, type_maps, checkpoint_state, args)
        return
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; choose a new path or pass --overwrite")

    source_bundle, target_bundle = bundles[args.source], bundles[args.target]
    source_types, target_types = type_maps[args.source], type_maps[args.target]
    target_valid = attack_pattern_queries(target_bundle, "valid", target_types)
    target_test = attack_pattern_queries(target_bundle, "test", target_types)
    target_test_all_count = len(target_test)
    if len(target_test) < args.min_valid_queries:
        raise RuntimeError(f"Target has only {len(target_test)} attack-pattern test queries")

    output = {
        "protocol": "pretrained zero-shot vs standard source fine-tuning vs source fine-tuning with auxiliary relation composition",
        "source": args.source, "target": args.target, "seeds": [],
        "epochs": args.epochs, "batch_size": args.batch_size,
        "num_negative": args.num_negative, "composition_batch_size": args.composition_batch_size,
        "composition_weight": args.composition_weight,
        "checkpoint": str(base.CHECKPOINT), "checkpoint_sha256": sha256(base.CHECKPOINT),
        "target_test_attack_pattern_queries": target_test_all_count,
        "selection_rule": "each fine-tuned variant selected by source validation attack-pattern MRR; target validation/test do not select checkpoints",
        "interpretation_limit": "The auxiliary task predicts direct relation labels from two-hop structural relation compositions. It does not establish causal attack sequences.",
        "runs": [],
    }

    for seed_text in args.seeds.split(","):
        if not seed_text.strip():
            continue
        seed = int(seed_text.strip())
        log(f"===== SEED {seed} started =====")
        zero = base.new_model(checkpoint_state).cpu()
        log(f"SEED {seed}: evaluating zero-shot target test")
        zero_test = evaluate_queries(zero, target_bundle, "test", target_test, args.eval_batch_size)
        standard, standard_info = train_one_variant(
            args.source, seed, source_bundle, source_types, checkpoint_state, args, False
        )
        composition, composition_info = train_one_variant(
            args.source, seed, source_bundle, source_types, checkpoint_state, args, True
        )
        standard_test = evaluate_queries(standard, target_bundle, "test", target_test, args.eval_batch_size)
        composition_test = evaluate_queries(composition, target_bundle, "test", target_test, args.eval_batch_size)
        run = {
            "seed": seed, "zero_shot_target_test": zero_test,
            "standard_finetuning": {"selection": standard_info, "target_test": standard_test},
            "relation_composition": {"selection": composition_info, "target_test": composition_test},
            "delta_composition_vs_zero_mrr": composition_test["mrr"] - zero_test["mrr"],
            "delta_standard_vs_zero_mrr": standard_test["mrr"] - zero_test["mrr"],
            "delta_composition_vs_standard_mrr": composition_test["mrr"] - standard_test["mrr"],
        }
        output["runs"].append(run)
        log(f"{args.source}->{args.target} seed={seed}: zero={zero_test['mrr']:.6f} "
            f"standard={standard_test['mrr']:.6f} composition={composition_test['mrr']:.6f} "
            f"composition-vs-standard={run['delta_composition_vs_standard_mrr']:+.6f}")

    if len(output["runs"]) < 2:
        raise ValueError("At least two seeds are required for a confidence interval")
    d_zero = [r["delta_composition_vs_zero_mrr"] for r in output["runs"]]
    d_standard = [r["delta_composition_vs_standard_mrr"] for r in output["runs"]]
    ci_zero, ci_standard = paired_ci(d_zero), paired_ci(d_standard)
    mean_zero = sum(d_zero) / len(d_zero)
    mean_standard = sum(d_standard) / len(d_standard)
    output["summary"] = {
        "mean_delta_composition_vs_zero_mrr": mean_zero,
        "seed_level_95pct_t_ci_vs_zero": ci_zero,
        "mean_delta_composition_vs_standard_mrr": mean_standard,
        "seed_level_95pct_t_ci_vs_standard": ci_standard,
        "acceptance_threshold_delta_mrr": 0.03,
        "primary_pass": bool(mean_zero >= 0.03 and ci_zero is not None and ci_zero[0] > 0),
        "incremental_objective_ci_above_zero": bool(ci_standard is not None and ci_standard[0] > 0),
        "note": "A primary pass does not establish that composition helps beyond standard fine-tuning; inspect the incremental comparison too.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    log(json.dumps(output["summary"], indent=2))
    log(f"Saved results: {args.output}")


if __name__ == "__main__":
    main()

