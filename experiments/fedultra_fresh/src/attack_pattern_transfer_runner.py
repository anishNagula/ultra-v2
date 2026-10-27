#!/usr/bin/env python3
"""Path-aware Enterprise ATT&CK <-> ICS ATT&CK transfer evaluation.

This is a new, separate experiment. It does not overwrite the previous aggregate
link-prediction experiment. The source fine-tuning objective gives explicit emphasis
to attack-pattern-related triples supported by alternate multi-hop paths. The primary
target metric is filtered MRR on held-out attack-pattern-related triples that have a
2..4-hop alternate path in the target TRAIN graph. Checkpoints are selected only by
source validation path-supported MRR.

STIX types are used only to select the research task; they are not input features to
ULTRA. The model/checkpoint architecture remains stock ULTRA in this first extension.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
import fedavg_runner as base
from evaluation_protocol import CLIENTS, load_client
from ultra.cybersecurity import (
    attack_pattern_indices,
    load_stix_type_map,
    path_supported_indices,
)

DEFAULT_JSON = {
    "enterprise": ROOT / "datasets/cybersecurity/mitre/enterprise/v19.2/enterprise-attack-19.2.json",
    "ics": ROOT / "datasets/cybersecurity/mitre/ics/v19.2/ics-attack-19.2.json",
}
DEFAULT_OUT = ROOT / "experiments/fedultra_fresh/results/attack_pattern_transfer.json"
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
       7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
       13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
       19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
       25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mrr_ci(values):
    n = len(values)
    if n < 2:
        return None
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    se = math.sqrt(variance / n)
    return [mean - T95.get(n - 1, 1.96) * se,
            mean + T95.get(n - 1, 1.96) * se]


def query_lists(bundle, split, type_map, max_hops):
    """Build attack-pattern and path-supported query lists in file order."""
    context = bundle["raw_triples"]["train"]
    queries = bundle["raw_triples"][split]
    ap_indices = attack_pattern_indices(queries, type_map)
    supported = set(path_supported_indices(
        context, queries, type_map, max_hops=max_hops, attack_pattern_only=True
    ))
    return {
        "all_attack_pattern": [queries[i] for i in ap_indices],
        "path_supported": [queries[i] for i in ap_indices if i in supported],
        "path_supported_indices": [i for i in ap_indices if i in supported],
    }


def encode_triples(triples, bundle):
    e2i, r2i = bundle["entity_to_id"], bundle["relation_to_id"]
    unknown = [triple for triple in triples
               if triple[0] not in e2i or triple[2] not in e2i or triple[1] not in r2i]
    if unknown:
        raise ValueError(f"Query contains entities/relations absent from train: {unknown[:3]}")
    return torch.tensor([[e2i[h], e2i[t], r2i[r]] for h, r, t in triples], dtype=torch.long)


@torch.no_grad()
def evaluate_queries(model, bundle, split, triples, eval_batch_size=4):
    if not triples:
        return {"mrr": None, "hits@1": None, "hits@3": None, "hits@10": None,
                "triples": 0, "ranking_queries": 0}
    target_data = bundle[f"{split}_data"]
    filter_graph = bundle[f"{split}_filter"]
    train_graph = bundle["train_data"]
    positives = encode_triples(triples, bundle)
    ranks = []
    model.eval()
    for start in range(0, len(positives), eval_batch_size):
        batch = positives[start:start + eval_batch_size]
        tail_batch, head_batch = base.tasks.all_negative(train_graph, batch)
        tail_scores = model(train_graph, tail_batch)
        head_scores = model(train_graph, head_batch)
        tail_mask, head_mask = base.tasks.strict_negative_mask(filter_graph, batch)
        tail_rank = base.tasks.compute_ranking(tail_scores, batch[:, 1], tail_mask).float()
        head_rank = base.tasks.compute_ranking(head_scores, batch[:, 0], head_mask).float()
        ranks.extend([tail_rank, head_rank])
    all_ranks = torch.cat(ranks)
    return {
        "mrr": float((1.0 / all_ranks).mean()),
        "hits@1": float((all_ranks <= 1).float().mean()),
        "hits@3": float((all_ranks <= 3).float().mean()),
        "hits@10": float((all_ranks <= 10).float().mean()),
        "triples": len(triples),
        "ranking_queries": int(all_ranks.numel()),
    }


def _loss_step(model, optimizer, train_data, positives, args, generator):
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
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite path-aware ULTRA loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads or any(not torch.isfinite(g).all().item() for g in grads):
        raise FloatingPointError("Missing/non-finite ULTRA gradients")
    optimizer.step()
    return float(loss.detach())


def train_path_aware(source, seed, bundles, type_maps, args, checkpoint_state):
    """Train one epoch over path-supported attack-pattern facts plus matched context facts.

    Every path-supported source training triple is visited once per epoch. An equally
    sized background set is sampled without replacement where possible. This avoids
    the prior 50-update-with-replacement protocol and reports the actual update count.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    bundle = bundles[source]
    source_types = type_maps[source]
    train_triples = bundle["raw_triples"]["train"]
    supported_idx = path_supported_indices(
        train_triples, train_triples, source_types,
        max_hops=args.max_hops, attack_pattern_only=True,
    )
    path_triples = [train_triples[i] for i in supported_idx]
    if len(path_triples) < args.min_path_train:
        raise RuntimeError(
            f"{source}: only {len(path_triples)} path-supported attack-pattern training triples; "
            f"need at least {args.min_path_train}. Run --audit-only and inspect the split/path definition."
        )
    all_train = encode_triples(train_triples, bundle)
    path_train = encode_triples(path_triples, bundle)
    path_train_set = set(supported_idx)
    background = [i for i in range(len(train_triples)) if i not in path_train_set]
    rng = random.Random(seed + 991)
    rng.shuffle(background)
    if not background:
        raise RuntimeError(f"{source}: no non-path background triples are available")
    if len(background) >= len(path_triples):
        background = background[:len(path_triples)]
    else:
        # Rare fallback for an unusually dense graph: keep equal objective weights.
        background = [rng.choice(background) for _ in range(len(path_triples))]
    background_train = encode_triples([train_triples[i] for i in background], bundle)

    model = base.new_model(checkpoint_state).cpu()
    optimizer = base.make_optimizer(model, args.lr, args.weight_decay)
    val_queries = query_lists(bundle, "valid", source_types, args.max_hops)["path_supported"]
    if len(val_queries) < args.min_path_valid:
        raise RuntimeError(
            f"{source}: only {len(val_queries)} path-supported attack-pattern validation queries; "
            f"need at least {args.min_path_valid}. Do not select checkpoints on target metrics."
        )
    best_mrr = evaluate_queries(model, bundle, "valid", val_queries, args.eval_batch_size)["mrr"]
    best_state = base.clone_state(model)
    best_epoch = 0
    history = [{"epoch": 0, "source_valid_path_mrr": best_mrr}]
    updates_total = 0

    for epoch in range(1, args.epochs + 1):
        gen = torch.Generator(device="cpu").manual_seed(seed * 1000003 + epoch * 7919)
        path_order = torch.randperm(len(path_train), generator=gen)
        bg_order = torch.randperm(len(background_train), generator=gen)
        # One pass over each path example and one matched-size background sample.
        combined = torch.cat((path_train[path_order], background_train[bg_order]), dim=0)
        combined = combined[torch.randperm(len(combined), generator=gen)]
        model.train()
        losses = []
        for start in range(0, len(combined), args.batch_size):
            positives = combined[start:start + args.batch_size]
            if len(positives) < 2:
                continue
            losses.append(_loss_step(model, optimizer, bundle["train_data"], positives, args, gen))
            updates_total += 1
        val_mrr = evaluate_queries(model, bundle, "valid", val_queries, args.eval_batch_size)["mrr"]
        history.append({"epoch": epoch, "mean_loss": sum(losses) / max(1, len(losses)),
                        "source_valid_path_mrr": val_mrr, "updates_total": updates_total})
        print(f"{source} seed={seed} epoch={epoch} updates={updates_total} "
              f"path_train={len(path_triples)} source_valid_path_mrr={val_mrr:.6f}")
        if val_mrr > best_mrr:
            best_mrr, best_state, best_epoch = val_mrr, base.clone_state(model), epoch
    model.load_state_dict(best_state, strict=True)
    return model, {
        "path_supported_source_train_triples": len(path_triples),
        "path_supported_source_valid_triples": len(val_queries),
        "updates_total": updates_total,
        "selected_source_epoch": best_epoch,
        "best_source_validation_path_mrr": best_mrr,
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=CLIENTS, default="enterprise")
    parser.add_argument("--target", choices=CLIENTS, default="ics")
    parser.add_argument("--enterprise-json", type=Path, default=DEFAULT_JSON["enterprise"])
    parser.add_argument("--ics-json", type=Path, default=DEFAULT_JSON["ics"])
    parser.add_argument("--seeds", default="20261027,20261028,20261029,20261030,20261031")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-negative", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--max-hops", type=int, default=4)
    parser.add_argument("--min-path-train", type=int, default=16)
    parser.add_argument("--min-path-valid", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.source == args.target:
        raise ValueError("source and target must differ")
    if args.epochs < 1 or args.batch_size < 2 or args.batch_size % 2:
        raise ValueError("epochs must be positive; batch-size must be even and >= 2")
    for name, path in (("enterprise", args.enterprise_json), ("ics", args.ics_json)):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name} STIX JSON: {path}. Pass the correct path explicitly.")
    type_maps = {"enterprise": load_stix_type_map(args.enterprise_json),
                 "ics": load_stix_type_map(args.ics_json)}
    bundles = {"enterprise": load_client("enterprise"), "ics": load_client("ics")}

    audit = {"protocol": "path-aware attack-pattern cross-graph transfer",
             "source": args.source, "target": args.target,
             "max_hops": args.max_hops, "epochs": args.epochs,
             "batch_size": args.batch_size, "num_negative": args.num_negative,
             "checkpoint": str(base.CHECKPOINT),
             "checkpoint_sha256": sha256(base.CHECKPOINT), "runs": []}
    for name in (args.source, args.target):
        bundle = bundles[name]
        type_map = type_maps[name]
        entry = {"graph": name, "stix_json_sha256": sha256(args.enterprise_json if name == "enterprise" else args.ics_json),
                 "split_counts": {s: len(bundle["raw_triples"][s]) for s in ("train", "valid", "test")}}
        for split in ("train", "valid", "test"):
            qs = query_lists(bundle, split, type_map, args.max_hops)
            entry[split] = {"attack_pattern_triples": len(qs["all_attack_pattern"]),
                            "path_supported_attack_pattern_triples": len(qs["path_supported"]),
                            "by_relation": {}}
            for triple in qs["path_supported"]:
                entry[split]["by_relation"][triple[1]] = entry[split]["by_relation"].get(triple[1], 0) + 1
        audit["runs"].append(entry)
    print(json.dumps(audit, indent=2))
    if args.audit_only:
        return
    if not base.CHECKPOINT.is_file():
        raise FileNotFoundError(f"Missing pretrained ULTRA checkpoint: {base.CHECKPOINT}")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; choose another output or pass --overwrite")
    checkpoint = torch.load(base.CHECKPOINT, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"Unexpected checkpoint format: {base.CHECKPOINT}")
    checkpoint_state = checkpoint["model"]
    results = []
    for seed in [int(x.strip()) for x in args.seeds.split(",") if x.strip()]:
        target_bundle = bundles[args.target]
        target_types = type_maps[args.target]
        test = query_lists(target_bundle, "test", target_types, args.max_hops)
        valid = query_lists(target_bundle, "valid", target_types, args.max_hops)
        if len(test["path_supported"]) < args.min_path_valid:
            raise RuntimeError(
                f"Target has only {len(test['path_supported'])} path-supported attack-pattern test queries; "
                "the path-specific result would be too small to interpret. Inspect audit-only output first."
            )
        model, selection = train_path_aware(args.source, seed, bundles, type_maps, args, checkpoint_state)
        zero = base.new_model(checkpoint_state).cpu()
        zero_valid = evaluate_queries(zero, target_bundle, "valid", valid["path_supported"], args.eval_batch_size)
        zero_test_all = evaluate_queries(zero, target_bundle, "test", test["all_attack_pattern"], args.eval_batch_size)
        zero_test_path = evaluate_queries(zero, target_bundle, "test", test["path_supported"], args.eval_batch_size)
        transfer_valid = evaluate_queries(model, target_bundle, "valid", valid["path_supported"], args.eval_batch_size)
        transfer_test_all = evaluate_queries(model, target_bundle, "test", test["all_attack_pattern"], args.eval_batch_size)
        transfer_test_path = evaluate_queries(model, target_bundle, "test", test["path_supported"], args.eval_batch_size)
        delta = transfer_test_path["mrr"] - zero_test_path["mrr"]
        result = {"seed": seed, "selection": selection,
                  "zero_shot_target_valid_path": zero_valid,
                  "transfer_target_valid_path": transfer_valid,
                  "zero_shot_target_test_attack_pattern": zero_test_all,
                  "transfer_target_test_attack_pattern": transfer_test_all,
                  "zero_shot_target_test_path_supported": zero_test_path,
                  "transfer_target_test_path_supported": transfer_test_path,
                  "delta_path_supported_mrr": delta}
        results.append(result)
        print(f"{args.source}->{args.target} seed={seed}: path-supported test "
              f"zero={zero_test_path['mrr']:.6f} transfer={transfer_test_path['mrr']:.6f} delta={delta:+.6f}")
    deltas = [r["delta_path_supported_mrr"] for r in results]
    audit["results"] = results
    audit["summary"] = {"mean_delta_path_supported_mrr": sum(deltas) / len(deltas),
                        "seed_level_95pct_t_ci": mrr_ci(deltas),
                        "acceptance_threshold_delta_mrr": 0.03,
                        "mean_gain_pass": sum(deltas) / len(deltas) >= 0.03,
                        "ci_entirely_above_zero": mrr_ci(deltas)[0] > 0 if mrr_ci(deltas) else False,
                        "claim_scope": "path-supported attack-pattern link prediction on the selected Enterprise/ICS graph direction only"}
    audit["protocol_note"] = (
        "Path-supported queries are defined by a 2..max_hops alternate path in the target training graph. "
        "Entity types select queries but are not model inputs. This is not evidence of full attack-campaign reconstruction."
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
