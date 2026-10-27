#!/usr/bin/env python3
"""Confirmatory cross-graph transfer evaluation for pretrained ULTRA on MITRE Enterprise and ICS, with per-query MRRs for paired bootstrap analysis.

For each direction, fine-tune the pretrained checkpoint on the source graph only,
select the checkpoint using source validation MRR, then evaluate it on the target
KG without any gradient updates on that target. Compare against pretrained zero-shot
ULTRA on the same target graph. Target validation/test data are never used to select
or tune the source-fine-tuned model.
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# This script lives beside fedavg_runner.py, so the existing tested data/model helpers
# are reused rather than copied or reimplemented.
import fedavg_runner as base
from evaluation_protocol import CLIENTS, load_client
import torch

T_CRITICAL_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045,
}


def ci95(values):
    n = len(values)
    if n < 2:
        return None
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    se = math.sqrt(variance / n)
    tcrit = T_CRITICAL_95.get(n - 1, 1.96)
    return [mean - tcrit * se, mean + tcrit * se]


def relation_metric_deltas(zero_shot, transferred):
    output = {}
    for relation in sorted(set(zero_shot) | set(transferred)):
        before = zero_shot.get(relation)
        after = transferred.get(relation)
        output[relation] = {
            "zero_shot": before,
            "transferred": after,
            "delta_mrr": (after["mrr"] - before["mrr"])
            if before is not None and after is not None else None,
        }
    return output


@torch.no_grad()
def evaluate_with_query_mrr(model, bundle, split, eval_batch_size=4, eval_limit=0):
    """Return aggregate, per-triple, and per-relation metrics in one scoring pass.

    Each triple's MRR is the mean of reciprocal head and tail ranks. Averaging
    those per-triple values reproduces standard MRR over the 2*N ranking queries.
    The same ranks feed per-relation summaries, avoiding redundant graph scoring.
    """
    if split == "valid":
        target_data, filter_graph = bundle["valid_data"], bundle["valid_filter"]
    elif split == "test":
        target_data, filter_graph = bundle["test_data"], bundle["test_filter"]
    else:
        raise ValueError(split)

    positives = base.triples_tensor(target_data)
    if eval_limit > 0:
        positives = positives[:eval_limit]
    train_graph = bundle["train_data"]
    relation_names = {idx: name for name, idx in bundle["relation_to_id"].items()}
    model.eval()
    all_head_ranks, all_tail_ranks, per_triple_mrr = [], [], []
    ranks_by_relation = {}
    for start in range(0, len(positives), eval_batch_size):
        batch = positives[start:start + eval_batch_size]
        tail_batch, head_batch = base.tasks.all_negative(train_graph, batch)
        tail_scores = model(train_graph, tail_batch)
        head_scores = model(train_graph, head_batch)
        tail_mask, head_mask = base.tasks.strict_negative_mask(filter_graph, batch)
        tail_rank = base.tasks.compute_ranking(tail_scores, batch[:, 1], tail_mask).float()
        head_rank = base.tasks.compute_ranking(head_scores, batch[:, 0], head_mask).float()
        all_tail_ranks.append(tail_rank)
        all_head_ranks.append(head_rank)
        per_triple_mrr.extend(((1.0 / tail_rank + 1.0 / head_rank) / 2.0).cpu().tolist())
        for i, triple in enumerate(batch):
            relation_id = int(triple[2])
            if relation_id not in relation_names:
                raise ValueError(f"Relation ID {relation_id} missing from loaded vocabulary")
            relation = relation_names[relation_id]
            ranks_by_relation.setdefault(relation, []).extend([
                tail_rank[i].reshape(1), head_rank[i].reshape(1)
            ])

    tail_ranks = torch.cat(all_tail_ranks)
    head_ranks = torch.cat(all_head_ranks)
    ranks = torch.cat((tail_ranks, head_ranks))
    result = {
        "mrr": float((1.0 / ranks).mean()),
        "hits@1": float((ranks <= 1).float().mean()),
        "hits@3": float((ranks <= 3).float().mean()),
        "hits@10": float((ranks <= 10).float().mean()),
        "triples_evaluated": int(len(positives)),
        "ranking_queries": int(len(ranks)),
    }
    if abs(sum(per_triple_mrr) / len(per_triple_mrr) - result["mrr"]) > 1e-6:
        raise AssertionError("Per-triple reciprocal ranks do not reproduce aggregate MRR")

    raw_split = bundle["raw_triples"][split][:len(positives)]
    per_query = [
        {"triple": list(triple), "mrr": float(value)}
        for triple, value in zip(raw_split, per_triple_mrr)
    ]
    by_relation = {}
    for relation, rank_parts in sorted(ranks_by_relation.items()):
        rr = torch.cat(rank_parts)
        by_relation[relation] = {
            "mrr": float((1.0 / rr).mean()),
            "hits@1": float((rr <= 1).float().mean()),
            "hits@3": float((rr <= 3).float().mean()),
            "hits@10": float((rr <= 10).float().mean()),
            "triples_evaluated": int(len(rank_parts) // 2),
            "ranking_queries": int(len(rr)),
        }
    return result, per_query, by_relation


def train_source_model(source, target, seed, bundles, args, checkpoint_state):
    # Reproducible source-only fine-tuning. No target validation/test value is used
    # for model selection or optimization.
    torch.manual_seed(seed)
    random.seed(seed)
    model = base.new_model(checkpoint_state).cpu()
    optimizer = base.make_optimizer(model, args.lr, args.weight_decay)
    source_bundle = bundles[source]
    initial = base.evaluate(
        model, source_bundle, "valid", args.eval_batch_size, args.eval_limit
    )
    best_val = initial["mrr"]
    best_state = base.clone_state(model)
    best_round = 0
    history = [{"round": 0, "source_validation_mrr": best_val}]

    for round_idx in range(1, args.rounds + 1):
        loss = base.train_updates(
            model,
            optimizer,
            source_bundle["train_data"],
            args.local_steps,
            args.batch_size,
            args.num_negative,
            seed,
            source,
            round_idx,
        )
        val = base.evaluate(
            model, source_bundle, "valid", args.eval_batch_size, args.eval_limit
        )["mrr"]
        history.append({"round": round_idx, "mean_training_loss": loss,
                        "source_validation_mrr": val})
        if val > best_val:
            best_val = val
            best_state = base.clone_state(model)
            best_round = round_idx
        print(f"{source}->{target} seed={seed} round={round_idx} "
              f"loss={loss:.6f} source_valid_mrr={val:.6f} best_round={best_round}")

    model.load_state_dict(best_state, strict=True)
    return model, {"best_source_validation_mrr": best_val,
                   "selected_source_round": best_round,
                   "source_validation_history": history}


def run_direction(source, target, seed, bundles, args, checkpoint_state):
    print(f"\n--- Transfer {source} -> {target}, seed {seed} ---")
    transferred_model, selection = train_source_model(
        source, target, seed, bundles, args, checkpoint_state
    )
    target_bundle = bundles[target]

    # Evaluate both models on exactly the same target graph and target test split.
    # The pretrained baseline has no gradient updates; the transfer model was tuned
    # using source data only. Both use target train triples as graph context.
    zero_shot_model = base.new_model(checkpoint_state).cpu()
    # Target validation is intentionally not scored: it is neither needed for
    # checkpoint selection nor part of the confirmatory test. It remains in the
    # filter graph so other known true facts are excluded from test negatives.
    zero_test, zero_test_per_query, zero_test_by_relation = evaluate_with_query_mrr(
        zero_shot_model, target_bundle, "test", args.eval_batch_size, args.eval_limit
    )
    transfer_test, transfer_test_per_query, transfer_test_by_relation = evaluate_with_query_mrr(
        transferred_model, target_bundle, "test", args.eval_batch_size, args.eval_limit
    )
    if [item["triple"] for item in zero_test_per_query] != [item["triple"] for item in transfer_test_per_query]:
        raise AssertionError("Baseline and transfer test queries are not aligned")
    per_query_comparison = [
        {"triple": before["triple"], "zero_shot_mrr": before["mrr"],
         "transferred_mrr": after["mrr"], "delta_mrr": after["mrr"] - before["mrr"]}
        for before, after in zip(zero_test_per_query, transfer_test_per_query)
    ]
    per_relation = relation_metric_deltas(zero_test_by_relation, transfer_test_by_relation)
    delta = transfer_test["mrr"] - zero_test["mrr"]
    print(f"ZERO-SHOT {target}: {json.dumps(zero_test, sort_keys=True)}")
    print(f"TRANSFER {source}->{target}: {json.dumps(transfer_test, sort_keys=True)}")
    print(f"TRANSFER DELTA {source}->{target}: {delta:+.6f} MRR")
    print(f"PER-RELATION TEST {source}->{target}: {json.dumps(per_relation, sort_keys=True)}")
    return {
        "seed": seed,
        "source": source,
        "target": target,
        "config": {
            "rounds": args.rounds,
            "local_steps_per_round": args.local_steps,
            "updates_on_source": args.rounds * args.local_steps,
            "batch_size": args.batch_size,
            "num_negative": args.num_negative,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "eval_limit": args.eval_limit,
        },
        "selection": selection,
        "zero_shot_target": {"test": zero_test},
        "source_finetuned_transfer": {"target_test": transfer_test},
        "target_validation_scored": False,
        "transfer_delta_mrr": delta,
        "zero_shot_target_test_by_relation": zero_test_by_relation,
        "transferred_target_test_by_relation": transfer_test_by_relation,
        "per_relation_test_comparison": per_relation,
        "per_query_test_comparison": per_query_comparison,
        "diagnostic_only": args.eval_limit > 0,
    }


def summarize_direction(results):
    deltas = [r["transfer_delta_mrr"] for r in results]
    n = len(deltas)
    mean = sum(deltas) / n
    ci = ci95(deltas)
    diagnostic_only = any(r["diagnostic_only"] for r in results)
    ci_positive = bool(ci and ci[0] > 0)
    mean_pass = mean >= 0.03
    enough_seeds = n >= 3
    return {
        "num_seeds": n,
        "per_seed_transfer_delta_mrr": deltas,
        "mean_transfer_delta_mrr": mean,
        "paired_seed_level_95pct_t_ci": ci,
        "acceptance_criteria": {
            "minimum_mean_transfer_gain": 0.03,
            "mean_gain_meets_threshold": mean_pass,
            "ci_supports_positive_gain": ci_positive,
            "at_least_three_seeds": enough_seeds,
            "diagnostic_only": diagnostic_only,
            "meets_full_success_criterion": bool(mean_pass and ci_positive and enough_seeds and not diagnostic_only),
        },
        "diagnostic_only": diagnostic_only,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="20261023,20261024,20261025")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-negative", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--eval-limit", type=int, default=0,
                        help="Positive value is diagnostic only; 0 evaluates full splits")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--output", default="experiments/fedultra_fresh/results/transfer_comparison.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.rounds < 1 or args.local_steps < 1 or args.batch_size < 2:
        raise ValueError("rounds/local_steps must be positive and batch-size >= 2")
    if args.batch_size % 2:
        raise ValueError("--batch-size must be even")
    if args.eval_batch_size < 1 or args.num_negative < 1 or args.eval_limit < 0:
        raise ValueError("invalid evaluation batch, negative count, or eval limit")
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must contain unique comma-separated integers")
    if not base.CHECKPOINT.is_file():
        raise FileNotFoundError(base.CHECKPOINT)

    output = (ROOT / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists; choose another --output or pass --overwrite")

    torch.set_num_threads(min(2, torch.get_num_threads()))
    checkpoint = torch.load(base.CHECKPOINT, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"Unexpected checkpoint format: {base.CHECKPOINT}")
    checkpoint_state = checkpoint["model"]
    bundles = {client: load_client(client) for client in CLIENTS}

    # Both directions help distinguish general transfer from a one-way dataset-size effect.
    directions = [("enterprise", "ics"), ("ics", "enterprise")]
    all_results = {f"{src}_to_{tgt}": [] for src, tgt in directions}
    for seed in seeds:
        for source, target in directions:
            result = run_direction(source, target, seed, bundles, args, checkpoint_state)
            all_results[f"{source}_to_{target}"].append(result)
            payload = {
                "protocol": "pretrained ULTRA zero-shot vs source-finetuned cross-graph transfer",
                "checkpoint": str(base.CHECKPOINT),
                "source_code_commit": "427966ad8ed60420eef034063d44f3153addff90",
                "message_passing": "target/source training triples only within each graph",
                "model_selection": "source validation MRR only; target validation/test never select checkpoints",
                "primary_metric": "target test MRR transfer delta = source-finetuned minus pretrained zero-shot",
                "per_relation_metrics": "filtered test MRR and Hits@K, grouped by target relation label",
                "acceptance_threshold": 0.03,
                "results": all_results,
                "summary": {key: summarize_direction(value) for key, value in all_results.items() if value},
            }
            output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            print(f"Saved progress: {output}")

    print("\n===== TRANSFER SUMMARY =====")
    print(json.dumps({key: summarize_direction(value) for key, value in all_results.items() if value}, indent=2))
    if args.eval_limit:
        print("DIAGNOSTIC RUN: do not use for acceptance claims.")
    else:
        print("FULL TRANSFER EVALUATION COMPLETE.")


if __name__ == "__main__":
    main()

