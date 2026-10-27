#!/usr/bin/env python3
"""Reproducible ULTRA local-vs-FedAvg experiment on the fresh MITRE clients.

Message passing uses training triples only. Validation filters train+valid; test
filters train+valid+test. Test results are only computed after validation-based
checkpoint selection. A positive --eval-limit is diagnostic only, not a benchmark.
FedAvg resets local AdamW state each round and selects its shared checkpoint by
mean per-client absolute validation-MRR gain from the common pretrained start.
"""
import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch import optim
from torch.nn import functional as F
from ultra import tasks
from ultra.models import Ultra
from evaluation_protocol import CLIENTS, load_client

CHECKPOINT = ROOT / "ckpts" / "ultra_4g.pth"
MODEL_CONFIG = {
    "relation_model": {
        "class": "RelNBFNet", "input_dim": 64,
        "hidden_dims": [64, 64, 64, 64, 64, 64],
        "message_func": "distmult", "aggregate_func": "sum",
        "short_cut": True, "layer_norm": True,
    },
    "entity_model": {
        "class": "EntityNBFNet", "input_dim": 64,
        "hidden_dims": [64, 64, 64, 64, 64, 64],
        "message_func": "distmult", "aggregate_func": "sum",
        "short_cut": True, "layer_norm": True,
    },
}
METRICS = ("mrr", "hits@1", "hits@3", "hits@10")
# Two-sided 95% Student-t critical values for small samples, indexed by df.
T_CRITICAL_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045,
}


def new_model(checkpoint_state=None):
    model = Ultra(
        rel_model_cfg=copy.deepcopy(MODEL_CONFIG["relation_model"]),
        entity_model_cfg=copy.deepcopy(MODEL_CONFIG["entity_model"]),
    )
    if checkpoint_state is None:
        state = torch.load(CHECKPOINT, map_location="cpu")
        if not isinstance(state, dict) or "model" not in state:
            raise ValueError(f"Unexpected checkpoint format: {CHECKPOINT}")
        checkpoint_state = state["model"]
    model.load_state_dict(checkpoint_state, strict=True)
    return model


def clone_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def average_states(states, weights):
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("FedAvg weights must sum to a positive number")
    result = {}
    keys = states[0].keys()
    for key in keys:
        values = [state[key] for state in states]
        if torch.is_floating_point(values[0]) or torch.is_complex(values[0]):
            acc = torch.zeros_like(values[0])
            for value, weight in zip(values, weights):
                acc.add_(value, alpha=float(weight) / total)
            result[key] = acc
        else:
            result[key] = values[0].clone()
    return result


def triples_tensor(data):
    return torch.stack(
        [data.target_edge_index[0], data.target_edge_index[1], data.target_edge_type],
        dim=1,
    )


def train_updates(model, optimizer, train_data, steps, batch_size, num_negative,
                  seed, client_name, round_index):
    if batch_size % 2:
        raise ValueError("--batch-size must be even because ULTRA splits each batch for head/tail negatives")
    # Separate deterministic stream per client and round.
    stream_seed = seed * 1000003 + (0 if client_name == "enterprise" else 7919) + round_index * 97
    # ULTRA's negative sampler may use the global PyTorch RNG. Reset it as well
    # as the explicit positive-sampling generator so local/FedAvg runs use the
    # same stochastic sampling stream for each seed/client/round.
    torch.manual_seed(stream_seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(stream_seed)
    positives = triples_tensor(train_data)
    model.train()
    losses = []
    for _ in range(steps):
        ids = torch.randint(len(positives), (batch_size,), generator=generator)
        batch_pos = positives[ids]
        batch = tasks.negative_sampling(
            train_data, batch_pos, num_negative=num_negative, strict=True
        )
        pred = model(train_data, batch)
        target = torch.zeros_like(pred)
        target[:, 0] = 1
        loss_each = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        weights = torch.ones_like(pred)
        if num_negative > 0:
            with torch.no_grad():
                weights[:, 1:] = F.softmax(pred[:, 1:], dim=-1)
        loss = ((loss_each * weights).sum(dim=-1) / weights.sum(dim=-1)).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{client_name}: non-finite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if not grads or any(not torch.isfinite(g).all().item() for g in grads):
            raise FloatingPointError(f"{client_name}: missing/non-finite gradients")
        optimizer.step()
        losses.append(float(loss.detach()))
    return sum(losses) / len(losses)


@torch.no_grad()
def evaluate(model, bundle, split, eval_batch_size=4, eval_limit=0):
    """Full filtered ranking unless eval_limit > 0 (diagnostic subset)."""
    if split == "valid":
        target_data, filter_graph = bundle["valid_data"], bundle["valid_filter"]
    elif split == "test":
        target_data, filter_graph = bundle["test_data"], bundle["test_filter"]
    else:
        raise ValueError(split)

    positives = triples_tensor(target_data)
    if eval_limit > 0:
        positives = positives[:eval_limit]
    model.eval()
    ranks = []
    for start in range(0, len(positives), eval_batch_size):
        batch = positives[start:start + eval_batch_size]
        # The model input is always the training-only graph.
        train_graph = bundle["train_data"]
        tail_batch, head_batch = tasks.all_negative(train_graph, batch)
        tail_scores = model(train_graph, tail_batch)
        head_scores = model(train_graph, head_batch)
        tail_mask, head_mask = tasks.strict_negative_mask(filter_graph, batch)
        tail_rank = tasks.compute_ranking(tail_scores, batch[:, 1], tail_mask)
        head_rank = tasks.compute_ranking(head_scores, batch[:, 0], head_mask)
        ranks.append(torch.cat([tail_rank, head_rank]).float())
    all_ranks = torch.cat(ranks)
    result = {
        "mrr": float((1.0 / all_ranks).mean()),
        "hits@1": float((all_ranks <= 1).float().mean()),
        "hits@3": float((all_ranks <= 3).float().mean()),
        "hits@10": float((all_ranks <= 10).float().mean()),
        "triples_evaluated": int(len(positives)),
        "ranking_queries": int(len(all_ranks)),
    }
    if not all(math.isfinite(result[m]) for m in METRICS):
        raise FloatingPointError(f"Non-finite {split} metrics: {result}")
    return result


def make_optimizer(model, lr, weight_decay):
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def run_seed(seed, bundles, args, checkpoint_state):
    print(f"\n===== SEED {seed} =====")
    torch.manual_seed(seed)
    random.seed(seed)

    # Local baselines: same pretrained initialization and total updates per client.
    local_models = {}
    local_optimizers = {}
    local_best_states = {}
    local_best_val = {}
    local_best_round = {}
    local_losses = {}
    for client in CLIENTS:
        model = new_model(checkpoint_state).cpu()
        optimizer = make_optimizer(model, args.lr, args.weight_decay)
        local_models[client] = model
        local_optimizers[client] = optimizer
        local_losses[client] = []
        initial_val = evaluate(model, bundles[client], "valid",
                               args.eval_batch_size, args.eval_limit)
        local_best_val[client] = initial_val["mrr"]
        local_best_states[client] = clone_state(model)
        local_best_round[client] = 0
        print(f"local {client} round=0 valid_mrr={initial_val['mrr']:.6f}")

    for round_idx in range(1, args.rounds + 1):
        for client in CLIENTS:
            loss = train_updates(
                local_models[client], local_optimizers[client],
                bundles[client]["train_data"], args.local_steps, args.batch_size,
                args.num_negative, seed, client, round_idx,
            )
            local_losses[client].append(loss)
            val = evaluate(local_models[client], bundles[client], "valid",
                           args.eval_batch_size, args.eval_limit)
            if val["mrr"] > local_best_val[client]:
                local_best_val[client] = val["mrr"]
                local_best_states[client] = clone_state(local_models[client])
                local_best_round[client] = round_idx
            print(f"local {client} round={round_idx} loss={loss:.6f} "
                  f"valid_mrr={val['mrr']:.6f} best_round={local_best_round[client]}")

    local_test = {}
    for client in CLIENTS:
        local_models[client].load_state_dict(local_best_states[client], strict=True)
        local_test[client] = evaluate(local_models[client], bundles[client], "test",
                                      args.eval_batch_size, args.eval_limit)
        print(f"LOCAL TEST {client}: " + json.dumps(local_test[client], sort_keys=True))

    # Federated model and client-side optimizers. Client optimizer state remains
    # local across rounds; only model weights are aggregated.
    global_model = new_model(checkpoint_state).cpu()
    global_state = clone_state(global_model)
    client_models = {c: new_model(global_state).cpu() for c in CLIENTS}
    client_optimizers = {
        c: make_optimizer(client_models[c], args.lr, args.weight_decay) for c in CLIENTS
    }
    train_weights = [len(bundles[c]["raw_triples"]["train"]) for c in CLIENTS]
    initial_val = {
        c: evaluate(global_model, bundles[c], "valid", args.eval_batch_size, args.eval_limit)
        for c in CLIENTS
    }
    # Select one shared checkpoint by equal-weighted absolute MRR gain from
    # the same pretrained starting point on each client. Averaging raw MRRs
    # would let the client with the larger baseline MRR dominate selection.
    best_global_score = 0.0
    best_global_state = clone_state(global_model)
    best_global_round = 0
    initial_mean_val = sum(initial_val[c]["mrr"] for c in CLIENTS) / len(CLIENTS)
    print(f"fedavg round=0 mean_valid_mrr={initial_mean_val:.6f} "
          f"mean_gain_from_pretrained={best_global_score:.6f}")

    fed_losses = {c: [] for c in CLIENTS}
    for round_idx in range(1, args.rounds + 1):
        states = []
        round_losses = []
        for client in CLIENTS:
            client_models[client].load_state_dict(global_state, strict=True)
            # FedAvg clients start each round from the same global weights.
            # Reset AdamW moments too; retaining moments from a discarded local
            # model mixes optimizer history with newly aggregated parameters.
            client_optimizers[client] = make_optimizer(
                client_models[client], args.lr, args.weight_decay
            )
            loss = train_updates(
                client_models[client], client_optimizers[client],
                bundles[client]["train_data"], args.local_steps, args.batch_size,
                args.num_negative, seed, client, round_idx,
            )
            fed_losses[client].append(loss)
            round_losses.append(loss)
            states.append(clone_state(client_models[client]))
        global_state = average_states(states, train_weights)
        global_model.load_state_dict(global_state, strict=True)
        val = {
            c: evaluate(global_model, bundles[c], "valid", args.eval_batch_size, args.eval_limit)
            for c in CLIENTS
        }
        mean_val = sum(val[c]["mrr"] for c in CLIENTS) / len(CLIENTS)
        mean_gain = sum(
            val[c]["mrr"] - initial_val[c]["mrr"] for c in CLIENTS
        ) / len(CLIENTS)
        if mean_gain > best_global_score:
            best_global_score = mean_gain
            best_global_state = clone_state(global_model)
            best_global_round = round_idx
        print(f"fedavg round={round_idx} mean_loss={sum(round_losses)/len(round_losses):.6f} "
              f"valid_mrr_enterprise={val['enterprise']['mrr']:.6f} "
              f"valid_mrr_ics={val['ics']['mrr']:.6f} mean_valid_mrr={mean_val:.6f} "
              f"mean_gain_from_pretrained={mean_gain:.6f} "
              f"best_round={best_global_round}")

    global_model.load_state_dict(best_global_state, strict=True)
    fed_test = {
        c: evaluate(global_model, bundles[c], "test", args.eval_batch_size, args.eval_limit)
        for c in CLIENTS
    }
    for client in CLIENTS:
        print(f"FEDAVG TEST {client}: " + json.dumps(fed_test[client], sort_keys=True))

    per_client_delta = {
        c: fed_test[c]["mrr"] - local_test[c]["mrr"] for c in CLIENTS
    }
    mean_delta = sum(per_client_delta.values()) / len(per_client_delta)
    return {
        "seed": seed,
        "config": {
            "rounds": args.rounds, "local_steps_per_round": args.local_steps,
            "updates_per_client": args.rounds * args.local_steps,
            "batch_size": args.batch_size, "num_negative": args.num_negative,
            "lr": args.lr, "weight_decay": args.weight_decay,
            "eval_limit": args.eval_limit,
        },
        "local": {
            "selected_round": local_best_round,
            "validation_mrr": local_best_val,
            "test": local_test,
        },
        "fedavg": {
            "selected_round": best_global_round,
            "selection_metric": "equal-weighted mean absolute validation MRR gain from pretrained checkpoint",
            "initial_validation_mrr": {c: initial_val[c]["mrr"] for c in CLIENTS},
            "validation_mean_mrr": sum(initial_val[c]["mrr"] for c in CLIENTS) / len(CLIENTS) + best_global_score,
            "validation_mean_gain_from_pretrained": best_global_score,
            "test": fed_test,
        },
        "delta_mrr_fedavg_minus_local": per_client_delta,
        "mean_client_delta_mrr": mean_delta,
        "diagnostic_only": args.eval_limit > 0,
    }


def summarize(seed_results):
    deltas = [r["mean_client_delta_mrr"] for r in seed_results]
    n = len(deltas)
    mean = sum(deltas) / n
    if n >= 2:
        variance = sum((x - mean) ** 2 for x in deltas) / (n - 1)
        se = math.sqrt(variance / n)
        tcrit = T_CRITICAL_95.get(n - 1, 1.96)
        ci = [mean - tcrit * se, mean + tcrit * se]
    else:
        ci = None
    client_means = {}
    for client in CLIENTS:
        vals = [r["delta_mrr_fedavg_minus_local"][client] for r in seed_results]
        client_means[client] = sum(vals) / len(vals)
    substantial_regression_cutoff = -0.02
    return {
        "num_seeds": n,
        "seed_mean_client_delta_mrr": deltas,
        "mean_mrr_delta": mean,
        "paired_seed_level_95pct_t_ci": ci,
        "mean_delta_by_client": client_means,
        "acceptance_criteria": {
            "minimum_mean_gain": 0.03,
            "mean_gain_meets_threshold": mean >= 0.03,
            "ci_excludes_zero": bool(ci and (ci[0] > 0 or ci[1] < 0)),
            "ci_supports_positive_gain": bool(ci and ci[0] > 0),
            "positive_mean_gain": mean > 0,
            "at_least_three_seeds": n >= 3,
            "no_client_mean_regression_below_minus_0_02": all(
                value >= substantial_regression_cutoff for value in client_means.values()
            ),
            "substantial_regression_cutoff": substantial_regression_cutoff,
        },
        "diagnostic_only": any(r["diagnostic_only"] for r in seed_results),
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", default="20261023,20261024,20261025",
                   help="Comma-separated random seeds (default: three seeds)")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--local-steps", type=int, default=10,
                   help="Optimizer updates per client per round")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-negative", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=4)
    p.add_argument("--eval-limit", type=int, default=0,
                   help="Diagnostic subset size per split; 0 means full evaluation")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--output", default="experiments/fedultra_fresh/results/fedavg_comparison.json")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.rounds < 1 or args.local_steps < 1 or args.batch_size < 2 or args.eval_batch_size < 1:
        raise ValueError("rounds/local_steps/eval_batch_size must be positive and batch_size >= 2")
    if args.batch_size % 2:
        raise ValueError("--batch-size must be even")
    if args.num_negative < 1 or args.eval_limit < 0:
        raise ValueError("--num-negative must be positive and --eval-limit non-negative")
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must contain unique comma-separated integers")
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {CHECKPOINT}")

    output = (ROOT / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} already exists; choose --output or pass --overwrite")
    torch.set_num_threads(min(2, torch.get_num_threads()))
    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"Unexpected checkpoint format: {CHECKPOINT}")
    checkpoint_state = checkpoint["model"]

    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Output: {output}")
    print(f"Seeds: {seeds}")
    print(f"Budget: {args.rounds} rounds x {args.local_steps} updates/client/round")
    print(f"Evaluation: {'DIAGNOSTIC SUBSET' if args.eval_limit else 'FULL'}")
    if args.eval_limit:
        print("WARNING: subset evaluation is diagnostic only; do not use it for acceptance claims.")
    bundles = {client: load_client(client) for client in CLIENTS}
    seed_results = []
    for seed in seeds:
        result = run_seed(seed, bundles, args, checkpoint_state)
        seed_results.append(result)
        payload = {
            "protocol": "ULTRA pretrained checkpoint; local fine-tuning vs FedAvg",
            "checkpoint": str(CHECKPOINT),
            "source_code_commit": "427966ad8ed60420eef034063d44f3153addff90",
            "runner_revision": "audit-fixed: round-local AdamW reset, balanced validation selection, paired sampling RNG",
            "client_weighting": "training triple count",
            "fedavg_optimizer": "AdamW state reset each round after global model reload",
            "fedavg_checkpoint_selection": "equal-weighted mean absolute validation MRR gain from pretrained checkpoint",
            "sampling": "deterministic per seed/client/round for positive and negative sampling RNG",
            "message_passing": "training triples only",
            "validation_filter": "train + validation facts",
            "test_filter": "train + validation + test facts",
            "results": seed_results,
            "summary": summarize(seed_results),
        }
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Saved progress: {output}")

    print("\n===== SUMMARY =====")
    print(json.dumps(summarize(seed_results), indent=2))
    if args.eval_limit:
        print("DIAGNOSTIC RUN COMPLETE: do not interpret these values as benchmark results.")
    else:
        print("FULL EVALUATION COMPLETE: inspect CI and each acceptance criterion before drawing conclusions.")


if __name__ == "__main__":
    main()

