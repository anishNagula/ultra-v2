#!/usr/bin/env python3
"""One arm of the real-transfer experiment on a C-time target.

  local          The registered local arm (transfer_conditions/src/conditions_runner.py::run_local),
                 starting from the pretrained checkpoint.
  sparse_source  Two stages, and only model weights cross from source to target.
                 1. The source client trains alone on sparse per-head views of its own graph
                    (src/sparse.py::train_source), learning to predict held-out actor `uses` facts.
                    It keeps its final round (--source-checkpoint final, the registered rule). Source
                    validation (a fixed 10% of its actor `uses` facts) is scored and recorded every round.
                    The pilot's rule a, --source-checkpoint source_valid, keeps the best such round instead.
                 2. The target runs the same registered local arm, starting from the source's weights.
  dense_source   Control for sparse_source: the source stage uses its complete graph.

The arms share the target stage: code, budget (rounds x local steps), sampling stream, validation-only
round selection (round 0 = the starting weights) and one test scoring. They differ only in the
weights the target starts from. Each seed's record also scores those starting weights on test with
no target training (zero-shot). --skip-test scores validation only (pilot).

The source stage never reads target data, so its weights depend only on the source configuration
and seed. --source-cache stores them, and another target fraction reuses them after checking the
configuration and the weights' SHA-256. --source-only fills the cache without running the target stage.

  python run_method.py --arm sparse_source --target ../../enterprise_split/data/v1/C-time/target_f0.10 \
      --seeds 20261401 --skip-test --output ../results/pilot/sparse_source_f0.10.json
"""
import argparse
import hashlib
import io
import json
import os
import time
from pathlib import Path

import torch

from env import ROOT, SOURCE, is_actor_use
import fedavg_runner as base
import transfer_runner as transfer
import conditions_runner as runner
from common import load_bundle, sha256
import sparse

PROTOCOL_VERSION = "real-transfer v1"
ARMS = ("local", "sparse_source", "dense_source")
SOURCE_TASK = "held-out actor `uses` facts"


def state_sha256(state):
    buffer = io.BytesIO()
    torch.save({k: state[k] for k in sorted(state)}, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def source_fraction(arm, args):
    return args.source_fraction if arm == "sparse_source" else 1.0


def source_key(arm, seed, args):
    """Everything the source stage's weights depend on."""
    return {"protocol": PROTOCOL_VERSION, "seed": seed, "source_fraction": source_fraction(arm, args),
            "source_rounds": args.source_rounds, "source_checkpoint": args.source_checkpoint,
            "local_steps": args.local_steps, "batch_size": args.batch_size,
            "num_negative": args.num_negative, "lr": args.lr, "weight_decay": args.weight_decay,
            "source_task": SOURCE_TASK, "threads": args.threads,
            "source_sha256": sha256(SOURCE / "train.txt"), "checkpoint_sha256": sha256(base.CHECKPOINT)}


def starting_state(arm, seed, source, args, checkpoint_state):
    """Weights the target starts from, plus a record of how they were produced."""
    if arm == "local":
        return checkpoint_state, {"stage": "pretrained checkpoint"}
    key = source_key(arm, seed, args)
    cached = None
    if args.source_cache:
        digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:24]
        cached = Path(args.source_cache).expanduser().resolve() / f"source_{digest}.pt"
        if cached.exists():
            saved = torch.load(cached, map_location="cpu")
            if saved["key"] != key or state_sha256(saved["state"]) != saved["record"]["state_sha256"]:
                raise ValueError(f"{cached} does not match this source configuration")
            return saved["state"], {**saved["record"], "loaded_from_cache": str(cached)}
    model = base.new_model(checkpoint_state)
    start = time.time()
    record = sparse.train_source(model, source, fraction=source_fraction(arm, args), rounds=args.source_rounds,
                                 steps=args.local_steps, batch_size=args.batch_size, num_negative=args.num_negative,
                                 seed=seed, lr=args.lr, weight_decay=args.weight_decay, task=is_actor_use,
                                 select=True, checkpoint=args.source_checkpoint)
    state = base.clone_state(model)
    rule = "final round" if args.source_checkpoint == "final" else "round chosen on source validation"
    record = {"stage": f"source-only training ({arm}), {rule}", "source_key": key,
              **record, "source_seconds": round(time.time() - start, 1), "state_sha256": state_sha256(state)}
    if cached:
        cached.parent.mkdir(parents=True, exist_ok=True)
        partial = cached.with_name(f"{cached.name}.{os.getpid()}.partial")
        torch.save({"key": key, "state": state, "record": record}, partial)
        partial.replace(cached)
    return state, record


def score(model, state, target, args, split="test"):
    """Fine-tuned (the validation-selected round, `model`) and zero-shot (`state`, no target training) on `split`."""
    fine, per_query, by_relation = transfer.evaluate_with_query_mrr(
        model, target, split, args.eval_batch_size, args.eval_limit)
    zero, z_per_query, z_by_relation = transfer.evaluate_with_query_mrr(
        base.new_model(state), target, split, args.eval_batch_size, args.eval_limit)
    return {"test": fine, "test_by_relation": by_relation, "per_query_test": per_query,
            "zero_shot": {"test": zero, "test_by_relation": z_by_relation, "per_query_test": z_per_query}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--target", type=Path, required=True)
    p.add_argument("--seeds", required=True)
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--local-steps", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-negative", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=4)
    p.add_argument("--eval-limit", type=int, default=0)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--source-rounds", type=int, default=30, help="source-stage rounds (x --local-steps steps)")
    p.add_argument("--source-fraction", type=float, default=0.10, help="per-head fraction kept in a sparse view")
    p.add_argument("--source-checkpoint", choices=("final", "source_valid"), default="final",
                   help="source weights handed to the target: the final round (registered) or the best "
                        "source-validation round (the pilot's rule a)")
    p.add_argument("--source-cache", type=Path, help="directory for source-stage weights, reused across targets")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--skip-test", action="store_true")
    p.add_argument("--source-only", action="store_true", help="only train and cache the source stage")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must be unique")
    if args.source_only and (args.arm == "local" or not args.source_cache):
        p.error("--source-only needs a source arm and --source-cache")
    if not args.source_only and not args.output:
        p.error("--output is required")

    torch.set_num_threads(args.threads)
    checkpoint_state = torch.load(base.CHECKPOINT, map_location="cpu")["model"]
    source = load_bundle(SOURCE) if args.arm != "local" else None
    if args.source_only:
        for seed in seeds:
            stage = starting_state(args.arm, seed, source, args, checkpoint_state)[1]
            print(f"{args.arm} seed={seed} source stage {'already cached' if 'loaded_from_cache' in stage else 'cached'}: "
                  f"source round {stage['selected_round']} ({stage['source_seconds']}s)", flush=True)
        return
    target = load_bundle(args.target.expanduser().resolve())
    target_dir = Path(target["folder"])
    config = {k: getattr(args, k) for k in ("rounds", "local_steps", "batch_size", "num_negative", "eval_limit",
                                            "lr", "weight_decay", "skip_test", "threads")}
    if args.arm != "local":
        config |= {"source_rounds": args.source_rounds, "source_task": SOURCE_TASK,
                   "source_fraction": source_fraction(args.arm, args), "source_checkpoint": args.source_checkpoint}
    output = args.output.expanduser().resolve()
    payload = json.loads(output.read_text()) if output.exists() else None
    if payload and (payload["config"] != config or payload["target"] != str(target_dir) or payload["arm"] != args.arm):
        raise FileExistsError(f"{output} exists with a different configuration")
    if not payload:
        payload = {
            "protocol": PROTOCOL_VERSION, "arm": args.arm, "target": str(target_dir), "config": config,
            "checkpoint": str(base.CHECKPOINT.relative_to(ROOT)), "checkpoint_sha256": sha256(base.CHECKPOINT),
            "target_sha256": {f.name: sha256(f) for f in sorted(target_dir.glob("*.txt"))},
            "source_sha256": sha256(SOURCE / "train.txt") if args.arm != "local" else None,
            "shared_between_clients": "model weights only, once, source to target" if args.arm != "local" else None,
            "selection": "best target round (0 = starting weights) by target validation MRR",
            "source_selection": None if args.arm == "local" else (
                "final source round; source validation MRR recorded every round" if args.source_checkpoint == "final"
                else "best source round (0 = pretrained) by source validation MRR, scored in the regime trained for"),
            "diagnostic_only": args.eval_limit > 0, "results": [],
        }
    done = {r["seed"] for r in payload["results"]}
    for seed in seeds:
        if seed in done:
            print(f"seed {seed} already in {output}; skipping")
            continue
        start = time.time()
        state, stage = starting_state(args.arm, seed, source, args, checkpoint_state)
        model, best_round, history = runner.run_local(seed, target, args, state)
        result = {"seed": seed, "starting_weights": stage, "selected_round": best_round,
                  "validation_mrr_history": history, "selected_validation_mrr": history[best_round],
                  "zero_shot_validation_mrr": history[0]}
        if not args.skip_test:
            result.update(score(model, state, target, args))
        result["seconds"] = round(time.time() - start, 1)
        payload["results"].append(result)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
        # Test metrics are written to the output file only, so progress logs never show them.
        print(f"{args.arm} seed={seed} start_valid={history[0]:.4f} selected_round={best_round} "
              f"valid={history[best_round]:.4f} test={'scored' if not args.skip_test else 'skipped'} "
              f"({result['seconds']}s) -> {output}", flush=True)


if __name__ == "__main__":
    main()
