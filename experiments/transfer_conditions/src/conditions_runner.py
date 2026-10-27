#!/usr/bin/env python3
"""Local fine-tuning or two-client FedAvg for one target client (transfer-conditions study).

Both arms start from the pretrained ULTRA checkpoint and give the target the same
number of updates (rounds x local steps) from the same sampling stream. Each arm
keeps the round (0 = pretrained) with the best target validation MRR, and target
test is scored once, for that round only. FedAvg weights the two clients equally
and resets AdamW each round, as in fedavg_runner.py; the local arm keeps its AdamW
state, as fedavg_runner.py's local baseline does. --skip-test scores validation
only (sanity checks). A positive --eval-limit is diagnostic only.
"""
import argparse
import json
import time
from pathlib import Path

import torch

from common import ROOT, load_bundle, sha256
import fedavg_runner as base
import transfer_runner as transfer
from evaluation_protocol import load_client

PROTOCOL_VERSION = "transfer-conditions v1 (pre-registered 2026-10-26)"


def client(spec):
    """'enterprise' means the frozen fresh-FedULTRA Enterprise split; anything else is a folder."""
    if spec == "enterprise":
        return load_client("enterprise")
    return load_bundle(Path(spec).expanduser().resolve())


def validation_mrr(model, target, args):
    return base.evaluate(model, target, "valid", args.eval_batch_size, args.eval_limit)["mrr"]


def run_local(seed, target, args, checkpoint_state):
    model = base.new_model(checkpoint_state)
    optimizer = base.make_optimizer(model, args.lr, args.weight_decay)
    history = [validation_mrr(model, target, args)]
    best_state, best_round = base.clone_state(model), 0
    for round_idx in range(1, args.rounds + 1):
        base.train_updates(model, optimizer, target["train_data"], args.local_steps,
                           args.batch_size, args.num_negative, seed, "target", round_idx)
        history.append(validation_mrr(model, target, args))
        if history[-1] > history[best_round]:
            best_state, best_round = base.clone_state(model), round_idx
        print(f"  local seed={seed} round={round_idx} valid_mrr={history[-1]:.4f} best_round={best_round}", flush=True)
    model.load_state_dict(best_state, strict=True)
    return model, best_round, history


def run_fedavg(seed, target, source, args, checkpoint_state):
    global_model = base.new_model(checkpoint_state)
    global_state = base.clone_state(global_model)
    worker = base.new_model(checkpoint_state)
    history = [validation_mrr(global_model, target, args)]
    best_state, best_round = global_state, 0
    for round_idx in range(1, args.rounds + 1):
        states = []
        # The target trains on the same stream as the local arm; the source on its own.
        for name, bundle, stream_seed in (("target", target, seed), ("source", source, seed + 1)):
            worker.load_state_dict(global_state, strict=True)
            optimizer = base.make_optimizer(worker, args.lr, args.weight_decay)
            base.train_updates(worker, optimizer, bundle["train_data"], args.local_steps,
                               args.batch_size, args.num_negative, stream_seed, "target", round_idx)
            states.append(base.clone_state(worker))
        global_state = base.average_states(states, [1.0, 1.0])
        global_model.load_state_dict(global_state, strict=True)
        history.append(validation_mrr(global_model, target, args))
        if history[-1] > history[best_round]:
            best_state, best_round = global_state, round_idx
        print(f"  fedavg seed={seed} round={round_idx} valid_mrr={history[-1]:.4f} best_round={best_round}", flush=True)
    global_model.load_state_dict(best_state, strict=True)
    return global_model, best_round, history


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=("local", "fedavg"), required=True)
    p.add_argument("--target", required=True, help="Target client folder")
    p.add_argument("--source", help="Source client folder, or 'enterprise' (fedavg only)")
    p.add_argument("--seeds", required=True, help="Comma-separated training seeds")
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--local-steps", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-negative", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=4)
    p.add_argument("--eval-limit", type=int, default=0)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--skip-test", action="store_true", help="Validation only (sanity checks)")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if (args.arm == "fedavg") != bool(args.source):
        raise ValueError("--source is required for fedavg and not allowed for local")
    if args.batch_size % 2 or args.batch_size < 2 or args.rounds < 1 or args.local_steps < 1:
        raise ValueError("invalid budget or batch size")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must be unique")

    torch.set_num_threads(args.threads)
    checkpoint_state = torch.load(base.CHECKPOINT, map_location="cpu")["model"]
    target = client(args.target)
    source = client(args.source) if args.source else None
    target_dir = Path(target["folder"])
    config = {k: getattr(args, k) for k in ("rounds", "local_steps", "batch_size", "num_negative",
                                            "eval_limit", "lr", "weight_decay", "skip_test")}
    output = args.output.expanduser().resolve()
    payload = json.loads(output.read_text()) if output.exists() else None
    if payload and (payload["config"] != config or payload["target"] != str(target_dir)
                    or payload["source"] != args.source or payload["arm"] != args.arm):
        raise FileExistsError(f"{output} exists with a different configuration")
    if not payload:
        payload = {
            "protocol": PROTOCOL_VERSION, "arm": args.arm, "target": str(target_dir), "source": args.source,
            "config": config, "client_weights": [1.0, 1.0] if args.arm == "fedavg" else None,
            "checkpoint": str(base.CHECKPOINT),
            "target_sha256": {f.name: sha256(f) for f in sorted(target_dir.glob("*.txt"))},
            "selection": "best round (0 = pretrained) by target validation MRR",
            "diagnostic_only": args.eval_limit > 0, "results": [],
        }
    done = {r["seed"] for r in payload["results"]}
    for seed in seeds:
        if seed in done:
            print(f"seed {seed} already in {output}; skipping")
            continue
        start = time.time()
        if args.arm == "local":
            model, best_round, history = run_local(seed, target, args, checkpoint_state)
        else:
            model, best_round, history = run_fedavg(seed, target, source, args, checkpoint_state)
        result = {"seed": seed, "selected_round": best_round, "validation_mrr_history": history,
                  "selected_validation_mrr": history[best_round]}
        if not args.skip_test:
            test, per_query, by_relation = transfer.evaluate_with_query_mrr(
                model, target, "test", args.eval_batch_size, args.eval_limit)
            result.update(test=test, test_by_relation=by_relation, per_query_test=per_query)
        result["seconds"] = round(time.time() - start, 1)
        payload["results"].append(result)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
        print(f"{args.arm} seed={seed} selected_round={best_round} valid={history[best_round]:.4f} "
              + (f"test_mrr={result['test']['mrr']:.4f} " if not args.skip_test else "")
              + f"({result['seconds']}s) -> {output}", flush=True)


if __name__ == "__main__":
    main()
