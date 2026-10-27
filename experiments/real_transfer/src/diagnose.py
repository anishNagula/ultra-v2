#!/usr/bin/env python3
"""Phase-2 diagnostics on the C-time benchmark. These score target validation only; test is never scored.

  --fraction sets the target fraction scored (default 0.10) and the source's sparse-view fraction.

  oracle_labels   Upper-bound probe. Target f graph for message passing, but the target's hidden
                  facts are added as positives (and as known facts for negative filtering). The
                  result shows how much better ULTRA's weights can do on the sparse graph given ~6x
                  more target facts (~10x more actor labels). ORACLE: it uses hidden facts and is
                  not a method.
  dense_weights   Train locally on the full target graph (f = 1.00, the registered local arm). Score the
                  weights on both the f = 1.00 and the f = 0.10 graphs. This tests whether weights trained
                  with dense evidence suit the sparse graph. ORACLE: f = 1.00 contains the hidden facts.
  source_sparse   Source-only training on sparse per-head views of the source's own graph
                  (the --fraction), predicting held-out actor `uses` facts. Scored zero-shot on
                  target validation at that fraction.
  source_dense    The same, but on the complete source graph: the regime the FedAvg source trained in.
  source_select   The source stage of run_method.py's sparse_source arm (source validation held out and
                  scored every round), unchanged, plus target validation after every round. Same seed,
                  same trajectory as that arm. It compares source-side and target-side round choices.
  fedavg_replay   The registered FedAvg arm (conditions_runner.run_fedavg), rebuilt here. Its validation
                  curve must match the registered run of the same seed; this checks the rebuild.
  fedavg_sparse   The same FedAvg, with one change: each round the source's 10 local steps use a fresh
                  per-head view of its graph (the --fraction) and the held-out facts of every relation
                  as positives, instead of the complete graph. The registered source positives (every
                  relation), sampling stream, client weights and target updates are unchanged.

  python diagnose.py --diagnostic source_sparse --seeds 20261401,20261402 --output ../results/diagnostics/source_sparse.json
"""
import argparse
import json
import time
from pathlib import Path

import torch

from env import SOURCE, TARGET, is_actor_use
import fedavg_runner as base
from common import load_bundle
import sparse

DIAGNOSTICS = ("oracle_labels", "dense_weights", "source_sparse", "source_dense", "source_select", "fedavg_replay",
               "fedavg_sparse")


def valid_mrr(model, bundle):
    return base.evaluate(model, bundle, "valid", 4, 0)["mrr"]


def fedavg(seed, args, ckpt, tgt, src, curve, losses):
    """conditions_runner.run_fedavg (weights [1, 1]); fedavg_sparse changes only the source's message graph."""
    global_model, worker = base.new_model(ckpt), base.new_model(ckpt)
    global_state = base.clone_state(global_model)
    key = f"target_f{args.fraction}_valid"
    for r in range(1, args.rounds + 1):
        worker.load_state_dict(global_state, strict=True)
        optimizer = base.make_optimizer(worker, args.lr, 0.0)
        base.train_updates(worker, optimizer, tgt["train_data"], args.steps, 8, 64, seed, "target", r)
        states = [base.clone_state(worker)]
        worker.load_state_dict(global_state, strict=True)
        optimizer = base.make_optimizer(worker, args.lr, 0.0)
        if args.diagnostic == "fedavg_replay":
            losses.append(base.train_updates(worker, optimizer, src["train_data"], args.steps, 8, 64, seed + 1, "target", r))
        else:
            keep, held = sparse.per_head_view(src["raw_triples"]["train"], float(args.fraction), key=f"fedavg:{seed}:{r}")
            losses.append(sparse.train_steps(worker, optimizer, sparse.message_graph(keep, src), src["train_data"],
                                             sparse.fact_tensor(held, src), args.steps, 8, 64,
                                             (seed + 1) * 1000003 + 7919 + r * 97))  # the registered source stream
        states.append(base.clone_state(worker))
        global_state = base.average_states(states, [1.0, 1.0])
        global_model.load_state_dict(global_state, strict=True)
        curve[key].append(valid_mrr(global_model, tgt))
        print(f"  {args.diagnostic} seed={seed} round={r} source_loss={losses[-1]:.4f} valid={curve[key][-1]:.4f}", flush=True)


def run(seed, args, ckpt, bundles):
    model = base.new_model(ckpt)
    tgt = bundles[args.fraction]
    key = f"target_f{args.fraction}_valid"
    curve = {key: [valid_mrr(model, tgt)]}
    losses = []
    if args.diagnostic == "oracle_labels":
        facts = tgt["raw_triples"]["train"] + tgt["hidden"]
        positives = sparse.fact_tensor(facts, tgt)
        negatives_from = sparse.filter_graph(facts, tgt)
        optimizer = base.make_optimizer(model, args.lr, 0.0)
        for r in range(1, args.rounds + 1):
            losses.append(sparse.train_steps(model, optimizer, tgt["train_data"], negatives_from, positives,
                                             args.steps, 8, 64, seed * 1000003 + 7919 + r * 97))
            curve[key].append(valid_mrr(model, tgt))
            print(f"  {args.diagnostic} seed={seed} round={r} loss={losses[-1]:.4f} "
                  f"valid={curve[key][-1]:.4f}", flush=True)
    elif args.diagnostic == "dense_weights":
        full = bundles["1.00"]
        curve["target_f1.00_valid"] = [valid_mrr(model, full)]
        optimizer = base.make_optimizer(model, args.lr, 0.0)
        for r in range(1, args.rounds + 1):
            losses.append(base.train_updates(model, optimizer, full["train_data"], args.steps, 8, 64, seed, "target", r))
            curve["target_f1.00_valid"].append(valid_mrr(model, full))
            curve[key].append(valid_mrr(model, tgt))
            print(f"  {args.diagnostic} seed={seed} round={r} loss={losses[-1]:.4f} "
                  f"valid f1.00={curve['target_f1.00_valid'][-1]:.4f} f{args.fraction}={curve[key][-1]:.4f}", flush=True)
    elif args.diagnostic.startswith("fedavg"):
        fedavg(seed, args, ckpt, tgt, bundles["source"], curve, losses)
    else:
        def on_round(r, loss):
            losses.append(loss)
            curve[key].append(valid_mrr(model, tgt))
            print(f"  {args.diagnostic} seed={seed} round={r} loss={loss:.4f} "
                  f"target f{args.fraction} valid (zero-shot)={curve[key][-1]:.4f}", flush=True)
        record = sparse.train_source(model, bundles["source"],
                                     fraction=1.0 if args.diagnostic == "source_dense" else float(args.fraction),
                                     rounds=args.rounds, steps=args.steps, batch_size=8, num_negative=64, seed=seed,
                                     lr=args.lr, weight_decay=0.0, task=is_actor_use,
                                     select=args.diagnostic == "source_select", on_round=on_round)
        if args.diagnostic == "source_select":
            curve["source_valid"] = record["source_valid_mrr"]
    return {"seed": seed, "curves": curve, "losses": losses,
            "best": {k: max(v) for k, v in curve.items()},
            "best_round": {k: max(range(len(v)), key=v.__getitem__) for k, v in curve.items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--diagnostic", choices=DIAGNOSTICS, required=True)
    p.add_argument("--seeds", required=True)
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--fraction", choices=("0.10", "0.25"), default="0.10",
                   help="target fraction scored (and the source's sparse-view fraction)")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    ckpt = torch.load(base.CHECKPOINT, map_location="cpu")["model"]
    bundles = {args.fraction: load_bundle(TARGET[args.fraction])}
    if args.diagnostic == "dense_weights":
        bundles["1.00"] = load_bundle(TARGET["1.00"])
    if args.diagnostic.startswith(("source", "fedavg")):
        bundles["source"] = load_bundle(SOURCE)
    out = {"diagnostic": args.diagnostic, "config": vars(args) | {"output": str(args.output)},
           "note": "validation only; test is never scored", "results": []}
    for seed in (int(s) for s in args.seeds.split(",")):
        start = time.time()
        res = run(seed, args, ckpt, bundles)
        res["seconds"] = round(time.time() - start, 1)
        out["results"].append(res)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
        print(f"{args.diagnostic} seed={seed} best={res['best']} at {res['best_round']} ({res['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
