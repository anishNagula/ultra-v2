#!/usr/bin/env python3
"""Validation-only probe: how far ULTRA can see on each target graph.

For every validation fact (h, r, t), this records the hop distance between h and t in the
undirected training graph. It scores the pretrained model and a per-relation popularity ranker
(pessimistic filtered ranks, the registered filters) and reports MRR by distance band.
ULTRA's entity model has 6 message-passing layers. An entity outside the head's 6-hop
neighbourhood receives the same representation as every other such entity, so its score is
tied with all of them, and pessimistic ranking puts it behind the whole tie.

  python reachability.py --output ../results/diagnostics/reachability.json
"""
import argparse
import json
from collections import Counter, defaultdict, deque
from pathlib import Path

import torch

from env import TARGET
import fedavg_runner as base
import transfer_runner as transfer
from common import load_bundle

BANDS = ((1, 2), (3, 3), (4, 6), (7, None))  # hop bands; (7, None) = more than 6 hops or disconnected


def hop_distance(adj, a, b, limit=12):
    if a == b:
        return 0
    seen, frontier, d = {a}, deque([a]), 0
    while frontier and d < limit:
        d += 1
        for _ in range(len(frontier)):
            for n in adj[frontier.popleft()]:
                if n == b:
                    return d
                if n not in seen:
                    seen.add(n)
                    frontier.append(n)
    return None


def popularity_mrr(bundle, split="valid"):
    """Rank tails of (h, r, ?) by how often they are tails of r in train, heads of (?, r, t) likewise."""
    train = bundle["raw_triples"]["train"]
    tail_pop = Counter((r, t) for _, r, t in train)
    head_pop = Counter((r, h) for h, r, _ in train)
    known = set(train) | set(bundle["hidden"]) | set(bundle["raw_triples"]["valid"])
    if split == "test":
        known |= set(bundle["raw_triples"]["test"])
    entities = list(bundle["entity_to_id"])
    out = []
    for h, r, t in bundle["raw_triples"][split]:
        def rank(true, score_of, is_known):
            s = score_of(true)
            return 1 + sum(1 for e in entities if e != true and not is_known(e) and score_of(e) >= s)
        rt = rank(t, lambda e: tail_pop[(r, e)], lambda e: (h, r, e) in known)
        rh = rank(h, lambda e: head_pop[(r, e)], lambda e: (e, r, t) in known)
        out.append((1 / rt + 1 / rh) / 2)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    model = base.new_model(torch.load(base.CHECKPOINT, map_location="cpu")["model"])
    out = {"note": "validation only; pretrained checkpoint (round 0 of every registered run)", "fractions": {}}
    for f, folder in TARGET.items():
        bundle = load_bundle(folder)
        adj = defaultdict(set)
        for h, _, t in bundle["raw_triples"]["train"]:
            adj[h].add(t)
            adj[t].add(h)
        facts = bundle["raw_triples"]["valid"]
        dist = [hop_distance(adj, h, t) for h, _, t in facts]
        _, per_query, _ = transfer.evaluate_with_query_mrr(model, bundle, "valid", 4, 0)
        ultra = [q["mrr"] for q in per_query]
        pop = popularity_mrr(bundle)
        bands = {}
        for lo, hi in BANDS:
            idx = [i for i, d in enumerate(dist) if (d is not None and d >= lo and (hi is None or d <= hi))
                   or (hi is None and d is None)]
            name = f"{lo}-{hi} hops" if hi else f">{lo - 1} hops or disconnected"
            bands[name] = {"share": len(idx) / len(facts),
                           "pretrained_mrr": sum(ultra[i] for i in idx) / max(1, len(idx)),
                           "popularity_mrr": sum(pop[i] for i in idx) / max(1, len(idx))}
        far = [i for i, d in enumerate(dist) if d is None or d > 6]
        out["fractions"][f"f{f}"] = {
            "validation_facts": len(facts), "bands": bands,
            "pretrained_mrr": sum(ultra) / len(ultra), "popularity_mrr": sum(pop) / len(pop),
            "mrr_share_from_beyond_6_hops": {"pretrained": sum(ultra[i] for i in far) / max(1e-12, sum(ultra)),
                                             "popularity": sum(pop[i] for i in far) / max(1e-12, sum(pop))}}
        print(f"f{f}: pretrained {out['fractions'][f'f{f}']['pretrained_mrr']:.4f} "
              f"popularity {out['fractions'][f'f{f}']['popularity_mrr']:.4f}")
        for name, b in bands.items():
            print(f"   {name:<26} share {b['share']:.3f}  pretrained {b['pretrained_mrr']:.4f}  popularity {b['popularity_mrr']:.4f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
