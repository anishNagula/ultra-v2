#!/usr/bin/env python3
"""Sanity check S3: can ULTRA tell the planted paths apart, and do roles align across clients?

For the query relation, colour-refines each client's relation graph the way ULTRA's
relation model propagates (fused rspmm convention) and lists relations that share a
colour: ULTRA gives those identical representations for any weights. Then compares
pretrained relation representations of same-named relations across clients.
Uses training graphs only; scores no validation or test triple.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from common import load_bundle
import fedavg_runner as base


def mix64(x):
    x = x.astype(np.uint64)
    x ^= x >> np.uint64(30)
    x *= np.uint64(0xBF58476D1CE4E5B9)
    x ^= x >> np.uint64(27)
    x *= np.uint64(0x94D049BB133111EB)
    x ^= x >> np.uint64(31)
    return x


def relation_colours(graph, query, iters=6):
    rg = graph.relation_graph
    src, nbr, col = rg.edge_index[0].numpy(), rg.edge_index[1].numpy(), rg.edge_type.numpy()
    c = np.zeros(rg.num_nodes, dtype=np.int64)
    c[query] = 1
    for _ in range(iters):
        key = mix64((col.astype(np.uint64) << np.uint64(32)) + c[nbr].astype(np.uint64) + np.uint64(1))
        sig = np.zeros(rg.num_nodes, dtype=np.uint64)
        np.add.at(sig, src, key)
        new = np.unique(np.stack([c.astype(np.uint64), sig], 1), axis=0, return_inverse=True)[1].reshape(-1)
        stable, c = new.max() == c.max(), new
        if stable:
            break
    return c


def names(bundle):
    n = len(bundle["relation_to_id"])
    inv = {i: r for r, i in bundle["relation_to_id"].items()}
    return [inv[i] for i in range(n)] + [inv[i] + "^-1" for i in range(n)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clients", nargs="+", required=True, help="Client folders; the first is the reference")
    p.add_argument("--query", default="uses")
    args = p.parse_args()
    model = base.new_model().eval()
    reps = {}
    for folder in args.clients:
        bundle = load_bundle(Path(folder))
        q = bundle["relation_to_id"][args.query]
        rel = names(bundle)
        colours = relation_colours(bundle["train_data"], q)
        classes = {}
        for i, c in enumerate(colours):
            classes.setdefault(c, []).append(rel[i])
        tied = [sorted(v) for v in classes.values() if len(v) > 1]
        with torch.no_grad():
            r = model.relation_model(bundle["train_data"].relation_graph, query=torch.tensor([q]))[0]
        reps[folder] = dict(zip(rel, r))
        print(f"[{Path(folder).name}] relations={len(rel)} relation-graph edges={bundle['train_data'].relation_graph.num_edges} "
              f"indistinguishable groups for query={args.query}: {tied if tied else 'none'}")
    ref = args.clients[0]
    for other in args.clients[1:]:
        same = [F.cosine_similarity(reps[ref][k], reps[other][k], dim=0).item() for k in reps[ref] if k in reps[other]]
        diff = [F.cosine_similarity(reps[ref][a], reps[other][b], dim=0).item()
                for a in reps[ref] for b in reps[other] if a != b]
        print(f"pretrained cosine {Path(ref).name} vs {Path(other).name}: same relation median={np.median(same):.3f} "
              f"min={min(same):.3f}; different relations median={np.median(diff):.3f}")


if __name__ == "__main__":
    main()
