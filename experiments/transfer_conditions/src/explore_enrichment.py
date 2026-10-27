#!/usr/bin/env python3
"""EXPLORATORY (not registered): does adding real STIX structure the KG dropped create transfer?

Adds technique->tactic (`in-tactic`, every domain) and technique->platform (`on-platform`,
Enterprise and Mobile; every ICS technique lists platform "None") from the v19.2 bundles to
each client's TRAINING graph. Validation queries are unchanged. Then runs the audit's
model-agnostic rule-transfer test: path rules r(X,Y) <- b1..bk (k<=3, simple paths) mined
on each training graph, and target validation ranked with the target's own rule statistics,
the source's, or both pooled (what centralised training on both clients could learn).
If pooling does not beat own statistics, the added structure gives the clients nothing
to exchange. Validation only; no test triple is read.
"""
import json
import time
from collections import defaultdict
from itertools import product

import numpy as np
import torch

from common import ROOT, STUDY
from evaluation_protocol import read_triples
import build_dataset as builder

PSEUDO, MIN_SUPPORT = 5.0, 2
BUNDLES = {
    "enterprise": ROOT / "datasets/cybersecurity/mitre/enterprise/v19.2/enterprise-attack-19.2.json",
    "ics": builder.DEFAULT_SOURCES["ics"],
    "mobile": ROOT / "datasets/cybersecurity/mitre/mobile/v19.2/mobile-attack-19.2.json",
}
FOLDERS = {
    "enterprise": ROOT / "experiments/fedultra_fresh/data_processed/enterprise",
    "ics": STUDY / "data/real/ics_f1.00", "ics_f0.25": STUDY / "data/real/ics_f0.25",
    "mobile": STUDY / "data/real/mobile_f1.00", "mobile_f0.25": STUDY / "data/real/mobile_f0.25",
}


def enrichment(domain, entities):
    objs = json.loads(BUNDLES[domain].read_text(encoding="utf-8"))["objects"]
    extra = set()
    for o in objs:
        if o.get("type") != "attack-pattern" or o["id"] not in entities:
            continue
        for phase in o.get("kill_chain_phases", []):
            extra.add((o["id"], "in-tactic", f"{domain}:tactic:{phase['phase_name']}"))
        for platform in o.get("x_mitre_platforms", []):
            if platform != "None":
                extra.add((o["id"], "on-platform", f"{domain}:platform:{platform}"))
    return extra


class Domain:
    def __init__(self, name, train, valid, hidden=()):
        self.name, self.train, self.valid = name, train, valid
        ents = sorted({e for h, _, t in train + valid + list(hidden) for e in (h, t)})
        self.e2i = {e: i for i, e in enumerate(ents)}
        self.n = len(ents)
        self.rels = sorted({r for _, r, _ in train})
        self.names = self.rels + [r + "^-1" for r in self.rels]
        self.A = {}
        for r in self.rels:
            M = torch.zeros(self.n, self.n)
            pairs = [(self.e2i[h], self.e2i[t]) for h, x, t in train if x == r]
            M[[a for a, _ in pairs], [b for _, b in pairs]] = 1.0
            self.A[r], self.A[r + "^-1"] = M, M.T.contiguous()
        self.tails, self.heads = defaultdict(set), defaultdict(set)
        for h, r, t in set(train) | set(valid) | set(hidden):
            self.tails[(self.e2i[h], r)].add(self.e2i[t])
            self.heads[(r, self.e2i[t])].add(self.e2i[h])

    def bodies(self):
        A, off = self.A, ~torch.eye(self.n, dtype=torch.bool)
        for b in self.names:
            yield (b,), (A[b] > 0) & off
        for b1, b2 in product(self.names, self.names):
            P = A[b1] @ A[b2]
            if not (P > 0.5).any():
                continue
            yield (b1, b2), (P > 0.5) & off
            D1 = torch.diagonal(P).clone()
            for b3 in self.names:
                W = P @ A[b3]
                if not (W > 0.5).any():
                    continue
                D2 = (A[b2] * A[b3].T).sum(1)
                W = W - D1[:, None] * A[b3] - A[b1] * D2[None, :] + A[b1] * A[b2].T * A[b3]
                B = (W > 0.5) & off
                if B.any():
                    yield (b1, b2, b3), B


def rule_stats(dom):
    body_count, support = {}, {}
    for body, B in dom.bodies():
        body_count[body] = int(B.sum())
        for r in dom.rels:
            if body != (r,):
                s = int((B & (dom.A[r] > 0)).sum())
                if s:
                    support[(r, body)] = s
    return body_count, support


def tables(target, t_stats, s_stats):
    (bt_, st_), (bs_, ss_) = t_stats, s_stats
    out = {"own": {}, "source": {}, "pooled": {}}
    for (r, body) in set(st_) | set(ss_):
        if r not in target.rels or not all(b in target.names for b in body):
            continue
        st, ss, bt, bs = st_.get((r, body), 0), ss_.get((r, body), 0), bt_.get(body, 0), bs_.get(body, 0)
        if st >= MIN_SUPPORT:
            out["own"][(r, body)] = st / (bt + PSEUDO)
        if ss >= MIN_SUPPORT:
            out["source"][(r, body)] = ss / (bs + PSEUDO)
        if st + ss >= MIN_SUPPORT:
            out["pooled"][(r, body)] = (st + ss) / (bt + bs + PSEUDO)
    return out


def evaluate(target, tabs):
    queries = defaultdict(list)
    for h, r, t in target.valid:
        queries[r].append((target.e2i[h], target.e2i[t]))
    acc = {(v, r, d, k): torch.zeros(len(qs), target.n) for v in tabs for r, qs in queries.items()
           for d in ("tail", "head") for k in ("max", "lnor")}
    for body, B in target.bodies():
        for r, qs in queries.items():
            rows = None
            for v, table in tabs.items():
                c = table.get((r, body))
                if not c:
                    continue
                if rows is None:
                    hs, ts = torch.tensor([h for h, _ in qs]), torch.tensor([t for _, t in qs])
                    rows = {"tail": B[hs].float(), "head": B[:, ts].T.float()}
                for d, x in rows.items():
                    acc[(v, r, d, "max")] = torch.maximum(acc[(v, r, d, "max")], x * c)
                    acc[(v, r, d, "lnor")] += x * float(np.log1p(-min(c, 0.999999)))
    rr = defaultdict(list)
    for v in tabs:
        for r, qs in queries.items():
            for d in ("tail", "head"):
                M, L = acc[(v, r, d, "max")].numpy(), acc[(v, r, d, "lnor")].numpy()
                for i, (h, t) in enumerate(qs):
                    ans, known = (t, target.tails[(h, r)]) if d == "tail" else (h, target.heads[(r, t)])
                    mask = np.ones(target.n, dtype=bool)
                    mask[list(known)] = False
                    m, l = M[i], L[i]
                    g = int((((m > m[ans]) | ((m == m[ans]) & (l < l[ans]))) & mask).sum())
                    k = int(((m == m[ans]) & (l == l[ans]) & mask).sum())
                    rr[v].append(float(np.mean(1.0 / (1 + g + np.arange(k + 1)))))
    return {v: float(np.mean(x)) for v, x in rr.items()}


def main():
    torch.set_num_threads(2)
    raw = {}
    for name, folder in FOLDERS.items():
        hidden = folder / "hidden.txt"
        raw[name] = (read_triples(folder / "train.txt"), read_triples(folder / "valid.txt"),
                     read_triples(hidden) if hidden.is_file() else [])
    extra = {}
    for name, (train, valid, hidden) in raw.items():
        domain = name.split("_")[0]
        known = {e for h, _, t in train + valid + hidden for e in (h, t)}
        extra[name] = sorted(enrichment(domain, known))
    print("added edges: " + ", ".join(f"{n}={len(x)}" for n, x in extra.items()))
    stats = {}
    for variant in ("as built", "with tactics+platforms"):
        t0 = time.time()
        doms = {n: Domain(n, tr + (extra[n] if variant != "as built" else []), va, hi) for n, (tr, va, hi) in raw.items()}
        for n, d in doms.items():
            stats[(variant, n)] = rule_stats(d)
        print(f"\n[{variant}] rule statistics in {time.time() - t0:.0f}s")
        for target in ("ics", "ics_f0.25", "mobile", "mobile_f0.25"):
            res = evaluate(doms[target], tables(doms[target], stats[(variant, target)], stats[(variant, "enterprise")]))
            print(f"   target={target:<13} validation rule MRR: own={res['own']:.4f} source(enterprise)={res['source']:.4f} "
                  f"pooled={res['pooled']:.4f}  pooled-own={res['pooled'] - res['own']:+.4f}", flush=True)


if __name__ == "__main__":
    main()
