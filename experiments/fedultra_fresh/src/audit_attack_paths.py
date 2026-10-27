#!/usr/bin/env python3
"""Audit shortest alternate paths supporting held-out ATT&CK attack-pattern triples.

This is diagnostic only: it does not train models or alter splits. It reports one
representative shortest path per query, relation-direction signatures, path lengths,
entity-type endpoints, and direct train/test overlap checks.
"""
from __future__ import annotations
import argparse, json, sys
from collections import Counter, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from evaluation_protocol import load_client
from ultra.cybersecurity import load_stix_type_map, attack_pattern_indices

DEFAULT_JSON = {
    "enterprise": ROOT / "datasets/cybersecurity/mitre/enterprise/v19.2/enterprise-attack-19.2.json",
    "ics": ROOT / "datasets/cybersecurity/mitre/ics/v19.2/ics-attack-19.2.json",
}

def build_adj(triples):
    adj = {}
    for h, r, t in triples:
        edge = (h, r, t)
        adj.setdefault(h, []).append((t, edge, 1))
        adj.setdefault(t, []).append((h, edge, -1))
    for node in adj:
        adj[node].sort(key=lambda x: (x[0], x[1][1], x[2]))
    return adj

def shortest_alternate_path(adj, query, max_hops=4):
    """Return one deterministic shortest 2..max_hops path, or None.

    Traversal is undirected, as ULTRA inserts inverse edges. Each edge is labelled
    with its original relation and traversal direction. The exact queried fact is
    removed from consideration.
    """
    h, _r, t = query
    if h == t:
        return None
    skip = tuple(query)
    # Start at the lower-degree endpoint to reduce BFS work; normalize path later.
    reverse_result = len(adj.get(h, ())) > len(adj.get(t, ()))
    start, goal = (t, h) if reverse_result else (h, t)
    queue = deque([(start, 0, [start], [])])
    best_depth = {start: 0}
    while queue:
        node, depth, nodes, steps = queue.popleft()
        if depth >= max_hops:
            continue
        for neighbor, edge, direction in adj.get(node, ()):
            if edge == skip or neighbor in nodes:
                continue
            nd = depth + 1
            step = (edge[1], direction, node, neighbor)
            if neighbor == goal:
                if nd >= 2:
                    found_steps = steps + [step]
                    if reverse_result:
                        found_steps = [(r, -d, v, u) for r, d, u, v in reversed(found_steps)]
                    return {"length": nd, "steps": found_steps}
                # A different direct edge must not prevent discovery of a longer path.
                continue
            if nd < max_hops and nd < best_depth.get(neighbor, max_hops + 1):
                best_depth[neighbor] = nd
                queue.append((neighbor, nd, nodes + [neighbor], steps + [step]))
    return None

def signature(path):
    parts = []
    for relation, direction, _u, _v in path["steps"]:
        parts.append(relation if direction == 1 else relation + "^-1")
    return " -> ".join(parts)

def audit_graph(name, json_path, max_hops):
    bundle = load_client(name)
    type_map = load_stix_type_map(json_path)
    train = [tuple(x) for x in bundle["raw_triples"]["train"]]
    train_set = set(train)
    adj = build_adj(train)
    result = {"graph": name, "max_hops": max_hops, "splits": {}}
    for split in ("valid", "test"):
        queries = [tuple(x) for x in bundle["raw_triples"][split]]
        ap_ids = attack_pattern_indices(queries, type_map)
        length_counts, sig_counts, relation_counts = Counter(), Counter(), Counter()
        supported = 0
        exact_overlap = 0
        unsupported_examples = []
        path_examples = []
        endpoint_type_counts = Counter()
        for i in ap_ids:
            q = queries[i]
            if q in train_set:
                exact_overlap += 1
            path = shortest_alternate_path(adj, q, max_hops)
            endpoint_type_counts[f"{type_map.get(q[0], 'unknown')} -> {type_map.get(q[2], 'unknown')}"] += 1
            relation_counts[q[1]] += 1
            if path:
                supported += 1
                length_counts[str(path["length"])] += 1
                sig = signature(path)
                sig_counts[sig] += 1
                if len(path_examples) < 12:
                    path_examples.append({"query": list(q), "query_endpoint_types": [type_map.get(q[0]), type_map.get(q[2])],
                                          "path_length": path["length"], "path_signature": sig,
                                          "path_nodes": [path["steps"][0][2]] + [s[3] for s in path["steps"]]})
            elif len(unsupported_examples) < 12:
                unsupported_examples.append({"query": list(q), "query_endpoint_types": [type_map.get(q[0]), type_map.get(q[2])]})
        total = len(ap_ids)
        result["splits"][split] = {
            "attack_pattern_queries": total,
            "path_supported_queries": supported,
            "path_support_fraction": supported / total if total else None,
            "unsupported_queries": total - supported,
            "exact_fact_overlap_with_train": exact_overlap,
            "shortest_alternate_path_length_counts": dict(sorted(length_counts.items())),
            "query_relation_counts": dict(relation_counts.most_common()),
            "query_endpoint_type_counts": dict(endpoint_type_counts.most_common()),
            "most_common_representative_shortest_path_signatures": [
                {"signature": s, "count": c} for s, c in sig_counts.most_common(25)
            ],
            "example_supported_queries": path_examples,
            "example_unsupported_queries": unsupported_examples,
            "note": "One deterministic representative shortest path per query; signatures are diagnostic, not exhaustive enumeration of all paths."
        }
    return result

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--enterprise-json", type=Path, default=DEFAULT_JSON["enterprise"])
    p.add_argument("--ics-json", type=Path, default=DEFAULT_JSON["ics"])
    p.add_argument("--max-hops", type=int, default=4)
    p.add_argument("--output", type=Path, default=ROOT / "experiments/fedultra_fresh/results/attack_path_audit.json")
    args = p.parse_args()
    if args.max_hops < 2:
        p.error("--max-hops must be >= 2")
    for name, path in (("enterprise", args.enterprise_json), ("ics", args.ics_json)):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name} STIX JSON: {path}")
    report = {"protocol": "alternate path audit for held-out attack-pattern-related triples",
              "max_hops": args.max_hops,
              "graphs": [audit_graph("enterprise", args.enterprise_json, args.max_hops),
                         audit_graph("ics", args.ics_json, args.max_hops)],
              "interpretation": [
                  "Only target TRAIN triples form the path context for VALID and TEST queries.",
                  "The queried fact itself is excluded from the path search.",
                  "Paths are traversed as undirected for structural support, with relation direction recorded.",
                  "A high support fraction alone does not establish semantic attack-path reasoning; inspect path signatures and endpoint types.",
                  "Exact query fact overlap with train should be zero; any nonzero value needs investigation before training."
              ]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved audit: {args.output}")
if __name__ == "__main__":
    main()
