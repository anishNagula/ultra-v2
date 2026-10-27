#!/usr/bin/env python3
"""Audit relation-specific, direction-aware ATT&CK meta-path support.

This is diagnostic only. It does not train models, alter splits, or change the
ULTRA checkpoint. Unlike the generic connectivity audit, it only counts a query
as semantically path-supported when an exact directed relation sequence from the
small, documented template catalog connects its endpoints in the TRAIN graph.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation_protocol import load_client
from ultra.cybersecurity import attack_pattern_indices, load_stix_type_map

DEFAULT_JSON = {
    "enterprise": ROOT / "datasets/cybersecurity/mitre/enterprise/v19.2/enterprise-attack-19.2.json",
    "ics": ROOT / "datasets/cybersecurity/mitre/ics/v19.2/ics-attack-19.2.json",
}

# A step is (relation name, direction): +1 follows the original STIX triple,
# -1 traverses it backwards. These are intentionally conservative templates.
TEMPLATES = {
    "uses": [
        {
            "name": "actor_attributed_to_group_uses_technique",
            "steps": [("attributed-to", 1), ("uses", 1)],
            "types": ["malware", "intrusion-set", "attack-pattern"],
            "rationale": "A malware object is attributed to a group that uses the candidate technique.",
        },
        {
            "name": "actor_uses_subtechnique_parent",
            "steps": [("uses", 1), ("subtechnique-of", 1)],
            "types": ["ACTOR", "attack-pattern", "attack-pattern"],
            "rationale": "The actor uses a known sub-technique whose parent is the candidate technique.",
        },
    ],
    "mitigates": [
        {
            "name": "mitigates_child_parent_technique",
            "steps": [("mitigates", 1), ("subtechnique-of", 1)],
            "types": ["course-of-action", "attack-pattern", "attack-pattern"],
            "rationale": "The mitigation is linked to a child technique of the candidate parent technique.",
        },
    ],
    "targets": [
        {
            "name": "child_technique_parent_targets_asset",
            "steps": [("subtechnique-of", 1), ("targets", 1)],
            "types": ["attack-pattern", "attack-pattern", "x-mitre-asset"],
            "rationale": "The parent technique of a sub-technique targets the candidate asset type.",
        },
    ],
    "subtechnique-of": [
        {
            "name": "transitive_subtechnique_hierarchy",
            "steps": [("subtechnique-of", 1), ("subtechnique-of", 1)],
            "types": ["attack-pattern", "attack-pattern", "attack-pattern"],
            "rationale": "A two-edge sub-technique hierarchy implies a transitive ancestor candidate.",
        },
    ],
}
ACTOR_TYPES = {"malware", "tool", "intrusion-set", "campaign"}


def build_directed_adj(triples):
    adj = defaultdict(list)
    for h, r, t in triples:
        # Forward edge follows the original relation; reverse edge is explicit.
        adj[h].append((t, r, 1, (h, r, t)))
        adj[t].append((h, r, -1, (h, r, t)))
    for node in adj:
        adj[node].sort(key=lambda x: (x[1], x[2], x[0]))
    return adj


def type_matches(expected, actual):
    if expected == "ACTOR":
        return actual in ACTOR_TYPES
    return expected == actual


def find_template_path(adj, query, type_map, template):
    h, query_relation, t = query
    expected_types = template["types"]
    if not type_matches(expected_types[0], type_map.get(h, "unknown")):
        return None
    if not type_matches(expected_types[-1], type_map.get(t, "unknown")):
        return None

    # DFS is bounded by the short, exact template length. It forbids repeated
    # nodes and cannot traverse the queried fact if that fact exists in context.
    steps_expected = template["steps"]
    def visit(node, depth, nodes, actual_steps):
        if depth == len(steps_expected):
            return actual_steps if node == t else None
        relation_expected, direction_expected = steps_expected[depth]
        for neighbor, relation, direction, edge in adj.get(node, []):
            if relation != relation_expected or direction != direction_expected:
                continue
            if edge == tuple(query) or neighbor in nodes:
                continue
            expected_next_type = expected_types[depth + 1]
            if not type_matches(expected_next_type, type_map.get(neighbor, "unknown")):
                continue
            result = visit(neighbor, depth + 1, nodes + [neighbor],
                           actual_steps + [{"relation": relation,
                                            "direction": "forward" if direction == 1 else "inverse",
                                            "from": node, "to": neighbor}])
            if result is not None:
                return result
        return None

    found = visit(h, 0, [h], [])
    if found is None:
        return None
    return {
        "template": template["name"],
        "signature": " -> ".join(s["relation"] if s["direction"] == "forward"
                                   else s["relation"] + "^-1" for s in found),
        "path_length": len(found),
        "steps": found,
        "path_nodes": [h] + [s["to"] for s in found],
    }


def audit_graph(name, json_path):
    bundle = load_client(name)
    type_map = load_stix_type_map(json_path)
    train = [tuple(x) for x in bundle["raw_triples"]["train"]]
    train_set = set(train)
    adj = build_directed_adj(train)
    report = {
        "graph": name,
        "split_counts": {s: len(bundle["raw_triples"][s]) for s in ("train", "valid", "test")},
        "semantic_path_templates": {},
        "splits": {},
    }
    for rel, templates in TEMPLATES.items():
        report["semantic_path_templates"][rel] = [
            {"name": t["name"], "signature": " -> ".join(r if d == 1 else r + "^-1" for r, d in t["steps"]),
             "expected_types": t["types"], "rationale": t["rationale"]}
            for t in templates
        ]

    for split in ("valid", "test"):
        queries = [tuple(x) for x in bundle["raw_triples"][split]]
        selected_indices = attack_pattern_indices(queries, type_map)
        selected = [queries[i] for i in selected_indices]
        relation_counts, endpoint_counts, support_by_rel = Counter(), Counter(), Counter()
        template_counts = Counter()
        supported = []
        unsupported_examples = []
        examples_by_template = defaultdict(list)
        exact_overlap = 0
        for q in selected:
            h, r, t = q
            relation_counts[r] += 1
            endpoint_counts[f"{type_map.get(h, 'unknown')} -> {type_map.get(t, 'unknown')}"] += 1
            exact_overlap += int(q in train_set)
            match = None
            for template in TEMPLATES.get(r, []):
                match = find_template_path(adj, q, type_map, template)
                if match:
                    break
            if match:
                supported.append(q)
                support_by_rel[r] += 1
                template_counts[match["template"]] += 1
                if len(examples_by_template[match["template"]]) < 5:
                    examples_by_template[match["template"]].append({
                        "query": list(q),
                        "endpoint_types": [type_map.get(h, "unknown"), type_map.get(t, "unknown")],
                        **match,
                    })
            elif len(unsupported_examples) < 20:
                unsupported_examples.append({"query": list(q), "endpoint_types": [type_map.get(h, "unknown"), type_map.get(t, "unknown")],
                                            "reason": "No matching directed semantic template for this relation/endpoints in TRAIN context."})
        total = len(selected)
        report["splits"][split] = {
            "attack_pattern_queries": total,
            "strict_semantic_path_supported_queries": len(supported),
            "strict_semantic_path_support_fraction": len(supported) / total if total else None,
            "unsupported_queries": total - len(supported),
            "exact_fact_overlap_with_train": exact_overlap,
            "query_relation_counts": dict(relation_counts.most_common()),
            "strict_supported_by_relation": dict(support_by_rel.most_common()),
            "query_endpoint_type_counts": dict(endpoint_counts.most_common()),
            "matched_template_counts": dict(template_counts.most_common()),
            "examples_by_template": dict(examples_by_template),
            "example_unsupported_queries": unsupported_examples,
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enterprise-json", type=Path, default=DEFAULT_JSON["enterprise"])
    parser.add_argument("--ics-json", type=Path, default=DEFAULT_JSON["ics"])
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/fedultra_fresh/results/semantic_attack_path_audit.json")
    args = parser.parse_args()
    for name, path in (("enterprise", args.enterprise_json), ("ics", args.ics_json)):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name} STIX JSON: {path}")
    report = {
        "protocol": "strict relation-specific direction-aware ATT&CK meta-path audit",
        "purpose": "Validate path query eligibility before any path-aware training; no model training is performed.",
        "context": "Only the target graph TRAIN triples are used as path context for VALID and TEST queries.",
        "path_direction": "Every template follows the directed source STIX relation sequence; inverse traversal is only allowed if a template explicitly includes ^-1.",
        "important_limit": "These templates are hand-specified structural hypotheses, not proof of causal attack progression. They must be reviewed against ATT&CK/STIX semantics before confirmatory claims.",
        "graphs": [audit_graph("enterprise", args.enterprise_json), audit_graph("ics", args.ics_json)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved audit: {args.output}")

if __name__ == "__main__":
    main()
