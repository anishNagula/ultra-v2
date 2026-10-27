import json
from collections import Counter
from pathlib import Path

SOURCES = {
    "enterprise": Path.home() / (
        "code/ULTRA/datasets/cybersecurity/mitre/"
        "enterprise/v19.2/enterprise-attack-19.2.json"
    ),
    "ics": Path.home() / (
        "code/ULTRA/datasets/cybersecurity/mitre/"
        "ics/v19.2/ics-attack-19.2.json"
    ),
}

NODE_TYPES = {
    "attack-pattern",
    "malware",
    "tool",
    "intrusion-set",
    "campaign",
    "course-of-action",
    "x-mitre-asset",
    "x-mitre-data-source",
    "x-mitre-data-component",
    "x-mitre-analytic",
    "x-mitre-detection-strategy",
    "x-mitre-tactic",
}

RELATION_TYPES = {
    "uses",
    "mitigates",
    "detects",
    "subtechnique-of",
    "attributed-to",
    "targets",
}

def active(obj):
    return (
        not obj.get("revoked", False)
        and not obj.get("x_mitre_deprecated", False)
    )

def preview(name, path):
    bundle = json.loads(path.read_text(encoding="utf-8"))
    objects = bundle["objects"]

    nodes = {
        obj["id"]: obj
        for obj in objects
        if obj.get("type") in NODE_TYPES
        and isinstance(obj.get("id"), str)
        and active(obj)
    }

    candidates = []
    rejected_type = Counter()
    rejected_inactive = Counter()
    rejected_endpoint = Counter()

    for obj in objects:
        if obj.get("type") != "relationship":
            continue

        if not active(obj):
            rejected_inactive[obj.get("relationship_type", "<missing>")] += 1
            continue

        rel = obj.get("relationship_type")
        if rel not in RELATION_TYPES:
            rejected_type[rel or "<missing>"] += 1
            continue

        h, t = obj.get("source_ref"), obj.get("target_ref")
        if h not in nodes or t not in nodes:
            rejected_endpoint[rel] += 1
            continue

        candidates.append((h, rel, t))

    unique = set(candidates)
    rel_counts = Counter(r for h, r, t in unique)
    type_pairs = Counter(
        (nodes[h]["type"], r, nodes[t]["type"])
        for h, r, t in unique
    )

    print(f"\n=== {name.upper()} ===")
    print("Active selected entities:", len(nodes))
    print("Candidate relationship objects:", len(candidates))
    print("Unique candidate triples:", len(unique))
    print("Duplicate triples removed:", len(candidates) - len(unique))
    print("Entity types:", dict(Counter(
        n["type"] for n in nodes.values()
    )))
    print("Relation counts:", dict(sorted(rel_counts.items())))
    print("Rejected by relationship type:", dict(rejected_type))
    print("Rejected inactive relationships:", dict(rejected_inactive))
    print("Rejected because endpoint is excluded/inactive:", dict(rejected_endpoint))
    print("Most frequent (head type, relation, tail type) patterns:")
    for pattern, count in type_pairs.most_common(20):
        print(f"  {pattern}: {count}")

    return nodes, unique

results = {
    name: preview(name, path)
    for name, path in SOURCES.items()
}

ent_nodes, ent_triples = results["enterprise"]
ics_nodes, ics_triples = results["ics"]

print("\n=== CROSS-DOMAIN CHECK ===")
print("Shared entity IDs:", len(set(ent_nodes) & set(ics_nodes)))
print("Shared unique triples:", len(ent_triples & ics_triples))
print("Enterprise-only triples:", len(ent_triples - ics_triples))
print("ICS-only triples:", len(ics_triples - ent_triples))
print("Shared relation labels:", sorted(
    {r for h, r, t in ent_triples}
    & {r for h, r, t in ics_triples}
))
print("Global relation vocabulary:", sorted(
    {r for h, r, t in ent_triples}
    | {r for h, r, t in ics_triples}
))
