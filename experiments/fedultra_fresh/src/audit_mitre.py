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

def audit(name, path):
    print(f"\n{'=' * 72}\n{name.upper()}: {path}\n{'=' * 72}")

    if not path.is_file():
        print("ERROR: Source file not found")
        return

    with path.open(encoding="utf-8") as f:
        bundle = json.load(f)

    objects = bundle.get("objects", [])
    by_type = Counter(
        obj.get("type", "<missing-type>") for obj in objects
    )
    by_relationship = Counter(
        obj.get("relationship_type", "<missing>")
        for obj in objects
        if obj.get("type") == "relationship"
    )

    ids = [
        obj["id"] for obj in objects
        if isinstance(obj.get("id"), str)
    ]
    id_counts = Counter(ids)
    duplicate_ids = {
        key: count for key, count in id_counts.items() if count > 1
    }

    object_ids = set(ids)
    relationships = [
        obj for obj in objects
        if obj.get("type") == "relationship"
    ]

    missing_source = sum(
        not isinstance(r.get("source_ref"), str)
        for r in relationships
    )
    missing_target = sum(
        not isinstance(r.get("target_ref"), str)
        for r in relationships
    )
    dangling_source = sum(
        isinstance(r.get("source_ref"), str)
        and r["source_ref"] not in object_ids
        for r in relationships
    )
    dangling_target = sum(
        isinstance(r.get("target_ref"), str)
        and r["target_ref"] not in object_ids
        for r in relationships
    )

    print("Bundle type:", bundle.get("type"))
    print("Objects:", len(objects))
    print("Unique object IDs:", len(object_ids))
    print("Duplicate IDs:", len(duplicate_ids))
    print("\nObject types:")
    for typ, count in by_type.most_common():
        print(f"  {typ}: {count}")

    print("\nRelationship types:")
    for typ, count in by_relationship.most_common():
        print(f"  {typ}: {count}")

    print("\nRelationship integrity:")
    print("  Total relationship objects:", len(relationships))
    print("  Missing source_ref:", missing_source)
    print("  Missing target_ref:", missing_target)
    print("  Source references not in bundle:", dangling_source)
    print("  Target references not in bundle:", dangling_target)

    print("\nSelected object metadata:")
    for typ in (
        "attack-pattern", "malware", "tool", "intrusion-set",
        "campaign", "course-of-action", "relationship"
    ):
        examples = [
            obj for obj in objects if obj.get("type") == typ
        ][:2]
        for obj in examples:
            print({
                "type": obj.get("type"),
                "id": obj.get("id"),
                "name": obj.get("name"),
                "relationship_type": obj.get("relationship_type"),
                "source_ref": obj.get("source_ref"),
                "target_ref": obj.get("target_ref"),
                "revoked": obj.get("revoked", False),
                "deprecated": obj.get("x_mitre_deprecated", False),
            })

for name, path in SOURCES.items():
    audit(name, path)
