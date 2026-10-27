#!/usr/bin/env python3
"""Build a reproducible CAPEC + CWE + NVD CVE knowledge-graph feasibility audit.

Inputs:
  data/raw/capec/capec_latest.zip
  data/raw/cwe/cwec_latest.xml.zip
  data/raw/nvd/nvdcve-2.0-YYYY.json.gz (selected years)

Outputs:
  data/processed/cyber_cti/triples.tsv
  data/processed/cyber_cti/audit.json

This is an initial feasibility graph, not a claim that all source relations are
semantically interchangeable. It only emits explicit catalog relationships,
CVE-to-CWE associations, and CVE-to-product CPE references.
"""
from __future__ import annotations
import argparse
import gzip
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

def lname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].split(":")[-1]

def xml_root_from_zip(path: Path, preferred_tokens: tuple[str, ...]) -> ET.Element:
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".xml")]
        if not names:
            raise RuntimeError(f"No XML file found in {path}")
        preferred = [n for n in names if any(t in n.lower() for t in preferred_tokens)]
        chosen = preferred[0] if preferred else names[0]
        with zf.open(chosen) as f:
            return ET.parse(f).getroot()

def child_text(parent: ET.Element, child_name: str) -> str | None:
    for child in list(parent):
        if lname(child.tag) == child_name and child.text:
            return child.text.strip()
    return None

def descendants(parent: ET.Element, name: str):
    for el in parent.iter():
        if lname(el.tag) == name:
            yield el

def add(triples: set[tuple[str, str, str]], s: str | None, r: str, o: str | None):
    if s and o and s != o:
        triples.add((s, r, o))

def parse_capec(path: Path, triples: set[tuple[str, str, str]], entities: dict[str, set[str]]):
    root = xml_root_from_zip(path, ("capec",))
    count = 0
    for ap in descendants(root, "Attack_Pattern"):
        capec_id = ap.attrib.get("ID")
        if not capec_id:
            continue
        s = f"CAPEC:{capec_id}"
        entities["CAPEC"].add(s)
        count += 1
        for rel in descendants(ap, "Related_Attack_Pattern"):
            target = rel.attrib.get("CAPEC_ID")
            nature = (rel.attrib.get("Nature") or "related").strip().lower().replace(" ", "_")
            if target:
                add(triples, s, f"capec_{nature}", f"CAPEC:{target}")
        for rel in descendants(ap, "Related_Weakness"):
            target = rel.attrib.get("CWE_ID")
            if target:
                add(triples, s, "exploits_or_relates_to_weakness", f"CWE:{target}")
        for ref in descendants(ap, "External_Reference"):
            source = (ref.attrib.get("Source") or "").strip().lower()
            external_id = ref.attrib.get("External_ID") or ref.attrib.get("Reference_ID")
            if external_id and ("attack" in source or re.fullmatch(r"T\d{4}(?:\.\d{3})?", external_id)):
                add(triples, s, "mapped_to_attack_technique", f"ATTACK:{external_id}")
    return count

def parse_cwe(path: Path, triples: set[tuple[str, str, str]], entities: dict[str, set[str]]):
    root = xml_root_from_zip(path, ("cwec", "cwe"))
    count = 0
    for weakness in descendants(root, "Weakness"):
        cwe_id = weakness.attrib.get("ID")
        if not cwe_id:
            continue
        s = f"CWE:{cwe_id}"
        entities["CWE"].add(s)
        count += 1
        for rel in descendants(weakness, "Related_Weakness"):
            target = rel.attrib.get("CWE_ID")
            nature = (rel.attrib.get("Nature") or "related").strip().lower().replace(" ", "_")
            if target:
                add(triples, s, f"cwe_{nature}", f"CWE:{target}")
        for rel in descendants(weakness, "Related_Attack_Pattern"):
            target = rel.attrib.get("CAPEC_ID")
            if target:
                add(triples, s, "related_attack_pattern", f"CAPEC:{target}")
    return count

def cpe_criteria_strings(node):
    # NVD 2.0 configuration objects nest nodes recursively.
    if isinstance(node, dict):
        for match in node.get("cpeMatch", []) or []:
            criteria = match.get("criteria")
            if criteria and isinstance(criteria, str) and criteria.startswith("cpe:2.3:"):
                yield criteria
        for child in node.get("nodes", []) or []:
            yield from cpe_criteria_strings(child)
        # Some schema versions use children directly in the configuration tree.
        for key in ("configurations", "children"):
            value = node.get(key)
            if isinstance(value, list):
                for child in value:
                    yield from cpe_criteria_strings(child)

def parse_nvd(path: Path, triples: set[tuple[str, str, str]], entities: dict[str, set[str]], max_cpes_per_cve: int):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    count = 0
    for item in data.get("vulnerabilities", []):
        cve = item.get("cve", {})
        cve_id = cve.get("id")
        if not cve_id:
            continue
        s = f"CVE:{cve_id}"
        entities["CVE"].add(s)
        count += 1
        cwe_ids = set()
        for weakness in cve.get("weaknesses", []) or []:
            for desc in weakness.get("description", []) or []:
                value = desc.get("value", "")
                if re.fullmatch(r"CWE-\d+", value):
                    cwe_ids.add(value.split("-", 1)[1])
        for cwe_id in cwe_ids:
            entities["CWE"].add(f"CWE:{cwe_id}")
            add(triples, s, "has_weakness", f"CWE:{cwe_id}")
        cpes = []
        for config in cve.get("configurations", []) or []:
            cpes.extend(cpe_criteria_strings(config))
        # Keep bounded and deterministic: CPEs sorted, one edge per distinct criterion.
        for cpe in sorted(set(cpes))[:max_cpes_per_cve]:
            product = f"CPE:{cpe}"
            entities["CPE"].add(product)
            add(triples, s, "affects_product_configuration", product)
    return count

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--capec", type=Path, default=Path("data/raw/capec/capec_latest.zip"))
    p.add_argument("--cwe", type=Path, default=Path("data/raw/cwe/cwec_latest.xml.zip"))
    p.add_argument("--nvd-dir", type=Path, default=Path("data/raw/nvd"))
    p.add_argument("--years", default="2020,2021,2022,2023,2024,2025")
    p.add_argument("--max-cpes-per-cve", type=int, default=10)
    p.add_argument("--out-dir", type=Path, default=Path("data/processed/cyber_cti"))
    args = p.parse_args()

    for required in (args.capec, args.cwe):
        if not required.exists():
            raise SystemExit(f"Missing {required}. Download the source data first.")
    years = [int(y.strip()) for y in args.years.split(",") if y.strip()]
    nvd_files = [args.nvd_dir / f"nvdcve-2.0-{year}.json.gz" for year in years]
    missing = [x for x in nvd_files if not x.exists()]
    if missing:
        raise SystemExit("Missing NVD feed(s): " + ", ".join(map(str, missing)))

    triples: set[tuple[str, str, str]] = set()
    entities: dict[str, set[str]] = defaultdict(set)
    capec_count = parse_capec(args.capec, triples, entities)
    cwe_count = parse_cwe(args.cwe, triples, entities)
    nvd_count_by_year = {}
    for path in nvd_files:
        nvd_count_by_year[path.stem.replace("nvdcve-2.0-", "")] = parse_nvd(
            path, triples, entities, args.max_cpes_per_cve
        )

    # Keep only explicit edges whose endpoint entities exist as catalog records,
    # except CPE nodes and ATTACK IDs, which are external identifiers.
    relation_counts = Counter(r for _, r, _ in triples)
    entity_counts = {k: len(v) for k, v in sorted(entities.items())}
    degrees = Counter()
    adjacency = defaultdict(set)
    for s, r, o in triples:
        degrees[s] += 1
        degrees[o] += 1
        adjacency[s].add(o)
        adjacency[o].add(s)
    unseen = set(adjacency)
    components = []
    while unseen:
        start = unseen.pop()
        stack = [start]
        size = 0
        while stack:
            cur = stack.pop()
            size += 1
            for nxt in adjacency[cur]:
                if nxt in unseen:
                    unseen.remove(nxt)
                    stack.append(nxt)
        components.append(size)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    triples_path = args.out_dir / "triples.tsv"
    with triples_path.open("w", encoding="utf-8") as f:
        f.write("head\trelation\ttail\n")
        for s, r, o in sorted(triples):
            f.write(f"{s}\t{r}\t{o}\n")

    audit = {
        "sources": {
            "capec_file": str(args.capec),
            "cwe_file": str(args.cwe),
            "nvd_files": [str(x) for x in nvd_files],
            "nvd_years": years,
            "nvd_cve_records_per_year": nvd_count_by_year,
            "capec_attack_pattern_records_seen": capec_count,
            "cwe_weakness_records_seen": cwe_count,
            "max_cpes_per_cve": args.max_cpes_per_cve,
        },
        "graph": {
            "unique_triples": len(triples),
            "entities_by_source_or_type": entity_counts,
            "relation_counts": dict(sorted(relation_counts.items())),
            "connected_components": len(components),
            "largest_component_nodes": max(components, default=0),
            "largest_component_fraction": (max(components) / len(adjacency)) if adjacency else 0,
            "mean_degree": (sum(degrees.values()) / len(degrees)) if degrees else 0,
            "median_degree": sorted(degrees.values())[len(degrees)//2] if degrees else 0,
            "entities_with_degree_1": sum(1 for d in degrees.values() if d == 1),
            "entities_in_edges": len(degrees),
        },
        "notes": [
            "CVE-to-CWE edges are based on NVD weakness descriptions where the value is a CWE identifier.",
            "CVE-to-CPE edges are based on NVD CPE criteria; at most max_cpes_per_cve sorted criteria are retained per CVE.",
            "CAPEC/CWE catalog links are emitted only when explicit source relationships are present.",
            "ATTACK IDs emitted from CAPEC references are external identifier nodes; merge with Enterprise ATT&CK only after verifying exact ID conventions.",
            "This is a feasibility extraction. Audit relation semantics, source versions, missing mappings, and leakage before model training.",
            "NVD feeds are split by CVE ID year; this gives a reproducible bounded snapshot, not every CVE published during those calendar years if the feeds later change."
        ],
        "output_triples_tsv": str(triples_path),
    }
    audit_path = args.out_dir / "audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    print(f"\nWrote {triples_path}")
    print(f"Wrote {audit_path}")

if __name__ == "__main__":
    main()

