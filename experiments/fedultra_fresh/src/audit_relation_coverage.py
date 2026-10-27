#!/usr/bin/env python3
"""Audit relation coverage and entity-degree constraints in fresh FedULTRA splits.

Read-only: does not modify data. Run from the repository root:
  python experiments/fedultra_fresh/src/audit_relation_coverage.py
"""
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path.cwd()
DATA = ROOT / "experiments" / "fedultra_fresh" / "data_processed"
SPLITS = ("train", "valid", "test")


def read_split(path):
    triples = set()
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            fields = line.split("\t")
            if len(fields) != 3:
                raise ValueError(f"{path}:{line_no}: expected 3 tab-separated fields")
            h, r, t = fields
            triples.add((h, r, t))
    return triples


def main():
    for client in ("enterprise", "ics"):
        folder = DATA / client
        missing = [str(folder / f"{s}.txt") for s in SPLITS
                   if not (folder / f"{s}.txt").is_file()]
        if missing:
            raise FileNotFoundError("Missing split files:\n  " + "\n  ".join(missing))

        split = {s: read_split(folder / f"{s}.txt") for s in SPLITS}
        all_triples = set().union(*split.values())
        if len(all_triples) != sum(map(len, split.values())):
            raise AssertionError(f"{client}: duplicate triple across splits")

        degree = Counter()
        rel_counts = {s: Counter(r for _, r, _ in split[s]) for s in SPLITS}
        rel_triples = defaultdict(list)
        for h, r, t in all_triples:
            degree[h] += 1
            degree[t] += 1
            rel_triples[r].append((h, r, t))

        print(f"\n===== {client.upper()} RELATION-COVERAGE AUDIT =====")
        print(f"Triples: total={len(all_triples)}, "
              f"train={len(split['train'])}, valid={len(split['valid'])}, "
              f"test={len(split['test'])}")
        print("relation\tall\ttrain\tvalid\ttest\theads\ttails\t"
              "degree1_endpoint_triples\tstatus")
        for r in sorted(rel_triples):
            triples = rel_triples[r]
            heads = {h for h, _, _ in triples}
            tails = {t for _, _, t in triples}
            degree1 = sum(1 for h, _, t in triples
                          if degree[h] == 1 or degree[t] == 1)
            counts = [rel_counts[s][r] for s in SPLITS]
            if counts[2] == 0:
                status = "NO_TEST_EXAMPLES"
            elif counts[1] == 0:
                status = "NO_VALID_EXAMPLES"
            else:
                status = "HELD_OUT_OK"
            print(f"{r}\t{len(triples)}\t{counts[0]}\t{counts[1]}\t"
                  f"{counts[2]}\t{len(heads)}\t{len(tails)}\t"
                  f"{degree1}\t{status}")

        degree1_entities = sum(1 for e, d in degree.items() if d == 1)
        heldout = split["valid"] | split["test"]
        heldout_degree1 = sum(
            1 for h, _, t in heldout if degree[h] == 1 or degree[t] == 1
        )
        print(f"Unique entities: {len(degree)}; degree-1 entities: {degree1_entities}")
        print(f"Held-out triples touching a degree-1 entity: "
              f"{heldout_degree1} (should be 0 for this entity-covered split)")
        train_relations = {r for _, r, _ in split["train"]}
        all_relations = {r for _, r, _ in all_triples}
        if train_relations != all_relations:
            print("WARNING: relation(s) absent from training:",
                  sorted(all_relations - train_relations))
        else:
            print("All relations represented in training: PASS")

    print("\nInterpretation:")
    print("- NO_TEST_EXAMPLES means per-relation test performance is not measurable.")
    print("- Degree-1 endpoint triples cannot be held out while requiring every entity")
    print("  to appear in training; report this limitation rather than fabricating facts.")
    print("- This audit does not change any files or alter the splits.")


if __name__ == "__main__":
    main()
