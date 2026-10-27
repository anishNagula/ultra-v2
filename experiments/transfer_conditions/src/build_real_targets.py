#!/usr/bin/env python3
"""Study R data: real ATT&CK target clients at reduced training-data levels.

Mobile ATT&CK v19.2 is extracted and split with the fresh FedULTRA builder
(same relation set, same split seed, triples identical to Enterprise removed,
as for ICS). ICS reuses its frozen split. For each target and fraction f, the
training split keeps, per relation, the first max(1, round(f * n_r)) facts of one
seeded shuffle, so smaller fractions are subsets of larger ones. Removed facts go
to hidden.txt (filtered at evaluation, never trained on); validation and test are
unchanged. Only real ATT&CK facts are written; no fact is synthesised.
"""
import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

from common import ROOT, STUDY, sha256, write_triples
import build_dataset as builder
from evaluation_protocol import read_triples

DEFAULT_OUT = STUDY / "data" / "real"
SUBSAMPLE_SEED = 20261102
FRACTIONS = (0.10, 0.25, 1.00)
ENTERPRISE_JSON = ROOT / "datasets/cybersecurity/mitre/enterprise/v19.2/enterprise-attack-19.2.json"
MOBILE_JSON = ROOT / "datasets/cybersecurity/mitre/mobile/v19.2/mobile-attack-19.2.json"
ICS_SPLIT = ROOT / "experiments/fedultra_fresh/data_processed/ics"


def subsample(train, fraction, seed):
    rng = random.Random(seed)
    by_relation = {}
    for triple in sorted(train):
        by_relation.setdefault(triple[1], []).append(triple)
    kept = []
    for relation, items in sorted(by_relation.items()):
        rng.shuffle(items)
        kept += items[:max(1, round(fraction * len(items)))]
    return set(kept)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    out = args.output_root
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{out} is not empty; pass --overwrite to rebuild")

    _, enterprise_triples, _ = builder.extract(ENTERPRISE_JSON)
    _, mobile_triples, rejected = builder.extract(MOBILE_JSON)
    shared = mobile_triples & enterprise_triples
    mobile_split = builder.split_triples(mobile_triples - shared, builder.SPLIT_SEED)
    splits = {
        "mobile": {name: sorted(items) for name, items in mobile_split.items()},
        "ics": {name: read_triples(ICS_SPLIT / f"{name}.txt") for name in ("train", "valid", "test")},
    }
    manifest = {
        "subsample_seed": SUBSAMPLE_SEED, "fractions": FRACTIONS,
        "mobile_source": {"path": str(MOBILE_JSON), "sha256": sha256(MOBILE_JSON),
                          "rejected": dict(rejected), "identical_enterprise_triples_removed": len(shared),
                          "split_seed": builder.SPLIT_SEED},
        "ics_source": {"split_dir": str(ICS_SPLIT),
                       "sha256": {f"{n}.txt": sha256(ICS_SPLIT / f"{n}.txt") for n in ("train", "valid", "test")}},
        "clients": {},
    }
    for target, split in splits.items():
        for fraction in FRACTIONS:
            kept = subsample(split["train"], fraction, SUBSAMPLE_SEED)
            hidden = set(split["train"]) - kept
            folder = out / f"{target}_f{fraction:.2f}"
            write_triples(folder / "train.txt", kept)
            if hidden:
                write_triples(folder / "hidden.txt", hidden)
            write_triples(folder / "valid.txt", split["valid"])
            write_triples(folder / "test.txt", split["test"])
            manifest["clients"][folder.name] = {
                "train": len(kept), "hidden": len(hidden),
                "valid": len(split["valid"]), "test": len(split["test"]),
                "train_relation_counts": dict(sorted(Counter(r for _, r, _ in kept).items())),
                "entities_with_train_edges": len({e for h, _, t in kept for e in (h, t)}),
                "sha256": {f.name: sha256(f) for f in sorted(folder.glob("*.txt"))},
            }
    for fraction in FRACTIONS:  # nesting check
        for target in splits:
            smaller = [f for f in FRACTIONS if f < fraction]
            for f in smaller:
                a = set(read_triples(out / f"{target}_f{f:.2f}" / "train.txt"))
                b = set(read_triples(out / f"{target}_f{fraction:.2f}" / "train.txt"))
                assert a <= b, (target, f, fraction)
    if (out / "ics_f1.00" / "train.txt").read_bytes() != (ICS_SPLIT / "train.txt").read_bytes():
        raise AssertionError("ICS at fraction 1.00 must equal the frozen ICS training split")
    shutil.copyfile(ICS_SPLIT.parent.parent / "data_manifest.json", out / "fedultra_fresh_data_manifest.json")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "sha256"}
                      for k, v in manifest["clients"].items()}, indent=1))


if __name__ == "__main__":
    main()
