#!/usr/bin/env python3
"""Confirmatory ULTRA transfer: Enterprise ATT&CK -> unbridged CAPEC/CWE/CVE KG.

The source and target have no shared entity IDs or asserted bridge edges. ULTRA is
fine-tuned on Enterprise only, selected on source validation MRR, then compared
against the unchanged pretrained checkpoint on the target graph. Target validation
and test never select the source checkpoint. Positive --eval-limit is diagnostic only.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import fedavg_runner as base
import transfer_runner as transfer
from evaluation_protocol import load_client

DEFAULT_TARGET_ROOT = ROOT / "data" / "processed" / "cyber_transfer"
DEFAULT_OUTPUT = ROOT / "experiments" / "fedultra_fresh" / "results" / "cyber_transfer_confirmatory.json"
SOURCE_SPLIT_ROOT = ROOT / "experiments" / "fedultra_fresh" / "data_processed" / "enterprise"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_target_audit(audit, target_root):
    if not audit:
        raise ValueError(f"Missing target audit: {target_root / 'audit.json'}")
    if "unbridged" not in audit.get("protocol", "").lower():
        raise ValueError("Target audit is not the expected unbridged transfer protocol")
    cross = audit.get("cross_domain", {})
    if cross.get("explicit_bridge_edges_added") != 0 or cross.get("source_target_entity_overlap_count") != 0:
        raise ValueError("Target audit violates the frozen no-bridge/entity-disjoint protocol")
    graph = audit.get("graph", {})
    split = audit.get("split", {})
    if graph.get("train_entity_coverage_fraction") != 1.0:
        raise ValueError("Target training graph does not cover all target entities")
    if not split.get("all_held_out_edges_are_has_weakness"):
        raise ValueError("Target valid/test contain relations other than has_weakness")
    if not split.get("all_valid_test_endpoints_present_in_train"):
        raise ValueError("Target valid/test endpoint coverage check failed")
    if audit.get("seed") != 20261026:
        raise ValueError("Confirmatory protocol requires frozen preparation/split seed 20261026")
    if audit.get("cve_sampling", {}).get("max_cves") != 20_000:
        raise ValueError("Confirmatory protocol requires the frozen max_cves=20000 target sample")
    split_config = audit.get("split_config", {})
    if split_config.get("valid_fraction_requested") != 0.10 or split_config.get("test_fraction_requested") != 0.10:
        raise ValueError("Confirmatory protocol requires the frozen 10% validation and 10% test split")
    if split.get("actual_valid_count", 0) < 1 or split.get("actual_test_count", 0) < 1:
        raise ValueError("Target validation and test splits must both be non-empty")

    base_graph_path = ROOT / "data" / "processed" / "cyber_cti" / "triples.tsv"
    expected_base_hash = audit.get("inputs", {}).get("base_triples_sha256")
    if not base_graph_path.is_file() or sha256(base_graph_path) != expected_base_hash:
        raise ValueError("The source CAPEC/CWE/NVD triples.tsv changed or is missing since target preparation")

    for name, expected in audit.get("source", {}).get("split_sha256", {}).items():
        actual = sha256(SOURCE_SPLIT_ROOT / name)
        if actual != expected:
            raise ValueError(f"Source split changed since target preparation: {name}")
    outputs = audit.get("outputs", {}).get("sha256", {})
    paths = {
        "target_graph_tsv": target_root / "target_graph.tsv",
        "target_train": target_root / "target" / "train.txt",
        "target_valid": target_root / "target" / "valid.txt",
        "target_test": target_root / "target" / "test.txt",
    }
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        if outputs.get(key) != sha256(path):
            raise ValueError(f"Target artifact does not match audit hash: {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-root", type=Path, default=DEFAULT_TARGET_ROOT,
                   help="Directory containing target/train.txt, valid.txt, test.txt")
    p.add_argument("--seeds", default="20261027,20261028,20261029,20261030,20261031,20261032,20261033,20261034,20261035,20261036")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--local-steps", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-negative", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=4)
    p.add_argument("--eval-limit", type=int, default=0,
                   help="Positive value is diagnostic only; 0 evaluates full splits")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if args.rounds < 1 or args.local_steps < 1 or args.batch_size < 2 or args.batch_size % 2:
        raise ValueError("rounds/local_steps must be positive and batch-size must be even and >= 2")
    if args.eval_batch_size < 1 or args.num_negative < 1 or args.eval_limit < 0:
        raise ValueError("invalid evaluation batch, negative count, or eval limit")
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must contain unique comma-separated integers")
    if not base.CHECKPOINT.is_file():
        raise FileNotFoundError(f"Missing pretrained checkpoint: {base.CHECKPOINT}")
    if not (args.target_root / "target" / "train.txt").is_file():
        raise FileNotFoundError(
            f"Missing {args.target_root / 'target' / 'train.txt'}. Run scripts/cybersecurity/prepare_attack_to_cve_transfer.py first."
        )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists; choose another --output or pass --overwrite")

    target_audit_path = args.target_root / "audit.json"
    if not target_audit_path.is_file():
        raise FileNotFoundError(f"Missing target audit: {target_audit_path}")
    target_audit = json.loads(target_audit_path.read_text(encoding="utf-8"))
    verify_target_audit(target_audit, args.target_root)

    torch.set_num_threads(min(2, torch.get_num_threads()))
    checkpoint = torch.load(base.CHECKPOINT, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"Unexpected checkpoint format: {base.CHECKPOINT}")
    checkpoint_state = checkpoint["model"]
    bundles = {
        "enterprise": load_client("enterprise"),
        "target": load_client("target", data_root=args.target_root),
    }

    results = []
    for seed in seeds:
        result = transfer.run_direction("enterprise", "target", seed, bundles, args, checkpoint_state)
        results.append(result)
        payload = {
            "protocol": "pretrained ULTRA vs Enterprise-source-finetuned ULTRA on unbridged, entity-disjoint CAPEC/CWE/CVE target",
            "checkpoint": str(base.CHECKPOINT),
            "checkpoint_sha256": sha256(base.CHECKPOINT),
            "source_graph": "MITRE Enterprise ATT&CK v19.2, frozen source train/valid/test splits",
            "target_graph_audit": target_audit,
            "message_passing": "source or target training triples only; all non-has_weakness target triples are context; no bridge edges",
            "model_selection": "source Enterprise validation MRR only; target validation/test never select source checkpoint",
            "primary_metric": "filtered target test MRR delta on held-out CVE has_weakness CWE triples, ranking heads and tails over all target entities",
            "acceptance_threshold": 0.03,
            "final_acceptance_requires": "mean delta-MRR >= 0.03 and both seed-level t CI and hierarchical paired-bootstrap 95% CI entirely above zero",
            "results": results,
            "summary": transfer.summarize_direction(results),
            "diagnostic_only": args.eval_limit > 0,
        }
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Saved progress: {output}")

    summary = transfer.summarize_direction(results)
    print("\n===== CYBER TRANSFER SUMMARY =====")
    print(json.dumps(summary, indent=2))
    if args.eval_limit:
        print("DIAGNOSTIC RUN ONLY: do not use for acceptance claims.")
    else:
        print("FULL TRANSFER EVALUATION COMPLETE.")


if __name__ == "__main__":
    main()
