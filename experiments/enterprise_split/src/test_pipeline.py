#!/usr/bin/env python3
"""Automated checks on the written Enterprise-split data and the federation code path.

  python test_pipeline.py            # checks data/v1/C-time
  DATA=../data/v1/C-rand python test_pipeline.py

Checks:
- Leakage L1-L4: exact, inverse-equivalent and entity-pair.
- Relation coverage.
- Nested fractions.
- Exact partition of the extracted facts.
- Deterministic rebuild.
- Loader isolation: message passing uses training edges only.
- FedAvg exchanges only weights, and each client trains on its own graph.
"""
import json
import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch

from env import BUNDLE, STUDY
import build_dataset as builder
import build_split
import conditions_runner
import fedavg_runner as base
from common import load_bundle, sha256
from evaluation_protocol import read_triples

DATA = Path(os.environ.get("DATA", STUDY / "data/v1/C-time")).resolve()
RELATIONS = {"attributed-to", "detects", "mitigates", "subtechnique-of", "uses"}


def read(folder, split):
    path = folder / f"{split}.txt"
    return set(read_triples(path)) if path.is_file() else set()


class Data(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nodes, cls.facts, _ = builder.extract(BUNDLE)
        cls.kind = {k: v["type"] for k, v in cls.nodes.items()}
        cls.manifest = json.loads((DATA / "manifest.json").read_text())
        cls.source = read(DATA / "source", "train")
        cls.t = {f: {s: read(DATA / f"target_f{f}", s) for s in ("train", "valid", "test", "hidden")}
                 for f in build_split.FRACTIONS}

    def target_all(self, f):
        return set().union(*self.t[f].values())

    def test_manifest_hashes_match_files(self):
        for name, info in self.manifest["files"].items():
            self.assertEqual(sha256(DATA / name), info["sha256"], name)
            self.assertEqual(len(read_triples(DATA / name)), info["triples"], name)
        self.assertEqual(self.manifest["bundle"]["sha256"], sha256(BUNDLE))

    def test_rebuild_is_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            clients, _ = build_split.build(self.nodes, self.facts, self.manifest["candidate"])
            for name, splits in clients.items():
                for split, facts in splits.items():
                    if facts:
                        path = Path(tmp) / name / f"{split}.txt"
                        build_split.write_triples(path, facts)
                        self.assertEqual(sha256(path), self.manifest["files"][f"{name}/{split}.txt"]["sha256"])

    def test_every_fact_in_exactly_one_client_and_none_invented(self):
        target = self.target_all("1.00")
        self.assertFalse(self.source & target)
        self.assertEqual(self.source | target, set(self.facts))

    def test_no_leakage_L1_to_L4(self):
        src_pairs = {frozenset((h, t)) for h, _, t in self.source}
        src_dir = {(h, t) for h, _, t in self.source}
        src_ents = {e for h, _, t in self.source for e in (h, t)}
        for f, s in self.t.items():
            everything = self.target_all(f)
            evaluation = s["valid"] | s["test"]
            self.assertFalse(self.source & everything, f"L1 exact, f={f}")
            self.assertFalse(src_dir & {(t, h) for h, _, t in everything}, f"L1 inverse-equivalent, f={f}")
            self.assertFalse(s["valid"] & s["test"], f"L2 valid/test, f={f}")
            self.assertFalse(evaluation & (s["train"] | s["hidden"]), f"L2 exact, f={f}")
            self.assertFalse({(h, t) for h, _, t in evaluation} & {(t, h) for h, _, t in s["train"] | s["hidden"]},
                             f"L2 inverse-equivalent, f={f}")
            self.assertFalse([x for x in evaluation if frozenset((x[0], x[2])) in src_pairs], f"L3, f={f}")
            tgt_ents = {e for h, _, t in everything for e in (h, t)}
            actor = lambda e: self.kind[e] in build_split.ACTOR_TYPES
            self.assertFalse({e for e in tgt_ents if actor(e)} & src_ents, f"L4 target actors in source, f={f}")
            self.assertFalse({e for e in src_ents if actor(e)} & tgt_ents, f"L4 source actors in target, f={f}")

    def test_actor_units_are_not_split(self):
        units, _ = build_split.actor_units(self.nodes, self.facts)
        src_ents = {e for h, _, t in self.source for e in (h, t)}
        for members in units:
            sides = {m in src_ents for m in members}
            self.assertEqual(len(sides), 1, members)

    def test_relation_coverage(self):
        self.assertEqual({r for _, r, _ in self.source}, RELATIONS)
        for f, s in self.t.items():
            self.assertEqual({r for _, r, _ in s["train"]}, RELATIONS, f"target train relations, f={f}")
            for x in s["valid"] | s["test"]:
                self.assertEqual(x[1], "uses")
                self.assertIn(self.kind[x[0]], build_split.ACTOR_TYPES)

    def test_fractions_are_nested_and_evaluation_is_fixed(self):
        tr = [self.t[f]["train"] for f in build_split.FRACTIONS]
        self.assertTrue(tr[0] <= tr[1] <= tr[2])
        for f in build_split.FRACTIONS:
            self.assertEqual(self.t[f]["train"] | self.t[f]["hidden"], tr[2])
            self.assertEqual(self.t[f]["valid"], self.t["1.00"]["valid"])
            self.assertEqual(self.t[f]["test"], self.t["1.00"]["test"])

    def test_every_head_keeps_a_training_fact(self):
        heads = {h for h, _, _ in self.t["1.00"]["train"]}
        for f in build_split.FRACTIONS:
            self.assertEqual({h for h, _, _ in self.t[f]["train"]}, heads, f)

    def test_loader_message_graph_is_training_only(self):
        for f in build_split.FRACTIONS:
            s = self.t[f]
            b = load_bundle(DATA / f"target_f{f}")
            e2i, r2i = b["entity_to_id"], b["relation_to_id"]
            n = len(s["train"])
            g = b["train_data"]
            self.assertEqual(g.edge_index.shape[1], 2 * n)
            enc = lambda facts: {(e2i[h], r2i[r], e2i[t]) for h, r, t in facts}
            seen = {(int(h), int(r), int(t)) for h, t, r in zip(g.edge_index[0][:n], g.edge_index[1][:n], g.edge_type[:n])}
            self.assertEqual(seen, enc(s["train"]))
            self.assertEqual(b["valid_filter"].edge_index.shape[1], n + len(s["hidden"]) + len(s["valid"]))
            self.assertEqual(b["test_filter"].edge_index.shape[1], n + len(s["hidden"]) + len(s["valid"]) + len(s["test"]))


class Federation(unittest.TestCase):
    def test_fedavg_shares_weights_only_and_clients_train_on_their_own_graph(self):
        target, source = load_bundle(DATA / "target_f0.10"), load_bundle(DATA / "source")
        calls, averaged = [], []
        train_updates, average_states = base.train_updates, base.average_states

        def spy_train(model, optimizer, train_data, *rest):
            calls.append(train_data)
            return train_updates(model, optimizer, train_data, *rest)

        def spy_average(states, weights):
            averaged.append((states, weights))
            return average_states(states, weights)

        args = Namespace(rounds=1, local_steps=1, batch_size=2, num_negative=4, eval_batch_size=4, eval_limit=2,
                         lr=5e-4, weight_decay=0.0)
        state = torch.load(base.CHECKPOINT, map_location="cpu")["model"]
        base.train_updates, base.average_states = spy_train, spy_average
        try:
            conditions_runner.run_fedavg(20261209, target, source, args, state)
        finally:
            base.train_updates, base.average_states = train_updates, average_states
        self.assertEqual([id(g) for g in calls], [id(target["train_data"]), id(source["train_data"])])
        self.assertEqual(len(averaged), 1)
        states, weights = averaged[0]
        self.assertEqual(weights, [1.0, 1.0])
        for st in states:
            self.assertEqual(set(st), set(state))
            self.assertTrue(all(isinstance(v, torch.Tensor) for v in st.values()))


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main(verbosity=2)
