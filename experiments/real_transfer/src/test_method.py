#!/usr/bin/env python3
"""Unit tests for the real-transfer code (python test_method.py). Uses the C-time data; trains a few steps only."""
import argparse
import json
import tempfile
import unittest

import torch

from env import SOURCE, TARGET, is_actor_use
import fedavg_runner as base
import conditions_runner as runner
from common import load_bundle
import analyze_transfer as analyze
import run_method
import sparse

CKPT = torch.load(base.CHECKPOINT, map_location="cpu")["model"]
TGT = load_bundle(TARGET["0.10"])
SRC = load_bundle(SOURCE)


def same_state(a, b):
    return all(torch.equal(a[k], b[k]) for k in a) and a.keys() == b.keys()


class TrainStep(unittest.TestCase):
    def test_matches_registered_train_updates(self):
        """With the registered inputs, train_steps is the registered local training step, bit for bit."""
        seed, round_index, steps = 20261211, 3, 3
        m1, m2 = base.new_model(CKPT), base.new_model(CKPT)
        o1, o2 = base.make_optimizer(m1, 5e-4, 0.0), base.make_optimizer(m2, 5e-4, 0.0)
        l1 = base.train_updates(m1, o1, TGT["train_data"], steps, 8, 64, seed, "target", round_index)
        data = TGT["train_data"]
        l2 = sparse.train_steps(m2, o2, data, data, base.triples_tensor(data), steps, 8, 64,
                                seed * 1000003 + 7919 + round_index * 97)
        self.assertEqual(l1, l2)
        self.assertTrue(same_state(base.clone_state(m1), base.clone_state(m2)))


class SparseView(unittest.TestCase):
    def test_view_partitions_source_facts(self):
        facts = SRC["raw_triples"]["train"]
        keep, held = sparse.per_head_view(facts, 0.10, key="7:1")
        self.assertEqual(sorted(keep + held), sorted(facts))
        self.assertFalse(set(keep) & set(held))
        heads = {h for h, _, _ in facts}
        self.assertEqual({h for h, _, _ in keep}, heads)  # every head keeps at least one fact

    def test_view_is_deterministic_and_redrawn_per_round(self):
        facts = SRC["raw_triples"]["train"]
        a, b = sparse.per_head_view(facts, 0.10, "7:1"), sparse.per_head_view(facts, 0.10, "7:1")
        c = sparse.per_head_view(facts, 0.10, "7:2")
        self.assertEqual(a, b)
        self.assertNotEqual(a[0], c[0])

    def test_view_graph_holds_only_kept_facts(self):
        keep, held = sparse.per_head_view(SRC["raw_triples"]["train"], 0.10, "7:1")
        g = sparse.message_graph(keep, SRC)
        self.assertEqual(g.edge_index.shape[1], 2 * len(keep))  # facts plus inverses
        self.assertEqual(g.num_nodes, len(SRC["entity_to_id"]))
        direct = {tuple(x) for x in sparse.fact_tensor(keep, SRC).tolist()}
        held_ids = {tuple(x) for x in sparse.fact_tensor(held, SRC).tolist()}
        self.assertFalse(direct & held_ids)


class SourceStage(unittest.TestCase):
    def test_source_holdout_is_fixed_and_disjoint(self):
        train, valid = sparse.source_holdout(SRC, is_actor_use)
        self.assertEqual((train, valid), sparse.source_holdout(SRC, is_actor_use))  # same for every seed
        self.assertFalse(set(train) & set(valid))
        self.assertEqual(sorted(train + valid), sorted(SRC["raw_triples"]["train"]))
        n_task = sum(map(is_actor_use, SRC["raw_triples"]["train"]))
        self.assertTrue(all(map(is_actor_use, valid)) and len(valid) == round(0.10 * n_task))

    def spy_run(self, fraction, checkpoint="source_valid"):
        seen, orig = [], sparse.train_steps

        def spy(model, optimizer, graph, negatives_from, positives, *rest):
            seen.append((graph, negatives_from, positives))
            return orig(model, optimizer, graph, negatives_from, positives, *rest)

        sparse.train_steps = spy
        try:
            model = base.new_model(CKPT)
            record = sparse.train_source(model, SRC, fraction=fraction, rounds=2, steps=1, batch_size=8,
                                         num_negative=8, seed=5, lr=5e-4, weight_decay=0.0, task=is_actor_use,
                                         select=True, checkpoint=checkpoint)
        finally:
            sparse.train_steps = orig
        return model, record, seen

    def test_source_stage_reads_only_source_training_facts(self):
        """Views and positives come from the source's own facts; source-validation facts are never trained on."""
        id_to_entity = {i: e for e, i in SRC["entity_to_id"].items()}
        id_to_rel = {i: r for r, i in SRC["relation_to_id"].items()}
        train, valid = map(set, sparse.source_holdout(SRC, is_actor_use))
        for fraction in (0.10, 1.0):
            _, record, seen = self.spy_run(fraction)
            self.assertEqual(len(record["source_valid_mrr"]), 3)  # round 0 + 2 rounds
            self.assertIn(record["selected_round"], (0, 1, 2))
            for graph, negatives_from, positives in seen:
                self.assertIs(negatives_from, SRC["train_data"])  # negatives filtered by every known source fact
                n = graph.edge_index.shape[1] // 2
                kept = {(id_to_entity[h], id_to_rel[r], id_to_entity[t])
                        for h, t, r in zip(graph.edge_index[0, :n].tolist(), graph.edge_index[1, :n].tolist(),
                                           graph.edge_type[:n].tolist())}
                pos = {(id_to_entity[h], id_to_rel[r], id_to_entity[t]) for h, t, r in positives.tolist()}
                self.assertTrue(kept <= train and pos <= train)
                self.assertTrue(all(map(is_actor_use, pos)))
                if fraction < 1:
                    self.assertFalse(pos & kept)  # sparse views hold positives out of the message graph

    def test_selection_leaves_the_best_source_round(self):
        model, record, _ = self.spy_run(0.10)
        scorer = sparse.source_validation(SRC, *sparse.source_holdout(SRC, is_actor_use), 0.10)
        best = max(record["source_valid_mrr"])
        self.assertEqual(record["source_valid_mrr"][record["selected_round"]], best)
        self.assertAlmostEqual(base.evaluate(model, scorer, "valid", 4, 0)["mrr"], best, places=6)

    def test_final_checkpoint_leaves_the_last_round(self):
        model, record, _ = self.spy_run(0.10, checkpoint="final")
        scorer = sparse.source_validation(SRC, *sparse.source_holdout(SRC, is_actor_use), 0.10)
        self.assertEqual(record["selected_round"], 2)
        self.assertEqual(record["checkpoint_rule"], "final")
        self.assertEqual(record["source_valid_mrr"][record["source_valid_best_round"]], max(record["source_valid_mrr"]))
        self.assertAlmostEqual(base.evaluate(model, scorer, "valid", 4, 0)["mrr"], record["source_valid_mrr"][-1], places=6)


def run_args(**kw):
    defaults = dict(rounds=2, local_steps=2, batch_size=8, num_negative=8, eval_batch_size=4, eval_limit=0, lr=5e-4,
                    weight_decay=0.0, source_rounds=1, source_fraction=0.10, source_checkpoint="final",
                    source_cache=None, threads=2)
    return argparse.Namespace(**(defaults | kw))


class Checkpoints(unittest.TestCase):
    def test_scoring_reloads_the_selected_and_starting_weights(self):
        """score() on validation reproduces the selected round's and round 0's validation MRR."""
        args = run_args()
        state = run_method.starting_state("sparse_source", 7, SRC, args, CKPT)[0]
        model, best_round, history = runner.run_local(7, TGT, args, state)
        scores = run_method.score(model, state, TGT, args, split="valid")
        self.assertAlmostEqual(scores["test"]["mrr"], history[best_round], places=6)
        self.assertAlmostEqual(scores["zero_shot"]["test"]["mrr"], history[0], places=6)
        self.assertEqual(len(scores["per_query_test"]), len(TGT["raw_triples"]["valid"]))
        json.dumps(scores)  # the record is saved as JSON

    def test_source_cache_returns_the_same_weights(self):
        with tempfile.TemporaryDirectory() as cache:
            args = run_args(source_cache=cache)
            fresh, record = run_method.starting_state("dense_source", 7, SRC, args, CKPT)
            again, loaded = run_method.starting_state("dense_source", 7, SRC, args, CKPT)
            self.assertTrue(same_state(fresh, again))
            self.assertNotIn("loaded_from_cache", record)
            self.assertEqual(loaded["state_sha256"], record["state_sha256"])
            self.assertIn("loaded_from_cache", loaded)
            # A different source configuration is a different cache entry, and the stage is deterministic.
            other = run_method.starting_state("dense_source", 8, SRC, args, CKPT)[1]
            self.assertNotIn("loaded_from_cache", other)
            rerun = run_method.starting_state("dense_source", 7, SRC, run_args(), CKPT)[0]
            self.assertTrue(same_state(fresh, rerun))


def fake_run(arm, fine, zero, skip_test=False):
    """A run file whose seed s has per-query test MRRs fine[s] (and zero[s] zero-shot) over the same triples."""
    triples = [[f"h{i}", "uses", f"t{i}"] for i in range(len(next(iter(fine.values()))))]

    def block(values):
        return {"mrr": sum(values) / len(values)}, [{"triple": t, "mrr": v} for t, v in zip(triples, values)]

    results = []
    for seed in fine:
        (test, per_query), (z_test, z_per_query) = block(fine[seed]), block(zero[seed])
        results.append({"seed": seed, "selected_round": 2, "selected_validation_mrr": 0.1,
                        "validation_mrr_history": [0.05, 0.08, 0.1], "starting_weights": {"selected_round": 4},
                        "test": test, "per_query_test": per_query,
                        "zero_shot": {"test": z_test, "per_query_test": z_per_query}})
    config = {"rounds": 20, "local_steps": 10, "skip_test": skip_test} | (
        {} if arm == "local" else {"source_rounds": 30, "source_fraction": 1.0 if arm == "dense_source" else 0.1})
    return {"protocol": run_method.PROTOCOL_VERSION, "arm": arm, "target": "t", "target_sha256": {"test.txt": "x"},
            "checkpoint_sha256": "c", "config": config, "diagnostic_only": False, "results": results}


class Analysis(unittest.TestCase):
    def run_analysis(self, baseline, candidate, *scores):
        with tempfile.TemporaryDirectory() as d:
            paths = [f"{d}/baseline.json", f"{d}/candidate.json"]
            for path, run in zip(paths, (baseline, candidate)):
                with open(path, "w") as f:
                    json.dump(run, f)
            return analyze.analyze(*paths, f"{d}/analysis.json", *scores)

    def test_criterion_on_known_deltas(self):
        seeds = (1, 2, 3)
        flat = {s: [0.1] * 6 for s in seeds}
        local = fake_run("local", flat, flat)
        gain = fake_run("sparse_source", {s: [0.1 + 0.04 + 0.01 * i] * 6 for i, s in enumerate(seeds)},
                        {s: [0.1 + 0.01 * i] * 6 for i, s in enumerate(seeds)})
        out = self.run_analysis(local, gain)
        self.assertAlmostEqual(out["mean_delta"], 0.05)
        self.assertGreater(out["seed_level_95pct_t_ci"][0], 0)
        self.assertGreater(out["bootstrap_95pct_ci"][0], 0)
        self.assertTrue(out["PASS"])
        zero = self.run_analysis(local, gain, "zero_shot", "zero_shot")  # +0.01 mean: below the threshold
        self.assertAlmostEqual(zero["mean_delta"], 0.01)
        self.assertFalse(zero["PASS"])
        self.assertEqual(zero["comparison"], "sparse_source[zero_shot] - local[zero_shot]")
        self.assertEqual(zero["per_seed"][0]["source_selected_round_candidate"], 4)

    def test_refuses_misaligned_or_validation_only_runs(self):
        flat = {s: [0.1] * 4 for s in (1, 2, 3)}
        local, cand = fake_run("local", flat, flat), fake_run("sparse_source", flat, flat)
        cand["results"][1]["per_query_test"].reverse()
        cand["results"][1]["per_query_test"][0]["triple"] = ["other", "uses", "x"]
        with self.assertRaises(AssertionError):
            analyze.pair(local, cand)
        with self.assertRaises(ValueError):
            analyze.pair(local, fake_run("sparse_source", flat, flat, skip_test=True))
        other = fake_run("sparse_source", flat, flat)
        other["config"]["rounds"] = 30  # a different target budget
        with self.assertRaises(ValueError):
            analyze.pair(local, other)
        # Two source arms may differ in their source settings.
        self.assertEqual(len(analyze.pair(fake_run("dense_source", flat, flat), fake_run("sparse_source", flat, flat))[0]), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
