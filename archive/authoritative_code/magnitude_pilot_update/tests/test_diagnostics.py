import contextlib
import io
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "synthetic_memory_pilot"))
from diagnostics import budget_diagnostics, cosine_pairs, distribution, summarize_extended
from longmemeval import duplicate_check, summarize_checks
from test_embedding_cache import FakeEncoder


def turn(mid, sid):
    return {"id": mid, "session": sid}


class DiagnosticTests(unittest.TestCase):
    def test_budget_oracle_matches_exhaustive_search(self):
        rng = np.random.default_rng(72)
        for _ in range(30):
            candidates = [turn(f"m{i}", f"s{i % 3}") for i in range(7)]
            costs = rng.integers(1, 15, size=7)
            gold = [t for t in candidates if rng.random() < .65] + [turn("missing", "s3")]
            gold_ids = {t["id"] for t in gold}
            budget = int(rng.integers(0, 30))
            oracle, session = budget_diagnostics(gold, candidates, costs, [], budget)
            best_turns = best_sessions = 0
            for bits in itertools.product([False, True], repeat=7):
                picked = [i for i, yes in enumerate(bits) if yes]
                if sum(costs[i] for i in picked) <= budget:
                    hits = [candidates[i] for i in picked if candidates[i]["id"] in gold_ids]
                    best_turns = max(best_turns, len(hits))
                    best_sessions = max(best_sessions, len({t["session"] for t in hits}))
            self.assertEqual(oracle["gold_turns_affordable"], best_turns)
            self.assertEqual(session["oracle_evidence_session_recall"], best_sessions / len({t["session"] for t in gold}))

    def test_loss_decomposition_distinguishes_retrieval_budget_and_selection(self):
        gold = [turn("a", "s1"), turn("b", "s2"), turn("c", "s3")]
        pool = [gold[0], gold[1], turn("filler", "s2")]
        oracle, _ = budget_diagnostics(gold, pool, [4, 9, 2], [2], 5)
        self.assertAlmostEqual(oracle["retrieval_loss"], 1 / 3)
        self.assertAlmostEqual(oracle["unavoidable_budget_loss"], 1 / 3)
        self.assertAlmostEqual(oracle["selection_gap"], 1 / 3)
        self.assertAlmostEqual(sum(oracle[k] for k in ["retrieval_loss", "unavoidable_budget_loss", "selection_gap"]), 1.0)

    def test_session_visit_is_not_evidence_and_partial_session_is_not_complete(self):
        gold = [turn("a", "s1"), turn("b", "s1"), turn("c", "s2")]
        pool = gold + [turn("irrelevant", "s2")]
        oracle, session = budget_diagnostics(gold, pool, [2, 2, 2, 1], [0, 3], 6)
        self.assertEqual(session["selected_sessions"], 2)
        self.assertEqual(session["selected_gold_sessions_any_turn"], 2)
        self.assertEqual(session["evidence_sessions_covered"], 1)
        self.assertEqual(session["evidence_sessions_fully_covered"], 0)
        self.assertEqual(session["evidence_session_state"], "some")
        self.assertEqual(session["oracle_evidence_session_recall"], 1.0)
        self.assertFalse(session["all_gold_turns_covered"])
        self.assertEqual(oracle["recall"], 1.0)

    def test_no_affordable_evidence_has_no_selection_gap(self):
        gold = [turn("a", "s1")]
        oracle, session = budget_diagnostics(gold, gold, [20], [], 5)
        self.assertEqual(oracle["recall"], 0)
        self.assertEqual(oracle["selection_gap"], 0)
        self.assertEqual(oracle["unavoidable_budget_loss"], 1)
        self.assertIsNone(session["largest_selected_session_share"])
        self.assertEqual(session["evidence_session_state"], "none")

    def test_invalid_selection_cannot_be_reported_as_feasible(self):
        gold = [turn("a", "s1")]
        with self.assertRaises(ValueError):
            budget_diagnostics(gold, gold, [20], [0], 5)

    def test_pair_percentiles_exclude_diagonal_and_empty_is_undefined(self):
        values = cosine_pairs(np.array([[1, 0], [0, 1], [-1, 0]], dtype=float))
        self.assertEqual(values, [0.0, -1.0, 0.0])
        self.assertEqual(distribution(values)["pairs"], 3)
        self.assertEqual(distribution(values)["p50"], 0)
        self.assertEqual(cosine_pairs(np.array([[1, 0]])), [])
        self.assertIsNone(distribution([])["p99"])

    def test_pooled_quantiles_are_not_averaged_question_quantiles(self):
        rows = [{"budgets": {"10": {"selected_pairwise_cosines": [0.0]}}},
                {"budgets": {"10": {"selected_pairwise_cosines": [1.0, 1.0, 1.0]}}}]
        summary = summarize_extended(rows, 10)
        self.assertEqual(summary["selected_pairwise_cosine"]["p50"], 1.0)
        self.assertEqual(summary["questions_with_selected_cosine_pairs"], 2)
        self.assertIsNone(summary["oracle_recall"])
        json.dumps(summary, allow_nan=False)

    def test_multisession_report_integration_and_cached_reuse(self):
        items = []
        for i in range(2):
            items.append(dict(question_id=f"q{i}", question_type="multi-session", question="Where do I live?",
                haystack_session_ids=["s1", "s2"],
                haystack_sessions=[
                    [{"role": "user", "content": "I live in Zurich.", "has_answer": True},
                     {"role": "user", "content": "Weather today is nice."}],
                    [{"role": "user", "content": "I previously lived in Bern.", "has_answer": True},
                     {"role": "assistant", "content": "You enjoy travel."}]]))
        with tempfile.TemporaryDirectory() as root, contextlib.redirect_stdout(io.StringIO()):
            with patch("trajectory_memory.encoders.MiniLMEncoder", side_effect=lambda *a: FakeEncoder()):
                cold = duplicate_check(items, {0: "validation", 1: "validation"}, "minilm", [5, 50], 4,
                                       False, "cosine", None, "test", 1, root)
            with patch("trajectory_memory.encoders.MiniLMEncoder", side_effect=lambda *a: FakeEncoder(fail_after=0)):
                warm = duplicate_check(items, {0: "validation", 1: "validation"}, "minilm", [5, 50], 4,
                                       False, "cosine", None, "test", 1, root)
        self.assertEqual(cold["cases"], warm["cases"])
        self.assertEqual(warm["embedding_cache"]["newly_encoded_texts"], 0)
        for r in cold["cases"]:
            for b, x in r["budgets"].items():
                self.assertLessEqual(x["evidence_recall"], x["oracle"]["recall"])
                o = x["oracle"]
                self.assertAlmostEqual(1 - x["evidence_recall"], o["retrieval_loss"] + o["unavoidable_budget_loss"] + o["selection_gap"])
        summary = summarize_checks(cold["cases"], [5, 50])
        self.assertEqual(summary["budgets"]["50"]["multi_session_evidence"]["questions"], 2)
        self.assertEqual(summary["budgets"]["50"]["multi_session_evidence"]["all_gold_turns_share"], 1.0)
        json.dumps(summary, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
