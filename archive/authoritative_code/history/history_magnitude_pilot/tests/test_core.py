import unittest
import numpy as np
from scipy.integrate import quad
from trajectory_memory.magnitude import (kernels, distances, magnitude, SchurState, Geometry,
                                         log_weights, calibrate_profiles, select, RIDGE)
from trajectory_memory.encoders import normalized, LSAEncoder
from trajectory_memory.data import public_query, query_texts
from trajectory_memory.experiment import Retrieval, selection_metrics
from trajectory_memory.reader import parse_answer, text_scores


class MagnitudeTests(unittest.TestCase):
    def test_two_points_analytic(self):
        for t in [.001, .1, 1, 10, 100]:
            k = kernels(np.array([[0., 2], [2, 0]]), [t])[0]
            self.assertAlmostEqual(magnitude(k), 2 / (1 + RIDGE + np.exp(-2 * t)), places=10)

    def test_duplicate_and_random_marginal_gains(self):
        rng = np.random.default_rng(35)
        for _ in range(8):
            x = normalized(rng.normal(size=(9, 5)))
            x[-1] = x[0]
            bank = kernels(distances(x), np.geomspace(.0001, 1000, 17))
            state, chosen = SchurState(bank), []
            for j in [0, 8, 2, 5, 6]:
                gains = state.gains()
                for a in set(range(len(x))) - set(chosen):
                    for si, k in enumerate(bank):
                        old = magnitude(k[np.ix_(chosen, chosen)])
                        new = magnitude(k[np.ix_(chosen + [a], chosen + [a])])
                        self.assertAlmostEqual(gains[si, a], new - old, places=6)
                state.add(j)
                chosen.append(j)
                np.testing.assert_allclose(state.values, [magnitude(k[np.ix_(chosen, chosen)]) for k in bank], atol=1e-6)

    def test_nonuniform_log_quadrature(self):
        scales = np.exp([0, .1, .7, 1.9, 4.])
        w = log_weights(scales)
        self.assertAlmostEqual(w.sum(), 1)
        self.assertTrue(np.all(w > 0))
        self.assertAlmostEqual(w @ (3 * np.log(scales) + 2), 8)

    def test_profile_integral_against_quad(self):
        x = np.array([[0., 0.], [.1, 0], [1, .5], [3, 2.]])
        d = distances(x)
        t = np.geomspace(.001, 100, 2049)
        g = Geometry.make(d, t)
        subset = [0, 3]
        values = np.array([magnitude(k[np.ix_(subset, subset)]) for k in g.bank]) / g.full
        def f(u):
            k = kernels(d, [np.exp(u)])[0]
            return magnitude(k[np.ix_(subset, subset)]) / magnitude(k)
        exact = quad(f, np.log(t[0]), np.log(t[-1]), epsabs=1e-9)[0] / np.log(t[-1] / t[0])
        self.assertAlmostEqual(log_weights(t) @ values, exact, places=5)

    def test_calibration_reports_unmet_cap(self):
        d = distances(np.array([[0, 0], [.01, 0], [.03, 0], [1, 0], [5, 1]]))
        result = calibrate_profiles([d], dense_nodes=65, max_nodes=3, tolerance=1e-6)
        self.assertFalse(result["tolerance_met"])
        self.assertEqual(len(result["scales"]), 3)

    def test_budget_and_objective_recompute(self):
        rng = np.random.default_rng(29)
        for _ in range(20):
            x = rng.normal(size=(12, 5))
            g = Geometry.make(distances(x), np.geomspace(.01, 10, 11))
            w = log_weights(g.scales)
            rel, cost = rng.random(12), rng.integers(10, 80, 12)
            result = select(g, rel, cost, 145, .4, w)
            s = result["indices"]
            self.assertEqual(len(s), len(set(s)))
            self.assertEqual(result["tokens"], sum(cost[s]))
            self.assertLessEqual(result["tokens"], 145)
            r = (rel - rel.min()) / np.ptp(rel)
            expected = .4 * r[s].sum() + .6 * len(x) * sum(wi * magnitude(k[np.ix_(s, s)]) / mi for wi, k, mi in zip(w, g.bank, g.full))
            self.assertAlmostEqual(result["objective"], expected, places=6)

    def test_no_fitting_memory_and_relevance_limit(self):
        g = Geometry.make(distances(np.eye(5)), [.1, 1, 10])
        r, c = np.arange(5.), np.array([10, 20, 15, 12, 18])
        self.assertEqual(select(g, r, c, 5, .5, np.ones(3)/3)["indices"], [])
        a = select(g, r, c, 40, 1, np.ones(3)/3)
        b = select(None, r, c, 40, method="relevance")
        self.assertEqual(a["indices"], b["indices"])


class DataTests(unittest.TestCase):
    def test_public_input_excludes_current_gold(self):
        row = dict(Context=["Which artist?", "Maya"], Conversation_no=1, Turn_no=2,
                   Question="Where was she born?", Answer="SECRET", Topic="SECRET", Rationale="SECRET")
        q = public_query(row)
        self.assertNotIn("SECRET", str(q))
        row.update(Answer="DIFFERENT", Topic="DIFFERENT", Rationale="DIFFERENT")
        self.assertEqual(q, public_query(row))
        self.assertLess(max(h["turn"] for h in q["history"]), q["turn"])

    def test_prefix_misalignment_rejected(self):
        with self.assertRaises(ValueError):
            public_query(dict(Context=["q", "a", "future", "answer"], Conversation_no=1, Turn_no=2, Question="q"))

    def test_history_changes_retrieval_without_gold(self):
        corpus = [dict(id="a", cost=10), dict(id="b", cost=10)]
        arrays = dict(corpus=np.eye(2), current=np.zeros((2, 2)), history=np.eye(2),
                      permuted=np.eye(2)[::-1], concat=np.eye(2))
        retrieval = Retrieval(corpus, arrays, 1)
        self.assertEqual(retrieval.pool(0, beta=1)["ids"], ["a"])
        self.assertEqual(retrieval.pool(1, beta=1)["ids"], ["b"])
        self.assertEqual(retrieval.pool(0, beta=0)["ids"], retrieval.pool(1, beta=0)["ids"])

    def test_missing_gold_not_inserted(self):
        pool = dict(ids=["a", "b"])
        metric = selection_metrics(pool, dict(indices=[0], tokens=10), ["missing"])
        self.assertEqual(metric["evidence_recall"], 0)
        self.assertEqual(metric["candidate_recall"], 0)
        self.assertIsNone(metric["conditional_recall"])
        self.assertEqual(pool["ids"], ["a", "b"])

    def test_lsa_vectors_and_cost(self):
        encoder = LSAEncoder(2, 4)
        encoder.fit(["Dogs play outdoors", "Cats play indoors", "Cooking fresh pasta", "Plants grow outdoors"])
        x = encoder.encode(["Dogs outdoors", "Cooking pasta"])
        np.testing.assert_allclose(np.linalg.norm(x, axis=1), 1, atol=1e-6)
        self.assertEqual(encoder.count("Hello, world!"), 4)

    def test_reader_parser_and_metrics(self):
        a, c, valid = parse_answer('```json\n{"answer":"Paris", "citations":["p_1"]}\n```')
        self.assertEqual((a, c, valid), ("Paris", ["p_1"], True))
        self.assertFalse(parse_answer("not json")[2])
        self.assertEqual(text_scores("The Paris", ["Paris"]), (1, 1))


if __name__ == "__main__":
    unittest.main()
