import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "synthetic_memory_pilot"))
from trajectory_memory.embedding_cache import CachedEncoder
from trajectory_memory.encoders import LSAEncoder
from longmemeval import assign_splits, duplicate_check, summarize_checks


class FakeEncoder:
    metadata = {"name": "test encoder"}

    def __init__(self, version="one", fail_after=None):
        self.cache_identity = {"kind": "test", "version": version}
        self.calls = []
        self.fail_after = fail_after

    def encode(self, texts):
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise KeyboardInterrupt()
        self.calls.append(list(texts))
        return np.array([np.frombuffer(hashlib.sha256((self.cache_identity["version"] + t).encode()).digest(),
                                       dtype=np.uint8)[:8] / 255 for t in texts], dtype=np.float32)


def fixture():
    items = []
    for kind in ["multi-session", "knowledge-update", "temporal-reasoning"]:
        for i in range(8):
            turns = [
                {"role": "user", "content": "I live in Zurich and study statistics.", "has_answer": True},
                {"role": "assistant", "content": "You study statistics in Zurich."},
                {"role": "user", "content": f"My current job is library assistant number {i}."},
                {"role": "user", "content": "I travelled to Bern last summer."},
                {"role": "assistant", "content": "The train trip was short and comfortable."},
            ]
            items.append(dict(question_id=f"{kind}_{i}", question_type=kind,
                              question="Where do I live and what do I study?", answer="Zurich, statistics",
                              haystack_sessions=[turns], haystack_session_ids=[f"session_{i}"],
                              answer_session_ids=[f"session_{i}"]))
    return items


class CacheTests(unittest.TestCase):
    def test_deduplicates_restores_order_and_reopens_without_inference(self):
        with tempfile.TemporaryDirectory() as root:
            enc = FakeEncoder()
            with CachedEncoder(enc, root, batch_size=2, verbose=False) as cache:
                actual = cache.encode(["alpha", "beta", "alpha", "gamma"])
                self.assertEqual(enc.calls, [["alpha", "beta"], ["gamma"]])
                self.assertEqual(cache.encoded, 3)
            with CachedEncoder(FakeEncoder(fail_after=0), root, verbose=False) as cache:
                warm = cache.encode(["gamma", "alpha", "beta", "alpha"])
                np.testing.assert_array_equal(warm, actual[[3, 0, 1, 2]])
                self.assertEqual(cache.encoded, 0)

    def test_interruption_retains_completed_batches(self):
        with tempfile.TemporaryDirectory() as root:
            with CachedEncoder(FakeEncoder(fail_after=1), root, batch_size=2, verbose=False) as cache:
                with self.assertRaises(KeyboardInterrupt):
                    cache.encode(["a", "b", "c", "d", "e"])
            enc = FakeEncoder()
            with CachedEncoder(enc, root, batch_size=2, verbose=False) as cache:
                cache.encode(["a", "b", "c", "d", "e"])
                self.assertEqual(enc.calls, [["c", "d"], ["e"]])

    def test_abrupt_process_exit_retains_committed_batch(self):
        with tempfile.TemporaryDirectory() as root:
            code = '''
import os, sys
from test_embedding_cache import FakeEncoder
from trajectory_memory.embedding_cache import CachedEncoder
class ExitEncoder(FakeEncoder):
    def encode(self, texts):
        if self.calls:
            os._exit(19)
        return super().encode(texts)
with CachedEncoder(ExitEncoder(), sys.argv[1], batch_size=2, verbose=False) as cache:
    cache.encode(["a", "b", "c"])
'''
            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "tests"), str(ROOT)])
            result = subprocess.run([sys.executable, "-c", code, root], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 19, result.stderr)
            enc = FakeEncoder()
            with CachedEncoder(enc, root, verbose=False) as cache:
                cache.encode(["a", "b", "c"])
                self.assertEqual(enc.calls, [["c"]])

    def test_model_and_exact_text_changes_do_not_hit_old_vectors(self):
        with tempfile.TemporaryDirectory() as root:
            with CachedEncoder(FakeEncoder("one"), root, verbose=False) as cache:
                old = cache.encode(["Text"])
            enc = FakeEncoder("two")
            with CachedEncoder(enc, root, verbose=False) as cache:
                new = cache.encode(["Text", "text", "Text "])
                self.assertEqual(cache.encoded, 3)
                self.assertFalse(np.array_equal(old[0], new[0]))

    def test_two_open_connections_observe_new_commits(self):
        with tempfile.TemporaryDirectory() as root:
            with CachedEncoder(FakeEncoder(), root, verbose=False) as a, CachedEncoder(FakeEncoder(), root, verbose=False) as b:
                x = a.encode(["shared"])
                np.testing.assert_array_equal(b.encode(["shared"]), x)
                self.assertEqual(b.encoded, 0)

    def test_invalid_batch_never_enters_cache(self):
        bad = FakeEncoder()
        bad.encode = lambda texts: np.full((len(texts), 8), np.nan, dtype=np.float32)
        with tempfile.TemporaryDirectory() as root:
            with CachedEncoder(bad, root, verbose=False) as cache:
                with self.assertRaises(ValueError):
                    cache.encode(["bad"])
            with CachedEncoder(FakeEncoder(), root, verbose=False) as cache:
                self.assertTrue(np.isfinite(cache.encode(["bad"])).all())
                self.assertEqual(cache.encoded, 1)

    def test_lsa_cached_and_direct_embeddings_agree_and_fit_changes_identity(self):
        texts = ["city Zurich statistics university", "city Bern rail transport", "physics biology laboratory",
                 "library reading research thesis", "hiking mountains climbing", "city Zurich statistics university"]
        with threadpool_limits(limits=1), tempfile.TemporaryDirectory() as root:
            enc = LSAEncoder(4, 2026)
            enc.fit(texts)
            reference = enc.encode(texts)
            with CachedEncoder(enc, root, batch_size=2, verbose=False) as cache:
                np.testing.assert_allclose(cache.encode(texts), reference, rtol=1e-6, atol=1e-7)
                identity = cache.namespace
            other = LSAEncoder(4, 2026)
            other.fit(texts + ["additional economics business marketing"])
            with CachedEncoder(other, root, verbose=False) as cache:
                self.assertNotEqual(identity, cache.namespace)

    def test_longmemeval_warm_run_and_type_subset_need_no_model_inference(self):
        items = fixture()
        split = assign_splits(items)
        with tempfile.TemporaryDirectory() as root, contextlib.redirect_stdout(io.StringIO()):
            with patch("trajectory_memory.encoders.MiniLMEncoder", side_effect=lambda *a: FakeEncoder()):
                first = duplicate_check(items, split, "minilm", [8, 32], 5, False, "cosine", None, "test", 1, root, 2)
            with patch("trajectory_memory.encoders.MiniLMEncoder", side_effect=lambda *a: FakeEncoder(fail_after=0)):
                second = duplicate_check(items, split, "minilm", [8, 32], 5, False, "cosine", None, "test", 1, root, 2)
                narrowed_split = {i: s if items[i]["question_type"] == "multi-session" else "test" for i, s in split.items()}
                subset = duplicate_check(items, narrowed_split, "minilm", [8, 32], 5, False, "cosine", None, "test", 1, root, 2)
            self.assertEqual(first["cases"], second["cases"])
            self.assertEqual(first["overall"], second["overall"])
            self.assertEqual(second["embedding_cache"]["newly_encoded_texts"], 0)
            self.assertEqual(subset["overall"], first["by_type"]["multi-session"])
            self.assertEqual(sum(v["questions"] for v in first["by_type"].values()), first["overall"]["questions"])

    def test_empty_evidence_skips_model_and_report_is_strict_json(self):
        items = fixture()
        for it in items:
            for session in it["haystack_sessions"]:
                for turn in session:
                    turn["has_answer"] = False
        with patch("trajectory_memory.encoders.MiniLMEncoder", side_effect=AssertionError("must not load model")), contextlib.redirect_stdout(io.StringIO()):
            report = duplicate_check(items, assign_splits(items), "minilm", [8], 5, False, "cosine", None, "test", 1)
        self.assertEqual(report["overall"]["questions"], 0)
        json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
