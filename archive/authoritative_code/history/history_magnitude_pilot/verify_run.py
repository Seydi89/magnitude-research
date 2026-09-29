"""Independent artifact audit: data identity, pools, budgets, scores and profile errors."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from trajectory_memory.data import digest


def verify(data, run):
    data, run = Path(data), Path(run)
    meta = json.loads((data / "manifest.json").read_text())
    for f, h in meta["files_sha256"].items():
        assert digest(data / f) == h, f
    manifest = json.loads((run / "manifest.json").read_text())
    for f, h in manifest["source_sha256"].items():
        assert digest(Path(__file__).parent / "trajectory_memory" / f) == h, f
    queries = json.loads((data / "queries.json").read_text())
    qmap = {q["id"]: q for q in queries}
    labels = json.loads((data / "labels.json").read_text())
    memories = json.loads((data / "corpus.json").read_text())
    corpus = {m["id"]: m for m in memories}
    rows = json.loads((run / "selections.json").read_text())
    arrays = np.load(data / "embeddings.npz", allow_pickle=False)
    frozen = json.loads((run / "frozen_validation.json").read_text())
    groups = {s: {q["conversation"] for q in queries if q["split"] == s} for s in ["calibration", "validation", "test"]}
    assert not groups["calibration"] & groups["validation"]
    assert not groups["calibration"] & groups["test"]
    assert not groups["validation"] & groups["test"]
    for q in queries:
        assert set(q) == {"id", "conversation", "turn", "question", "history", "split"}
        assert all(h["turn"] < q["turn"] for h in q["history"])
    pool_rows = json.loads((run / "candidate_pools.json").read_text())
    position = {q["id"]: i for i, q in enumerate(queries)}
    corpus_matrix = np.asarray(arrays["corpus"], float)
    for record in pool_rows:
        i = position[record["case"]]
        for mode in ["current", "history", "concat", "permuted"]:
            q = arrays["current"][i].astype(float)
            if mode in ["history", "permuted"]:
                # Match original float32 addition before normalization exactly.
                q = arrays["current"][i] + frozen["history_beta"] * arrays[mode][i]
                q = q.astype(float)
                q /= max(np.linalg.norm(q), 1e-12)
            elif mode == "concat":
                q = arrays["concat"][i].astype(float)
            scores = corpus_matrix @ q
            order = np.argsort(-scores, kind="stable")[:len(record[mode]["ids"])]
            assert record[mode]["ids"] == [memories[j]["id"] for j in order]
            np.testing.assert_allclose(record[mode]["relevance"], scores[order], atol=1e-12)
    poolmap = {p["case"]: p for p in pool_rows}
    for r in rows:
        q = qmap[r["case"]]
        assert q["split"] == "test"
        ids = r["selected_ids"]
        assert len(ids) == len(set(ids))
        assert set(ids) <= set(r["candidate_ids"])
        assert r["tokens"] == sum(corpus[mid]["cost"] for mid in ids) <= r["budget"]
        gold = set(labels[r["case"]]["gold_ids"])
        assert r["evidence_recall"] == len(set(ids) & gold) / len(gold)
        assert r["candidate_recall"] == len(set(r["candidate_ids"]) & gold) / len(gold)
        assert r["selected_count"] == len(ids)
        assert r["complete"] == int(gold <= set(ids))
        mode = {"current_relevance": "current", "concat_relevance": "concat", "permuted_history_relevance": "permuted"}.get(r["method"], "history")
        assert r["candidate_ids"] == poolmap[r["case"]][mode]["ids"]
        assert r["relevance"] == poolmap[r["case"]][mode]["relevance"]
        assert ids == [r["candidate_ids"][t["index"]] for t in r["trace"]]
    scales = json.loads((run / "scale_calibration.json").read_text())
    x = np.log(scales["dense_scales"])
    indices = np.array(scales["indices"])
    for audit in json.loads((run / "profiles.json").read_text()):
        ratio = np.array(audit["selected_profile"]) / np.array(audit["full_profile"])
        error = np.max(abs(ratio - np.interp(x, x[indices], ratio[indices])))
        assert abs(error - audit["selected_profile_interpolation_error"]) < 1e-12
    summary = list(csv.DictReader((run / "summary.csv").open()))
    for row in summary:
        selected = [r for r in rows if r["method"] == row["method"] and r["budget"] == int(row["budget"]) and r["turn"] > 1]
        per_conversation = defaultdict(list)
        for r in selected:
            per_conversation[r["conversation"]].append(r["evidence_recall"])
        score = np.mean([np.mean(values) for values in per_conversation.values()])
        assert abs(score - float(row["evidence_recall"])) < 1e-12
    return dict(queries=len(queries), test_cases=len(pool_rows), selections=len(rows),
                checks="source/data hashes, disjoint splits, public schema, independent candidate ranking, shared pools, costs, evidence scores, trace IDs, profile errors and conversation-weighted summaries")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/prepared")
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        print(json.dumps(verify(args.data, args.run), indent=2))
