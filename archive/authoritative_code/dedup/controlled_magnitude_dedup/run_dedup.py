"""Evaluate multiscale leave-one-out magnitude as a deduplication signal.

The experiment separates pool cleaning from budgeted selection.  Cosine and
magnitude methods clean a pool, then the same raw-relevance packer selects from
what remains.  MMR selects directly from the uncleaned pool.  Oracle dedup uses
only the synthetic equivalence labels and is an upper-bound control.
"""

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent))
if (HERE.parent / "updated_project").exists():
    sys.path.insert(0, str(HERE.parent / "updated_project"))

from trajectory_memory.magnitude import (
    calibrate_profiles, distances, kernels, log_weights, magnitude,
)


REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
BUDGETS = [80, 120]
COSINE_THRESHOLDS = [0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99, 1.01]
MAGNITUDE_SIM_GATES = [0.70, 0.80, 0.85, 0.90, 1.01]
MAGNITUDE_DELTAS = [
    0.0, 0.0001, 0.001, 0.005, 0.01, 0.025, 0.05, 0.10, 0.20,
    0.30, 0.40, 0.60, 0.80, 1.00,
]
MMR_LAMBDAS = [0.50, 0.70, 0.85, 0.95, 0.99, 1.00]
METHODS = [
    "relevance", "exact-text-dedup", "cosine-dedup",
    "magnitude-loo-dedup", "MMR", "oracle-dedup",
]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(
        json.dumps(value, indent=1, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def word_cost(text):
    return max(1, len(re.findall(r"\w+|[^\w\s]", text)))


def minmax(values):
    values = np.asarray(values, float)
    spread = float(np.ptp(values))
    return np.ones(len(values)) if spread <= 1e-12 else (values - values.min()) / spread


def load_dataset(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("manifest", {}).get("dataset") != "controlled matched memory deduplication":
        raise ValueError("--data is not the controlled matched deduplication dataset")
    cases = payload.get("cases", [])
    if not cases:
        raise ValueError("Dataset has no cases")
    split_by_profile = defaultdict(set)
    for case in cases:
        split_by_profile[case["profile_id"]].add(case["split"])
    if any(len(value) != 1 for value in split_by_profile.values()):
        raise ValueError("Profile leakage across splits")
    return payload["manifest"], cases


def build_cases(records, cached):
    inputs = sorted(
        {memory["text"] for case in records for memory in case["memories"]}
        | {case["query"] for case in records}
        | {aspect["text"] for case in records for aspect in case.get("aspects", [])}
    )
    vectors = cached.encode(inputs).astype(np.float64)
    encoded = {text: vectors[index] for index, text in enumerate(inputs)}
    cases = []
    for raw in records:
        emb = np.array([encoded[memory["text"]] for memory in raw["memories"]], dtype=np.float64)
        emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
        query = encoded[raw["query"]].astype(np.float64)
        query /= max(float(np.linalg.norm(query)), 1e-12)
        aspect_emb = np.array(
            [encoded[aspect["text"]] for aspect in raw.get("aspects", [])],
            dtype=np.float64,
        )
        if len(aspect_emb):
            aspect_emb /= np.maximum(np.linalg.norm(aspect_emb, axis=1, keepdims=True), 1e-12)
        cases.append({
            "case_id": raw["case_id"],
            "profile_id": raw["profile_id"],
            "split": raw["split"],
            "copies": int(raw["copies"]),
            "query": raw["query"],
            "gold_fact_ids": set(raw["gold_fact_ids"]),
            "aspects": raw.get("aspects", []),
            "aspect_emb": aspect_emb,
            "memories": raw["memories"],
            "emb": emb,
            "rel": emb @ query,
            "sim": np.clip(emb @ emb.T, -1.0, 1.0),
            "costs": np.array([cached.count(memory["text"]) for memory in raw["memories"]], dtype=int),
        })
    return cases


def calibrate(cases, calibration_pools):
    probes = [
        case for case in cases
        if case["split"] == "calibration" and case["copies"] == 8
    ][:calibration_pools]
    if not probes:
        raise ValueError("No calibration cases at eight copies")
    matrices = [distances(case["emb"]) for case in probes]
    positive = np.concatenate([
        matrix[np.triu_indices(len(matrix), 1)] for matrix in matrices
    ])
    positive = positive[positive > 1e-10]
    unit = float(np.median(positive))
    normalized = [matrix / unit for matrix in matrices]
    config = calibrate_profiles(normalized, 129, 33, 0.002)
    scales = np.asarray(config["dense_scales"], dtype=float)
    return {
        "distance_unit": unit,
        "scales": scales,
        "weights": log_weights(scales),
        "calibration_pools": len(probes),
        "retained_nodes": len(config["indices"]),
        "calibration_error": config["calibration_max_interpolation_error"],
    }


def leave_one_out_profile(case, geometry):
    r"""Return delta_i(t)=Mag(C)-Mag(C\i), using one inverse per scale.

    For inverse Q=K^-1 and magnitude weights w=Q1, deletion contribution is
    w_i^2/Q_ii.  This is algebraically equivalent to recomputing every deleted
    subset but is much cheaper.
    """
    distance = distances(case["emb"], geometry["distance_unit"])
    bank = kernels(distance, geometry["scales"])
    output = np.empty((len(bank), len(case["memories"])), dtype=np.float64)
    identity = np.eye(len(case["memories"]))
    ones = np.ones(len(case["memories"]))
    for scale_index, matrix in enumerate(bank):
        factor = cho_factor(matrix, lower=True, check_finite=False)
        inverse = cho_solve(factor, identity, check_finite=False)
        weighting = inverse @ ones
        output[scale_index] = weighting ** 2 / np.maximum(np.diag(inverse), 1e-15)
    return np.maximum(output, 0.0)


def equip(case, geometry):
    profile = leave_one_out_profile(case, geometry)
    case["loo_profile"] = profile
    case["loo_score"] = geometry["weights"] @ profile


def discard(case):
    case.pop("loo_profile", None)
    case.pop("loo_score", None)


def relevance_order(case, allowed=None):
    allowed = range(len(case["memories"])) if allowed is None else allowed
    return sorted(allowed, key=lambda i: (-case["rel"][i], case["costs"][i], i))


def no_clean(case):
    return list(range(len(case["memories"])))


def exact_text_clean(case):
    kept, seen = [], set()
    for index in relevance_order(case):
        text = case["memories"][index]["text"]
        if text in seen:
            continue
        kept.append(index)
        seen.add(text)
    return kept


def cosine_clean(case, threshold):
    kept = []
    for index in relevance_order(case):
        if kept and max(case["sim"][index, kept]) >= threshold:
            continue
        kept.append(index)
    return kept


def magnitude_clean(case, similarity_gate, delta_threshold):
    kept = []
    for index in relevance_order(case):
        close = kept and max(case["sim"][index, kept]) >= similarity_gate
        redundant = case["loo_score"][index] <= delta_threshold
        if close and redundant:
            continue
        kept.append(index)
    return kept


def oracle_clean(case):
    groups = defaultdict(list)
    for index, memory in enumerate(case["memories"]):
        groups[memory["equivalence_group"]].append(index)
    kept = []
    for indices in groups.values():
        # Labels define equivalence only. Relevance chooses the representative.
        kept.append(relevance_order(case, indices)[0])
    return sorted(kept)


def pack_relevance(case, allowed, budget):
    selected, spent = [], 0
    for index in relevance_order(case, allowed):
        if spent + case["costs"][index] <= budget:
            selected.append(index)
            spent += int(case["costs"][index])
    return selected


def pack_mmr(case, budget, lam):
    relevance = minmax(case["rel"])
    selected, spent = [], 0
    while True:
        fits = [
            i for i in range(len(case["memories"]))
            if i not in selected and spent + case["costs"][i] <= budget
        ]
        if not fits:
            return selected
        if selected:
            redundancy = np.maximum(case["sim"][:, selected].max(axis=1), 0.0)
            gain = lam * relevance - (1.0 - lam) * redundancy
        else:
            gain = relevance
        # With alpha=0 this is raw gain. The cheaper tie-break prevents the
        # negative-gain/cost inversion found in the earlier experiments.
        index = max(fits, key=lambda i: (gain[i], case["rel"][i], -case["costs"][i], -i))
        selected.append(index)
        spent += int(case["costs"][index])


def fact_coverage(case, indices):
    covered = set()
    for index in indices:
        memory = case["memories"][index]
        if memory["valid"]:
            covered.update(memory["fact_ids"])
    return covered & case["gold_fact_ids"]


def cleaning_metrics(case, retained):
    retained_set = set(retained)
    removed = set(range(len(case["memories"]))) - retained_set
    groups = defaultdict(list)
    for index, memory in enumerate(case["memories"]):
        groups[memory["equivalence_group"]].append(index)
    redundant_total, correct_removed = 0, 0
    exact_total = paraphrase_total = 0
    exact_removed = paraphrase_removed = 0
    for indices in groups.values():
        extras = max(0, len(indices) - 1)
        redundant_total += extras
        kind = case["memories"][indices[0]]["kind"]
        if kind == "exact-duplicate":
            exact_total += extras
        elif kind == "paraphrase-duplicate":
            paraphrase_total += extras
        removed_here = len(set(indices) & removed)
        retained_here = len(set(indices) & retained_set)
        if retained_here:
            correct_here = min(removed_here, len(indices) - 1)
            correct_removed += correct_here
            if kind == "exact-duplicate":
                exact_removed += correct_here
            elif kind == "paraphrase-duplicate":
                paraphrase_removed += correct_here
    precision = correct_removed / len(removed) if removed else (1.0 if redundant_total == 0 else 0.0)
    recall = correct_removed / redundant_total if redundant_total else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    complement = [i for i, m in enumerate(case["memories"]) if m["kind"] == "complementary"]
    current = [i for i, m in enumerate(case["memories"]) if m["kind"] == "update-current"]
    old = [i for i, m in enumerate(case["memories"]) if m["kind"] == "update-old"]
    original_cost = int(case["costs"].sum())
    retained_cost = int(case["costs"][retained].sum()) if retained else 0
    return {
        "removed": len(removed),
        "correct_duplicate_removals": correct_removed,
        "true_redundant_extras": redundant_total,
        "false_removals": len(removed) - correct_removed,
        "correct_exact_removals": exact_removed,
        "true_exact_extras": exact_total,
        "correct_paraphrase_removals": paraphrase_removed,
        "true_paraphrase_extras": paraphrase_total,
        "duplicate_removal_precision": precision,
        "duplicate_removal_recall": recall,
        "duplicate_removal_f1": f1,
        "gold_pool_recall_after_cleaning": len(fact_coverage(case, retained)) / len(case["gold_fact_ids"]),
        "complement_preservation": np.mean([i in retained_set for i in complement]) if complement else 1.0,
        "current_update_preservation": np.mean([i in retained_set for i in current]) if current else 1.0,
        "obsolete_removal": np.mean([i not in retained_set for i in old]) if old else 0.0,
        "pool_token_reduction": 1.0 - retained_cost / original_cost,
        "retained_pool_tokens": retained_cost,
    }


def score(case, method, retained, selected, budget):
    cleaning = cleaning_metrics(case, retained)
    covered = fact_coverage(case, selected)
    selected_groups = Counter(case["memories"][i]["equivalence_group"] for i in selected)
    selected_extras = sum(max(0, count - 1) for count in selected_groups.values())
    obsolete = sum(case["memories"][i]["kind"] == "update-old" for i in selected)
    return {
        "case_id": case["case_id"],
        "profile_id": case["profile_id"],
        "split": case["split"],
        "copies": case["copies"],
        "budget": budget,
        "method": method,
        "evidence_recall": len(covered) / len(case["gold_fact_ids"]),
        "complete_coverage": len(covered) == len(case["gold_fact_ids"]),
        "selected": len(selected),
        "selected_tokens": int(case["costs"][selected].sum()) if selected else 0,
        "selected_duplicate_extras": selected_extras,
        "obsolete_selected": obsolete,
        **cleaning,
    }


def plan(case, method, budget, parameters):
    if method == "relevance":
        retained = no_clean(case)
        selected = pack_relevance(case, retained, budget)
    elif method == "exact-text-dedup":
        retained = exact_text_clean(case)
        selected = pack_relevance(case, retained, budget)
    elif method == "cosine-dedup":
        retained = cosine_clean(case, parameters["cosine_threshold"])
        selected = pack_relevance(case, retained, budget)
    elif method == "magnitude-loo-dedup":
        retained = magnitude_clean(
            case, parameters["magnitude_similarity_gate"], parameters["magnitude_delta"]
        )
        selected = pack_relevance(case, retained, budget)
    elif method == "MMR":
        retained = no_clean(case)
        selected = pack_mmr(case, budget, parameters["mmr_lambda"])
    elif method == "oracle-dedup":
        retained = oracle_clean(case)
        selected = pack_relevance(case, retained, budget)
    else:
        raise ValueError(method)
    return score(case, method, retained, selected, budget)


def mean(rows, key):
    return float(np.mean([row[key] for row in rows]))


def aggregate(rows):
    keys = [
        "evidence_recall", "complete_coverage", "selected", "selected_tokens",
        "selected_duplicate_extras", "obsolete_selected", "removed",
        "duplicate_removal_precision", "duplicate_removal_recall",
        "duplicate_removal_f1", "gold_pool_recall_after_cleaning",
        "complement_preservation", "current_update_preservation",
        "obsolete_removal", "pool_token_reduction", "retained_pool_tokens",
    ]
    result = {"cases": len(rows), **{key: mean(rows, key) for key in keys}}
    correct = sum(row["correct_duplicate_removals"] for row in rows)
    removed = sum(row["removed"] for row in rows)
    redundant = sum(row["true_redundant_extras"] for row in rows)
    precision = correct / removed if removed else 1.0
    recall = correct / redundant if redundant else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    result.update(
        correct_duplicate_removals=int(correct),
        true_redundant_extras=int(redundant),
        false_removals=int(sum(row["false_removals"] for row in rows)),
        duplicate_removal_precision=float(precision),
        duplicate_removal_recall=float(recall),
        duplicate_removal_f1=float(f1),
        exact_duplicate_recall=float(
            sum(row["correct_exact_removals"] for row in rows)
            / max(1, sum(row["true_exact_extras"] for row in rows))
        ),
        paraphrase_duplicate_recall=float(
            sum(row["correct_paraphrase_removals"] for row in rows)
            / max(1, sum(row["true_paraphrase_extras"] for row in rows))
        ),
    )
    return result


def tune_key(rows):
    values = aggregate(rows)
    # Evidence is lexicographically protected. Among recall-equivalent settings,
    # prefer complete coverage, dedup quality, and then greater token reduction.
    return (
        round(values["evidence_recall"], 12),
        round(values["complete_coverage"], 12),
        round(values["duplicate_removal_f1"], 12),
        round(values["gold_pool_recall_after_cleaning"], 12),
        round(values["pool_token_reduction"], 12),
    )


def tune(validation, geometry):
    for number, case in enumerate(validation, 1):
        equip(case, geometry)
        if number % 10 == 0 or number == len(validation):
            print(f"  validation geometry {number}/{len(validation)}", flush=True)
    frozen, curves = {}, {}
    for budget in BUDGETS:
        key = str(budget)
        frozen[key], curves[key] = {}, {}

        cosine_rows = {}
        for threshold in COSINE_THRESHOLDS:
            params = {"cosine_threshold": threshold}
            cosine_rows[threshold] = [plan(case, "cosine-dedup", budget, params) for case in validation]
        best_cosine = max(COSINE_THRESHOLDS, key=lambda value: (tune_key(cosine_rows[value]), value))
        frozen[key]["cosine_threshold"] = best_cosine
        curves[key]["cosine"] = {str(value): aggregate(rows) for value, rows in cosine_rows.items()}

        magnitude_rows = {}
        for gate in MAGNITUDE_SIM_GATES:
            for delta in MAGNITUDE_DELTAS:
                params = {"magnitude_similarity_gate": gate, "magnitude_delta": delta}
                magnitude_rows[(gate, delta)] = [
                    plan(case, "magnitude-loo-dedup", budget, params) for case in validation
                ]
        best_magnitude = max(
            magnitude_rows,
            key=lambda pair: (tune_key(magnitude_rows[pair]), pair[0], -pair[1]),
        )
        frozen[key]["magnitude_similarity_gate"] = best_magnitude[0]
        frozen[key]["magnitude_delta"] = best_magnitude[1]
        curves[key]["magnitude"] = {
            f"gate={gate}|delta={delta}": aggregate(rows)
            for (gate, delta), rows in magnitude_rows.items()
        }

        mmr_rows = {}
        for lam in MMR_LAMBDAS:
            params = {"mmr_lambda": lam}
            mmr_rows[lam] = [plan(case, "MMR", budget, params) for case in validation]
        best_mmr = max(MMR_LAMBDAS, key=lambda value: (tune_key(mmr_rows[value]), value))
        frozen[key]["mmr_lambda"] = best_mmr
        curves[key]["mmr"] = {str(value): aggregate(rows) for value, rows in mmr_rows.items()}
        print(f"  frozen budget {budget}: {frozen[key]}", flush=True)

    for case in validation:
        discard(case)
    return frozen, curves


def paired_bootstrap(left, right, seed):
    by_left, by_right = defaultdict(list), defaultdict(list)
    for row in left:
        by_left[row["profile_id"]].append(row["evidence_recall"])
    for row in right:
        by_right[row["profile_id"]].append(row["evidence_recall"])
    profiles = sorted(set(by_left) & set(by_right))
    differences = np.array([
        np.mean(by_left[profile]) - np.mean(by_right[profile]) for profile in profiles
    ])
    samples = np.random.default_rng(seed).choice(
        differences, (4000, len(differences)), replace=True
    ).mean(axis=1)
    return {
        "mean": float(differences.mean()),
        "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        "profiles": len(profiles),
    }


def evaluate(test, geometry, frozen):
    rows = {
        str(budget): {method: [] for method in METHODS}
        for budget in BUDGETS
    }
    for number, case in enumerate(test, 1):
        equip(case, geometry)
        for budget in BUDGETS:
            key = str(budget)
            for method in METHODS:
                rows[key][method].append(plan(case, method, budget, frozen[key]))
        discard(case)
        if number % 20 == 0 or number == len(test):
            print(f"  test {number}/{len(test)}", flush=True)
    summary, by_copies, comparisons = {}, {}, {}
    for budget in BUDGETS:
        key = str(budget)
        summary[key] = {method: aggregate(method_rows) for method, method_rows in rows[key].items()}
        by_copies[key] = {
            str(copies): {
                method: aggregate([row for row in method_rows if row["copies"] == copies])
                for method, method_rows in rows[key].items()
            }
            for copies in [1, 2, 4, 8]
        }
        baseline = rows[key]["relevance"]
        comparisons[key] = {
            f"{method}_minus_relevance": paired_bootstrap(rows[key][method], baseline, 700 + budget + i)
            for i, method in enumerate(METHODS) if method != "relevance"
        }
        comparisons[key]["magnitude_minus_cosine"] = paired_bootstrap(
            rows[key]["magnitude-loo-dedup"], rows[key]["cosine-dedup"], 900 + budget
        )
        comparisons[key]["magnitude_minus_MMR"] = paired_bootstrap(
            rows[key]["magnitude-loo-dedup"], rows[key]["MMR"], 1100 + budget
        )
    return rows, summary, by_copies, comparisons


def effect_text(effect):
    lo, hi = effect["ci95"]
    return f"{effect['mean']:+.4f} [{lo:+.4f}, {hi:+.4f}]"


def report_text(result):
    print("\nControlled multiscale magnitude deduplication")
    print("Frozen parameters:", json.dumps(result["frozen"], indent=1))
    for budget in BUDGETS:
        key = str(budget)
        print(f"\nBudget {budget}")
        print(
            f"{'method':27s} {'recall':>7s} {'complete':>9s} {'dup F1':>7s} "
            f"{'exact/para':>11s} {'gold':>7s} {'comp':>7s} {'current':>8s} {'saved':>7s}"
        )
        for method in METHODS:
            row = result["summary"][key][method]
            print(
                f"{method:27s} {row['evidence_recall']:7.3f} "
                f"{row['complete_coverage']:9.1%} {row['duplicate_removal_f1']:7.3f} "
                f"{row['exact_duplicate_recall']:.0%}/{row['paraphrase_duplicate_recall']:.0%} "
                f"{row['gold_pool_recall_after_cleaning']:7.1%} "
                f"{row['complement_preservation']:7.1%} "
                f"{row['current_update_preservation']:8.1%} "
                f"{row['pool_token_reduction']:7.1%}"
            )
        print("Paired profile-level recall effects:")
        for name, effect in result["comparisons"][key].items():
            print(f"  {name}: {effect_text(effect)}")
        print("Recall by copy count:")
        for copies in [1, 2, 4, 8]:
            values = result["by_copies"][key][str(copies)]
            line = "  ".join(f"{method}={values[method]['evidence_recall']:.3f}" for method in METHODS)
            print(f"  copies {copies}: {line}")


def self_test():
    # Two coincident points should each have negligible leave-one-out contribution;
    # an isolated third point should have a much larger contribution.
    case = {
        "emb": np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        "memories": [{}, {}, {}],
    }
    scales = np.geomspace(1 / 64, 16, 65)
    geometry = {"distance_unit": 1.0, "scales": scales, "weights": log_weights(scales)}
    profile = leave_one_out_profile(case, geometry)
    score = geometry["weights"] @ profile
    assert score[2] > score[0] * 1000
    assert np.allclose(profile[:, 0], profile[:, 1], atol=1e-7)
    # Check the inverse identity against explicit deletion.
    bank = kernels(distances(case["emb"]), scales)
    for scale_index in [0, len(scales) // 2, len(scales) - 1]:
        full = magnitude(bank[scale_index])
        for index in range(3):
            keep = [j for j in range(3) if j != index]
            brute = full - magnitude(bank[scale_index][np.ix_(keep, keep)])
            assert abs(profile[scale_index, index] - brute) < 1e-7

    # Magnitude cleaning never removes the first representative of a close group.
    clean_case = {
        "memories": [{"id": "a"}, {"id": "b"}],
        "rel": np.array([0.9, 0.8]),
        "costs": np.array([10, 10]),
        "sim": np.array([[1.0, 0.99], [0.99, 1.0]]),
        "loo_score": np.array([0.001, 0.001]),
    }
    assert magnitude_clean(clean_case, 0.9, 0.01) == [0]
    exact_case = {
        "memories": [{"text": "same"}, {"text": "same"}, {"text": "different"}],
        "rel": np.array([0.9, 0.8, 0.7]),
        "costs": np.array([10, 10, 10]),
    }
    assert exact_text_clean(exact_case) == [0, 2]
    print("Runner self-tests passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", default="controlled_magnitude_dedup/data/controlled_magnitude_dedup.json"
    )
    parser.add_argument("--report", default="results/controlled_magnitude_dedup.json")
    parser.add_argument("--encoder", choices=["lsa", "minilm"], default="minilm")
    parser.add_argument("--model-dir")
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--onnx-threads", type=int, default=4)
    parser.add_argument("--cache-dir")
    parser.add_argument("--cache-batch-size", type=int, default=64)
    parser.add_argument("--calibration-pools", type=int, default=12)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.summary_only:
        report_text(json.loads(Path(args.report).read_text(encoding="utf-8")))
        return
    if Path(args.report).exists():
        parser.error("Report already exists; choose a new --report path")

    started = time.monotonic()
    manifest, records = load_dataset(args.data)
    if args.encoder == "minilm":
        from trajectory_memory.encoders import MiniLMEncoder
        encoder = MiniLMEncoder(args.model_dir, args.revision, args.onnx_threads)
    else:
        from trajectory_memory.encoders import LSAEncoder
        encoder = LSAEncoder(128, 2026)
        corpus = sorted({
            text for case in records
            for text in [case["query"]] + [memory["text"] for memory in case["memories"]]
        })
        encoder.fit(corpus)

    from trajectory_memory.embedding_cache import CachedEncoder, DEFAULT_CACHE_DIR
    with CachedEncoder(
        encoder, args.cache_dir or DEFAULT_CACHE_DIR, args.cache_batch_size
    ) as cached:
        print("Embedding cache:", cached.path, flush=True)
        cases = build_cases(records, cached)
        cache_stats = cached.stats()
    print("Cases:", dict(Counter(case["split"] for case in cases)), flush=True)
    geometry = calibrate(cases, args.calibration_pools)
    validation = [case for case in cases if case["split"] == "validation"]
    test = [case for case in cases if case["split"] == "test"]
    frozen, curves = tune(validation, geometry)
    rows, summary, by_copies, comparisons = evaluate(test, geometry, frozen)

    result = {
        "experiment": "controlled_multiscale_magnitude_deduplication",
        "schema_version": 2,
        "arguments": vars(args),
        "dataset_manifest": manifest,
        "encoder": encoder.metadata,
        "embedding_cache": cache_stats,
        "splits": dict(Counter(case["split"] for case in cases)),
        "budgets": BUDGETS,
        "methods": METHODS,
        "geometry": {
            "scales_evaluated": len(geometry["scales"]),
            "distance_unit": geometry["distance_unit"],
            "scale_low": float(geometry["scales"][0]),
            "scale_high": float(geometry["scales"][-1]),
            "calibration_pools": geometry["calibration_pools"],
            "retained_nodes_reported_but_not_used": geometry["retained_nodes"],
            "calibration_error": geometry["calibration_error"],
        },
        "grids": {
            "cosine_threshold": COSINE_THRESHOLDS,
            "magnitude_similarity_gate": MAGNITUDE_SIM_GATES,
            "magnitude_leave_one_out_delta": MAGNITUDE_DELTAS,
            "mmr_lambda": MMR_LAMBDAS,
        },
        "frozen": frozen,
        "validation_curves": curves,
        "summary": summary,
        "by_copies": by_copies,
        "comparisons": comparisons,
        "rows": rows,
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "notes": [
            "Profiles, facts, distractors, queries, and fixed controls are matched across copy counts.",
            "Only the number of exact/paraphrased copies changes between conditions.",
            "Magnitude dedup uses multiscale leave-one-out contribution w_i^2/(K^-1)_ii.",
            "Cosine and magnitude clean pools before an identical raw-relevance packer.",
            "MMR selects directly from the uncleaned pool; oracle dedup uses equivalence labels only.",
            "Validation tuning protects evidence recall lexicographically before deduplication metrics.",
            "All 129 calibrated scales are evaluated; reported adaptive nodes are not substituted.",
            "Confidence intervals resample complete profiles, retaining all four copy conditions.",
            "Costs are encoder tokens for MiniLM and regex units for LSA.",
        ],
    }
    atomic_json(args.report, result)
    report_text(result)
    print("Saved:", args.report)


if __name__ == "__main__":
    main()
