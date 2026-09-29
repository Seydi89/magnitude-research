"""Test semantic aspect coverage as an alternative to geometric diversity.

Aspect descriptions are answer-free subquestions derived from the current
query.  A deployable selector estimates memory-to-aspect support using the same
encoder as retrieval.  A temporal-cue variant downweights explicitly obsolete
memories.  A label-based support matrix is evaluated only as an oracle ceiling.
"""

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent))
if (HERE.parent / "updated_project").exists():
    sys.path.insert(0, str(HERE.parent / "updated_project"))

import run_dedup as base


TEMPERATURES = [0.02, 0.05, 0.10, 0.20, 0.40]
MIXTURES = [0.00, 0.10, 0.25, 0.50, 0.75, 0.90, 1.00]
MMR_LAMBDAS = base.MMR_LAMBDAS
METHODS = [
    "relevance", "MMR", "aspect-coverage",
    "aspect-coverage-temporal", "oracle-aspect-coverage",
]


def temporal_validity(text):
    """Query-independent cue available from memory text, not gold metadata."""
    value = text.lower()
    obsolete = ["obsolete", "previously", "no longer", "retired configuration", "used to run"]
    return 0.05 if any(cue in value for cue in obsolete) else 1.0


def embedding_support(case, temperature, use_temporal=False):
    if not len(case["aspect_emb"]):
        raise ValueError(f"{case['case_id']}: dataset has no aspect descriptions")
    similarity = np.clip(case["aspect_emb"] @ case["emb"].T, -1.0, 1.0)
    # Best candidate for each answer-free aspect receives support 1. Other
    # candidates decay according to their distance from that within-pool best.
    support = np.exp((similarity - similarity.max(axis=1, keepdims=True)) / temperature)
    if use_temporal:
        validity = np.array([temporal_validity(memory["text"]) for memory in case["memories"]])
        support *= validity[None, :]
    return np.clip(support, 0.0, 1.0)


def oracle_support(case):
    """Gold support matrix used only to measure the objective's ceiling."""
    support = np.zeros((len(case["aspects"]), len(case["memories"])), dtype=float)
    for aspect_index, aspect in enumerate(case["aspects"]):
        gold = aspect["gold_fact_id"]
        for memory_index, memory in enumerate(case["memories"]):
            if memory["valid"] and gold in memory["fact_ids"]:
                support[aspect_index, memory_index] = 1.0
    return support


def select_coverage(case, budget, support, relevance_mixture):
    """Greedy probabilistic aspect coverage plus modular relevance.

    C_a(S)=1-prod_{m in S}(1-p_am).  Its marginal is
    p_ai*prod_{m in S}(1-p_am), so repeated support for one aspect saturates.
    """
    relevance = base.minmax(case["rel"])
    uncovered = np.ones(len(support), dtype=float)
    selected, spent = [], 0
    while True:
        fits = [
            index for index in range(len(case["memories"]))
            if index not in selected and spent + case["costs"][index] <= budget
        ]
        if not fits:
            return selected, float(np.mean(1.0 - uncovered))
        marginal = np.mean(uncovered[:, None] * support, axis=0)
        gain = relevance_mixture * relevance + (1.0 - relevance_mixture) * marginal
        index = max(
            fits,
            key=lambda i: (gain[i], marginal[i], case["rel"][i], -case["costs"][i], -i),
        )
        selected.append(index)
        spent += int(case["costs"][index])
        uncovered *= 1.0 - support[:, index]


def score(case, method, selected, budget, predicted_coverage):
    row = base.score(case, method, base.no_clean(case), selected, budget)
    row["predicted_aspect_coverage"] = predicted_coverage
    return row


def run_method(case, method, budget, parameters):
    if method == "relevance":
        selected = base.pack_relevance(case, base.no_clean(case), budget)
        predicted = None
    elif method == "MMR":
        selected = base.pack_mmr(case, budget, parameters["mmr_lambda"])
        predicted = None
    elif method in {"aspect-coverage", "aspect-coverage-temporal"}:
        support = embedding_support(
            case, parameters["temperature"], method == "aspect-coverage-temporal"
        )
        selected, predicted = select_coverage(
            case, budget, support, parameters["relevance_mixture"]
        )
    elif method == "oracle-aspect-coverage":
        selected, predicted = select_coverage(case, budget, oracle_support(case), 0.0)
    else:
        raise ValueError(method)
    return score(case, method, selected, budget, predicted)


def mean(rows, key):
    values = [row[key] for row in rows if row[key] is not None]
    return float(np.mean(values)) if values else None


def aggregate(rows):
    return {
        "cases": len(rows),
        **{
            key: mean(rows, key)
            for key in [
                "evidence_recall", "complete_coverage", "selected", "selected_tokens",
                "selected_duplicate_extras", "obsolete_selected",
                "predicted_aspect_coverage",
            ]
        },
    }


def tune_quality(rows):
    values = aggregate(rows)
    return (
        round(values["evidence_recall"], 12),
        round(values["complete_coverage"], 12),
        -round(values["obsolete_selected"], 12),
    )


def tune(validation):
    frozen, curves = {}, {}
    for budget in base.BUDGETS:
        key = str(budget)
        frozen[key], curves[key] = {}, {}
        for method in ["aspect-coverage", "aspect-coverage-temporal"]:
            candidates = {}
            for temperature in TEMPERATURES:
                for mixture in MIXTURES:
                    params = {"temperature": temperature, "relevance_mixture": mixture}
                    candidates[(temperature, mixture)] = [
                        run_method(case, method, budget, params) for case in validation
                    ]
            # Prefer more relevance and a middle temperature only after exact
            # evidence/complete-coverage ties.
            best = max(
                candidates,
                key=lambda pair: (
                    tune_quality(candidates[pair]), pair[1], -abs(pair[0] - 0.10)
                ),
            )
            frozen[key][method] = {
                "temperature": best[0], "relevance_mixture": best[1]
            }
            curves[key][method] = {
                f"temperature={temperature}|relevance={mixture}": aggregate(rows)
                for (temperature, mixture), rows in candidates.items()
            }

        mmr = {}
        for lam in MMR_LAMBDAS:
            params = {"mmr_lambda": lam}
            mmr[lam] = [run_method(case, "MMR", budget, params) for case in validation]
        best_mmr = max(MMR_LAMBDAS, key=lambda value: (tune_quality(mmr[value]), value))
        frozen[key]["mmr_lambda"] = best_mmr
        curves[key]["MMR"] = {str(value): aggregate(rows) for value, rows in mmr.items()}
        print(f"  frozen budget {budget}: {frozen[key]}", flush=True)
    return frozen, curves


def evaluate(test, frozen):
    rows = {
        str(budget): {method: [] for method in METHODS}
        for budget in base.BUDGETS
    }
    for number, case in enumerate(test, 1):
        for budget in base.BUDGETS:
            key = str(budget)
            for method in METHODS:
                if method in {"aspect-coverage", "aspect-coverage-temporal"}:
                    params = frozen[key][method]
                elif method == "MMR":
                    params = {"mmr_lambda": frozen[key]["mmr_lambda"]}
                else:
                    params = {}
                rows[key][method].append(run_method(case, method, budget, params))
        if number % 20 == 0 or number == len(test):
            print(f"  test {number}/{len(test)}", flush=True)

    summary, by_copies, comparisons = {}, {}, {}
    for budget in base.BUDGETS:
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
            f"{method}_minus_relevance": base.paired_bootstrap(
                rows[key][method], baseline, 1700 + budget + index
            )
            for index, method in enumerate(METHODS) if method != "relevance"
        }
        comparisons[key]["temporal_minus_plain_aspect"] = base.paired_bootstrap(
            rows[key]["aspect-coverage-temporal"], rows[key]["aspect-coverage"],
            1900 + budget,
        )
    return rows, summary, by_copies, comparisons


def effect_text(effect):
    low, high = effect["ci95"]
    return f"{effect['mean']:+.4f} [{low:+.4f}, {high:+.4f}]"


def report_text(result):
    print("\nControlled semantic aspect-coverage selection")
    print("Frozen parameters:", json.dumps(result["frozen"], indent=1))
    for budget in base.BUDGETS:
        key = str(budget)
        print(f"\nBudget {budget}")
        print(
            f"{'method':29s} {'recall':>7s} {'complete':>9s} {'tokens':>7s} "
            f"{'dup picks':>9s} {'old picks':>9s}"
        )
        for method in METHODS:
            row = result["summary"][key][method]
            print(
                f"{method:29s} {row['evidence_recall']:7.3f} "
                f"{row['complete_coverage']:9.1%} {row['selected_tokens']:7.1f} "
                f"{row['selected_duplicate_extras']:9.2f} {row['obsolete_selected']:9.2f}"
            )
        print("Paired profile-level recall effects:")
        for name, effect in result["comparisons"][key].items():
            print(f"  {name}: {effect_text(effect)}")
        print("Recall by copy count:")
        for copies in [1, 2, 4, 8]:
            values = result["by_copies"][key][str(copies)]
            line = "  ".join(
                f"{method}={values[method]['evidence_recall']:.3f}" for method in METHODS
            )
            print(f"  copies {copies}: {line}")


def self_test():
    case = {
        "memories": [
            {"text": "alpha", "fact_ids": ["a"], "valid": True, "equivalence_group": "a1", "kind": "x"},
            {"text": "alpha duplicate", "fact_ids": ["a"], "valid": True, "equivalence_group": "a2", "kind": "x"},
            {"text": "beta", "fact_ids": ["b"], "valid": True, "equivalence_group": "b", "kind": "x"},
        ],
        "rel": np.array([0.9, 0.8, 0.7]),
        "costs": np.array([10, 10, 10]),
    }
    support = np.array([[1.0, 0.9, 0.0], [0.0, 0.0, 1.0]])
    selected, coverage = select_coverage(case, 20, support, 0.0)
    assert selected == [0, 2]
    assert coverage == 1.0
    assert temporal_validity("This configuration is obsolete.") == 0.05
    assert temporal_validity("This is the current configuration.") == 1.0
    print("Aspect-coverage self-tests passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", default="controlled_magnitude_dedup/data/controlled_magnitude_dedup.json"
    )
    parser.add_argument("--report", default="results/controlled_aspect_coverage.json")
    parser.add_argument("--encoder", choices=["lsa", "minilm"], default="minilm")
    parser.add_argument("--model-dir")
    parser.add_argument("--revision", default=base.REVISION)
    parser.add_argument("--onnx-threads", type=int, default=4)
    parser.add_argument("--cache-dir")
    parser.add_argument("--cache-batch-size", type=int, default=64)
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
    manifest, records = base.load_dataset(args.data)
    if any(not case.get("aspects") for case in records):
        parser.error("Dataset lacks aspect descriptions; regenerate it with the updated generator")
    if args.encoder == "minilm":
        from trajectory_memory.encoders import MiniLMEncoder
        encoder = MiniLMEncoder(args.model_dir, args.revision, args.onnx_threads)
    else:
        from trajectory_memory.encoders import LSAEncoder
        encoder = LSAEncoder(128, 2026)
        corpus = sorted({
            text for case in records
            for text in (
                [case["query"]]
                + [memory["text"] for memory in case["memories"]]
                + [aspect["text"] for aspect in case["aspects"]]
            )
        })
        encoder.fit(corpus)

    from trajectory_memory.embedding_cache import CachedEncoder, DEFAULT_CACHE_DIR
    with CachedEncoder(
        encoder, args.cache_dir or DEFAULT_CACHE_DIR, args.cache_batch_size
    ) as cached:
        print("Embedding cache:", cached.path, flush=True)
        cases = base.build_cases(records, cached)
        cache_stats = cached.stats()
    print("Cases:", dict(Counter(case["split"] for case in cases)), flush=True)
    validation = [case for case in cases if case["split"] == "validation"]
    test = [case for case in cases if case["split"] == "test"]
    frozen, curves = tune(validation)
    rows, summary, by_copies, comparisons = evaluate(test, frozen)
    result = {
        "experiment": "controlled_semantic_aspect_coverage",
        "schema_version": 1,
        "arguments": vars(args),
        "dataset_manifest": manifest,
        "encoder": encoder.metadata,
        "embedding_cache": cache_stats,
        "splits": dict(Counter(case["split"] for case in cases)),
        "budgets": base.BUDGETS,
        "methods": METHODS,
        "grids": {
            "support_temperature": TEMPERATURES,
            "relevance_mixture": MIXTURES,
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
            "Aspect descriptions are derived from the query and contain no answer values.",
            "Embedding support is normalized relative to the best candidate for each aspect.",
            "Aspect coverage uses 1-prod(1-p_am), giving repeated support diminishing returns.",
            "The temporal variant uses only explicit textual cues and never reads valid labels.",
            "Oracle aspect coverage uses fact/validity labels and is a non-deployable ceiling.",
            "Parameters are tuned on validation profiles and frozen for test.",
            "Confidence intervals resample complete profiles with all copy conditions.",
        ],
    }
    base.atomic_json(args.report, result)
    report_text(result)
    print("Saved:", args.report)


if __name__ == "__main__":
    main()
