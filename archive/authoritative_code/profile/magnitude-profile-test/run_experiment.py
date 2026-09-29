from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from magprofile.dataset import generate_dataset
from magprofile.embeddings import make_embedder
from magprofile.evaluate import evaluate_models, paired_bootstrap_difference
from magprofile.features import compute_feature_table
from magprofile.plots import plot_model_comparison, plot_profiles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/run"))
    parser.add_argument("--groups", type=int, default=300, help="Scenario groups; seven examples per group")
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument(
        "--split-by", choices=("template_family", "group_id"), default="template_family",
        help="Template-family holdout is the primary anti-leakage evaluation.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--scales", type=int, default=32)
    parser.add_argument("--scale-min", type=float, default=0.05)
    parser.add_argument("--scale-max", type=float, default=100.0)
    parser.add_argument("--backend", choices=("hashing", "sentence-transformer"), default="hashing")
    parser.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = generate_dataset(groups=args.groups, seed=args.seed)
    if data[args.split_by].nunique() < args.folds:
        raise ValueError(f"{args.split_by} must contain at least --folds distinct groups")
    data.to_json(args.output / "dataset.jsonl", orient="records", lines=True)
    scales = np.geomspace(args.scale_min, args.scale_max, args.scales)
    embedder = make_embedder(args.backend, args.model)
    feature_data, blocks = compute_feature_table(data, embedder, scales, args.output, args.seed)
    aggregate, folds = evaluate_models(
        feature_data, blocks, folds=args.folds, seed=args.seed, split_by=args.split_by
    )
    aggregate.to_csv(args.output / "model_results.csv", index=False)
    folds.to_csv(args.output / "fold_results.csv", index=False)

    pairs = [
        ("07_sim_stats_plus_profile", "05_sim_stats_plus_single_mag"),
        ("16_complete_geometry_plus_profile", "15_complete_geometry"),
        ("25_complete_geometry_plus_profile_rf", "24_complete_geometry_random_forest"),
        ("06_profile_only", "17_sigmoid_params"),
        ("19_sigmoid_params_plus_residual", "17_sigmoid_params"),
        ("07_sim_stats_plus_profile", "18_sim_stats_plus_sigmoid_params"),
        ("27_sim_surface_plus_profile", "26_sim_surface_plus_single_mag"),
    ]
    comparisons = [
        paired_bootstrap_difference(folds, a, b, metric=metric, seed=args.seed)
        for a, b in pairs
        for metric in ("roc_auc", "r2", "type_macro_f1")
    ]
    comparisons.extend(
        [
            paired_bootstrap_difference(folds, "06_profile_only", "11_sorted_profile_values", seed=args.seed),
            paired_bootstrap_difference(folds, "06_profile_only", "12_per_example_shuffled_profile", seed=args.seed),
        ]
    )
    (args.output / "paired_comparisons.json").write_text(json.dumps(comparisons, indent=2))
    plot_profiles(feature_data, blocks["profile"], scales, args.output / "profiles_by_type.png")
    plot_model_comparison(aggregate, args.output / "model_comparison.png")

    key = aggregate.set_index("model")["roc_auc_mean"]
    verdict = {
        "profile_beats_single_magnitude_given_similarity": bool(
            key["07_sim_stats_plus_profile"] > key["05_sim_stats_plus_single_mag"]
        ),
        "profile_adds_beyond_complete_geometry_linear": bool(
            key["16_complete_geometry_plus_profile"] > key["15_complete_geometry"]
        ),
        "profile_adds_beyond_complete_geometry_random_forest": bool(
            key["25_complete_geometry_plus_profile_rf"] > key["24_complete_geometry_random_forest"]
        ),
        "full_profile_beats_sigmoid_parameters": bool(
            key["06_profile_only"] > key["17_sigmoid_params"]
        ),
        "sigmoid_residual_adds_information": bool(
            key["19_sigmoid_params_plus_residual"] > key["17_sigmoid_params"]
        ),
        "scale_order_beats_sorted_values": bool(key["06_profile_only"] > key["11_sorted_profile_values"]),
        "scale_order_beats_per_example_shuffle": bool(
            key["06_profile_only"] > key["12_per_example_shuffled_profile"]
        ),
        "sorting_actually_changed_any_profile": bool(
            np.any(np.abs(blocks["profile"] - blocks["sorted_profile"]) > 1e-10)
        ),
        "warning": "Directional checks are not claims of significance; inspect paired_comparisons.json and transfer tests.",
        "split_by": args.split_by,
    }
    (args.output / "verdict.json").write_text(json.dumps(verdict, indent=2))
    print(aggregate.sort_values("roc_auc_mean", ascending=False).to_string(index=False))
    print("\nVerdict:\n" + json.dumps(verdict, indent=2))
    print(f"\nArtifacts written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
