from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer


def _model_specs(blocks: dict[str, np.ndarray]) -> list[tuple[str, np.ndarray, str]]:
    geometry = blocks["complete_geometry"]
    geometry_profile = np.hstack([geometry, blocks["profile"]])
    base = {
        "01_sim_max": blocks["sim_max"],
        "02_sim_stats": blocks["sim_stats"],
        "03_full_sorted_sim_vector": blocks["sim_vector"],
        "04_single_magnitude": blocks["single_magnitude"],
        "05_sim_stats_plus_single_mag": np.hstack([blocks["sim_stats"], blocks["single_magnitude"]]),
        "06_profile_only": blocks["profile"],
        "07_sim_stats_plus_profile": np.hstack([blocks["sim_stats"], blocks["profile"]]),
        "08_full_sim_vector_plus_profile": np.hstack([blocks["sim_vector"], blocks["profile"]]),
        "09_profile_shape": blocks["shape"],
        "10_sim_stats_plus_shape": np.hstack([blocks["sim_stats"], blocks["shape"]]),
        "11_sorted_profile_values": blocks["sorted_profile"],
        "12_per_example_shuffled_profile": blocks["shuffled_profile"],
        "13_surface_only": blocks["surface"],
        "14_sim_stats_plus_surface": np.hstack([blocks["sim_stats"], blocks["surface"]]),
        "15_complete_geometry": geometry,
        "16_complete_geometry_plus_profile": geometry_profile,
        "17_sigmoid_params": blocks["sigmoid_params"],
        "18_sim_stats_plus_sigmoid_params": np.hstack([blocks["sim_stats"], blocks["sigmoid_params"]]),
        "19_sigmoid_params_plus_residual": np.hstack([blocks["sigmoid_params"], blocks["profile_residual"]]),
        "20_sim_stats_plus_profile_residual": np.hstack([blocks["sim_stats"], blocks["profile_residual"]]),
    }
    return [(name, x, "linear") for name, x in base.items()] + [
        ("21_sim_stats_polynomial", blocks["sim_stats"], "polynomial"),
        ("22_sim_vector_splines", blocks["sim_vector"], "spline"),
        ("23_sim_vector_random_forest", blocks["sim_vector"], "forest"),
        ("24_complete_geometry_random_forest", geometry, "forest"),
        ("25_complete_geometry_plus_profile_rf", geometry_profile, "forest"),
        (
            "26_sim_surface_plus_single_mag",
            np.hstack([blocks["sim_stats"], blocks["surface"], blocks["single_magnitude"]]),
            "linear",
        ),
        (
            "27_sim_surface_plus_profile",
            np.hstack([blocks["sim_stats"], blocks["surface"], blocks["profile"]]),
            "linear",
        ),
    ]


def _estimators(kind: str, seed: int):
    if kind == "forest":
        return (
            RandomForestClassifier(
                n_estimators=300, min_samples_leaf=4, class_weight="balanced_subsample",
                random_state=seed, n_jobs=-1,
            ),
            RandomForestClassifier(
                n_estimators=300, min_samples_leaf=4, class_weight="balanced_subsample",
                random_state=seed + 1, n_jobs=-1,
            ),
            RandomForestRegressor(
                n_estimators=300, min_samples_leaf=4, random_state=seed, n_jobs=-1,
            ),
        )
    transform = []
    if kind == "polynomial":
        transform = [PolynomialFeatures(degree=2, include_bias=False)]
    elif kind == "spline":
        transform = [SplineTransformer(n_knots=4, degree=3, include_bias=False)]
    classifier = make_pipeline(
        SimpleImputer(), *transform, StandardScaler(),
        LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced"),
    )
    type_classifier = clone(classifier)
    regressor = make_pipeline(SimpleImputer(), *transform, StandardScaler(), Ridge(alpha=10.0))
    return classifier, type_classifier, regressor


def evaluate_models(
    data: pd.DataFrame,
    blocks: dict[str, np.ndarray],
    folds: int = 5,
    seed: int = 13,
    split_by: str = "template_family",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if split_by not in data.columns:
        raise ValueError(f"Unknown split column {split_by!r}")
    groups = data[split_by].to_numpy()
    y_binary = data["has_new_information"].to_numpy(dtype=int)
    y_delta = data["info_delta"].to_numpy(dtype=float)
    y_type = data["candidate_type"].to_numpy()
    splitter = GroupKFold(n_splits=folds)
    model_specs = _model_specs(blocks)
    aggregate_rows, fold_rows = [], []

    for name, x, estimator_kind in model_specs:
        binary_metrics: dict[str, list[float]] = defaultdict(list)
        regression_metrics: dict[str, list[float]] = defaultdict(list)
        type_metrics: dict[str, list[float]] = defaultdict(list)
        for fold, (train, test) in enumerate(splitter.split(x, y_binary, groups)):
            classifier, type_classifier, regressor = _estimators(estimator_kind, seed + fold)
            classifier.fit(x[train], y_binary[train])
            type_classifier.fit(x[train], y_type[train])
            regressor.fit(x[train], y_delta[train])
            probability = classifier.predict_proba(x[test])[:, 1]
            prediction = (probability >= 0.5).astype(int)
            delta_prediction = regressor.predict(x[test])
            type_prediction = type_classifier.predict(x[test])
            values = {
                "roc_auc": roc_auc_score(y_binary[test], probability),
                "balanced_accuracy": balanced_accuracy_score(y_binary[test], prediction),
                "f1": f1_score(y_binary[test], prediction),
                "mae": mean_absolute_error(y_delta[test], delta_prediction),
                "r2": r2_score(y_delta[test], delta_prediction),
                "type_macro_f1": f1_score(y_type[test], type_prediction, average="macro"),
                "type_balanced_accuracy": balanced_accuracy_score(y_type[test], type_prediction),
            }
            fold_rows.append({"model": name, "estimator": estimator_kind, "split_by": split_by, "fold": fold, **values})
            for key in ("roc_auc", "balanced_accuracy", "f1"):
                binary_metrics[key].append(values[key])
            for key in ("mae", "r2"):
                regression_metrics[key].append(values[key])
            for key in ("type_macro_f1", "type_balanced_accuracy"):
                type_metrics[key].append(values[key])

        row: dict[str, float | str] = {"model": name, "estimator": estimator_kind, "split_by": split_by}
        for metric, vals in {**binary_metrics, **regression_metrics, **type_metrics}.items():
            row[f"{metric}_mean"] = float(np.mean(vals))
            row[f"{metric}_std"] = float(np.std(vals, ddof=1))
        aggregate_rows.append(row)
    return pd.DataFrame(aggregate_rows), pd.DataFrame(fold_rows)


def paired_bootstrap_difference(
    fold_results: pd.DataFrame,
    model_a: str,
    model_b: str,
    metric: str = "roc_auc",
    iterations: int = 20_000,
    seed: int = 13,
) -> dict[str, float | str]:
    a = fold_results.loc[fold_results.model == model_a].sort_values("fold")[metric].to_numpy()
    b = fold_results.loc[fold_results.model == model_b].sort_values("fold")[metric].to_numpy()
    if len(a) != len(b):
        raise ValueError("Models do not have matching fold counts")
    differences = a - b
    rng = np.random.default_rng(seed)
    samples = rng.choice(differences, size=(iterations, len(differences)), replace=True).mean(axis=1)
    return {
        "model_a": model_a,
        "model_b": model_b,
        "metric": metric,
        "mean_difference": float(differences.mean()),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "probability_a_gt_b": float(np.mean(samples > 0)),
    }
