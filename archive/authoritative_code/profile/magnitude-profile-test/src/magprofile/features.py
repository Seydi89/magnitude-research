from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .embeddings import Embedder
from .magnitude import (
    complete_geometry_features,
    marginal_profile,
    profile_shape_features,
    sigmoid_profile_decomposition,
    similarity_features,
    surface_features,
)


def compute_feature_table(
    data: pd.DataFrame,
    embedder: Embedder,
    scales: np.ndarray,
    output_dir: Path,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    all_texts = sorted(
        set(data["candidate"].tolist())
        | {m for memories in data["base_memories"] for m in memories}
    )
    embedded = embedder.encode(all_texts)
    lookup = dict(zip(all_texts, embedded, strict=True))
    max_set_size = max(len(x) for x in data["base_memories"])
    rng = np.random.default_rng(seed)

    sim_summary_rows, sim_vector_rows, profile_rows, shape_rows = [], [], [], []
    geometry_rows, surface_rows, sigmoid_rows, profile_residual_rows = [], [], [], []
    sorted_profile_rows, shuffled_profile_rows = [], []
    for row in data.itertuples(index=False):
        base = np.vstack([lookup[x] for x in row.base_memories])
        candidate = lookup[row.candidate]
        sim_summary, sim_vector = similarity_features(base, candidate, max_set_size)
        profile = marginal_profile(base, candidate, scales)
        sigmoid_params, profile_residual = sigmoid_profile_decomposition(profile, scales)
        sim_summary_rows.append(sim_summary)
        sim_vector_rows.append(sim_vector)
        profile_rows.append(profile)
        shape_rows.append(profile_shape_features(profile, scales))
        geometry_rows.append(complete_geometry_features(base, candidate, max_set_size))
        surface_rows.append(surface_features(row.candidate))
        sigmoid_rows.append(sigmoid_params)
        profile_residual_rows.append(profile_residual)
        sorted_profile_rows.append(np.sort(profile))
        shuffled_profile_rows.append(profile[rng.permutation(len(profile))])

    blocks = {
        "sim_max": np.asarray(sim_summary_rows)[:, :1],
        "sim_stats": np.asarray(sim_summary_rows),
        "sim_vector": np.asarray(sim_vector_rows),
        "single_magnitude": np.asarray(profile_rows)[:, [len(scales) // 2]],
        "profile": np.asarray(profile_rows),
        "sorted_profile": np.asarray(sorted_profile_rows),
        "shuffled_profile": np.asarray(shuffled_profile_rows),
        "shape": np.asarray(shape_rows),
        "complete_geometry": np.asarray(geometry_rows),
        "surface": np.asarray(surface_rows),
        "sigmoid_params": np.asarray(sigmoid_rows),
        "profile_residual": np.asarray(profile_residual_rows),
    }
    feature_data = data.copy()
    for i, scale in enumerate(scales):
        feature_data[f"profile_{i:02d}_scale_{scale:.6g}"] = blocks["profile"][:, i]
    for i, name in enumerate(["sim_max", "sim_mean", "sim_std", "sim_median", "sim_top3_mean", "set_size"]):
        feature_data[name] = blocks["sim_stats"][:, i]
    feature_data.to_json(output_dir / "features.jsonl", orient="records", lines=True)
    np.savez_compressed(output_dir / "feature_blocks.npz", scales=scales, **blocks)
    (output_dir / "feature_manifest.json").write_text(
        json.dumps({"embedder": embedder.name, "scales": scales.tolist()}, indent=2)
    )
    return feature_data, blocks
