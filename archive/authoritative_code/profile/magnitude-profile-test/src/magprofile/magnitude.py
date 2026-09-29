from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import least_squares
from scipy.special import expit


def cosine_distance_matrix(x: np.ndarray) -> np.ndarray:
    """Cosine distances for row-normalized or unnormalized vectors."""
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.maximum(norms, 1e-12)
    return np.clip(1.0 - x @ x.T, 0.0, 2.0)


def magnitude_from_distances(
    distances: np.ndarray,
    scale: float,
    ridge: float = 1e-8,
) -> float:
    """Magnitude of a finite metric space using Z_ij=exp(-scale*d_ij).

    A tiny relative ridge and least-squares fallback make nearly duplicated
    points numerically safe without changing ordinary well-conditioned cases.
    """
    d = np.asarray(distances, dtype=np.float64)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distances must be a square matrix")
    z = np.exp(-float(scale) * d)
    reg = ridge * max(float(np.trace(z)) / max(len(z), 1), 1.0)
    z = z + reg * np.eye(len(z))
    ones = np.ones(len(z), dtype=np.float64)
    try:
        weights = np.linalg.solve(z, ones)
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(z, ones, rcond=1e-10)[0]
    return float(weights.sum())


def marginal_profile(
    base_embeddings: np.ndarray,
    candidate_embedding: np.ndarray,
    scales: np.ndarray,
    ridge: float = 1e-8,
) -> np.ndarray:
    """Return M_scale(S union {x}) - M_scale(S) at every scale."""
    base = np.asarray(base_embeddings, dtype=np.float64)
    candidate = np.asarray(candidate_embedding, dtype=np.float64).reshape(1, -1)
    if base.ndim != 2 or base.shape[1] != candidate.shape[1]:
        raise ValueError("base and candidate embedding dimensions must agree")
    d_base = cosine_distance_matrix(base)
    d_aug = cosine_distance_matrix(np.vstack([base, candidate]))
    return np.asarray(
        [
            magnitude_from_distances(d_aug, s, ridge)
            - magnitude_from_distances(d_base, s, ridge)
            for s in scales
        ],
        dtype=np.float64,
    )


def similarity_features(
    base_embeddings: np.ndarray,
    candidate_embedding: np.ndarray,
    max_set_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return scalar summaries and a padded sorted similarity vector."""
    base = np.asarray(base_embeddings, dtype=np.float64)
    candidate = np.asarray(candidate_embedding, dtype=np.float64)
    base = base / np.maximum(np.linalg.norm(base, axis=1, keepdims=True), 1e-12)
    candidate = candidate / max(np.linalg.norm(candidate), 1e-12)
    similarities = np.clip(base @ candidate, -1.0, 1.0)
    sorted_sims = np.sort(similarities)[::-1]
    padded = np.full(max_set_size, -1.0, dtype=np.float64)
    padded[: len(sorted_sims)] = sorted_sims
    top3 = sorted_sims[: min(3, len(sorted_sims))]
    summaries = np.asarray(
        [
            sorted_sims[0],
            similarities.mean(),
            similarities.std(),
            np.median(similarities),
            top3.mean(),
            float(len(similarities)),
        ],
        dtype=np.float64,
    )
    return summaries, padded


def profile_shape_features(profile: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Interpretable features of an ordered profile on log scale."""
    g = np.asarray(profile, dtype=np.float64)
    log_s = np.log(np.asarray(scales, dtype=np.float64))
    grad = np.gradient(g, log_s)
    curvature = np.gradient(grad, log_s)
    midpoint = len(g) // 2
    coarse = float(np.mean(g[:midpoint]))
    fine = float(np.mean(g[midpoint:]))
    return np.asarray(
        [
            trapezoid(g, log_s),
            g.sum(),
            g.max(),
            log_s[int(np.argmax(g))],
            fine / (abs(coarse) + 1e-9),
            grad.mean(),
            np.max(np.abs(grad)),
            curvature.mean(),
            np.max(np.abs(curvature)),
        ],
        dtype=np.float64,
    )


def complete_geometry_features(
    base_embeddings: np.ndarray,
    candidate_embedding: np.ndarray,
    max_set_size: int,
) -> np.ndarray:
    """Canonicalized full augmented cosine geometry, including S-to-S relations.

    Base points are ordered by their similarity to the candidate. The feature
    vector contains the padded candidate-to-S similarities, the upper triangle
    of the correspondingly reordered S-to-S matrix, and set size. Except for
    exact ties, this retains the complete pairwise geometry available to
    magnitude while remaining invariant to the original memory order.
    """
    base = np.asarray(base_embeddings, dtype=np.float64)
    candidate = np.asarray(candidate_embedding, dtype=np.float64)
    base = base / np.maximum(np.linalg.norm(base, axis=1, keepdims=True), 1e-12)
    candidate = candidate / max(np.linalg.norm(candidate), 1e-12)
    x_sim = base @ candidate
    order = np.argsort(-x_sim, kind="stable")
    x_sim = x_sim[order]
    s_sim = base[order] @ base[order].T

    padded_x = np.full(max_set_size, -1.0, dtype=np.float64)
    padded_x[: len(x_sim)] = x_sim
    pair_count = max_set_size * (max_set_size - 1) // 2
    padded_pairs = np.full(pair_count, -1.0, dtype=np.float64)
    values = s_sim[np.triu_indices(len(s_sim), k=1)]
    padded_pairs[: len(values)] = values
    return np.concatenate([padded_x, padded_pairs, [float(len(base))]])


def surface_features(text: str) -> np.ndarray:
    """Cheap controls for length, punctuation and clause/template artifacts."""
    words = text.split()
    lower = text.lower()
    return np.asarray(
        [
            len(words),
            len(text),
            len(set(w.strip(".,;:!?\"'").lower() for w in words)),
            text.count(",") + text.count(";") + text.count(":"),
            sum(lower.count(token) for token in (" not ", " instead", " from ", " to ", " recently")),
            sum(ch.isdigit() for ch in text),
        ],
        dtype=np.float64,
    )


def sigmoid_profile_decomposition(
    profile: np.ndarray,
    scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit amplitude*sigmoid(slope*(log(scale)-location)).

    Returns five interpretable parameters [amplitude, slope, location, RMSE,
    area] and the residual vector. If those parameters match the full profile,
    apparent multiscale value is mostly a shifted/squeezed transition curve.
    """
    g = np.asarray(profile, dtype=np.float64)
    x = np.log(np.asarray(scales, dtype=np.float64))
    amplitude0 = max(float(np.max(g)), 0.0)
    if amplitude0 < 1e-10:
        fitted = np.zeros_like(g)
        params = np.asarray([0.0, 0.0, float(np.mean(x)), 0.0, 0.0])
        return params, g.copy()

    target = np.clip(g, 0.0, None)
    half = amplitude0 / 2.0
    location0 = float(x[int(np.argmin(np.abs(target - half)))])

    def residual(theta: np.ndarray) -> np.ndarray:
        amplitude, slope, location = theta
        return amplitude * expit(slope * (x - location)) - target

    fit = least_squares(
        residual,
        x0=np.asarray([amplitude0, 2.0, location0]),
        bounds=([0.0, 0.01, x.min() - 5.0], [2.0, 50.0, x.max() + 5.0]),
        max_nfev=500,
    )
    amplitude, slope, location = fit.x
    fitted = amplitude * expit(slope * (x - location))
    rmse = float(np.sqrt(np.mean((target - fitted) ** 2)))
    area = float(trapezoid(target, x))
    params = np.asarray([amplitude, slope, location, rmse, area], dtype=np.float64)
    return params, target - fitted
