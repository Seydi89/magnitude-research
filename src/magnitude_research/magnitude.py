"""Single authoritative finite metric-space magnitude implementation."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist

DEFAULT_RIDGE = 1e-8


def pairwise_distances(x: np.ndarray, metric: str = "euclidean", unit: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or len(x) == 0 or not np.isfinite(x).all() or unit <= 0:
        raise ValueError("x must be a finite nonempty matrix and unit must be positive")
    if metric == "cosine":
        x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        d = np.clip(1 - x @ x.T, 0, 2)
    elif metric == "euclidean":
        d = cdist(x, x, metric="euclidean") / unit
    else:
        raise ValueError("metric must be 'euclidean' or 'cosine'")
    np.fill_diagonal(d, 0)
    return d


def similarity_kernel(distances: np.ndarray, scale: float, ridge: float = DEFAULT_RIDGE) -> np.ndarray:
    d = np.asarray(distances, dtype=np.float64)
    if d.ndim != 2 or d.shape[0] != d.shape[1] or np.any(d < 0):
        raise ValueError("distances must be a nonnegative square matrix")
    if not np.allclose(d, d.T) or not np.allclose(np.diag(d), 0):
        raise ValueError("distances must be symmetric with zero diagonal")
    if scale <= 0 or ridge <= 0:
        raise ValueError("scale and ridge must be positive")
    return np.exp(-scale * d) + ridge * np.eye(len(d))


def weighting(kernel: np.ndarray) -> np.ndarray:
    if len(kernel) == 0:
        return np.empty(0)
    one = np.ones(len(kernel))
    try:
        return cho_solve(cho_factor(kernel, lower=True, check_finite=False), one,
                         check_finite=False)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(kernel, one, rcond=1e-10)[0]


def magnitude(kernel: np.ndarray) -> float:
    return float(weighting(kernel).sum())


def magnitude_profile(distances: np.ndarray, scales: np.ndarray,
                      ridge: float = DEFAULT_RIDGE) -> np.ndarray:
    return np.asarray([magnitude(similarity_kernel(distances, t, ridge)) for t in scales])


def log_quadrature_weights(scales: np.ndarray) -> np.ndarray:
    scales = np.asarray(scales, dtype=float)
    if len(scales) < 2 or np.any(scales <= 0) or np.any(np.diff(scales) <= 0):
        raise ValueError("scales must be strictly increasing and positive")
    widths = np.diff(np.log(scales)); w = np.zeros(len(scales))
    w[:-1] += widths / 2; w[1:] += widths / 2
    return w / widths.sum()


@dataclass(frozen=True)
class MagnitudeGeometry:
    distances: np.ndarray
    scales: np.ndarray
    kernels: np.ndarray
    profile: np.ndarray

    @classmethod
    def from_points(cls, x: np.ndarray, scales: np.ndarray, metric: str = "euclidean",
                    unit: float = 1.0, ridge: float = DEFAULT_RIDGE) -> "MagnitudeGeometry":
        d = pairwise_distances(x, metric=metric, unit=unit)
        s = np.asarray(scales, dtype=float)
        k = np.asarray([similarity_kernel(d, t, ridge) for t in s])
        return cls(d, s, k, np.asarray([magnitude(z) for z in k]))


def leave_one_out_contributions(kernel: np.ndarray) -> np.ndarray:
    """Exact leave-one-out identity: Mag(X)-Mag(X minus i) = w_i^2 / inv(K)_ii."""
    inv = np.linalg.inv(kernel)
    w = inv @ np.ones(len(kernel))
    return (w * w) / np.maximum(np.diag(inv), 1e-15)


def multiscale_leave_one_out(geometry: MagnitudeGeometry) -> np.ndarray:
    return np.asarray([leave_one_out_contributions(k) for k in geometry.kernels])
