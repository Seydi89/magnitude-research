"""Regularized Euclidean magnitude, exact marginal gains, and log-scale quadrature.

Magnitude is a geometric coverage objective, not a probability of relevance.
The greedy knapsack procedure has no asserted global approximation guarantee.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist

RIDGE = 1e-8


def distances(x, unit=1.0):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or len(x) == 0 or not np.isfinite(x).all() or unit <= 0:
        raise ValueError("Finite, nonempty embedding matrix and positive unit required")
    d = cdist(x, x, metric="euclidean") / unit
    np.fill_diagonal(d, 0)
    return d


def kernels(d, scales, ridge=RIDGE):
    d, scales = np.asarray(d, float), np.asarray(scales, float)
    if d.ndim != 2 or d.shape[0] != d.shape[1] or not np.isfinite(d).all():
        raise ValueError("A finite square distance matrix is required")
    if not np.allclose(d, d.T) or not np.allclose(np.diag(d), 0) or np.any(d < 0):
        raise ValueError("Distances must be symmetric and nonnegative, with zero diagonal")
    if np.any(scales <= 0) or not np.isfinite(scales).all() or ridge <= 0:
        raise ValueError("Positive finite scales and ridge required")
    return np.exp(-scales[:, None, None] * d) + ridge * np.eye(len(d))[None]


def magnitude(k):
    if len(k) == 0:
        return 0.0
    one = np.ones(len(k))
    return float(one @ cho_solve(cho_factor(k, lower=True, check_finite=False), one,
                                check_finite=False))


def profile(d, scales):
    return np.array([magnitude(k) for k in kernels(d, scales)])


def log_weights(scales):
    """Normalized trapezoidal quadrature for integration against d(log scale)."""
    s = np.asarray(scales, float)
    if len(s) < 2 or np.any(s <= 0) or np.any(np.diff(s) <= 0):
        raise ValueError("Need at least two strictly increasing positive scales")
    widths = np.diff(np.log(s))
    weights = np.zeros(len(s))
    weights[:-1] += widths / 2
    weights[1:] += widths / 2
    return weights / widths.sum()


class SchurState:
    """Batched rank-one Schur updates; O(scales * n^2) per selected item."""
    def __init__(self, bank):
        self.conditional = np.array(bank, dtype=np.float64, copy=True)
        self.residual = np.ones(bank.shape[:2])
        self.used = np.zeros(bank.shape[1], dtype=bool)
        self.values = np.zeros(bank.shape[0])

    def gains(self):
        diagonal = np.diagonal(self.conditional, axis1=1, axis2=2)
        if np.any(diagonal[:, ~self.used] <= 0):
            raise ArithmeticError("Nonpositive Schur denominator: inspect geometry/ridge")
        result = np.zeros_like(self.residual)
        result[:, ~self.used] = (self.residual[:, ~self.used] ** 2 /
                                diagonal[:, ~self.used])
        return result

    def add(self, index):
        if self.used[index]:
            raise ValueError("Cannot select an item twice")
        v = self.conditional[:, :, index].copy()
        denominator = v[:, index].copy()
        numerator = self.residual[:, index].copy()
        if np.any(denominator <= 0):
            raise ArithmeticError("Nonpositive pivot")
        gain = numerator ** 2 / denominator
        self.residual -= v * (numerator / denominator)[:, None]
        self.conditional -= v[:, :, None] * v[:, None, :] / denominator[:, None, None]
        self.values += gain
        self.used[index] = True
        return gain


@dataclass
class Geometry:
    bank: np.ndarray
    full: np.ndarray
    scales: np.ndarray

    @classmethod
    def make(cls, d, scales):
        bank = kernels(d, scales)
        return cls(bank, np.array([magnitude(k) for k in bank]), np.array(scales))


def select(geometry, relevance, costs, budget, lam=0.5, weights=None,
           method="magnitude", mmr_lambda=0.5, similarity=None):
    """Greedy marginal gain per token; compare with best singleton for F objectives.

    F(S) = lam sum r_i + (1-lam) n sum_j w_j Mag_j(S)/Mag_j(C).
    Relevance and geometry-only scale choices are fixed before selection.
    The same cost-aware rule applies to relevance and magnitude baselines.
    """
    rel, costs = np.asarray(relevance, float), np.asarray(costs, int)
    if len(rel) == 0 or len(rel) != len(costs) or np.any(costs <= 0) or budget <= 0:
        raise ValueError("Nonempty aligned candidates and positive costs/budget required")
    if not 0 <= lam <= 1 or not 0 <= mmr_lambda <= 1:
        raise ValueError("Mixture weights must be in [0, 1]")
    r = (rel - rel.min()) / max(float(np.ptp(rel)), 1e-12)
    if method not in {"magnitude", "relevance", "mmr"}:
        raise ValueError("Unknown selection method")
    n = len(r)
    if method == "magnitude":
        w = np.asarray(weights, float)
        if w.shape != geometry.full.shape or np.any(w < 0) or not np.isclose(w.sum(), 1):
            raise ValueError("Scale weights must be nonnegative and sum to one")
        state = SchurState(geometry.bank)
    else:
        state = None
    chosen, trace, spent, total = [], [], 0, 0.0
    while True:
        fits = [i for i in range(n) if i not in chosen and spent + costs[i] <= budget]
        if not fits:
            break
        if method == "magnitude":
            geometric = n * (w @ (state.gains() / geometry.full[:, None]))
            gains = lam * r + (1 - lam) * geometric
        elif method == "mmr" and chosen:
            redundancy = np.maximum(similarity[:, chosen].max(axis=1), 0)
            gains = mmr_lambda * r - (1 - mmr_lambda) * redundancy
            geometric = np.zeros(n)
        else:
            gains, geometric = r.copy(), np.zeros(n)
        j = max(fits, key=lambda i: (gains[i] / costs[i], rel[i], -i))
        if method == "mmr" and chosen and gains[j] <= 0:
            break
        chosen.append(j)
        spent += int(costs[j])
        total += float(gains[j])
        if state is not None:
            state.add(j)
        trace.append(dict(index=j, tokens=int(costs[j]), marginal_gain=float(gains[j]),
                          gain_per_token=float(gains[j] / costs[j]),
                          relevance=float(r[j]), magnitude_gain=float(geometric[j])))
    # Avoid the familiar long valuable singleton failure of density-only packing.
    if method != "mmr":
        fits = [i for i in range(n) if costs[i] <= budget]
        if fits:
            if method == "magnitude":
                singleton_mag = n * float(w @ (1 / np.diagonal(geometry.bank, axis1=1,
                                                             axis2=2)[:, 0] / geometry.full))
                singleton = lam * r + (1 - lam) * singleton_mag
            else:
                singleton, singleton_mag = r, 0.0
            j = max(fits, key=lambda i: (singleton[i], rel[i], -i))
            if singleton[j] > total + 1e-10:
                chosen, spent, total = [j], int(costs[j]), float(singleton[j])
                trace = [dict(index=j, tokens=int(costs[j]), marginal_gain=total,
                              gain_per_token=total / costs[j], relevance=float(r[j]),
                              magnitude_gain=float(singleton_mag), best_singleton=True)]
    return dict(indices=chosen, tokens=spent, objective=total, trace=trace)


def calibrate_profiles(matrices, dense_nodes=129, max_nodes=33, tolerance=0.002):
    """Freeze a global finite scale interval and adaptive quadrature grid.

    Calibration monitors full pools, ranked prefixes and seeded random subsets.
    Error is checked on a dense log grid, not guaranteed over all real scales or
    unseen subsets. Dense audits of held-out selected subsets are separate.
    """
    if dense_nodes < 17 or max_nodes < 3 or max_nodes > dense_nodes or tolerance <= 0:
        raise ValueError("Invalid profile grid settings")
    matrices = list(matrices)
    if not matrices:
        raise ValueError("Calibration geometry required")
    low, high = 1 / 64, 16.0
    def coarse_error(t):
        return max(abs(profile(d, [t])[0] - 1) / max(1, len(d) - 1) for d in matrices)
    def fine_error(t):
        errors = []
        for d in matrices:
            # Zero-distance duplicate rows share a point in the limiting space.
            _, counts = np.unique(d, axis=0, return_counts=True)
            limit = np.sum(counts / (counts + RIDGE))
            errors.append(abs(profile(d, [t])[0] - limit) / max(1, limit))
        return max(errors)
    while coarse_error(low) > tolerance and low > 2 ** -20:
        low /= 2
    while fine_error(high) > tolerance and high < 2 ** 20:
        high *= 2
    scales = np.geomspace(low, high, dense_nodes)
    monitored = []
    rng = np.random.default_rng(72)
    for d in matrices:
        bank = kernels(d, scales)
        full = np.array([magnitude(k) for k in bank])
        monitored.append(full / len(d))
        sets = [list(range(min(n, len(d)))) for n in [1, 2, 4, 8, 16]]
        sets += [rng.choice(len(d), min(5, len(d)), replace=False).tolist()]
        for subset in sets:
            part = np.array([magnitude(k[np.ix_(subset, subset)]) for k in bank])
            monitored.append(part / full)
    curves = np.array(monitored)
    log_t = np.log(scales)
    selected = [0, len(scales) - 1]
    while True:
        approximation = np.array([np.interp(log_t, log_t[selected], y[selected]) for y in curves])
        errors = np.max(np.abs(curves - approximation), axis=0)
        worst = int(np.argmax(errors))
        max_error = float(errors[worst])
        if max_error <= tolerance or len(selected) >= max_nodes:
            break
        selected = sorted(selected + [worst])
    return dict(dense_scales=scales.tolist(), indices=selected,
                scales=scales[selected].tolist(), weights=log_weights(scales[selected]).tolist(),
                tolerance=tolerance, calibration_max_interpolation_error=max_error,
                tolerance_met=max_error <= tolerance, monitored_curves=len(curves),
                coarse_tail_error=coarse_error(low), fine_tail_error=fine_error(high),
                measure="uniform in log(scale), finite calibration-derived interval")
