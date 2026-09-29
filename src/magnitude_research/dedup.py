"""Magnitude is used only as redundancy evidence; relevance remains separate."""
from __future__ import annotations
import numpy as np
from .magnitude import MagnitudeGeometry, log_quadrature_weights, multiscale_leave_one_out


def magnitude_redundancy_scores(geometry: MagnitudeGeometry) -> np.ndarray:
    contributions=multiscale_leave_one_out(geometry)
    return log_quadrature_weights(geometry.scales) @ contributions


def retain_nonredundant(geometry: MagnitudeGeometry, relevance, similarity,
                        contribution_threshold: float, similarity_threshold: float):
    contribution=magnitude_redundancy_scores(geometry); relevance=np.asarray(relevance)
    order=np.argsort(-relevance,kind="stable"); kept=[]; removed=[]
    for i in order:
        close=any(similarity[i,j]>=similarity_threshold for j in kept)
        if close and contribution[i]<=contribution_threshold: removed.append(int(i))
        else: kept.append(int(i))
    return {"kept":kept,"removed":removed,"contribution":contribution.tolist()}

