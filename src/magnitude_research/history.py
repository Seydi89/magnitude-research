"""History-aware query construction kept independent of geometric selection."""
from __future__ import annotations
import numpy as np


def weighted_history_query(current: np.ndarray, previous: list[np.ndarray], decay: float) -> np.ndarray:
    if not 0<=decay<=1: raise ValueError("decay must be in [0,1]")
    out=np.asarray(current,float).copy(); weight=decay
    for vector in reversed(previous): out+=weight*np.asarray(vector); weight*=decay
    return out/max(np.linalg.norm(out),1e-12)


def transition_weights(scope_changes: np.ndarray, scales: np.ndarray, temperature: float=1.0):
    """Experimental controller: larger change shifts mass toward finer scales."""
    change=float(np.mean(np.asarray(scope_changes,float)))
    z=(np.log(scales)-np.log(scales).mean())*change/max(temperature,1e-12)
    w=np.exp(z-z.max()); return w/w.sum()

