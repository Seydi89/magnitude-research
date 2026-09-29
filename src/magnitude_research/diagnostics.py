"""Profile and geometry diagnostics shared by controlled experiments."""
from __future__ import annotations
import numpy as np
from scipy.integrate import trapezoid


def profile_shape(profile: np.ndarray, scales: np.ndarray) -> dict[str, float]:
    y=np.asarray(profile,float); x=np.log(np.asarray(scales,float)); dy=np.gradient(y,x)
    ddy=np.gradient(dy,x); mid=len(y)//2
    return {"area":float(trapezoid(y,x)),"maximum":float(y.max()),
            "max_slope":float(np.max(np.abs(dy))),"max_curvature":float(np.max(np.abs(ddy))),
            "fine_to_coarse":float(y[mid:].mean()/max(abs(y[:mid].mean()),1e-12))}


def effective_rank(x: np.ndarray) -> float:
    values=np.linalg.svd(np.asarray(x)-np.mean(x,axis=0),compute_uv=False)**2
    if values.sum()==0:return 0.0
    p=values/values.sum(); return float(np.exp(-np.sum(p[p>0]*np.log(p[p>0]))))


def profile_drift(reference: np.ndarray, current: np.ndarray) -> dict[str,float]:
    delta=np.asarray(current)-np.asarray(reference)
    return {"l1":float(np.mean(np.abs(delta))),"l2":float(np.sqrt(np.mean(delta**2))),
            "maximum":float(np.max(np.abs(delta)))}

