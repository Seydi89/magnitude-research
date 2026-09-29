"""Cost-aware relevance, MMR, and whole-set magnitude selection."""
from __future__ import annotations
import numpy as np
from .magnitude import MagnitudeGeometry, log_quadrature_weights


class _SchurState:
    def __init__(self, kernels: np.ndarray):
        self.conditional = kernels.copy(); self.residual = np.ones(kernels.shape[:2])
        self.used = np.zeros(kernels.shape[1], dtype=bool)

    def gains(self) -> np.ndarray:
        diag = np.diagonal(self.conditional, axis1=1, axis2=2)
        out = np.zeros_like(self.residual)
        out[:, ~self.used] = self.residual[:, ~self.used] ** 2 / diag[:, ~self.used]
        return out

    def add(self, i: int) -> None:
        v = self.conditional[:, :, i].copy(); den = v[:, i]; num = self.residual[:, i]
        self.residual -= v * (num / den)[:, None]
        self.conditional -= v[:, :, None] * v[:, None, :] / den[:, None, None]
        self.used[i] = True


def select(relevance, costs, budget, method="relevance", geometry=None, similarity=None,
           relevance_weight=0.5, mmr_weight=0.5, scale_weights=None):
    relevance=np.asarray(relevance,float); costs=np.asarray(costs,int); n=len(relevance)
    if len(costs)!=n or np.any(costs<=0) or budget<=0: raise ValueError("invalid costs/budget")
    r=(relevance-relevance.min())/max(float(np.ptp(relevance)),1e-12)
    if method=="magnitude":
        if not isinstance(geometry,MagnitudeGeometry): raise ValueError("geometry required")
        sw=log_quadrature_weights(geometry.scales) if scale_weights is None else np.asarray(scale_weights)
        state=_SchurState(geometry.kernels)
    else: state=None
    chosen=[]; spent=0; trace=[]
    while True:
        fits=[i for i in range(n) if i not in chosen and spent+costs[i]<=budget]
        if not fits: break
        if method=="magnitude":
            g=n*(sw@(state.gains()/geometry.profile[:,None])); score=relevance_weight*r+(1-relevance_weight)*g
        elif method=="mmr" and chosen:
            red=np.maximum(np.asarray(similarity)[:,chosen].max(1),0); g=-red
            score=mmr_weight*r-(1-mmr_weight)*red
        elif method in {"relevance","mmr"}: g=np.zeros(n); score=r
        else: raise ValueError("unknown method")
        i=max(fits,key=lambda j:(score[j]/costs[j],relevance[j],-j))
        if method=="mmr" and chosen and score[i]<=0: break
        chosen.append(i); spent+=int(costs[i])
        if state is not None: state.add(i)
        trace.append({"index":i,"score":float(score[i]),"cost":int(costs[i])})
    return {"indices":chosen,"cost":spent,"trace":trace}

