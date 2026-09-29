#!/usr/bin/env python3
import json, numpy as np
from magnitude_research.magnitude import MagnitudeGeometry
from magnitude_research.diagnostics import profile_shape

rng=np.random.default_rng(9); base=rng.normal(size=(8,12)); duplicate=base[0]+rng.normal(0,.01,12); novel=rng.normal(size=12)
scales=np.geomspace(.01,100,32)
def marginal(candidate):
    before=MagnitudeGeometry.from_points(base,scales,metric="cosine").profile
    after=MagnitudeGeometry.from_points(np.vstack([base,candidate]),scales,metric="cosine").profile
    p=after-before; return {"profile":p.tolist(),"shape":profile_shape(p,scales)}
print(json.dumps({"duplicate":marginal(duplicate),"novel":marginal(novel)},indent=2))

