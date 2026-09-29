#!/usr/bin/env python3
import json, numpy as np
from magnitude_research.magnitude import MagnitudeGeometry
from magnitude_research.selection import select

rng=np.random.default_rng(7); x=rng.normal(size=(12,6)); x[6:9]=x[0]+rng.normal(0,.02,(3,6))
rel=np.array([.95,.9,.8,.75,.7,.6,.94,.93,.92,.55,.5,.45]); costs=np.full(12,20)
g=MagnitudeGeometry.from_points(x,np.geomspace(1/64,16,33),metric="cosine")
sim=1-g.distances
out={m:select(rel,costs,100,m,geometry=g,similarity=sim) for m in ("relevance","mmr","magnitude")}
print(json.dumps(out,indent=2))

