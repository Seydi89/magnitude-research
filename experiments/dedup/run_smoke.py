#!/usr/bin/env python3
import json, numpy as np
from magnitude_research.magnitude import MagnitudeGeometry
from magnitude_research.dedup import retain_nonredundant

rng=np.random.default_rng(11); core=rng.normal(size=(6,8)); x=np.vstack([core,core[0]+.005,core[0]+.01])
g=MagnitudeGeometry.from_points(x,np.geomspace(1/64,16,33),metric="cosine")
result=retain_nonredundant(g,np.linspace(1,.5,len(x)),1-g.distances,.2,.98)
print(json.dumps(result,indent=2))

