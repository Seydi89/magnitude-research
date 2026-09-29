import numpy as np
from magnitude_research.magnitude import MagnitudeGeometry
from magnitude_research.selection import select


def test_budget_and_unique_selection():
    rng=np.random.default_rng(1); x=rng.normal(size=(8,4)); costs=np.array([3,4,5,6,7,8,9,10])
    g=MagnitudeGeometry.from_points(x,np.geomspace(.02,8,9))
    out=select(np.linspace(1,0,8),costs,20,"magnitude",geometry=g)
    assert len(out["indices"])==len(set(out["indices"]))
    assert out["cost"]<=20

