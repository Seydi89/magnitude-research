import numpy as np
from magnitude_research.magnitude import (MagnitudeGeometry, leave_one_out_contributions,
                                           magnitude, similarity_kernel)


def test_limits_and_duplicates():
    x=np.array([[0.,0.],[0.,0.],[4.,0.]])
    g=MagnitudeGeometry.from_points(x,np.array([1e-4,100.]))
    assert 0.99 < g.profile[0] < 1.02
    assert 1.9 < g.profile[1] < 2.1


def test_leave_one_out_identity():
    d=np.array([[0.,.4,.8],[.4,0.,.5],[.8,.5,0.]])
    k=similarity_kernel(d,2.0); full=magnitude(k); loo=leave_one_out_contributions(k)
    explicit=np.array([full-magnitude(k[np.ix_([j for j in range(3) if j!=i],[j for j in range(3) if j!=i])]) for i in range(3)])
    np.testing.assert_allclose(loo,explicit,rtol=1e-7,atol=1e-7)

