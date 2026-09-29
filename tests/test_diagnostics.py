import numpy as np
from magnitude_research.diagnostics import effective_rank, profile_drift


def test_drift_zero_and_rank():
    p=np.arange(8,dtype=float)
    assert profile_drift(p,p)=={"l1":0.0,"l2":0.0,"maximum":0.0}
    assert 1.9 < effective_rank(np.eye(3)) <= 3

