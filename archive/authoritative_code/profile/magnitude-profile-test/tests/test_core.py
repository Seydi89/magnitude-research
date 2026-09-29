import numpy as np

from magprofile.dataset import generate_dataset
from magprofile.magnitude import (
    complete_geometry_features,
    magnitude_from_distances,
    marginal_profile,
    sigmoid_profile_decomposition,
)


def test_singleton_magnitude_is_one():
    assert np.isclose(magnitude_from_distances(np.zeros((1, 1)), 1.0), 1.0, atol=1e-6)


def test_duplicate_has_nearly_zero_marginal_gain():
    base = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    profile = marginal_profile(base, base[0], np.geomspace(0.1, 10.0, 8))
    assert np.max(np.abs(profile)) < 1e-5


def test_farther_candidate_has_larger_gain_at_large_scale():
    base = np.asarray([[1.0, 0.0], [0.98, 0.2]])
    near = np.asarray([0.95, 0.31])
    far = np.asarray([-1.0, 0.0])
    scales = np.asarray([5.0, 10.0])
    assert np.all(marginal_profile(base, far, scales) > marginal_profile(base, near, scales))


def test_dataset_keeps_all_variants_in_one_group():
    data = generate_dataset(groups=5)
    counts = data.groupby("group_id")["candidate_type"].nunique()
    assert (counts == 7).all()
    assert set(data["has_new_information"]) == {0, 1}
    assert data["template_family"].nunique() == 5


def test_geometry_feature_is_invariant_to_original_base_order():
    base = np.asarray([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    candidate = np.asarray([0.8, 0.2])
    a = complete_geometry_features(base, candidate, max_set_size=4)
    b = complete_geometry_features(base[[2, 0, 1]], candidate, max_set_size=4)
    assert np.allclose(a, b)


def test_sigmoid_decomposition_recovers_simple_curve():
    scales = np.geomspace(0.05, 100.0, 32)
    x = np.log(scales)
    profile = 0.9 / (1.0 + np.exp(-2.5 * (x - 1.2)))
    params, residual = sigmoid_profile_decomposition(profile, scales)
    assert np.allclose(params[:3], [0.9, 2.5, 1.2], atol=1e-3)
    assert np.sqrt(np.mean(residual**2)) < 1e-5
