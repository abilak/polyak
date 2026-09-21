import numpy as np

from sparse_ecp.domains import HardThresholdedBall, SparseBall, SparseBox


def test_sparse_ball_sampler_respects_geometry():
    domain = SparseBall(dimension=20, sparsity=3, radius=2.0)
    draws = domain.sample_many(np.random.default_rng(7), 500)
    assert np.all(np.count_nonzero(draws, axis=1) == 3)
    assert np.all(np.linalg.norm(draws, axis=1) <= 2.0 + 1e-12)
    assert domain.support_count == 1140


def test_explicit_support_weights_are_used():
    supports = [(0, 1), (2, 3)]
    domain = SparseBall(
        4,
        2,
        supports=supports,
        support_weights={(0, 1): 0.0, (2, 3): 1.0},
    )
    rng = np.random.default_rng(1)
    assert {domain.sample_support(rng) for _ in range(20)} == {(2, 3)}


def test_hard_threshold_ablation_is_sparse_but_not_support_first():
    domain = HardThresholdedBall(dimension=10, sparsity=2, radius=1.0)
    draws = np.vstack([domain.sample(np.random.default_rng(seed)) for seed in range(20)])
    assert np.all(np.count_nonzero(draws, axis=1) == 2)
    assert np.all(np.linalg.norm(draws, axis=1) <= 1.0 + 1e-12)


def test_mixed_sparse_box_uses_requested_support_size_prior():
    domain = SparseBox(8, 4, size_weights={1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0})
    draws = domain.sample_many(np.random.default_rng(4), 50)
    assert np.all(np.count_nonzero(draws, axis=1) == 1)
    assert np.all(np.abs(draws) <= 1.0)
