from math import comb

from sparse_ecp.theory import (
    complexity_size_weights,
    filter_gain,
    minimax_bounds,
    noisy_sparse_ecp_upper_bound,
    proposal_complexity_bound,
    sparse_ecp_upper_bound,
    structured_ecp_upper_bound,
)
from sparse_ecp.theory_experiments import _run_noiseless


def test_full_dimension_recovers_n_to_minus_one_over_d_rate():
    result = minimax_bounds(3, 3, 1000)
    assert result.support_count == 1
    assert result.deterministic_upper <= 3 * 1000 ** (-1 / 3) + 1e-12


def test_support_factor_is_combinatorial():
    result = minimax_bounds(12, 2, 1_000_000)
    assert result.support_count == comb(12, 2)
    assert result.local_regime


def test_ecp_bound_accounts_for_unfiltered_first_query():
    result = sparse_ecp_upper_bound(
        5,
        1,
        20,
        lipschitz=1.0,
        epsilon0=1.0,
        tau=2.0,
    )
    assert result.first_post_hit_query == 2
    assert result.guaranteed_post_hit_evaluations == 19


def test_uniform_structured_prior_recovers_sparse_ecp_bound():
    uniform = sparse_ecp_upper_bound(8, 2, 1000, epsilon0=1.0, tau=1.01)
    structured = structured_ecp_upper_bound(
        2,
        1000,
        1.0 / comb(8, 2),
        epsilon0=1.0,
        tau=1.01,
    )
    assert uniform == structured


def test_complexity_prior_normalizes_and_filter_gain_is_bounded():
    weights = complexity_size_weights(8)
    assert abs(sum(weights.values()) - 1.0) < 1e-12
    assert 1.0 < filter_gain(0.25, 10) <= 10.0


def test_noisy_and_proposal_bounds_report_finite_components():
    noisy = noisy_sparse_ecp_upper_bound(
        2,
        2,
        500,
        10,
        epsilon0=1.0,
        tau=1.1,
        sigma=0.2,
    )
    assert noisy.design_count == 50
    assert noisy.high_probability == noisy.search_term + noisy.estimation_term
    proposals = proposal_complexity_bound(
        100,
        2,
        objective_range=2.0,
        epsilon0=1.0,
        tau=1.1,
        rejection_threshold=20,
    )
    assert proposals.high_probability >= proposals.expected


def test_noiseless_endpoint_only_mode_skips_large_trajectory_table():
    trajectories, endpoints, _, _ = _run_noiseless(
        {
            "dimensions": [2],
            "sparsities": [1],
            "upper_sparsity": 1,
            "budgets": [2, 3],
            "seed_count": 1,
            "proposal_modes": ["known_support"],
            "save_trajectories": False,
        }
    )

    assert trajectories.empty
    assert len(endpoints) == 4
