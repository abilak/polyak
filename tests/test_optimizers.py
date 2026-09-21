import numpy as np

from sparse_ecp.domains import SparseBall
from sparse_ecp.objectives import SparseCone
from sparse_ecp.optimizers import ECP, RandomSearch, ReplicatedECP
from sparse_ecp.spaces import ContinuousOracle, FiniteOracle


def test_ecp_consumes_exact_evaluation_budget_and_tracks_proposals():
    center = np.array([0.5, 0.0, 0.0])
    oracle = ContinuousOracle(SparseBall(3, 1), SparseCone(center), true_max=0.0)
    trace = ECP(epsilon0=0.1, tau=1.2, rejection_growth=10).run(oracle, 12, seed=3)
    assert len(trace.values) == 12
    assert len(trace.proposals) == 12
    assert sum(trace.proposals) >= 12
    assert np.all(trace.to_frame(true_max=0.0)["simple_regret"] >= 0)


def test_finite_oracle_never_repeats_evaluated_candidates():
    x = np.eye(6)
    values = np.arange(6, dtype=float)
    supports = np.array([f"s{i}" for i in range(6)], dtype=object)
    oracle = FiniteOracle(x, values, supports)
    trace = RandomSearch(proposal="uniform_candidates").run(oracle, 6, seed=2)
    assert len({candidate.key for candidate in trace.candidates}) == 6


def test_oracles_report_sparse_intrinsic_dimension():
    continuous = ContinuousOracle(SparseBall(20, 2), lambda x: float(x.sum()))
    assert continuous.dimension == 20
    assert continuous.intrinsic_dimension == 2

    x = np.zeros((3, 20))
    x[0, [1, 4]] = 1.0
    x[1, [2, 9]] = 1.0
    x[2, [3, 7]] = 1.0
    finite = FiniteOracle(x, np.arange(3, dtype=float), np.array(["a", "b", "c"]))
    assert finite.dimension == 20
    assert finite.intrinsic_dimension == 2


def test_replicated_ecp_respects_noisy_oracle_call_budget():
    center = np.array([0.5, 0.0, 0.0])
    oracle = ContinuousOracle(SparseBall(3, 1), SparseCone(center), true_max=0.0)
    trace = ReplicatedECP(
        replications=4,
        sigma=0.1,
        epsilon0=1.0,
        tau=1.1,
        rejection_growth=10,
    ).run(oracle, 40, seed=3)
    frame = trace.to_frame(true_max=0.0)
    assert len(frame) == 10
    assert frame.iloc[-1]["cumulative_oracle_calls"] == 40
    assert "recommendation_regret" in frame
