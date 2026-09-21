from __future__ import annotations

import warnings
from collections.abc import Hashable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

from .spaces import Candidate, OracleSpace

FloatArray = NDArray[np.float64]


@dataclass
class OptimizationTrace:
    algorithm: str
    candidates: list[Candidate] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    proposals: list[int] = field(default_factory=list)
    epsilons: list[float] = field(default_factory=list)
    observed_values: list[float] = field(default_factory=list)
    oracle_calls: list[int] = field(default_factory=list)
    recommendation_values: list[float] = field(default_factory=list)

    @property
    def best_value(self) -> float:
        return float(np.max(self.values))

    def to_frame(
        self,
        true_max: float | None = None,
        true_min: float | None = None,
    ) -> pd.DataFrame:
        return trace_to_frame(self, true_max=true_max, true_min=true_min)


def _trace_frame_fix(frame: pd.DataFrame, true_max: float | None, true_min: float | None) -> pd.DataFrame:
    if true_max is not None and true_min is not None and true_max > true_min:
        frame["fraction_of_range"] = np.clip(
            (frame["best_value"] - float(true_min)) / (float(true_max) - float(true_min)),
            0.0,
            1.0,
        )
    return frame


def trace_to_frame(
    trace: OptimizationTrace,
    true_max: float | None = None,
    true_min: float | None = None,
) -> pd.DataFrame:
    """Public frame conversion kept separate for backwards-stable result files."""
    values = np.asarray(trace.values, dtype=float)
    best = np.maximum.accumulate(values)
    payload = {
            "algorithm": trace.algorithm,
            "evaluation": np.arange(1, len(values) + 1),
            "value": values,
            "best_value": best,
            "proposals": trace.proposals,
            "epsilon": trace.epsilons,
            "candidate_id": [candidate.key for candidate in trace.candidates],
            "support_id": [candidate.support for candidate in trace.candidates],
        }
    if trace.observed_values:
        payload["observed_value"] = trace.observed_values
    if trace.oracle_calls:
        payload["oracle_calls"] = trace.oracle_calls
        payload["cumulative_oracle_calls"] = np.cumsum(trace.oracle_calls)
    if trace.recommendation_values:
        payload["recommendation_value"] = trace.recommendation_values
    frame = pd.DataFrame(payload)
    if true_max is not None:
        frame["simple_regret"] = np.maximum(0.0, float(true_max) - best)
        if trace.recommendation_values:
            frame["recommendation_regret"] = np.maximum(
                0.0, float(true_max) - np.asarray(trace.recommendation_values, dtype=float)
            )
    return _trace_frame_fix(frame, true_max, true_min)


def _record(
    trace: OptimizationTrace,
    space: OracleSpace,
    candidate: Candidate,
    proposals: int,
    epsilon: float,
    *,
    observed_value: float | None = None,
    oracle_calls: int = 1,
    recommendation_value: float | None = None,
) -> None:
    value = space.evaluate(candidate)
    trace.candidates.append(candidate)
    trace.values.append(value)
    trace.proposals.append(int(proposals))
    trace.epsilons.append(float(epsilon))
    trace.observed_values.append(value if observed_value is None else float(observed_value))
    trace.oracle_calls.append(int(oracle_calls))
    if recommendation_value is None:
        recommendation_value = max(trace.values)
    trace.recommendation_values.append(float(recommendation_value))


@dataclass
class RandomSearch:
    proposal: str = "support_first"
    name: str = "support_random"

    def run(self, space: OracleSpace, budget: int, seed: int) -> OptimizationTrace:
        if budget < 1:
            raise ValueError("budget must be positive")
        rng = np.random.default_rng(seed)
        trace = OptimizationTrace(self.name)
        evaluated: set[Hashable] = set()
        for _ in range(budget):
            candidate = space.sample(rng, self.proposal, evaluated)
            _record(trace, space, candidate, proposals=1, epsilon=np.nan)
            if candidate.key is not None:
                evaluated.add(candidate.key)
        return trace


@dataclass
class ECP:
    """ECP with an explicit proposal measure.

    `proposal="support_first"` is the corrected Sparse ECP.  The acceptance
    condition is unchanged.  Rejected proposals do not consume the black-box
    evaluation budget, but are counted for computational-cost reporting.
    """

    epsilon0: float = 1e-2
    tau: float | None = None
    rejection_growth: int = 1000
    proposal: str = "support_first"
    name: str = "sparse_ecp"
    # This is a computational safety guard, not an evaluation-budget parameter.
    # Slow-growth ablations (notably tau=1.001) can validly exceed five million
    # cheap proposals before completing 300 expensive evaluations.
    max_total_proposals: int = 50_000_000

    def run(self, space: OracleSpace, budget: int, seed: int) -> OptimizationTrace:
        if budget < 1:
            raise ValueError("budget must be positive")
        if (
            self.epsilon0 <= 0
            or self.rejection_growth < 1
            or self.max_total_proposals < budget
        ):
            raise ValueError(
                "epsilon0/rejection_growth must be positive and the proposal cap "
                "must cover the evaluation budget"
            )
        rng = np.random.default_rng(seed)
        epsilon = float(self.epsilon0)
        intrinsic_dimension = int(getattr(space, "intrinsic_dimension", space.dimension))
        effective_tau = self.tau or max(
            1.001,
            1.0 + 1.0 / (budget * intrinsic_dimension),
        )
        if effective_tau <= 1.0:
            raise ValueError("tau must exceed one")

        trace = OptimizationTrace(self.name)
        evaluated: set[Hashable] = set()
        first = space.sample(rng, self.proposal, evaluated)
        _record(trace, space, first, proposals=1, epsilon=epsilon)
        if first.key is not None:
            evaluated.add(first.key)

        x_history = [first.x]
        y_history = [trace.values[-1]]
        total_proposals = 1

        while len(trace.values) < budget:
            rejected_since_growth = 0
            proposals_this_evaluation = 0
            # The accepted history is constant throughout this rejection loop.
            # Materializing it once avoids millions of identical allocations in
            # the slow-tolerance ablations.
            x_hist = np.asarray(x_history)
            y_hist = np.asarray(y_history)
            best_y = float(np.max(y_hist))
            while True:
                if total_proposals >= self.max_total_proposals:
                    raise RuntimeError(
                        "ECP proposal cap reached "
                        f"after {len(trace.values)}/{budget} evaluations, "
                        f"{total_proposals:,} proposals, and epsilon={epsilon:.6g}; "
                        "increase max_total_proposals or use a faster tolerance schedule"
                    )
                candidate = space.sample(rng, self.proposal, evaluated)
                proposals_this_evaluation += 1
                total_proposals += 1
                distances = np.linalg.norm(x_hist - candidate.x, axis=1)
                accepted = float(np.min(y_hist + epsilon * distances)) >= best_y
                if accepted:
                    _record(trace, space, candidate, proposals_this_evaluation, epsilon)
                    if candidate.key is not None:
                        evaluated.add(candidate.key)
                    x_history.append(candidate.x)
                    y_history.append(trace.values[-1])
                    epsilon *= effective_tau
                    break
                rejected_since_growth += 1
                if rejected_since_growth >= self.rejection_growth:
                    epsilon *= effective_tau
                    rejected_since_growth = 0
        return trace


@dataclass
class ReplicatedRandomSearch:
    """Sparse Random Search with replicated Gaussian observations.

    ``budget`` is the total noisy-oracle-call budget, not the number of
    distinct design points.  Latent objective values are retained only for
    retrospective regret scoring.
    """

    replications: int = 4
    sigma: float = 1.0
    proposal: str = "support_first"
    name: str = "noisy_support_random"

    def run(self, space: OracleSpace, budget: int, seed: int) -> OptimizationTrace:
        if self.replications < 1 or self.sigma < 0:
            raise ValueError("replications must be positive and sigma nonnegative")
        designs = budget // self.replications
        if designs < 1:
            raise ValueError("budget must fund at least one replicated design")
        rng = np.random.default_rng(seed)
        trace = OptimizationTrace(self.name)
        evaluated: set[Hashable] = set()
        observed: list[float] = []
        for _ in range(designs):
            candidate = space.sample(rng, self.proposal, evaluated)
            latent = float(space.evaluate(candidate))
            mean = latent + float(rng.normal(0.0, self.sigma, size=self.replications).mean())
            observed.append(mean)
            recommendation = (
                trace.values[int(np.argmax(observed[:-1]))]
                if len(observed) > 1
                else latent
            )
            if mean >= max(observed):
                recommendation = latent
            _record(
                trace,
                space,
                candidate,
                proposals=1,
                epsilon=np.nan,
                observed_value=mean,
                oracle_calls=self.replications,
                recommendation_value=recommendation,
            )
            if candidate.key is not None:
                evaluated.add(candidate.key)
        return trace


@dataclass
class ReplicatedECP:
    """Confidence-adjusted replicated Sparse ECP from equation (15)."""

    replications: int = 4
    sigma: float = 1.0
    delta: float = 0.05
    epsilon0: float = 1e-2
    tau: float | None = None
    rejection_growth: int = 1000
    proposal: str = "support_first"
    name: str = "noisy_sparse_ecp"
    max_total_proposals: int = 50_000_000

    def run(self, space: OracleSpace, budget: int, seed: int) -> OptimizationTrace:
        if self.replications < 1 or self.sigma < 0:
            raise ValueError("replications must be positive and sigma nonnegative")
        if not 0 < self.delta < 1:
            raise ValueError("delta must lie in (0,1)")
        designs = budget // self.replications
        if designs < 1:
            raise ValueError("budget must fund at least one replicated design")
        if (
            self.epsilon0 <= 0
            or self.rejection_growth < 1
            or self.max_total_proposals < designs
        ):
            raise ValueError(
                "epsilon0/rejection_growth must be positive and the proposal cap "
                "must cover the design budget"
            )
        rng = np.random.default_rng(seed)
        intrinsic = int(getattr(space, "intrinsic_dimension", space.dimension))
        effective_tau = self.tau or max(1.001, 1.0 + 1.0 / (designs * intrinsic))
        if effective_tau <= 1:
            raise ValueError("tau must exceed one")
        beta = self.sigma * np.sqrt(2.0 * np.log(4.0 * designs / self.delta) / self.replications)
        epsilon = float(self.epsilon0)
        trace = OptimizationTrace(self.name)
        evaluated: set[Hashable] = set()
        first = space.sample(rng, self.proposal, evaluated)
        first_latent = float(space.evaluate(first))
        first_mean = first_latent + float(
            rng.normal(0.0, self.sigma, size=self.replications).mean()
        )
        _record(
            trace,
            space,
            first,
            proposals=1,
            epsilon=epsilon,
            observed_value=first_mean,
            oracle_calls=self.replications,
            recommendation_value=first_latent,
        )
        if first.key is not None:
            evaluated.add(first.key)
        x_history = [first.x]
        observed_history = [first_mean]
        total_proposals = 1

        while len(trace.values) < designs:
            rejected_since_growth = 0
            proposals_this_evaluation = 0
            x_hist = np.asarray(x_history)
            observed_hist = np.asarray(observed_history)
            lower_incumbent = float(np.max(observed_hist - beta))
            while True:
                if total_proposals >= self.max_total_proposals:
                    raise RuntimeError(
                        "noisy ECP proposal cap reached "
                        f"after {len(trace.values)}/{designs} designs, "
                        f"{total_proposals:,} proposals, and epsilon={epsilon:.6g}; "
                        "increase max_total_proposals or use a faster tolerance schedule"
                    )
                candidate = space.sample(rng, self.proposal, evaluated)
                proposals_this_evaluation += 1
                total_proposals += 1
                distances = np.linalg.norm(x_hist - candidate.x, axis=1)
                accepted = float(
                    np.min(observed_hist + beta + epsilon * distances)
                ) >= lower_incumbent
                if accepted:
                    latent = float(space.evaluate(candidate))
                    mean = latent + float(
                        rng.normal(0.0, self.sigma, size=self.replications).mean()
                    )
                    observed_history.append(mean)
                    existing_latent = trace.values + [latent]
                    recommendation = existing_latent[int(np.argmax(observed_history))]
                    _record(
                        trace,
                        space,
                        candidate,
                        proposals_this_evaluation,
                        epsilon,
                        observed_value=mean,
                        oracle_calls=self.replications,
                        recommendation_value=recommendation,
                    )
                    if candidate.key is not None:
                        evaluated.add(candidate.key)
                    x_history.append(candidate.x)
                    epsilon *= effective_tau
                    break
                rejected_since_growth += 1
                if rejected_since_growth >= self.rejection_growth:
                    epsilon *= effective_tau
                    rejected_since_growth = 0
        return trace


@dataclass
class SpaceFillingSearch:
    pool_size: int = 2048
    name: str = "space_filling"

    def run(self, space: OracleSpace, budget: int, seed: int) -> OptimizationTrace:
        if budget < 1:
            raise ValueError("budget must be positive")
        rng = np.random.default_rng(seed)
        trace = OptimizationTrace(self.name)
        evaluated: set[Hashable] = set()
        first = space.sample(rng, "support_first", evaluated)
        _record(trace, space, first, proposals=1, epsilon=np.nan)
        if first.key is not None:
            evaluated.add(first.key)
        history = [first.x]
        while len(trace.values) < budget:
            pool = space.candidate_pool(rng, self.pool_size, evaluated)
            if not pool:
                break
            matrix = np.asarray([candidate.x for candidate in pool])
            hist = np.asarray(history)
            nearest = np.min(np.linalg.norm(matrix[:, None, :] - hist[None, :, :], axis=2), axis=1)
            candidate = pool[int(np.argmax(nearest))]
            _record(trace, space, candidate, proposals=len(pool), epsilon=np.nan)
            if candidate.key is not None:
                evaluated.add(candidate.key)
            history.append(candidate.x)
        return trace


@dataclass
class GaussianProcessUCB:
    initial_points: int = 6
    pool_size: int = 5000
    beta: float = 2.0
    refit_interval: int = 1
    name: str = "gp_ucb"

    def run(self, space: OracleSpace, budget: int, seed: int) -> OptimizationTrace:
        if budget < 1:
            raise ValueError("budget must be positive")
        if self.initial_points < 1 or self.pool_size < 1 or self.refit_interval < 1:
            raise ValueError("initial_points, pool_size, and refit_interval must be positive")
        rng = np.random.default_rng(seed)
        trace = OptimizationTrace(self.name)
        evaluated: set[Hashable] = set()
        initial = min(self.initial_points, budget)
        for _ in range(initial):
            candidate = space.sample(rng, "support_first", evaluated)
            _record(trace, space, candidate, proposals=1, epsilon=np.nan)
            if candidate.key is not None:
                evaluated.add(candidate.key)

        gp = None
        scaler = None
        while len(trace.values) < budget:
            since_initial = len(trace.values) - initial
            if gp is None or since_initial % self.refit_interval == 0:
                x_train = np.asarray([candidate.x for candidate in trace.candidates])
                y_train = np.asarray(trace.values)
                scaler = StandardScaler().fit(x_train)
                x_scaled = scaler.transform(x_train)
                kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(
                    length_scale=np.ones(space.dimension),
                    length_scale_bounds="fixed",
                    nu=2.5,
                ) + WhiteKernel(noise_level=1e-6, noise_level_bounds="fixed")
                gp = GaussianProcessRegressor(
                    kernel=kernel,
                    normalize_y=True,
                    alpha=1e-8,
                    optimizer=None,
                    random_state=seed,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    gp.fit(x_scaled, y_train)
            pool = space.candidate_pool(rng, self.pool_size, evaluated)
            if not pool:
                break
            assert scaler is not None and gp is not None
            pool_matrix = scaler.transform(np.asarray([candidate.x for candidate in pool]))
            mean, std = gp.predict(pool_matrix, return_std=True)
            candidate = pool[int(np.argmax(mean + self.beta * std))]
            _record(trace, space, candidate, proposals=len(pool), epsilon=np.nan)
            if candidate.key is not None:
                evaluated.add(candidate.key)
        return trace


def make_optimizer(name: str, options: dict | None = None):
    options = dict(options or {})
    if name == "sparse_ecp":
        return ECP(name=name, proposal="support_first", **options)
    if name == "structured_sparse_ecp":
        return ECP(name=name, proposal="structured", **options)
    if name == "ecp_uniform":
        return ECP(name=name, proposal="uniform_candidates", **options)
    if name == "ecp_hard_threshold":
        return ECP(name=name, proposal="uniform_candidates", **options)
    if name == "dense_ecp":
        return ECP(name=name, proposal="uniform_candidates", **options)
    if name == "support_random":
        return RandomSearch(name=name, proposal="support_first", **options)
    if name == "random":
        return RandomSearch(name=name, proposal="uniform_candidates", **options)
    if name == "dense_random":
        return RandomSearch(name=name, proposal="uniform_candidates", **options)
    if name == "hard_threshold_random":
        return RandomSearch(name=name, proposal="uniform_candidates", **options)
    if name == "space_filling":
        return SpaceFillingSearch(name=name, **options)
    if name == "gp_ucb":
        return GaussianProcessUCB(name=name, **options)
    if name == "noisy_sparse_ecp":
        return ReplicatedECP(name=name, proposal="support_first", **options)
    if name == "noisy_support_random":
        return ReplicatedRandomSearch(name=name, proposal="support_first", **options)
    raise ValueError(f"unknown optimizer: {name}")
