from __future__ import annotations

from dataclasses import dataclass
from math import ceil, comb, gamma, log, log2, pi, sqrt


def unit_ball_volume(dimension: int) -> float:
    return pi ** (dimension / 2.0) / gamma(dimension / 2.0 + 1.0)


def eta_s(sparsity: int) -> float:
    """Explicit local-regime constant used by the sparse packing proof."""
    if sparsity < 1:
        raise ValueError("sparsity must be positive")
    return unit_ball_volume(sparsity) / (
        8.0 * sparsity * unit_ball_volume(sparsity - 1)
    )


@dataclass(frozen=True)
class MinimaxBounds:
    deterministic_lower: float
    deterministic_upper: float
    randomized_lower: float
    support_count: int
    local_regime: bool


@dataclass(frozen=True)
class ECPUpperBound:
    high_probability: float
    expected: float
    guaranteed_post_hit_evaluations: int
    first_post_hit_query: int


@dataclass(frozen=True)
class NoisyECPUpperBound:
    high_probability: float
    search_term: float
    estimation_term: float
    design_count: int
    replications: int
    guaranteed_post_hit_designs: int


@dataclass(frozen=True)
class ProposalComplexityBound:
    expected: float
    high_probability: float
    tolerance_growths: int
    half_acceptance_tolerance: float


def tolerance_burn_in(
    lipschitz: float,
    epsilon0: float,
    tau: float,
) -> tuple[int, int]:
    """Return ``(T_k, growths)`` from equation (4) of the paper."""
    if lipschitz <= 0 or epsilon0 <= 0 or tau <= 1:
        raise ValueError("lipschitz/epsilon0 must be positive and tau must exceed one")
    growths = max(0, ceil(log(lipschitz / epsilon0) / log(tau)))
    return growths + 2, growths


def uniform_size_weights(max_sparsity: int) -> dict[int, float]:
    if max_sparsity < 1:
        raise ValueError("max_sparsity must be positive")
    return {size: 1.0 / max_sparsity for size in range(1, max_sparsity + 1)}


def complexity_size_weights(max_sparsity: int) -> dict[int, float]:
    """The scale-free prior w_r=(s+1)/(s r(r+1)) from Corollary 15."""
    if max_sparsity < 1:
        raise ValueError("max_sparsity must be positive")
    normalizer = (max_sparsity + 1.0) / max_sparsity
    return {
        size: normalizer / (size * (size + 1.0))
        for size in range(1, max_sparsity + 1)
    }


def filter_gain(acceptance_mass: float, rejection_threshold: int) -> float:
    """Compute Psi_C(q) in Theorem 9 with stable endpoint handling."""
    if not 0 < acceptance_mass <= 1:
        raise ValueError("acceptance_mass must lie in (0,1]")
    if rejection_threshold < 1:
        raise ValueError("rejection_threshold must be positive")
    q = float(acceptance_mass)
    return float((1.0 - (1.0 - q) ** rejection_threshold) / q)


def minimax_bounds(
    dimension: int,
    sparsity: int,
    evaluations: int,
    lipschitz: float = 1.0,
    radius: float = 1.0,
) -> MinimaxBounds:
    """Finite-n bounds proved in docs/THEORY_AUDIT.md.

    They apply to noiseless simple regret over B_2^d(radius) intersect B_0(s).
    """
    if not 1 <= sparsity <= dimension:
        raise ValueError("sparsity must lie in [1, dimension]")
    if evaluations < 1 or lipschitz <= 0 or radius <= 0:
        raise ValueError("evaluations, lipschitz, and radius must be positive")
    supports = comb(dimension, sparsity)
    eta = eta_s(sparsity)
    det_scale = (supports / (4.0 * evaluations)) ** (1.0 / sparsity)
    rand_scale = (supports / (8.0 * evaluations)) ** (1.0 / sparsity)
    upper_scale = 3.0 * (supports / evaluations) ** (1.0 / sparsity)
    return MinimaxBounds(
        deterministic_lower=lipschitz * radius * min(eta, det_scale),
        deterministic_upper=lipschitz * radius * min(1.0, upper_scale),
        randomized_lower=0.25 * lipschitz * radius * min(eta, rand_scale),
        support_count=supports,
        local_regime=det_scale <= eta,
    )


def sample_complexity(
    dimension: int,
    sparsity: int,
    target_regret: float,
    lipschitz: float = 1.0,
    radius: float = 1.0,
) -> float:
    """Rate-level evaluation count M (L B / epsilon)^s.

    This is an asymptotic scaling law, not an exact integer threshold.
    """
    if target_regret <= 0:
        raise ValueError("target_regret must be positive")
    return comb(dimension, sparsity) * (lipschitz * radius / target_regret) ** sparsity


def sparse_ecp_upper_bound(
    dimension: int,
    sparsity: int,
    evaluations: int,
    *,
    lipschitz: float = 1.0,
    radius: float = 1.0,
    epsilon0: float = 1e-2,
    tau: float = 1.001,
    delta: float = 0.05,
) -> ECPUpperBound:
    """Correct support-first Sparse ECP upper bound with explicit burn-in.

    Query 1 is unfiltered. Query j>=2 is selected using a tolerance at least
    epsilon0 * tau**(j-2). Rejection-triggered growth can only make the hitting
    query earlier. See docs/THEORY_AUDIT.md for the tail proof.
    """
    if not 1 <= sparsity <= dimension:
        raise ValueError("sparsity must lie in [1, dimension]")
    if evaluations < 1 or lipschitz <= 0 or radius <= 0 or epsilon0 <= 0:
        raise ValueError("positive evaluations/constants are required")
    if tau <= 1 or not 0 < delta < 1:
        raise ValueError("tau must exceed one and delta must lie in (0,1)")
    first_post_hit_query, _ = tolerance_burn_in(lipschitz, epsilon0, tau)
    safe = max(0, evaluations - first_post_hit_query + 1)
    trivial = 2.0 * lipschitz * radius
    if safe == 0:
        return ECPUpperBound(trivial, trivial, safe, first_post_hit_query)
    supports = comb(dimension, sparsity)
    hp = 2.0 * lipschitz * radius * min(
        1.0,
        (supports * log(1.0 / delta) / safe) ** (1.0 / sparsity),
    )
    expected = 2.0 * lipschitz * radius * min(
        1.0,
        gamma(1.0 + 1.0 / sparsity) * (supports / safe) ** (1.0 / sparsity),
    )
    return ECPUpperBound(hp, expected, safe, first_post_hit_query)


def structured_ecp_upper_bound(
    sparsity: int,
    evaluations: int,
    optimal_support_probability: float,
    *,
    lipschitz: float = 1.0,
    radius: float = 1.0,
    epsilon0: float = 1e-2,
    tau: float = 1.001,
    delta: float = 0.05,
) -> ECPUpperBound:
    """Instance-dependent bound when a fixed prior gives S* probability pi(S*).

    This is not a structure-free minimax improvement: its gain depends on side
    information assigning the unknown optimal support non-negligible mass.
    """
    if sparsity < 1 or evaluations < 1:
        raise ValueError("sparsity and evaluations must be positive")
    if not 0 < optimal_support_probability <= 1:
        raise ValueError("optimal_support_probability must lie in (0,1]")
    if lipschitz <= 0 or radius <= 0 or epsilon0 <= 0:
        raise ValueError("positive Lipschitz/radius/tolerance constants are required")
    if tau <= 1 or not 0 < delta < 1:
        raise ValueError("tau must exceed one and delta must lie in (0,1)")
    first_post_hit_query, _ = tolerance_burn_in(lipschitz, epsilon0, tau)
    safe = max(0, evaluations - first_post_hit_query + 1)
    trivial = 2.0 * lipschitz * radius
    if safe == 0:
        return ECPUpperBound(trivial, trivial, safe, first_post_hit_query)
    inverse_mass = 1.0 / optimal_support_probability
    high_probability = 2.0 * lipschitz * radius * min(
        1.0,
        (inverse_mass * log(1.0 / delta) / safe) ** (1.0 / sparsity),
    )
    expected = 2.0 * lipschitz * radius * min(
        1.0,
        gamma(1.0 + 1.0 / sparsity) * (inverse_mass / safe) ** (1.0 / sparsity),
    )
    return ECPUpperBound(high_probability, expected, safe, first_post_hit_query)


def approximate_sparsity_oracle_bound(
    dimension: int,
    max_sparsity: int,
    evaluations: int,
    approximation_errors: dict[int, float],
    *,
    size_weights: dict[int, float] | None = None,
    lipschitz: float = 1.0,
    radius: float = 1.0,
    epsilon0: float = 1e-2,
    tau: float = 1.001,
    delta: float = 0.05,
) -> float:
    """Evaluate the finite-sample oracle inequality in Theorem 16 on a cube.

    ``approximation_errors[r]`` is a_r.  The best r-sparse comparator is
    assumed to use exactly r nonzeros, as it does in the supplied experiment.
    """
    if not 1 <= max_sparsity <= dimension or evaluations < 1:
        raise ValueError("invalid dimension, sparsity, or evaluation count")
    if not 0 < delta < 1 or lipschitz <= 0 or radius <= 0:
        raise ValueError("invalid confidence or scale")
    weights = size_weights or complexity_size_weights(max_sparsity)
    first, _ = tolerance_burn_in(lipschitz, epsilon0, tau)
    post = max(0, evaluations - first + 1)
    log_term = log(max_sparsity / delta)
    fallback = min(
        float(approximation_errors[size]) + lipschitz * 2.0 * radius * sqrt(max_sparsity)
        for size in approximation_errors
        if 1 <= size <= max_sparsity
    )
    candidates: list[float] = []
    for size, approximation_error in approximation_errors.items():
        if not 1 <= size <= max_sparsity:
            continue
        support_count = comb(dimension, size)
        weight = float(weights.get(size, 0.0))
        if weight <= 0 or post * weight / support_count < log_term:
            continue
        diameter = 2.0 * radius * sqrt(size)
        statistical = lipschitz * diameter * (
            support_count * log_term / (weight * post)
        ) ** (1.0 / size)
        candidates.append(float(approximation_error) + statistical)
    return min(candidates, default=fallback)


def noisy_sparse_ecp_upper_bound(
    dimension: int,
    sparsity: int,
    oracle_budget: int,
    replications: int,
    *,
    support_size_weight: float = 1.0,
    lipschitz: float = 1.0,
    radius: float = 1.0,
    sigma: float = 1.0,
    epsilon0: float = 1e-2,
    tau: float = 1.001,
    delta: float = 0.05,
) -> NoisyECPUpperBound:
    """Theorem 20 specialized to exact supports on ``[-radius,radius]^d``."""
    if not 1 <= sparsity <= dimension:
        raise ValueError("sparsity must lie in [1, dimension]")
    if oracle_budget < 1 or replications < 1 or sigma < 0:
        raise ValueError("oracle budget/replications must be positive and sigma nonnegative")
    if not 0 < support_size_weight <= 1 or not 0 < delta < 1:
        raise ValueError("invalid support weight or confidence")
    designs = oracle_budget // replications
    if designs < 1:
        raise ValueError("oracle budget must fund at least one replicated design")
    first, _ = tolerance_burn_in(lipschitz, epsilon0, tau)
    post = max(0, designs - first + 1)
    support_count = comb(dimension, sparsity)
    diameter = 2.0 * radius * sqrt(sparsity)
    estimation = 2.0 * sigma * sqrt(2.0 * log(4.0 * designs / delta) / replications)
    if post == 0 or post * support_size_weight / support_count < log(2.0 / delta):
        search = lipschitz * 2.0 * radius * sqrt(sparsity)
    else:
        search = lipschitz * diameter * (
            support_count * log(2.0 / delta) / (support_size_weight * post)
        ) ** (1.0 / sparsity)
    return NoisyECPUpperBound(
        high_probability=search + estimation,
        search_term=search,
        estimation_term=estimation,
        design_count=designs,
        replications=replications,
        guaranteed_post_hit_designs=post,
    )


def oracle_balanced_design_count(
    dimension: int,
    sparsity: int,
    oracle_budget: int,
    *,
    lipschitz: float = 1.0,
    radius: float = 1.0,
    sigma: float = 1.0,
    known_support: bool = False,
) -> int:
    """Rate-level allocation from Corollary 21, clipped to ``[1,N]``."""
    if sigma <= 0:
        return oracle_budget
    support_count = 1 if known_support else comb(dimension, sparsity)
    diameter = 2.0 * radius * sqrt(sparsity)
    designs = (
        (lipschitz * diameter / sigma) ** (2.0 * sparsity / (sparsity + 2.0))
        * support_count ** (2.0 / (sparsity + 2.0))
        * oracle_budget ** (sparsity / (sparsity + 2.0))
    )
    return max(1, min(oracle_budget, round(designs)))


def proposal_complexity_bound(
    evaluations: int,
    sparsity: int,
    *,
    objective_range: float,
    radius: float = 1.0,
    epsilon0: float = 1e-2,
    tau: float = 1.001,
    rejection_threshold: int = 1000,
    delta: float = 0.05,
) -> ProposalComplexityBound:
    """Expected and high-probability proposal bounds from Theorem 25."""
    if evaluations < 1 or sparsity < 1 or objective_range < 0 or radius <= 0:
        raise ValueError("invalid evaluation, sparsity, range, or radius")
    if epsilon0 <= 0 or tau <= 1 or rejection_threshold < 1 or not 0 < delta < 1:
        raise ValueError("invalid ECP or confidence parameter")
    half_tolerance = (objective_range / (2.0 * radius)) * (
        2.0 * evaluations * unit_ball_volume(sparsity)
    ) ** (1.0 / sparsity)
    growths = max(0, ceil(log(max(half_tolerance / epsilon0, 1.0)) / log(tau)))
    prefix = (rejection_threshold + 1) * growths
    return ProposalComplexityBound(
        expected=float(prefix + 2 * evaluations),
        high_probability=float(prefix + evaluations * ceil(log2(evaluations / delta))),
        tolerance_growths=growths,
        half_acceptance_tolerance=half_tolerance,
    )
