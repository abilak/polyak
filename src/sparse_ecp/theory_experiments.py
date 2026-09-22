from __future__ import annotations

from math import ceil, comb, log, log2, sqrt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .domains import SparseBox
from .objectives import SparseCone, make_sparse_center
from .optimizers import ECP, RandomSearch, ReplicatedECP, ReplicatedRandomSearch
from .plotting import (
    plot_filter_calibration,
    plot_noisy_scaling,
    plot_proposal_complexity,
    plot_theory_scaling,
)
from .spaces import ContinuousOracle
from .theory import (
    approximate_sparsity_oracle_bound,
    complexity_size_weights,
    filter_gain,
    noisy_sparse_ecp_upper_bound,
    oracle_balanced_design_count,
    proposal_complexity_bound,
    tolerance_burn_in,
    uniform_size_weights,
)


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise TypeError("theory-validation configuration must be a mapping")
    return payload


def _seeds(section: dict[str, Any], default_count: int) -> list[int]:
    if "seeds" in section:
        return [int(value) for value in section["seeds"]]
    count = int(section.get("seed_count", default_count))
    if count < 1:
        raise ValueError("seed_count must be positive")
    return list(range(count))


def _proposal_mode(
    mode: str,
    dimension: int,
    true_sparsity: int,
    upper_sparsity: int,
    radius: float,
    support: tuple[int, ...],
) -> tuple[SparseBox, float, int]:
    if mode == "known_support":
        return (
            SparseBox(
                dimension,
                true_sparsity,
                radius=radius,
                supports=[support],
            ),
            1.0,
            1,
        )
    if mode == "exact_unknown":
        return (
            SparseBox(dimension, true_sparsity, radius=radius),
            1.0,
            comb(dimension, true_sparsity),
        )
    if mode == "uniform_adaptive":
        weights = uniform_size_weights(upper_sparsity)
        return (
            SparseBox(
                dimension,
                upper_sparsity,
                radius=radius,
                size_weights=weights,
            ),
            weights[true_sparsity],
            comb(dimension, true_sparsity),
        )
    if mode == "complexity_adaptive":
        weights = complexity_size_weights(upper_sparsity)
        return (
            SparseBox(
                dimension,
                upper_sparsity,
                radius=radius,
                size_weights=weights,
            ),
            weights[true_sparsity],
            comb(dimension, true_sparsity),
        )
    raise ValueError(f"unknown proposal mode: {mode}")


def _finite_regret_bound(
    *,
    evaluations: int,
    true_sparsity: int,
    upper_sparsity: int,
    support_count: int,
    size_weight: float,
    radius: float,
    lipschitz: float,
    epsilon0: float,
    tau: float,
    delta: float,
) -> float:
    first, _ = tolerance_burn_in(lipschitz, epsilon0, tau)
    post = max(0, evaluations - first + 1)
    global_diameter = 2.0 * radius * sqrt(upper_sparsity)
    if post == 0 or post * size_weight / support_count < log(1.0 / delta):
        return lipschitz * global_diameter
    slice_diameter = 2.0 * radius * sqrt(true_sparsity)
    return lipschitz * slice_diameter * (
        support_count * log(1.0 / delta) / (size_weight * post)
    ) ** (1.0 / true_sparsity)


def _filter_records(
    trace,
    domain: SparseBox,
    objective: SparseCone,
    *,
    seed: int,
    target_regret: float,
    lipschitz: float,
    rejection_threshold: int,
    samples: int,
    stride: int,
) -> list[dict[str, float | int]]:
    records: list[dict[str, float | int]] = []
    target_value = objective.maximum - target_regret
    diagnostic_rng = np.random.default_rng(np.random.SeedSequence([seed, 99173]))
    for next_index in range(1, len(trace.values), max(1, stride)):
        epsilon = float(trace.epsilons[next_index])
        history_values = np.asarray(trace.values[:next_index], dtype=float)
        incumbent = float(np.max(history_values))
        if epsilon < lipschitz or incumbent >= target_value:
            continue
        proposals = domain.sample_many(diagnostic_rng, samples)
        history_x = np.asarray([candidate.x for candidate in trace.candidates[:next_index]])
        distances = np.linalg.norm(proposals[:, None, :] - history_x[None, :, :], axis=2)
        accepted = np.min(history_values[None, :] + epsilon * distances, axis=1) >= incumbent
        proposal_values = np.asarray([objective(point) for point in proposals], dtype=float)
        target = proposal_values >= target_value
        q = float(np.mean(accepted))
        p = float(np.mean(target))
        if q <= 0:
            continue
        records.append(
            {
                "seed": seed,
                "next_evaluation": next_index + 1,
                "acceptance_mass": q,
                "target_mass": p,
                "psi_c": filter_gain(q, rejection_threshold),
                "predicted_success_lower": p * filter_gain(q, rejection_threshold),
                "target_hit": int(trace.values[next_index] >= target_value),
                "proposals_used": int(trace.proposals[next_index]),
            }
        )
    return records


def _run_noiseless(
    section: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dimensions = [int(value) for value in section.get("dimensions", [8, 16])]
    sparsities = [int(value) for value in section.get("sparsities", [1, 2, 3])]
    budgets = sorted(int(value) for value in section.get("budgets", [50, 100, 200, 400]))
    seeds = _seeds(section, 100)
    modes = [
        str(value)
        for value in section.get(
            "proposal_modes",
            ["known_support", "exact_unknown", "uniform_adaptive", "complexity_adaptive"],
        )
    ]
    upper_sparsity = int(section.get("upper_sparsity", max(sparsities)))
    radius = float(section.get("radius", 1.0))
    lipschitz = float(section.get("lipschitz", 1.0))
    epsilon0 = float(section.get("epsilon0", lipschitz))
    tau = float(section.get("tau", 1.01))
    rejection_threshold = int(section.get("rejection_threshold", 200))
    delta = float(section.get("delta", 0.05))
    diagnostics = dict(section.get("filter_diagnostics", {}))
    diagnostic_samples = int(diagnostics.get("samples", 1000))
    diagnostic_stride = int(diagnostics.get("stride", 5))
    target_regret = float(diagnostics.get("target_regret", 0.25))
    save_trajectories = bool(section.get("save_trajectories", True))
    max_budget = max(budgets)
    trajectory_frames: list[pd.DataFrame] = []
    endpoint_records: list[dict[str, Any]] = []
    filter_records: list[dict[str, Any]] = []
    proposal_records: list[dict[str, Any]] = []

    for dimension in dimensions:
        for true_sparsity in sparsities:
            if true_sparsity > dimension or true_sparsity > upper_sparsity:
                continue
            for seed in seeds:
                center_rng = np.random.default_rng(
                    np.random.SeedSequence([seed, dimension, true_sparsity, 104729])
                )
                center = make_sparse_center(
                    dimension,
                    true_sparsity,
                    radius * 0.65,
                    center_rng,
                )
                support = tuple(np.flatnonzero(center).tolist())
                objective = SparseCone(center, lipschitz=lipschitz)
                for mode in modes:
                    domain, size_weight, support_count = _proposal_mode(
                        mode,
                        dimension,
                        true_sparsity,
                        upper_sparsity,
                        radius,
                        support,
                    )
                    oracle = ContinuousOracle(domain, objective, true_max=0.0)
                    algorithms = [
                        (
                            f"sparse_ecp/{mode}",
                            ECP(
                                epsilon0=epsilon0,
                                tau=tau,
                                rejection_growth=rejection_threshold,
                                name=f"sparse_ecp/{mode}",
                            ),
                        ),
                        (
                            f"sparse_random/{mode}",
                            RandomSearch(name=f"sparse_random/{mode}"),
                        ),
                    ]
                    for algorithm_name, algorithm in algorithms:
                        trace = algorithm.run(oracle, max_budget, seed)
                        frame = trace.to_frame(true_max=0.0)
                        frame["experiment"] = "noiseless_theory"
                        frame["algorithm"] = algorithm_name
                        frame["proposal_mode"] = mode
                        frame["dimension"] = dimension
                        frame["true_sparsity"] = true_sparsity
                        frame["upper_sparsity"] = upper_sparsity
                        frame["seed"] = seed
                        if save_trajectories:
                            trajectory_frames.append(frame)
                        for budget in budgets:
                            row = frame.iloc[budget - 1]
                            theory_bound = _finite_regret_bound(
                                evaluations=budget,
                                true_sparsity=true_sparsity,
                                upper_sparsity=upper_sparsity,
                                support_count=support_count,
                                size_weight=size_weight,
                                radius=radius,
                                lipschitz=lipschitz,
                                epsilon0=epsilon0,
                                tau=tau,
                                delta=delta,
                            )
                            endpoint_records.append(
                                {
                                    "dimension": dimension,
                                    "true_sparsity": true_sparsity,
                                    "upper_sparsity": upper_sparsity,
                                    "proposal_mode": mode,
                                    "algorithm": algorithm_name,
                                    "seed": seed,
                                    "budget": budget,
                                    "simple_regret": float(row["simple_regret"]),
                                    "theory_high_probability_bound": theory_bound,
                                    "within_theory_bound": int(
                                        float(row["simple_regret"]) <= theory_bound + 1e-12
                                    ),
                                    "support_count": support_count,
                                    "size_weight": size_weight,
                                }
                            )
                        if algorithm_name == "sparse_ecp/exact_unknown":
                            for record in _filter_records(
                                trace,
                                domain,
                                objective,
                                seed=seed,
                                target_regret=target_regret,
                                lipschitz=lipschitz,
                                rejection_threshold=rejection_threshold,
                                samples=diagnostic_samples,
                                stride=diagnostic_stride,
                            ):
                                record.update(
                                    {
                                        "dimension": dimension,
                                        "true_sparsity": true_sparsity,
                                        "target_regret": target_regret,
                                    }
                                )
                                filter_records.append(record)
                            objective_range = 2.0 * lipschitz * radius * sqrt(true_sparsity)
                            for budget in budgets:
                                bound = proposal_complexity_bound(
                                    budget,
                                    true_sparsity,
                                    objective_range=objective_range,
                                    radius=radius,
                                    epsilon0=epsilon0,
                                    tau=tau,
                                    rejection_threshold=rejection_threshold,
                                    delta=delta,
                                )
                                actual = int(sum(trace.proposals[:budget]))
                                proposal_records.append(
                                    {
                                        "dimension": dimension,
                                        "true_sparsity": true_sparsity,
                                        "seed": seed,
                                        "budget": budget,
                                        "actual_proposals": actual,
                                        "expected_upper": bound.expected,
                                        "high_probability_upper": bound.high_probability,
                                        "within_high_probability_bound": int(
                                            actual <= bound.high_probability
                                        ),
                                    }
                                )

    trajectories = (
        pd.concat(trajectory_frames, ignore_index=True)
        if trajectory_frames
        else pd.DataFrame()
    )
    return (
        trajectories,
        pd.DataFrame.from_records(endpoint_records),
        pd.DataFrame.from_records(filter_records),
        pd.DataFrame.from_records(proposal_records),
    )


def _run_approximate(section: dict[str, Any]) -> pd.DataFrame:
    dimension = int(section.get("dimension", 12))
    max_sparsity = int(section.get("max_sparsity", 4))
    budgets = sorted(int(value) for value in section.get("budgets", [100, 250, 500, 1000]))
    seeds = _seeds(section, 100)
    radius = float(section.get("radius", 1.0))
    lipschitz = float(section.get("lipschitz", 1.0))
    epsilon0 = float(section.get("epsilon0", lipschitz))
    tau = float(section.get("tau", 1.01))
    rejection_threshold = int(section.get("rejection_threshold", 200))
    delta = float(section.get("delta", 0.05))
    decay = float(section.get("coefficient_decay", 1.5))
    coefficients = np.arange(1, dimension + 1, dtype=float) ** (-decay)
    center = radius * 0.8 * coefficients / coefficients.max()
    objective = SparseCone(center, lipschitz=lipschitz)
    weights = complexity_size_weights(max_sparsity)
    approximation_errors = {
        size: lipschitz * float(np.linalg.norm(center[size:]))
        for size in range(1, max_sparsity + 1)
    }
    records: list[dict[str, Any]] = []
    for seed in seeds:
        domain = SparseBox(
            dimension,
            max_sparsity,
            radius=radius,
            size_weights=weights,
        )
        oracle = ContinuousOracle(domain, objective, true_max=0.0)
        optimizer = ECP(
            epsilon0=epsilon0,
            tau=tau,
            rejection_growth=rejection_threshold,
            name="sparse_ecp/complexity_prior",
        )
        trace = optimizer.run(oracle, max(budgets), seed)
        frame = trace.to_frame(true_max=0.0)
        for budget in budgets:
            bound = approximate_sparsity_oracle_bound(
                dimension,
                max_sparsity,
                budget,
                approximation_errors,
                size_weights=weights,
                lipschitz=lipschitz,
                radius=radius,
                epsilon0=epsilon0,
                tau=tau,
                delta=delta,
            )
            regret = float(frame.iloc[budget - 1]["simple_regret"])
            records.append(
                {
                    "dimension": dimension,
                    "max_sparsity": max_sparsity,
                    "seed": seed,
                    "budget": budget,
                    "simple_regret": regret,
                    "oracle_bound": bound,
                    "within_oracle_bound": int(regret <= bound + 1e-12),
                    **{
                        f"approximation_error_r{size}": approximation_errors[size]
                        for size in approximation_errors
                    },
                }
            )
    return pd.DataFrame.from_records(records)


def _run_multiscale_noisy_ecp(
    oracle: ContinuousOracle,
    oracle_budget: int,
    seed: int,
    *,
    sigma: float,
    delta: float,
    epsilon0: float,
    tau: float,
    rejection_threshold: int,
) -> dict[str, Any]:
    """Implement the dyadic replication/validation construction in Theorem 22."""
    scales = max(1, ceil(log2(oracle_budget)))
    block_budget = oracle_budget // (2 * scales)
    validation_calls = oracle_budget // (2 * scales)
    candidates = []
    latent_values = []
    replication_levels = []
    total_designs = 0
    calls_used = 0
    for scale in range(1, scales + 1):
        replications = 2**scale
        if block_budget < replications:
            continue
        run_seed = int(
            np.random.SeedSequence([seed, oracle_budget, scale, 32452843]).generate_state(1)[0]
        )
        optimizer = ReplicatedECP(
            replications=replications,
            sigma=sigma,
            delta=delta / (2.0 * scales),
            epsilon0=epsilon0,
            tau=tau,
            rejection_growth=rejection_threshold,
            name="noisy_multiscale_ecp_base",
        )
        trace = optimizer.run(oracle, block_budget, run_seed)
        recommendation_index = int(np.argmax(trace.observed_values))
        candidates.append(trace.candidates[recommendation_index])
        latent_values.append(float(trace.values[recommendation_index]))
        replication_levels.append(replications)
        total_designs += len(trace.values)
        calls_used += int(sum(trace.oracle_calls))
    if not candidates:
        raise ValueError("oracle budget is too small for a multiscale noisy run")
    validation_rng = np.random.default_rng(
        np.random.SeedSequence([seed, oracle_budget, 49979687])
    )
    validation_means = [
        latent
        + float(validation_rng.normal(0.0, sigma, size=validation_calls).mean())
        for latent in latent_values
    ]
    selected = int(np.argmax(validation_means))
    calls_used += validation_calls * len(candidates)
    return {
        "recommendation_regret": max(0.0, float(oracle.true_max) - latent_values[selected]),
        "designs": total_designs,
        "replications": replication_levels[selected],
        "calls_used": calls_used,
        "multiscale_runs": len(candidates),
        "validation_calls_per_candidate": validation_calls,
    }


def _run_noisy(section: dict[str, Any]) -> pd.DataFrame:
    dimension = int(section.get("dimension", 8))
    sparsities = [int(value) for value in section.get("sparsities", [1, 2, 3])]
    budgets = sorted(int(value) for value in section.get("budgets", [128, 256, 512, 1024]))
    seeds = _seeds(section, 100)
    radius = float(section.get("radius", 1.0))
    lipschitz = float(section.get("lipschitz", 1.0))
    sigma = float(section.get("sigma", 0.25))
    epsilon0 = float(section.get("epsilon0", lipschitz))
    tau = float(section.get("tau", 1.01))
    rejection_threshold = int(section.get("rejection_threshold", 200))
    delta = float(section.get("delta", 0.05))
    configured_support_modes = section.get("known_support", True)
    if isinstance(configured_support_modes, list):
        support_modes = [bool(value) for value in configured_support_modes]
    else:
        support_modes = [bool(configured_support_modes)]
    include_multiscale = bool(section.get("multiscale", True))
    records: list[dict[str, Any]] = []
    for sparsity in sparsities:
        if sparsity > dimension:
            continue
        for seed in seeds:
            center_rng = np.random.default_rng(
                np.random.SeedSequence([seed, dimension, sparsity, 130363])
            )
            center = make_sparse_center(dimension, sparsity, radius * 0.65, center_rng)
            support = tuple(np.flatnonzero(center).tolist())
            objective = SparseCone(center, lipschitz=lipschitz)
            for known_support in support_modes:
                for budget in budgets:
                    designs = oracle_balanced_design_count(
                        dimension,
                        sparsity,
                        budget,
                        lipschitz=lipschitz,
                        radius=radius,
                        sigma=sigma,
                        known_support=known_support,
                    )
                    replications = max(1, budget // designs)
                    domain = SparseBox(
                        dimension,
                        sparsity,
                        radius=radius,
                        supports=[support] if known_support else None,
                    )
                    oracle = ContinuousOracle(domain, objective, true_max=0.0)
                    algorithms = [
                        ReplicatedECP(
                            replications=replications,
                            sigma=sigma,
                            delta=delta,
                            epsilon0=epsilon0,
                            tau=tau,
                            rejection_growth=rejection_threshold,
                        ),
                        ReplicatedRandomSearch(
                            replications=replications,
                            sigma=sigma,
                        ),
                    ]
                    for optimizer in algorithms:
                        trace = optimizer.run(oracle, budget, seed)
                        frame = trace.to_frame(true_max=0.0)
                        bound_dimension = sparsity if known_support else dimension
                        theory = noisy_sparse_ecp_upper_bound(
                            bound_dimension,
                            sparsity,
                            budget,
                            replications,
                            lipschitz=lipschitz,
                            radius=radius,
                            sigma=sigma,
                            epsilon0=epsilon0,
                            tau=tau,
                            delta=delta,
                        )
                        recommendation_regret = float(
                            frame.iloc[-1]["recommendation_regret"]
                        )
                        records.append(
                            {
                                "dimension": dimension,
                                "sparsity": sparsity,
                                "known_support": known_support,
                                "algorithm": trace.algorithm,
                                "seed": seed,
                                "oracle_budget": budget,
                                "designs": len(frame),
                                "replications": replications,
                                "calls_used": int(
                                    frame.iloc[-1]["cumulative_oracle_calls"]
                                ),
                                "recommendation_regret": recommendation_regret,
                                "theory_high_probability_bound": theory.high_probability,
                                "within_theory_bound": int(
                                    optimizer.name != "noisy_sparse_ecp"
                                    or recommendation_regret
                                    <= theory.high_probability + 1e-12
                                ),
                            }
                        )
                    if include_multiscale:
                        multiscale = _run_multiscale_noisy_ecp(
                            oracle,
                            budget,
                            seed,
                            sigma=sigma,
                            delta=delta,
                            epsilon0=epsilon0,
                            tau=tau,
                            rejection_threshold=rejection_threshold,
                        )
                        records.append(
                            {
                                "dimension": dimension,
                                "sparsity": sparsity,
                                "known_support": known_support,
                                "algorithm": "noisy_multiscale_ecp",
                                "seed": seed,
                                "oracle_budget": budget,
                                "theory_high_probability_bound": np.nan,
                                "within_theory_bound": np.nan,
                                **multiscale,
                            }
                        )
    return pd.DataFrame.from_records(records)


def _slope_summary(
    frame: pd.DataFrame,
    *,
    budget_column: str,
    regret_column: str,
    group_columns: list[str],
    theory_exponent,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for labels, group in frame.groupby(group_columns, dropna=False):
        if not isinstance(labels, tuple):
            labels = (labels,)
        medians = group.groupby(budget_column)[regret_column].median()
        medians = medians[medians > 0]
        if len(medians) < 2:
            continue
        slope, intercept = np.polyfit(np.log(medians.index), np.log(medians.values), 1)
        record = dict(zip(group_columns, labels))
        record.update(
            {
                "fitted_slope": float(slope),
                "intercept": float(intercept),
                "theory_slope": float(theory_exponent(record)),
                "budget_points": len(medians),
            }
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


def run_theory_validation(config_path: str | Path) -> dict[str, Path]:
    """Run the theorem-aligned experiment battery described in docs/EXPERIMENTS.md."""
    config = _load_config(config_path)
    output_dir = Path(config.get("output_dir", "results/theory_validation"))
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    noiseless_section = dict(config.get("noiseless", {}))
    if noiseless_section.get("enabled", True):
        noiseless, endpoints, filters, proposals = _run_noiseless(noiseless_section)
        for label, frame in {
            "noiseless_trajectories": noiseless,
            "noiseless_endpoints": endpoints,
            "filter_diagnostics": filters,
            "proposal_complexity": proposals,
        }.items():
            if frame.empty and label == "noiseless_trajectories":
                continue
            path = output_dir / f"{label}.csv"
            frame.to_csv(path, index=False)
            paths[label] = path

        noiseless_slopes = _slope_summary(
            endpoints,
            budget_column="budget",
            regret_column="simple_regret",
            group_columns=["dimension", "true_sparsity", "proposal_mode", "algorithm"],
            theory_exponent=lambda record: -1.0 / float(record["true_sparsity"]),
        )
        path = output_dir / "noiseless_slopes.csv"
        noiseless_slopes.to_csv(path, index=False)
        paths["noiseless_slopes"] = path
        paths["noiseless_plot"] = plot_theory_scaling(
            endpoints, output_dir / "noiseless_scaling.png"
        )
        if not filters.empty:
            filters = filters.copy()
            filters["acceptance_mass_bin"] = pd.cut(
                filters["acceptance_mass"],
                bins=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
                include_lowest=True,
            ).astype(str)
            calibration = filters.groupby("acceptance_mass_bin", observed=True).agg(
                histories=("target_hit", "size"),
                acceptance_mass=("acceptance_mass", "mean"),
                empirical_success=("target_hit", "mean"),
                theorem_lower=("predicted_success_lower", "mean"),
            ).reset_index()
        else:
            calibration = pd.DataFrame()
        calibration_path = output_dir / "filter_calibration.csv"
        calibration.to_csv(calibration_path, index=False)
        paths["filter_calibration"] = calibration_path
        if not calibration.empty:
            paths["filter_plot"] = plot_filter_calibration(
                calibration, output_dir / "filter_gain.png"
            )
        if not proposals.empty:
            paths["proposal_plot"] = plot_proposal_complexity(
                proposals, output_dir / "proposal_complexity.png"
            )
    else:
        endpoints = pd.DataFrame()
        proposals = pd.DataFrame()

    approximate_section = dict(config.get("approximate_sparsity", {}))
    if approximate_section.get("enabled", True):
        approximate = _run_approximate(approximate_section)
        approximate_path = output_dir / "approximate_sparsity.csv"
        approximate.to_csv(approximate_path, index=False)
        paths["approximate_sparsity"] = approximate_path
    else:
        approximate = pd.DataFrame()

    noisy_section = dict(config.get("noisy", {}))
    if noisy_section.get("enabled", True):
        noisy = _run_noisy(noisy_section)
        noisy_path = output_dir / "noisy_endpoints.csv"
        noisy.to_csv(noisy_path, index=False)
        paths["noisy_endpoints"] = noisy_path
        noisy_slopes = _slope_summary(
            noisy,
            budget_column="oracle_budget",
            regret_column="recommendation_regret",
            group_columns=["sparsity", "known_support", "algorithm"],
            theory_exponent=lambda record: -1.0 / (float(record["sparsity"]) + 2.0),
        )
        noisy_slope_path = output_dir / "noisy_slopes.csv"
        noisy_slopes.to_csv(noisy_slope_path, index=False)
        paths["noisy_slopes"] = noisy_slope_path
        paths["noisy_plot"] = plot_noisy_scaling(noisy, output_dir / "noisy_scaling.png")
    else:
        noisy = pd.DataFrame()

    coverage_records = []
    studies = []
    if not endpoints.empty:
        studies.append(
            (
                "noiseless",
                endpoints[endpoints["algorithm"].str.startswith("sparse_ecp")],
                "within_theory_bound",
            )
        )
    if not approximate.empty:
        studies.append(("approximate_sparsity", approximate, "within_oracle_bound"))
    if not noisy.empty:
        studies.append(
            (
                "noisy",
                noisy[noisy["algorithm"] == "noisy_sparse_ecp"],
                "within_theory_bound",
            )
        )
    if not proposals.empty:
        studies.append(("proposal_complexity", proposals, "within_high_probability_bound"))
    for study, frame, indicator in studies:
        coverage_records.append(
            {
                "study": study,
                "trials": len(frame),
                "empirical_coverage": float(frame[indicator].mean()) if len(frame) else np.nan,
            }
        )
    coverage = pd.DataFrame.from_records(coverage_records)
    coverage_path = output_dir / "coverage_summary.csv"
    coverage.to_csv(coverage_path, index=False)
    paths["coverage_summary"] = coverage_path
    return paths
