from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits

from .data import PreparedData, load_prepared, robust_across_tasks
from .domains import DenseBall, DenseBox, HardThresholdedBall, SparseBall
from .hpo import make_krr_objective
from .metrics import (
    aggregate_summary,
    fit_loglog_slopes,
    paired_algorithm_summary,
    summarize_runs,
)
from .objectives import make_sparse_center, make_sparse_objective
from .optimizers import make_optimizer
from .plotting import plot_fraction, plot_hpo_mse, plot_regret, plot_scaling
from .spaces import ContinuousOracle

_PARALLEL_WORKER = False
_BIOLOGY_DATA: PreparedData | None = None
_BIOLOGY_SUPPORT_WEIGHTS: dict[str, float] | None = None
_MATERIALS_DATA: PreparedData | None = None
_MATERIALS_SUPPORT_WEIGHTS: dict[str, float] | None = None
_AUTO_WORKER_CAP = 4


@dataclass(frozen=True)
class _SyntheticJob:
    dimension: int
    sparsity: int
    seed: int
    objective_index: int
    objective_name: str
    support_mode: str
    algorithm_spec: str | dict[str, Any]
    radius: float
    lipschitz: float
    max_budget: int


@dataclass(frozen=True)
class _BiologyJob:
    task_id: str
    seed: int
    algorithm_spec: str | dict[str, Any]
    budget: int
    dataset_name: str


@dataclass(frozen=True)
class _MaterialsJob:
    task_id: str
    seed: int
    algorithm_spec: str | dict[str, Any]
    budget: int
    dataset_name: str


@dataclass(frozen=True)
class _ECPHPOJob:
    dataset_name: str
    data_path: str
    seed: int
    algorithm_spec: str | dict[str, Any]
    budget: int
    folds: int


def _init_parallel_worker() -> None:
    global _PARALLEL_WORKER
    _PARALLEL_WORKER = True


def _init_biology_worker(
    prepared: PreparedData,
    support_weights: dict[str, float] | None,
) -> None:
    global _BIOLOGY_DATA, _BIOLOGY_SUPPORT_WEIGHTS
    _init_parallel_worker()
    _BIOLOGY_DATA = prepared
    _BIOLOGY_SUPPORT_WEIGHTS = support_weights


def _init_materials_worker(
    prepared: PreparedData,
    support_weights: dict[str, float] | None,
) -> None:
    global _MATERIALS_DATA, _MATERIALS_SUPPORT_WEIGHTS
    _init_parallel_worker()
    _MATERIALS_DATA = prepared
    _MATERIALS_SUPPORT_WEIGHTS = support_weights


def _numeric_thread_context():
    # Each process owns one experiment run, so nested BLAS/OpenMP threads only
    # oversubscribe the machine. Serial execution retains library defaults.
    return threadpool_limits(limits=1) if _PARALLEL_WORKER else nullcontext()


def _resolve_workers(value: int | str | None, job_count: int) -> int:
    if job_count < 1:
        return 1
    if value is None:
        return 1
    if isinstance(value, bool):
        raise TypeError("workers must be a positive integer or 'auto'")
    if isinstance(value, str):
        if value.strip().lower() != "auto":
            try:
                value = int(value)
            except ValueError as error:
                raise ValueError("workers must be a positive integer or 'auto'") from error
        else:
            available = os.cpu_count() or 1
            # "auto" should keep an interactive laptop responsive. Explicit
            # worker counts remain available for dedicated compute machines.
            conservative = min(_AUTO_WORKER_CAP, max(1, available // 2))
            return min(job_count, conservative)
    workers = int(value)
    if workers < 1:
        raise ValueError("workers must be a positive integer or 'auto'")
    return min(job_count, workers)


def _run_synthetic_job(job: _SyntheticJob) -> pd.DataFrame:
    with _numeric_thread_context():
        seed_sequence = np.random.SeedSequence(
            [job.seed, job.dimension, job.sparsity, job.objective_index]
        )
        center_rng = np.random.default_rng(seed_sequence)
        center = make_sparse_center(
            job.dimension,
            job.sparsity,
            job.radius * 0.8,
            center_rng,
        )
        objective = make_sparse_objective(job.objective_name, center, job.lipschitz)

        name, domain_kind, options = _algorithm_spec(job.algorithm_spec)
        center_support = tuple(np.flatnonzero(center).tolist())
        if domain_kind == "dense":
            domain = DenseBall(job.dimension, job.radius)
        elif domain_kind == "hard_threshold":
            domain = HardThresholdedBall(job.dimension, job.sparsity, job.radius)
        else:
            domain = SparseBall(
                job.dimension,
                job.sparsity,
                job.radius,
                supports=[center_support] if job.support_mode == "known" else None,
            )
        oracle = ContinuousOracle(domain, objective, true_max=objective.maximum)
        try:
            trace = make_optimizer(name, options).run(oracle, job.max_budget, job.seed)
        except RuntimeError as error:
            raise RuntimeError(
                f"synthetic job failed: algorithm={_algorithm_label(job.algorithm_spec, name)}, "
                f"objective={job.objective_name}, support_mode={job.support_mode}, "
                f"dimension={job.dimension}, sparsity={job.sparsity}, seed={job.seed}: {error}"
            ) from error
        trace.algorithm = _algorithm_label(job.algorithm_spec, name)
        frame = trace.to_frame(true_max=objective.maximum)
        frame["experiment"] = "synthetic_scaling"
        frame["objective_name"] = job.objective_name
        frame["support_mode"] = job.support_mode
        frame["dimension"] = job.dimension
        frame["sparsity"] = job.sparsity
        frame["seed"] = job.seed
        frame["domain"] = domain_kind
        return frame


def _run_biology_job(job: _BiologyJob) -> pd.DataFrame:
    if _BIOLOGY_DATA is None:
        raise RuntimeError("biology worker was not initialized")
    with _numeric_thread_context():
        name, _, options = _algorithm_spec(job.algorithm_spec)
        oracle = _BIOLOGY_DATA.oracle(job.task_id)
        if _BIOLOGY_SUPPORT_WEIGHTS is not None:
            oracle.support_weights = _BIOLOGY_SUPPORT_WEIGHTS
        effective_budget = min(job.budget, len(oracle.values))
        trace = make_optimizer(name, options).run(oracle, effective_budget, job.seed)
        trace.algorithm = _algorithm_label(job.algorithm_spec, name)
        frame = trace.to_frame(true_max=oracle.true_max, true_min=oracle.true_min)
        frame["experiment"] = "retrospective_biology"
        frame["dataset"] = job.dataset_name
        frame["task_id"] = job.task_id
        frame["dimension"] = oracle.dimension
        frame["sparsity"] = int(max(np.count_nonzero(row) for row in oracle.features))
        frame["total_supports"] = len(np.unique(oracle.support_ids))
        frame["seed"] = job.seed
        frame["top_1pct_threshold"] = float(np.quantile(oracle.values, 0.99))
        return frame


def _run_materials_job(job: _MaterialsJob) -> pd.DataFrame:
    if _MATERIALS_DATA is None:
        raise RuntimeError("materials worker was not initialized")
    with _numeric_thread_context():
        name, _, options = _algorithm_spec(job.algorithm_spec)
        oracle = _MATERIALS_DATA.oracle(job.task_id)
        if _MATERIALS_SUPPORT_WEIGHTS is not None:
            oracle.support_weights = _MATERIALS_SUPPORT_WEIGHTS
        effective_budget = min(job.budget, len(oracle.values))
        trace = make_optimizer(name, options).run(oracle, effective_budget, job.seed)
        trace.algorithm = _algorithm_label(job.algorithm_spec, name)
        frame = trace.to_frame(true_max=oracle.true_max, true_min=oracle.true_min)
        frame["experiment"] = "retrospective_materials"
        frame["dataset"] = job.dataset_name
        frame["task_id"] = job.task_id
        frame["dimension"] = oracle.dimension
        frame["sparsity"] = int(max(np.count_nonzero(row) for row in oracle.features))
        frame["total_supports"] = len(np.unique(oracle.support_ids))
        frame["seed"] = job.seed
        frame["top_1pct_threshold"] = float(np.quantile(oracle.values, 0.99))
        return frame


def _run_ecp_hpo_job(job: _ECPHPOJob) -> pd.DataFrame:
    with _numeric_thread_context():
        objective = make_krr_objective(job.dataset_name, job.data_path, job.folds)
        domain = DenseBox(np.full(2, -1.0), np.full(2, 1.0))
        oracle = ContinuousOracle(domain, objective)
        name, _, options = _algorithm_spec(job.algorithm_spec)
        try:
            trace = make_optimizer(name, options).run(oracle, job.budget, job.seed)
        except RuntimeError as error:
            raise RuntimeError(
                f"ECP HPO job failed: dataset={job.dataset_name}, "
                f"algorithm={_algorithm_label(job.algorithm_spec, name)}, "
                f"seed={job.seed}: {error}"
            ) from error
        trace.algorithm = _algorithm_label(job.algorithm_spec, name)
        frame = trace.to_frame()
        frame["experiment"] = "ecp_hpo_control"
        frame["dataset"] = job.dataset_name
        frame["task_id"] = job.dataset_name
        frame["support_mode"] = "dense_2d"
        frame["dimension"] = 2
        frame["seed"] = job.seed
        frame["mse"] = -frame["value"]
        frame["best_mse"] = -frame["best_value"]
        return frame


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError("configuration must be a mapping")
    return config


def _algorithm_spec(spec: str | dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    if isinstance(spec, str):
        return spec, "sparse", {}
    return str(spec["name"]), str(spec.get("domain", "sparse")), dict(spec.get("options", {}))


def _algorithm_label(spec: str | dict[str, Any], default: str) -> str:
    if isinstance(spec, dict) and "label" in spec:
        return str(spec["label"])
    return default


def _write_bundle(
    results: pd.DataFrame,
    output_dir: Path,
    *,
    budgets: list[int] | None = None,
    pair_reference: str = "sparse_ecp",
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "results": output_dir / "trajectories.csv",
        "runs": output_dir / "run_summary.csv",
        "aggregate": output_dir / "aggregate_summary.csv",
        "paired": output_dir / "paired_algorithm_summary.csv",
    }
    results.to_csv(paths["results"], index=False)
    run_summary = summarize_runs(results)
    run_summary.to_csv(paths["runs"], index=False)
    aggregate_summary(run_summary).to_csv(paths["aggregate"], index=False)
    paired_algorithm_summary(run_summary, reference=pair_reference).to_csv(
        paths["paired"], index=False
    )
    if budgets is not None and {"sparsity", "simple_regret"}.issubset(results.columns):
        slopes = fit_loglog_slopes(results, budgets)
        slope_path = output_dir / "scaling_slopes.csv"
        slopes.to_csv(slope_path, index=False)
        paths["slopes"] = slope_path
        if not slopes.empty:
            paths["scaling_plot"] = plot_scaling(slopes, output_dir / "scaling_slopes.png")
    return paths


def run_synthetic(
    config_path: str | Path,
    *,
    workers: int | str | None = None,
) -> dict[str, Path]:
    config = load_config(config_path)
    output_dir = Path(config.get("output_dir", "results/synthetic"))
    dimensions = [int(value) for value in config.get("dimensions", [12])]
    sparsities = [int(value) for value in config.get("sparsities", [1, 2, 3])]
    budgets = [int(value) for value in config.get("budgets", [25, 50, 100, 200])]
    max_budget = max(budgets)
    seeds = [
        int(value)
        for value in config.get("seeds", list(range(int(config.get("seed_count", 10)))))
    ]
    radius = float(config.get("radius", 1.0))
    lipschitz = float(config.get("lipschitz", 1.0))
    objective_names = list(config.get("objectives", ["cone"]))
    support_modes = [str(value) for value in config.get("support_modes", ["unknown"])]
    if not set(support_modes).issubset({"unknown", "known"}):
        raise ValueError("support_modes entries must be 'unknown' or 'known'")
    algorithms = list(config.get("algorithms", ["sparse_ecp", "support_random"]))
    jobs = []
    for dimension in dimensions:
        for sparsity in sparsities:
            if sparsity > dimension:
                continue
            for seed in seeds:
                for objective_index, objective_name in enumerate(objective_names):
                    for support_mode in support_modes:
                        for algorithm_spec in algorithms:
                            jobs.append(
                                _SyntheticJob(
                                    dimension=dimension,
                                    sparsity=sparsity,
                                    seed=seed,
                                    objective_index=objective_index,
                                    objective_name=objective_name,
                                    support_mode=support_mode,
                                    algorithm_spec=algorithm_spec,
                                    radius=radius,
                                    lipschitz=lipschitz,
                                    max_budget=max_budget,
                                )
                            )
    worker_count = _resolve_workers(
        config.get("workers", 1) if workers is None else workers,
        len(jobs),
    )
    if worker_count == 1:
        frames = [_run_synthetic_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_init_parallel_worker,
        ) as executor:
            frames = list(executor.map(_run_synthetic_job, jobs, chunksize=1))
    results = pd.concat(frames, ignore_index=True)
    paths = _write_bundle(
        results,
        output_dir,
        budgets=budgets,
        pair_reference=str(config.get("pair_reference", "sparse_ecp")),
    )
    representative = results[
        (results["objective_name"] == objective_names[0])
        & (results["support_mode"] == support_modes[0])
        & (results["dimension"] == max(dimensions))
        & (results["sparsity"] == max(s for s in sparsities if s <= max(dimensions)))
    ]
    paths["regret_plot"] = plot_regret(
        representative,
        output_dir / "regret.png",
        (
            f"Synthetic {objective_names[0]} "
            f"(d={max(dimensions)}, s={int(representative['sparsity'].iloc[0])})"
        ),
    )
    return paths


def run_biology(
    config_path: str | Path,
    *,
    workers: int | str | None = None,
) -> dict[str, Path]:
    config = load_config(config_path)
    prepared_path = Path(config["prepared_data"])
    prepared = load_prepared(prepared_path)
    dataset_name = str(config.get("dataset", prepared_path.stem))
    output_dir = Path(config.get("output_dir", f"results/{dataset_name}"))
    budget = int(config.get("budget", 200))
    seeds = [int(value) for value in config.get("seeds", list(range(10)))]
    algorithms = list(
        config.get(
            "algorithms",
            ["sparse_ecp", "support_random", "random", "space_filling", "gp_ucb"],
        )
    )
    requested_tasks = config.get("task_ids", "all")
    task_ids = prepared.task_ids if requested_tasks == "all" else [str(value) for value in requested_tasks]
    if config.get("robust_across_tasks", False):
        prepared = robust_across_tasks(prepared, task_ids)
        task_ids = prepared.task_ids
    max_tasks = config.get("max_tasks")
    if max_tasks is not None:
        task_ids = task_ids[: int(max_tasks)]
    weights_path = config.get("support_weights")
    support_weights = None
    if weights_path:
        weights = pd.read_csv(weights_path)
        if not {"support_id", "weight"}.issubset(weights.columns):
            raise ValueError("support weight CSV needs support_id and weight columns")
        support_weights = dict(zip(weights["support_id"].astype(str), weights["weight"]))

    jobs = []
    for task_id in task_ids:
        for seed in seeds:
            for algorithm_spec in algorithms:
                jobs.append(
                    _BiologyJob(
                        task_id=task_id,
                        seed=seed,
                        algorithm_spec=algorithm_spec,
                        budget=budget,
                        dataset_name=dataset_name,
                    )
                )
    if not jobs:
        raise ValueError("no biology tasks were run")
    worker_count = _resolve_workers(
        config.get("workers", 1) if workers is None else workers,
        len(jobs),
    )
    if worker_count == 1:
        global _BIOLOGY_DATA, _BIOLOGY_SUPPORT_WEIGHTS
        _BIOLOGY_DATA = prepared
        _BIOLOGY_SUPPORT_WEIGHTS = support_weights
        frames = [_run_biology_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_init_biology_worker,
            initargs=(prepared, support_weights),
        ) as executor:
            frames = list(executor.map(_run_biology_job, jobs, chunksize=1))
    results = pd.concat(frames, ignore_index=True)
    paths = _write_bundle(results, output_dir)
    paths["fraction_plot"] = plot_fraction(
        results,
        output_dir / "fraction_of_optimum.png",
        f"{dataset_name}: retrospective query efficiency",
    )
    return paths


def run_materials(
    config_path: str | Path,
    *,
    workers: int | str | None = None,
) -> dict[str, Path]:
    """Run finite-table materials or process-optimization benchmarks."""
    config = load_config(config_path)
    prepared_path = Path(config["prepared_data"])
    prepared = load_prepared(prepared_path)
    dataset_name = str(config.get("dataset", prepared_path.stem))
    output_dir = Path(config.get("output_dir", f"results/{dataset_name}"))
    budget = int(config.get("budget", 200))
    seeds = [int(value) for value in config.get("seeds", list(range(10)))]
    algorithms = list(
        config.get(
            "algorithms",
            ["sparse_ecp", "support_random", "random", "space_filling", "gp_ucb"],
        )
    )
    requested_tasks = config.get("task_ids", "all")
    task_ids = prepared.task_ids if requested_tasks == "all" else [str(value) for value in requested_tasks]
    max_tasks = config.get("max_tasks")
    if max_tasks is not None:
        task_ids = task_ids[: int(max_tasks)]
    weights_path = config.get("support_weights")
    support_weights = None
    if weights_path:
        weights = pd.read_csv(weights_path)
        if not {"support_id", "weight"}.issubset(weights.columns):
            raise ValueError("support weight CSV needs support_id and weight columns")
        support_weights = dict(zip(weights["support_id"].astype(str), weights["weight"]))

    jobs = [
        _MaterialsJob(task_id, seed, algorithm_spec, budget, dataset_name)
        for task_id in task_ids
        for seed in seeds
        for algorithm_spec in algorithms
    ]
    if not jobs:
        raise ValueError("no materials tasks were run")
    worker_count = _resolve_workers(
        config.get("workers", 1) if workers is None else workers,
        len(jobs),
    )
    if worker_count == 1:
        global _MATERIALS_DATA, _MATERIALS_SUPPORT_WEIGHTS
        _MATERIALS_DATA = prepared
        _MATERIALS_SUPPORT_WEIGHTS = support_weights
        frames = [_run_materials_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_init_materials_worker,
            initargs=(prepared, support_weights),
        ) as executor:
            frames = list(executor.map(_run_materials_job, jobs, chunksize=1))
    results = pd.concat(frames, ignore_index=True)
    paths = _write_bundle(results, output_dir)
    paths["fraction_plot"] = plot_fraction(
        results,
        output_dir / "fraction_of_optimum.png",
        f"{dataset_name}: retrospective query efficiency",
    )
    return paths


def run_ecp_hpo(
    config_path: str | Path,
    *,
    workers: int | str | None = None,
) -> dict[str, Path]:
    """Run the ECP paper's defensible UCI kernel-ridge HPO controls.

    These are dense two-parameter optimization problems. They provide a direct
    comparison to ECP, but they are intentionally kept separate from the sparse
    theorem-validation and application evidence.
    """
    config = load_config(config_path)
    output_dir = Path(config.get("output_dir", "results/ecp_uci_hpo"))
    budget = int(config.get("budget", 50))
    folds = int(config.get("folds", 3))
    seeds = [
        int(value)
        for value in config.get("seeds", list(range(int(config.get("seed_count", 100)))))
    ]
    dataset_paths = config.get("datasets")
    if not isinstance(dataset_paths, dict) or not dataset_paths:
        raise ValueError("ECP HPO config needs a non-empty datasets mapping")
    algorithms = list(
        config.get(
            "algorithms",
            ["dense_ecp", "dense_random", "space_filling", "gp_ucb"],
        )
    )
    jobs = [
        _ECPHPOJob(
            dataset_name=str(dataset_name),
            data_path=str(data_path),
            seed=seed,
            algorithm_spec=algorithm_spec,
            budget=budget,
            folds=folds,
        )
        for dataset_name, data_path in dataset_paths.items()
        for seed in seeds
        for algorithm_spec in algorithms
    ]
    worker_count = _resolve_workers(
        config.get("workers", 1) if workers is None else workers,
        len(jobs),
    )
    if worker_count == 1:
        frames = [_run_ecp_hpo_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_init_parallel_worker,
        ) as executor:
            frames = list(executor.map(_run_ecp_hpo_job, jobs, chunksize=1))
    results = pd.concat(frames, ignore_index=True)
    paths = _write_bundle(
        results,
        output_dir,
        pair_reference=str(config.get("pair_reference", "dense_ecp")),
    )
    for dataset_name in dataset_paths:
        selected = results[results["dataset"] == str(dataset_name)]
        paths[f"{dataset_name}_plot"] = plot_hpo_mse(
            selected,
            output_dir / f"{dataset_name}_best_mse.png",
            f"{dataset_name}: Gaussian KRR hyperparameter optimization",
        )
    return paths
