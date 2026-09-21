from __future__ import annotations

import numpy as np
import pandas as pd


def first_evaluation_at(frame: pd.DataFrame, fraction: float) -> float:
    reached = frame.loc[frame["fraction_of_range"] >= fraction, "evaluation"]
    return float(reached.iloc[0]) if len(reached) else np.nan


def summarize_runs(results: pd.DataFrame) -> pd.DataFrame:
    keys = [
        column
        for column in ["experiment", "dataset", "objective_name", "support_mode", "dimension", "sparsity", "task_id", "algorithm", "seed"]
        if column in results
    ]
    records = []
    for labels, frame in results.groupby(keys, dropna=False):
        if not isinstance(labels, tuple):
            labels = (labels,)
        record = dict(zip(keys, labels))
        last = frame.sort_values("evaluation").iloc[-1]
        record.update(
            {
                "budget": int(last["evaluation"]),
                "final_best": float(last["best_value"]),
                "final_regret": float(last.get("simple_regret", np.nan)),
                "total_proposals": int(frame["proposals"].sum()),
                "proposal_overhead": float(frame["proposals"].sum() / len(frame)),
                "unique_supports": int(frame["support_id"].nunique()),
            }
        )
        if "total_supports" in frame:
            record["support_coverage"] = float(
                frame["support_id"].nunique() / int(frame["total_supports"].iloc[0])
            )
        if "fraction_of_range" in frame:
            record["queries_to_90"] = first_evaluation_at(frame, 0.90)
            record["queries_to_95"] = first_evaluation_at(frame, 0.95)
            record["queries_to_99"] = first_evaluation_at(frame, 0.99)
            record["auc_fraction"] = float(frame["fraction_of_range"].mean())
        if "top_1pct_threshold" in frame:
            reached = frame.loc[frame["best_value"] >= frame["top_1pct_threshold"], "evaluation"]
            record["queries_to_top_1pct"] = float(reached.iloc[0]) if len(reached) else np.nan
        records.append(record)
    return pd.DataFrame.from_records(records)


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    confidence: float = 0.95,
    repetitions: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(repetitions, len(values)), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(values.mean()), float(np.quantile(samples, alpha)), float(np.quantile(samples, 1 - alpha))


def aggregate_summary(run_summary: pd.DataFrame) -> pd.DataFrame:
    group_keys = [
        column
        for column in ["experiment", "dataset", "objective_name", "support_mode", "dimension", "sparsity", "algorithm"]
        if column in run_summary
    ]
    metrics = [
        column
        for column in [
            "final_regret",
            "queries_to_90",
            "queries_to_95",
            "queries_to_99",
            "queries_to_top_1pct",
            "auc_fraction",
            "proposal_overhead",
            "unique_supports",
            "support_coverage",
        ]
        if column in run_summary
    ]
    records = []
    query_metrics = {
        "queries_to_90",
        "queries_to_95",
        "queries_to_99",
        "queries_to_top_1pct",
    }
    for labels, frame in run_summary.groupby(group_keys, dropna=False):
        if not isinstance(labels, tuple):
            labels = (labels,)
        record = dict(zip(group_keys, labels))
        record["runs"] = len(frame)
        for metric in metrics:
            values = frame[metric].to_numpy(dtype=float)
            if metric in query_metrics:
                reached = np.isfinite(values)
                record[f"{metric}_reach_rate"] = float(reached.mean())
                record[f"{metric}_conditional_mean"] = (
                    float(np.mean(values[reached])) if reached.any() else np.nan
                )
                censor_at = frame["budget"].to_numpy(dtype=float) + 1.0
                values = np.where(reached, values, censor_at)
            mean, low, high = bootstrap_mean_ci(values, seed=17)
            record[f"{metric}_mean"] = mean
            record[f"{metric}_ci_low"] = low
            record[f"{metric}_ci_high"] = high
        records.append(record)
    return pd.DataFrame.from_records(records)


def paired_algorithm_summary(
    run_summary: pd.DataFrame,
    *,
    reference: str = "sparse_ecp",
) -> pd.DataFrame:
    """Paired per-seed differences; positive values favor the reference method."""
    pair_keys = [
        column
        for column in [
            "experiment",
            "dataset",
            "objective_name",
            "support_mode",
            "dimension",
            "sparsity",
            "task_id",
            "seed",
        ]
        if column in run_summary
    ]
    reference_rows = run_summary[run_summary["algorithm"] == reference]
    records = []
    lower_is_better = [
        "final_regret",
        "queries_to_90",
        "queries_to_95",
        "queries_to_99",
        "queries_to_top_1pct",
        "proposal_overhead",
    ]
    higher_is_better = ["auc_fraction"]
    for competitor in sorted(set(run_summary["algorithm"]) - {reference}):
        competitor_rows = run_summary[run_summary["algorithm"] == competitor]
        paired = reference_rows.merge(
            competitor_rows,
            on=pair_keys,
            suffixes=("_reference", "_competitor"),
        )
        grouping = [key for key in pair_keys if key not in {"task_id", "seed"}]
        groups = paired.groupby(grouping, dropna=False) if grouping else [((), paired)]
        for labels, frame in groups:
            if not isinstance(labels, tuple):
                labels = (labels,)
            record = dict(zip(grouping, labels))
            record.update({"reference": reference, "competitor": competitor, "paired_runs": len(frame)})
            for metric in lower_is_better:
                ref_column = f"{metric}_reference"
                cmp_column = f"{metric}_competitor"
                if ref_column not in frame or cmp_column not in frame:
                    continue
                reference_values = frame[ref_column].to_numpy(float)
                competitor_values = frame[cmp_column].to_numpy(float)
                if metric.startswith("queries_to_"):
                    reference_values = np.where(
                        np.isfinite(reference_values),
                        reference_values,
                        frame["budget_reference"].to_numpy(float) + 1.0,
                    )
                    competitor_values = np.where(
                        np.isfinite(competitor_values),
                        competitor_values,
                        frame["budget_competitor"].to_numpy(float) + 1.0,
                    )
                differences = competitor_values - reference_values
                mean, low, high = bootstrap_mean_ci(differences, seed=23)
                record[f"{metric}_advantage_mean"] = mean
                record[f"{metric}_advantage_ci_low"] = low
                record[f"{metric}_advantage_ci_high"] = high
            for metric in higher_is_better:
                ref_column = f"{metric}_reference"
                cmp_column = f"{metric}_competitor"
                if ref_column not in frame or cmp_column not in frame:
                    continue
                differences = frame[ref_column].to_numpy(float) - frame[cmp_column].to_numpy(float)
                mean, low, high = bootstrap_mean_ci(differences, seed=23)
                record[f"{metric}_advantage_mean"] = mean
                record[f"{metric}_advantage_ci_low"] = low
                record[f"{metric}_advantage_ci_high"] = high
            records.append(record)
    return pd.DataFrame.from_records(records)


def fit_loglog_slopes(results: pd.DataFrame, budgets: list[int]) -> pd.DataFrame:
    records = []
    group_columns = [
        column
        for column in ["objective_name", "support_mode", "dimension", "algorithm", "sparsity"]
        if column in results
    ]
    for labels, frame in results.groupby(group_columns):
        if not isinstance(labels, tuple):
            labels = (labels,)
        group = dict(zip(group_columns, labels))
        medians = []
        used_budgets = []
        for budget in budgets:
            at_budget = frame[frame["evaluation"] == budget]["simple_regret"]
            positive = at_budget[at_budget > 0]
            if len(positive):
                medians.append(float(positive.median()))
                used_budgets.append(budget)
        if len(medians) < 2:
            continue
        slope, intercept = np.polyfit(np.log(used_budgets), np.log(medians), 1)
        records.append(
            {
                **group,
                "sparsity": int(group["sparsity"]),
                "fitted_slope": float(slope),
                "intercept": float(intercept),
                "theory_slope": -1.0 / float(group["sparsity"]),
                "points": len(medians),
            }
        )
    return pd.DataFrame.from_records(records)
