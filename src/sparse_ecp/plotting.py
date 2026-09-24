from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter

matplotlib.use("Agg")


COLORS = {
    "sparse_ecp": "#0057B8",
    "structured_sparse_ecp": "#7A3E9D",
    "support_random": "#777777",
    "random": "#A0A0A0",
    "space_filling": "#E17C05",
    "gp_ucb": "#2E8B57",
    "ecp_uniform": "#B33A3A",
    "ecp_hard_threshold": "#B33A3A",
    "hard_threshold_random": "#D66A6A",
    "dense_ecp": "#5B2C83",
    "dense_random": "#999999",
}


def plot_regret(results: pd.DataFrame, output: str | Path, title: str) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for algorithm, frame in results.groupby("algorithm"):
        pivot = frame.pivot_table(index="evaluation", columns=["seed", "task_id"] if "task_id" in frame else "seed", values="simple_regret")
        x = pivot.index.to_numpy()
        median = pivot.median(axis=1).to_numpy()
        low = pivot.quantile(0.25, axis=1).to_numpy()
        high = pivot.quantile(0.75, axis=1).to_numpy()
        color = COLORS.get(str(algorithm), None)
        axis.plot(x, np.maximum(median, 1e-12), label=str(algorithm), color=color, linewidth=2)
        axis.fill_between(x, np.maximum(low, 1e-12), np.maximum(high, 1e-12), color=color, alpha=0.16)
    axis.set_yscale("log")
    axis.set_xlabel("Black-box evaluations")
    axis.set_ylabel("Simple regret (median; IQR)")
    axis.set_title(title)
    axis.grid(True, which="both", alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output, dpi=220)
    plt.close(figure)
    return output


def plot_fraction(results: pd.DataFrame, output: str | Path, title: str) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for algorithm, frame in results.groupby("algorithm"):
        grouped = frame.groupby("evaluation")["fraction_of_range"]
        x = np.array(sorted(frame["evaluation"].unique()))
        median = grouped.median().reindex(x).to_numpy()
        low = grouped.quantile(0.25).reindex(x).to_numpy()
        high = grouped.quantile(0.75).reindex(x).to_numpy()
        color = COLORS.get(str(algorithm), None)
        axis.plot(x, median, label=str(algorithm), color=color, linewidth=2)
        axis.fill_between(x, low, high, color=color, alpha=0.16)
    axis.axhline(0.95, color="black", linestyle="--", linewidth=1, alpha=0.5)
    axis.set_ylim(0, 1.02)
    axis.set_xlabel("Wet-lab-equivalent queries")
    axis.set_ylabel("Fraction of observed response range found")
    axis.set_title(title)
    axis.grid(True, alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output, dpi=220)
    plt.close(figure)
    return output


def plot_hpo_mse(results: pd.DataFrame, output: str | Path, title: str) -> Path:
    """Plot the best cross-validation MSE found so far."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for algorithm, frame in results.groupby("algorithm"):
        pivot = frame.pivot_table(index="evaluation", columns="seed", values="best_mse")
        x = pivot.index.to_numpy()
        median = pivot.median(axis=1).to_numpy()
        low = pivot.quantile(0.25, axis=1).to_numpy()
        high = pivot.quantile(0.75, axis=1).to_numpy()
        color = COLORS.get(str(algorithm), None)
        axis.plot(x, median, label=str(algorithm), color=color, linewidth=2)
        axis.fill_between(x, low, high, color=color, alpha=0.16)
    axis.set_xlabel("Hyperparameter evaluations")
    axis.set_ylabel("Best 3-fold CV MSE (median; IQR)")
    axis.set_title(title)
    axis.grid(True, alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output, dpi=220)
    plt.close(figure)
    return output


def plot_scaling(slopes: pd.DataFrame, output: str | Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.2, 5.2))
    for algorithm, frame in slopes.groupby("algorithm"):
        summary = frame.groupby("sparsity")["fitted_slope"].agg(
            median="median",
            lower=lambda values: values.quantile(0.25),
            upper=lambda values: values.quantile(0.75),
        )
        x = summary.index.to_numpy(dtype=float)
        color = COLORS.get(str(algorithm), None)
        (line,) = axis.plot(
            x,
            summary["median"],
            marker="o",
            label=str(algorithm),
            color=color,
            linewidth=1.8,
        )
        band_color = color or line.get_color()
        axis.fill_between(
            x,
            summary["lower"].to_numpy(),
            summary["upper"].to_numpy(),
            color=band_color,
            alpha=0.10,
        )
    theory = slopes[["sparsity", "theory_slope"]].drop_duplicates().sort_values("sparsity")
    axis.plot(
        theory["sparsity"],
        theory["theory_slope"],
        "k--",
        linewidth=2,
        label="minimax theory",
    )
    axis.set_xlabel("Sparsity s")
    axis.set_ylabel("Fitted log-regret / log-budget slope (median; IQR)")
    axis.set_title("Empirical scaling slopes")
    axis.grid(True, alpha=0.2)
    axis.legend(frameon=False, fontsize=8, ncol=2)
    figure.tight_layout(pad=1.2)
    figure.savefig(output, dpi=220)
    plt.close(figure)
    return output


def plot_theory_scaling(results: pd.DataFrame, output: str | Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    selected_dimension = int(results["dimension"].max())
    selected = results[
        (results["dimension"] == selected_dimension)
        & (results["proposal_mode"].isin(["known_support", "exact_unknown"]))
    ]
    for (sparsity, mode, algorithm), frame in selected.groupby(
        ["true_sparsity", "proposal_mode", "algorithm"]
    ):
        medians = frame.groupby("budget")["simple_regret"].median()
        label = f"s={sparsity}, {mode}, {algorithm.split('/')[0]}"
        axes[0].loglog(medians.index, np.maximum(medians.values, 1e-12), marker="o", label=label)
    axes[0].set_xlabel("Expensive evaluations")
    axes[0].set_ylabel("Median simple regret")
    axes[0].set_title(f"Noiseless rate (d={selected_dimension})")
    axes[0].grid(True, which="both", alpha=0.2)
    axes[0].set_xticks(sorted(selected["budget"].unique()))
    axes[0].xaxis.set_major_formatter(ScalarFormatter())
    axes[0].legend(frameon=False, fontsize=7)

    final_budget = int(results["budget"].max())
    support = results[
        (results["budget"] == final_budget)
        & (results["proposal_mode"] == "exact_unknown")
        & (results["algorithm"] == "sparse_ecp/exact_unknown")
    ]
    medians = support.groupby(["dimension", "true_sparsity"], as_index=False)[
        "simple_regret"
    ].median()
    for sparsity, frame in medians.groupby("true_sparsity"):
        axes[1].plot(frame["dimension"], frame["simple_regret"], marker="o", label=f"s={sparsity}")
    axes[1].set_xlabel("Ambient dimension d")
    axes[1].set_ylabel("Median simple regret")
    axes[1].set_title(f"Unknown-support price (n={final_budget})")
    axes[1].grid(True, alpha=0.2)
    axes[1].legend(frameon=False)
    if medians["dimension"].nunique() == 1:
        only_dimension = float(medians["dimension"].iloc[0])
        axes[1].set_xlim(only_dimension - 0.5, only_dimension + 0.5)
    figure.tight_layout()
    figure.savefig(output, dpi=220)
    plt.close(figure)
    return output


def plot_filter_calibration(calibration: pd.DataFrame, output: str | Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.5, 4.6))
    x = np.arange(len(calibration))
    axis.plot(x, calibration["empirical_success"], marker="o", label="Empirical next-query success")
    axis.plot(x, calibration["theorem_lower"], marker="s", label="Theorem 9 lower bound")
    labels = [
        f"{interval}\n(n={histories:,})"
        for interval, histories in zip(
            calibration["acceptance_mass_bin"], calibration["histories"], strict=True
        )
    ]
    axis.set_xticks(x, labels, rotation=25, ha="right")
    axis.set_xlabel("Estimated acceptance-mass bin")
    axis.set_ylabel("Target-hit probability")
    axis.set_title("ECP filtering-gain calibration")
    axis.grid(True, alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output, dpi=220)
    plt.close(figure)
    return output


def plot_proposal_complexity(results: pd.DataFrame, output: str | Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.5, 4.6))
    summary = results.groupby("budget").agg(
        actual=("actual_proposals", "median"),
        high_probability=("high_probability_upper", "median"),
    )
    axis.loglog(summary.index, summary["actual"], marker="o", label="Median observed proposals")
    axis.loglog(
        summary.index,
        summary["high_probability"],
        linestyle="--",
        label="Theorem 25 high-probability bound",
    )
    axis.set_xlabel("Accepted evaluations")
    axis.set_ylabel("Candidate proposals")
    axis.set_title("Proposal complexity")
    axis.grid(True, which="both", alpha=0.2)
    axis.set_xticks(summary.index)
    axis.xaxis.set_major_formatter(ScalarFormatter())
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output, dpi=220)
    plt.close(figure)
    return output


def plot_noisy_scaling(results: pd.DataFrame, output: str | Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    support_modes = sorted(results["known_support"].dropna().unique(), reverse=True)
    figure, axes = plt.subplots(
        1,
        len(support_modes),
        figsize=(6.6 * len(support_modes), 4.8),
        sharey=True,
        squeeze=False,
    )
    short_names = {
        "noisy_sparse_ecp": "Sparse ECP",
        "noisy_support_random": "Sparse random",
        "noisy_multiscale_ecp": "Multiscale ECP",
    }
    line_styles = {
        "noisy_sparse_ecp": ("--", "s"),
        "noisy_support_random": (":", "^"),
        "noisy_multiscale_ecp": ("-", "o"),
    }
    sparsities = sorted(results["sparsity"].unique())
    sparsity_colors = {
        sparsity: plt.get_cmap("tab10")(index)
        for index, sparsity in enumerate(sparsities)
    }
    budgets = sorted(results["oracle_budget"].unique())
    for column, known_support in enumerate(support_modes):
        axis = axes[0, column]
        selected = results[results["known_support"] == known_support]
        for (sparsity, algorithm), frame in selected.groupby(["sparsity", "algorithm"]):
            medians = frame.groupby("oracle_budget")["recommendation_regret"].median()
            label = f"s={sparsity}, {short_names.get(str(algorithm), algorithm)}"
            linestyle, marker = line_styles.get(str(algorithm), ("-", "o"))
            axis.loglog(
                medians.index,
                np.maximum(medians.values, 1e-12),
                color=sparsity_colors[sparsity],
                linestyle=linestyle,
                marker=marker,
                label=label,
            )
        axis.set_xlabel("Noisy oracle calls")
        axis.set_title("Known support" if known_support else "Unknown support")
        axis.grid(True, which="both", alpha=0.2)
        axis.set_xticks(budgets, labels=[str(value) for value in budgets], rotation=30)
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.legend(frameon=False, fontsize=7, ncol=2)
    axes[0, 0].set_ylabel("Median recommendation regret")
    figure.suptitle("Noisy sparse optimization scaling")
    figure.tight_layout()
    figure.savefig(output, dpi=220)
    plt.close(figure)
    return output
