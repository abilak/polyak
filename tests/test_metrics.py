import numpy as np
import pandas as pd

from sparse_ecp.metrics import aggregate_summary, paired_algorithm_summary


def test_threshold_summaries_report_failures_and_use_budget_censoring():
    runs = pd.DataFrame(
        {
            "experiment": ["x", "x"],
            "algorithm": ["sparse_ecp", "sparse_ecp"],
            "seed": [0, 1],
            "budget": [10, 10],
            "queries_to_95": [2.0, np.nan],
        }
    )
    summary = aggregate_summary(runs).iloc[0]
    assert summary["queries_to_95_reach_rate"] == 0.5
    assert summary["queries_to_95_conditional_mean"] == 2.0
    assert summary["queries_to_95_mean"] == 6.5


def test_paired_query_advantage_censors_unreached_runs():
    runs = pd.DataFrame(
        {
            "experiment": ["x", "x"],
            "algorithm": ["sparse_ecp", "random"],
            "seed": [0, 0],
            "budget": [10, 10],
            "queries_to_95": [3.0, np.nan],
        }
    )
    summary = paired_algorithm_summary(runs).iloc[0]
    assert summary["queries_to_95_advantage_mean"] == 8.0
