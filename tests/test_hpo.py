from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from sparse_ecp.experiments import run_ecp_hpo
from sparse_ecp.hpo import KRRHyperparameterObjective, load_ecp_dataset


def _write_auto_mpg(path: Path) -> Path:
    rng = np.random.default_rng(7)
    rows = []
    for index in range(392):
        cylinders = 4 + 2 * (index % 3)
        displacement = 90.0 + index
        horsepower = 50.0 + index % 100
        weight = 1800.0 + 4.0 * index
        acceleration = 8.0 + (index % 20) / 2.0
        model_year = 70 + index % 13
        origin = 1 + index % 3
        mpg = 55.0 - 0.006 * weight + 0.05 * acceleration + rng.normal(0, 0.1)
        rows.append(
            f"{mpg} {cylinders} {displacement} {horsepower} {weight} "
            f"{acceleration} {model_year} {origin} car_{index}"
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_ecp_auto_mpg_objective_is_deterministic(tmp_path):
    dataset = load_ecp_dataset("auto_mpg", _write_auto_mpg(tmp_path / "auto-mpg.data"))
    assert dataset.features.shape == (392, 7)
    objective = KRRHyperparameterObjective(dataset, folds=3)
    first = objective(np.array([0.0, 0.0]))
    second = objective(np.array([0.0, 0.0]))
    assert np.isfinite(first)
    assert first == second
    assert first <= 0.0


def test_concrete_loader_excludes_other_outputs(tmp_path):
    frame = pd.DataFrame(
        {
            "No": np.arange(1, 104),
            "Cement": np.arange(103),
            "Slag": np.arange(103) + 1,
            "Fly ash": np.arange(103) + 2,
            "Water": np.arange(103) + 3,
            "SP": np.arange(103) + 4,
            "Coarse Aggr.": np.arange(103) + 5,
            "Fine Aggr.": np.arange(103) + 6,
            "SLUMP(cm)": np.arange(103) + 7,
            "FLOW(cm)": np.arange(103) + 8,
            "Compressive Strength (28-day)(Mpa)": np.arange(103) + 9,
        }
    )
    path = tmp_path / "slump_test.data"
    frame.to_csv(path, index=False)
    dataset = load_ecp_dataset("concrete_slump", path)
    assert dataset.features.shape == (103, 7)
    assert np.array_equal(dataset.features[:, -1], frame["Fine Aggr."].to_numpy())
    assert np.array_equal(dataset.targets, frame["SLUMP(cm)"].to_numpy())


def test_ecp_hpo_runner_writes_comparable_outputs(tmp_path):
    data_path = _write_auto_mpg(tmp_path / "auto-mpg.data")
    config = {
        "output_dir": str(tmp_path / "hpo"),
        "budget": 3,
        "folds": 3,
        "seeds": [0, 1],
        "pair_reference": "dense_random",
        "datasets": {"auto_mpg": str(data_path)},
        "algorithms": [
            "dense_random",
            {"name": "space_filling", "options": {"pool_size": 8}},
        ],
    }
    config_path = tmp_path / "hpo.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    outputs = run_ecp_hpo(config_path, workers=1)
    trajectories = pd.read_csv(outputs["results"])
    summary = pd.read_csv(outputs["aggregate"])
    assert set(trajectories["experiment"]) == {"ecp_hpo_control"}
    assert set(trajectories["support_mode"]) == {"dense_2d"}
    assert trajectories["best_mse"].ge(0).all()
    assert "final_mse_mean" in summary
