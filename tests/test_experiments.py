from pathlib import Path

import pandas as pd
import yaml

from sparse_ecp.experiments import _resolve_workers, run_biology, run_materials, run_synthetic


def _write_yaml(path: Path, payload: dict) -> Path:
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_worker_resolution(monkeypatch):
    monkeypatch.setattr("sparse_ecp.experiments.os.cpu_count", lambda: 6)
    assert _resolve_workers(None, 20) == 1
    assert _resolve_workers("auto", 20) == 3
    assert _resolve_workers("auto", 3) == 3
    assert _resolve_workers("2", 20) == 2
    assert _resolve_workers(50, 4) == 4

    monkeypatch.setattr("sparse_ecp.experiments.os.cpu_count", lambda: 32)
    assert _resolve_workers("auto", 20) == 4


def test_synthetic_parallel_matches_serial(tmp_path):
    config = {
        "output_dir": str(tmp_path / "synthetic"),
        "dimensions": [4],
        "sparsities": [1],
        "budgets": [4, 8],
        "seeds": [0, 1],
        "objectives": ["cone"],
        "support_modes": ["unknown"],
        "algorithms": ["support_random", "space_filling"],
    }
    config_path = _write_yaml(tmp_path / "synthetic.yaml", config)
    serial_path = run_synthetic(config_path, workers=1)["results"]
    serial = pd.read_csv(serial_path)
    parallel_path = run_synthetic(config_path, workers=2)["results"]
    parallel = pd.read_csv(parallel_path)
    pd.testing.assert_frame_equal(serial, parallel)


def test_synthetic_uses_configured_pair_reference(tmp_path):
    config = {
        "output_dir": str(tmp_path / "paired"),
        "dimensions": [4],
        "sparsities": [1],
        "budgets": [4],
        "seeds": [0, 1],
        "objectives": ["cone"],
        "support_modes": ["unknown"],
        "pair_reference": "random_baseline",
        "algorithms": [
            {"name": "support_random", "label": "random_baseline"},
            "space_filling",
        ],
    }
    config_path = _write_yaml(tmp_path / "paired.yaml", config)
    paired_path = run_synthetic(config_path, workers=1)["paired"]
    paired = pd.read_csv(paired_path)
    assert set(paired["reference"]) == {"random_baseline"}
    assert set(paired["competitor"]) == {"space_filling"}


def test_biology_parallel_matches_serial(tmp_path):
    prepared_path = tmp_path / "prepared.csv"
    pd.DataFrame(
        {
            "task_id": ["a", "a", "a", "b", "b", "b"],
            "support_id": ["A|B", "A|C", "B|C", "A|B", "A|C", "B|C"],
            "objective": [1.0, 3.0, 2.0, 2.0, 1.0, 4.0],
            "x_0": [1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
            "x_1": [1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            "x_2": [0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        }
    ).to_csv(prepared_path, index=False)
    config = {
        "dataset": "tiny",
        "prepared_data": str(prepared_path),
        "output_dir": str(tmp_path / "biology"),
        "budget": 3,
        "seeds": [0, 1],
        "task_ids": "all",
        "algorithms": ["support_random", "random"],
    }
    config_path = _write_yaml(tmp_path / "biology.yaml", config)
    serial_path = run_biology(config_path, workers=1)["results"]
    serial = pd.read_csv(serial_path)
    parallel_path = run_biology(config_path, workers=2)["results"]
    parallel = pd.read_csv(parallel_path)
    pd.testing.assert_frame_equal(serial, parallel)


def test_materials_parallel_matches_serial(tmp_path):
    prepared_path = tmp_path / "materials.csv"
    pd.DataFrame(
        {
            "task_id": ["yield"] * 4,
            "support_id": ["A", "A|B", "B", "A|C"],
            "objective": [1.0, 4.0, 2.0, 3.0],
            "x_0": [1.0, 0.5, 0.0, 0.5],
            "x_1": [0.0, 0.5, 1.0, 0.0],
            "x_2": [0.0, 0.0, 0.0, 0.5],
        }
    ).to_csv(prepared_path, index=False)
    config = {
        "dataset": "tiny_materials",
        "prepared_data": str(prepared_path),
        "output_dir": str(tmp_path / "materials"),
        "budget": 4,
        "seeds": [0, 1],
        "algorithms": ["support_random", "random"],
    }
    config_path = _write_yaml(tmp_path / "materials.yaml", config)
    serial_path = run_materials(config_path, workers=1)["results"]
    serial = pd.read_csv(serial_path)
    parallel_path = run_materials(config_path, workers=2)["results"]
    parallel = pd.read_csv(parallel_path)
    pd.testing.assert_frame_equal(serial, parallel)
    assert set(parallel["experiment"]) == {"retrospective_materials"}
