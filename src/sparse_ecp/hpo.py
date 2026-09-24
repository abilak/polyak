from __future__ import annotations

import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ECPDataset:
    name: str
    display_name: str
    features: FloatArray
    targets: FloatArray


_DISPLAY_NAMES = {
    "auto_mpg": "Auto-MPG",
    "breast_cancer_wisconsin": "Breast Cancer Wisconsin (Diagnostic)",
    "concrete_slump": "Concrete Slump Test",
    "yacht_hydrodynamics": "Yacht Hydrodynamics",
}

_EXPECTED_SHAPES = {
    "auto_mpg": (392, 7),
    "breast_cancer_wisconsin": (569, 30),
    "concrete_slump": (103, 7),
    "yacht_hydrodynamics": (308, 6),
}


def _read_yacht(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".zip":
        with (
            zipfile.ZipFile(path) as archive,
            archive.open("yacht_hydrodynamics.data") as handle,
        ):
            return pd.read_csv(handle, sep=r"\s+", header=None)
    return pd.read_csv(path, sep=r"\s+", header=None)


@lru_cache(maxsize=16)
def load_ecp_dataset(name: str, path: str | Path) -> ECPDataset:
    """Load one of the defensible UCI controls used by the ECP paper.

    The Concrete Slump loader deliberately uses only the seven declared input
    variables. The public ECP reference code also includes the other two
    outputs as predictors, which leaks outcome information into the model.
    """
    normalized = str(name).strip().lower()
    source = Path(path)
    if normalized not in _DISPLAY_NAMES:
        raise ValueError(f"unknown ECP HPO dataset: {name!r}")
    if not source.is_file():
        raise FileNotFoundError(source)

    if normalized == "auto_mpg":
        columns = [
            "mpg",
            "cylinders",
            "displacement",
            "horsepower",
            "weight",
            "acceleration",
            "model_year",
            "origin",
            "car_name",
        ]
        frame = pd.read_csv(source, sep=r"\s+", names=columns, na_values="?").dropna()
        features = frame[
            [
                "cylinders",
                "displacement",
                "horsepower",
                "weight",
                "acceleration",
                "model_year",
                "origin",
            ]
        ].to_numpy(dtype=float)
        targets = frame["mpg"].to_numpy(dtype=float)
    elif normalized == "breast_cancer_wisconsin":
        frame = pd.read_csv(source, header=None)
        features = frame.iloc[:, 2:].to_numpy(dtype=float)
        targets = frame.iloc[:, 1].map({"M": 1.0, "B": 0.0}).to_numpy(dtype=float)
    elif normalized == "concrete_slump":
        frame = pd.read_csv(source)
        input_columns = [
            "Cement",
            "Slag",
            "Fly ash",
            "Water",
            "SP",
            "Coarse Aggr.",
            "Fine Aggr.",
        ]
        features = frame[input_columns].to_numpy(dtype=float)
        targets = frame["SLUMP(cm)"].to_numpy(dtype=float)
    else:
        frame = _read_yacht(source)
        features = frame.iloc[:, :6].to_numpy(dtype=float)
        targets = frame.iloc[:, 6].to_numpy(dtype=float)

    expected = _EXPECTED_SHAPES[normalized]
    if features.shape != expected or targets.shape != (expected[0],):
        raise ValueError(
            f"unexpected shape for {normalized}: X={features.shape}, y={targets.shape}; "
            f"expected X={expected}, y={(expected[0],)}"
        )
    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise ValueError(f"{normalized} contains non-finite values after preprocessing")
    features.setflags(write=False)
    targets.setflags(write=False)
    return ECPDataset(normalized, _DISPLAY_NAMES[normalized], features, targets)


class KRRHyperparameterObjective:
    """Negative fixed-fold CV MSE for Gaussian kernel ridge regression.

    The two coordinates are ``log(alpha)`` and ``log(sigma)``. Feature
    standardization is fitted separately inside every training fold, matching
    the protocol in the ECP paper while avoiding validation leakage.
    """

    def __init__(self, dataset: ECPDataset, folds: int = 3) -> None:
        if folds < 2 or folds > len(dataset.targets):
            raise ValueError("folds must lie between 2 and the number of rows")
        self.dataset = dataset
        splitter = KFold(n_splits=folds, shuffle=False)
        self._folds: list[tuple[FloatArray, FloatArray, FloatArray, FloatArray]] = []
        for train_indices, test_indices in splitter.split(dataset.features):
            scaler = StandardScaler()
            x_train = scaler.fit_transform(dataset.features[train_indices])
            x_test = scaler.transform(dataset.features[test_indices])
            self._folds.append(
                (
                    np.asarray(x_train, dtype=float),
                    dataset.targets[train_indices],
                    np.asarray(x_test, dtype=float),
                    dataset.targets[test_indices],
                )
            )

    def __call__(self, x: FloatArray) -> float:
        parameters = np.asarray(x, dtype=float)
        if parameters.shape != (2,) or not np.isfinite(parameters).all():
            raise ValueError("KRR HPO expects two finite log-parameters")
        alpha = float(np.exp(parameters[0]))
        sigma = float(np.exp(parameters[1]))
        gamma = 1.0 / (2.0 * sigma * sigma)
        scores = []
        for x_train, y_train, x_test, y_test in self._folds:
            model = KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma)
            model.fit(x_train, y_train)
            scores.append(mean_squared_error(y_test, model.predict(x_test)))
        return -float(np.mean(scores))


@lru_cache(maxsize=16)
def make_krr_objective(
    dataset_name: str,
    path: str | Path,
    folds: int = 3,
) -> KRRHyperparameterObjective:
    return KRRHyperparameterObjective(load_ecp_dataset(dataset_name, path), folds=folds)
