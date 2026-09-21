from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def make_sparse_center(
    dimension: int,
    sparsity: int,
    radius: float,
    rng: np.random.Generator,
) -> FloatArray:
    support = np.sort(rng.choice(dimension, sparsity, replace=False))
    direction = rng.normal(size=sparsity)
    direction /= np.linalg.norm(direction)
    x = np.zeros(dimension, dtype=float)
    x[support] = radius * direction
    return x


@dataclass(frozen=True)
class SparseCone:
    """A worst-case-style Lipschitz cone with a hidden sparse maximizer."""

    center: FloatArray
    lipschitz: float = 1.0

    def __post_init__(self) -> None:
        if self.lipschitz <= 0:
            raise ValueError("lipschitz must be positive")

    def __call__(self, x: FloatArray) -> float:
        return -self.lipschitz * float(np.linalg.norm(np.asarray(x) - self.center))

    @property
    def maximum(self) -> float:
        return 0.0


@dataclass(frozen=True)
class SparseRastrigin:
    """Multimodal objective with a sparse translated global maximizer.

    It is used as a practical stress test, not for the minimax-rate theorem.
    """

    center: FloatArray
    scale: float = 0.25
    off_support_penalty: float = 1.0

    def __call__(self, x: FloatArray) -> float:
        z = (np.asarray(x, dtype=float) - self.center) / self.scale
        value = 10.0 * z.size + np.sum(z * z - 10.0 * np.cos(2.0 * np.pi * z))
        return -float(value)

    @property
    def maximum(self) -> float:
        return 0.0


@dataclass(frozen=True)
class SparseQuadratic:
    """Smooth translated bowl, negated so the sparse center is the maximizer."""

    center: FloatArray

    def __call__(self, x: FloatArray) -> float:
        difference = np.asarray(x, dtype=float) - self.center
        return -float(np.dot(difference, difference))

    @property
    def maximum(self) -> float:
        return 0.0


@dataclass(frozen=True)
class SparseAckley:
    center: FloatArray
    scale: float = 0.35

    def __call__(self, x: FloatArray) -> float:
        z = (np.asarray(x, dtype=float) - self.center) / self.scale
        mean_square = float(np.mean(z * z))
        mean_cosine = float(np.mean(np.cos(2.0 * np.pi * z)))
        loss = -20.0 * np.exp(-0.2 * np.sqrt(mean_square)) - np.exp(mean_cosine) + 20.0 + np.e
        return -float(loss)

    @property
    def maximum(self) -> float:
        return 0.0


@dataclass(frozen=True)
class SparseGriewank:
    center: FloatArray
    scale: float = 0.08

    def __call__(self, x: FloatArray) -> float:
        z = (np.asarray(x, dtype=float) - self.center) / self.scale
        indices = np.sqrt(np.arange(1, z.size + 1, dtype=float))
        loss = np.sum(z * z) / 4000.0 - np.prod(np.cos(z / indices)) + 1.0
        return -float(loss)

    @property
    def maximum(self) -> float:
        return 0.0


@dataclass(frozen=True)
class SparseLevy:
    center: FloatArray
    scale: float = 0.25

    def __call__(self, x: FloatArray) -> float:
        z = (np.asarray(x, dtype=float) - self.center) / self.scale
        w = 1.0 + z / 4.0
        first = np.sin(np.pi * w[0]) ** 2
        middle = np.sum((w[:-1] - 1.0) ** 2 * (1.0 + 10.0 * np.sin(np.pi * w[:-1] + 1.0) ** 2))
        last = (w[-1] - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * w[-1]) ** 2)
        return -float(first + middle + last)

    @property
    def maximum(self) -> float:
        return 0.0


@dataclass(frozen=True)
class SparseSalomon:
    center: FloatArray
    scale: float = 0.2

    def __call__(self, x: FloatArray) -> float:
        radius = float(np.linalg.norm((np.asarray(x, dtype=float) - self.center) / self.scale))
        loss = 1.0 - np.cos(2.0 * np.pi * radius) + 0.1 * radius
        return -float(loss)

    @property
    def maximum(self) -> float:
        return 0.0


def make_sparse_objective(name: str, center: FloatArray, lipschitz: float = 1.0):
    """Build one of the preregistered translated sparse benchmark landscapes."""
    factories = {
        "cone": lambda: SparseCone(center, lipschitz=lipschitz),
        "quadratic": lambda: SparseQuadratic(center),
        "rastrigin": lambda: SparseRastrigin(center),
        "ackley": lambda: SparseAckley(center),
        "griewank": lambda: SparseGriewank(center),
        "levy": lambda: SparseLevy(center),
        "salomon": lambda: SparseSalomon(center),
    }
    try:
        return factories[name.lower()]()
    except KeyError as error:
        raise ValueError(f"unknown synthetic objective: {name}") from error


@dataclass(frozen=True)
class MaxOfCones:
    centers: FloatArray
    heights: FloatArray
    lipschitz: float = 1.0

    def __call__(self, x: FloatArray) -> float:
        distances = np.linalg.norm(self.centers - np.asarray(x), axis=1)
        return float(np.max(self.heights - self.lipschitz * distances))

    @property
    def maximum(self) -> float:
        return float(np.max(self.heights))
