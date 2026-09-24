from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from math import comb

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _uniform_ball(rng: np.random.Generator, dimension: int, radius: float) -> FloatArray:
    """Sample exactly uniformly from a Euclidean ball."""
    direction = rng.normal(size=dimension)
    norm = np.linalg.norm(direction)
    while norm == 0.0:
        direction = rng.normal(size=dimension)
        norm = np.linalg.norm(direction)
    radial = radius * rng.random() ** (1.0 / dimension)
    return np.asarray(direction * (radial / norm), dtype=float)


@dataclass(frozen=True)
class DenseBox:
    """Uniform sampler on an axis-aligned box."""

    lower: FloatArray
    upper: FloatArray

    def __post_init__(self) -> None:
        lower = np.asarray(self.lower, dtype=float)
        upper = np.asarray(self.upper, dtype=float)
        if lower.ndim != 1 or upper.shape != lower.shape or lower.size < 1:
            raise ValueError("lower and upper must be non-empty vectors of equal shape")
        if not np.isfinite(lower).all() or not np.isfinite(upper).all():
            raise ValueError("box bounds must be finite")
        if np.any(upper <= lower):
            raise ValueError("each upper bound must exceed its lower bound")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @property
    def dimension(self) -> int:
        return int(self.lower.size)

    def sample(self, rng: np.random.Generator) -> FloatArray:
        return np.asarray(rng.uniform(self.lower, self.upper), dtype=float)

    @property
    def diameter(self) -> float:
        return float(np.linalg.norm(self.upper - self.lower))


@dataclass(frozen=True)
class DenseBall:
    dimension: int
    radius: float = 1.0

    def __post_init__(self) -> None:
        if self.dimension < 1 or self.radius <= 0:
            raise ValueError("dimension and radius must be positive")

    def sample(self, rng: np.random.Generator) -> FloatArray:
        return _uniform_ball(rng, self.dimension, self.radius)

    @property
    def diameter(self) -> float:
        return 2.0 * self.radius


@dataclass(frozen=True)
class HardThresholdedBall:
    """Original heuristic: threshold a dense uniform draw to its largest entries.

    This is a useful ablation, but it is not the support-first measure used by
    the corrected minimax proof.
    """

    dimension: int
    sparsity: int
    radius: float = 1.0

    def __post_init__(self) -> None:
        if not 1 <= self.sparsity <= self.dimension:
            raise ValueError("sparsity must lie in [1, dimension]")
        if self.radius <= 0:
            raise ValueError("radius must be positive")

    def sample(self, rng: np.random.Generator) -> FloatArray:
        dense = _uniform_ball(rng, self.dimension, self.radius)
        keep = np.argpartition(np.abs(dense), -self.sparsity)[-self.sparsity :]
        sparse = np.zeros(self.dimension, dtype=float)
        sparse[keep] = dense[keep]
        return sparse

    @property
    def diameter(self) -> float:
        return 2.0 * self.radius


@dataclass
class SparseBall:
    """Support-first sampler for B_2^d(radius) intersected with B_0(s).

    Selecting a size-s support first and then sampling uniformly in its
    s-dimensional ball is the proposal measure used by the corrected theorem.
    It is deliberately different from hard-thresholding a dense draw.
    """

    dimension: int
    sparsity: int
    radius: float = 1.0
    supports: Sequence[Sequence[int]] | None = None
    support_weights: Mapping[tuple[int, ...], float] | None = None
    size_weights: Mapping[int, float] | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.sparsity <= self.dimension:
            raise ValueError("sparsity must lie in [1, dimension]")
        if self.radius <= 0:
            raise ValueError("radius must be positive")
        if self.supports is not None:
            normalized = [tuple(sorted(int(j) for j in support)) for support in self.supports]
            if any(len(support) != self.sparsity for support in normalized):
                raise ValueError("every explicit support must have exactly sparsity entries")
            if any(min(support) < 0 or max(support) >= self.dimension for support in normalized):
                raise ValueError("support coordinate out of range")
            if len(set(normalized)) != len(normalized):
                raise ValueError("supports must be unique")
            self.supports = normalized
            if self.size_weights is not None:
                raise ValueError("size_weights cannot be combined with explicit supports")
        if self.support_weights is not None:
            if self.supports is None:
                raise ValueError("explicit supports are required when support_weights are supplied")
            missing = set(self.supports) - set(self.support_weights)
            if missing:
                raise ValueError(f"support_weights missing {len(missing)} supports")
            weights = np.array([self.support_weights[s] for s in self.supports], dtype=float)
            if np.any(weights < 0) or not np.isfinite(weights).all() or weights.sum() <= 0:
                raise ValueError("support weights must be finite, nonnegative, and not all zero")
        if self.size_weights is not None:
            sizes = np.array(sorted(int(size) for size in self.size_weights), dtype=int)
            if sizes.size == 0 or sizes[0] < 1 or sizes[-1] > self.sparsity:
                raise ValueError("size_weights keys must lie in [1, sparsity]")
            weights = np.array([self.size_weights[int(size)] for size in sizes], dtype=float)
            if np.any(weights < 0) or not np.isfinite(weights).all() or weights.sum() <= 0:
                raise ValueError("size weights must be finite, nonnegative, and not all zero")

    @property
    def support_count(self) -> int:
        if self.supports is not None:
            return len(self.supports)
        if self.size_weights is None:
            return comb(self.dimension, self.sparsity)
        return sum(
            comb(self.dimension, int(size))
            for size, weight in self.size_weights.items()
            if float(weight) > 0
        )

    @property
    def diameter(self) -> float:
        return 2.0 * self.radius

    def enumerate_supports(self, limit: int = 1_000_000) -> list[tuple[int, ...]]:
        if self.supports is not None:
            return list(self.supports)
        if self.support_count > limit:
            raise ValueError(f"refusing to enumerate {self.support_count:,} supports")
        if self.size_weights is None:
            return list(combinations(range(self.dimension), self.sparsity))
        return [
            support
            for size, weight in sorted(self.size_weights.items())
            if float(weight) > 0
            for support in combinations(range(self.dimension), int(size))
        ]

    def sample_support(self, rng: np.random.Generator) -> tuple[int, ...]:
        if self.supports is None:
            size = self.sparsity
            if self.size_weights is not None:
                sizes = np.array(sorted(self.size_weights), dtype=int)
                probabilities = np.array(
                    [self.size_weights[int(value)] for value in sizes], dtype=float
                )
                probabilities /= probabilities.sum()
                size = int(rng.choice(sizes, p=probabilities))
            return tuple(sorted(rng.choice(self.dimension, size, replace=False).tolist()))
        if self.support_weights is None:
            return self.supports[int(rng.integers(len(self.supports)))]
        probabilities = np.array([self.support_weights[s] for s in self.supports], dtype=float)
        probabilities /= probabilities.sum()
        return self.supports[int(rng.choice(len(self.supports), p=probabilities))]

    def sample(self, rng: np.random.Generator) -> FloatArray:
        support = self.sample_support(rng)
        x = np.zeros(self.dimension, dtype=float)
        x[np.asarray(support)] = _uniform_ball(rng, len(support), self.radius)
        return x

    def sample_many(self, rng: np.random.Generator, count: int) -> FloatArray:
        return np.vstack([self.sample(rng) for _ in range(count)])


@dataclass(frozen=True)
class SparseBox:
    """Support-first sampler for the centered sparse cube in the paper.

    ``sparsity`` is the largest allowed support size.  Without ``size_weights``
    the sampler uses exactly that size.  Supplying weights implements the
    proposal mixture Q_w from Section 2.3 of the Sparse ECP paper.
    """

    dimension: int
    sparsity: int
    radius: float = 1.0
    size_weights: Mapping[int, float] | None = None
    supports: Sequence[Sequence[int]] | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.sparsity <= self.dimension:
            raise ValueError("sparsity must lie in [1, dimension]")
        if self.radius <= 0:
            raise ValueError("radius must be positive")
        if self.supports is not None:
            normalized = tuple(tuple(sorted(int(j) for j in support)) for support in self.supports)
            if not normalized or any(len(support) < 1 for support in normalized):
                raise ValueError("explicit supports must be non-empty")
            if any(len(support) > self.sparsity for support in normalized):
                raise ValueError("explicit support exceeds the sparsity upper bound")
            if any(min(support) < 0 or max(support) >= self.dimension for support in normalized):
                raise ValueError("support coordinate out of range")
            if len(set(normalized)) != len(normalized):
                raise ValueError("supports must be unique")
            object.__setattr__(self, "supports", normalized)
            if self.size_weights is not None:
                raise ValueError("size_weights cannot be combined with explicit supports")
        if self.size_weights is not None:
            sizes = sorted(int(size) for size in self.size_weights)
            if not sizes or sizes[0] < 1 or sizes[-1] > self.sparsity:
                raise ValueError("size_weights keys must lie in [1, sparsity]")
            weights = np.array([self.size_weights[size] for size in sizes], dtype=float)
            if np.any(weights < 0) or not np.isfinite(weights).all() or weights.sum() <= 0:
                raise ValueError("size weights must be finite, nonnegative, and not all zero")

    def sample_support(self, rng: np.random.Generator) -> tuple[int, ...]:
        if self.supports is not None:
            return self.supports[int(rng.integers(len(self.supports)))]
        size = self.sparsity
        if self.size_weights is not None:
            sizes = np.array(sorted(self.size_weights), dtype=int)
            probabilities = np.array([self.size_weights[int(size)] for size in sizes], dtype=float)
            probabilities /= probabilities.sum()
            size = int(rng.choice(sizes, p=probabilities))
        return tuple(sorted(rng.choice(self.dimension, size, replace=False).tolist()))

    def sample(self, rng: np.random.Generator) -> FloatArray:
        support = self.sample_support(rng)
        x = np.zeros(self.dimension, dtype=float)
        x[np.asarray(support)] = rng.uniform(-self.radius, self.radius, size=len(support))
        return x

    @property
    def diameter(self) -> float:
        return float(2.0 * self.radius * np.sqrt(self.sparsity))

    @property
    def support_count(self) -> int:
        if self.supports is not None:
            return len(self.supports)
        if self.size_weights is None:
            return comb(self.dimension, self.sparsity)
        return sum(
            comb(self.dimension, int(size))
            for size, weight in self.size_weights.items()
            if float(weight) > 0
        )

    def sample_many(self, rng: np.random.Generator, count: int) -> FloatArray:
        return np.vstack([self.sample(rng) for _ in range(count)])
