from __future__ import annotations

from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Candidate:
    x: FloatArray
    key: Hashable | None = None
    support: Hashable | None = None


class OracleSpace(Protocol):
    dimension: int
    intrinsic_dimension: int
    true_max: float | None
    true_min: float | None

    def sample(
        self,
        rng: np.random.Generator,
        proposal: str,
        excluded: set[Hashable],
    ) -> Candidate: ...

    def candidate_pool(
        self,
        rng: np.random.Generator,
        count: int,
        excluded: set[Hashable],
    ) -> list[Candidate]: ...

    def evaluate(self, candidate: Candidate) -> float: ...


@dataclass
class ContinuousOracle:
    domain: object
    objective: Callable[[FloatArray], float]
    true_max: float | None = None
    true_min: float | None = None

    @property
    def dimension(self) -> int:
        return int(self.domain.dimension)

    @property
    def intrinsic_dimension(self) -> int:
        """Dimension of one proposal slice, not the ambient embedding."""
        return int(getattr(self.domain, "sparsity", self.dimension))

    def sample(
        self,
        rng: np.random.Generator,
        proposal: str,
        excluded: set[Hashable],
    ) -> Candidate:
        del proposal, excluded
        x = np.asarray(self.domain.sample(rng), dtype=float)
        support = tuple(np.flatnonzero(np.abs(x) > 1e-15).tolist())
        return Candidate(x=x, support=support)

    def candidate_pool(
        self,
        rng: np.random.Generator,
        count: int,
        excluded: set[Hashable],
    ) -> list[Candidate]:
        return [self.sample(rng, "support_first", excluded) for _ in range(count)]

    def evaluate(self, candidate: Candidate) -> float:
        return float(self.objective(candidate.x))


@dataclass
class FiniteOracle:
    """A retrospective oracle that reveals only selected historical measurements."""

    features: FloatArray
    values: FloatArray
    support_ids: NDArray[np.object_]
    metadata: pd.DataFrame | None = None
    support_weights: dict[Hashable, float] | None = None

    def __post_init__(self) -> None:
        self.features = np.asarray(self.features, dtype=float)
        self.values = np.asarray(self.values, dtype=float)
        self.support_ids = np.asarray(self.support_ids, dtype=object)
        if self.features.ndim != 2:
            raise ValueError("features must be a 2D array")
        if len(self.features) != len(self.values) or len(self.values) != len(self.support_ids):
            raise ValueError("features, values, and support_ids must have equal length")
        if len(self.values) == 0 or not np.isfinite(self.values).all():
            raise ValueError("values must be non-empty and finite")
        if self.metadata is not None and len(self.metadata) != len(self.values):
            raise ValueError("metadata must align with features")
        grouped: dict[Hashable, list[int]] = {}
        for index, support in enumerate(self.support_ids.tolist()):
            grouped.setdefault(support, []).append(index)
        self._by_support = {
            support: np.asarray(indices, dtype=np.int64) for support, indices in grouped.items()
        }

    @property
    def dimension(self) -> int:
        return int(self.features.shape[1])

    @property
    def intrinsic_dimension(self) -> int:
        nonzero_counts = np.count_nonzero(np.abs(self.features) > 1e-15, axis=1)
        return max(1, int(np.max(nonzero_counts)))

    @property
    def true_max(self) -> float:
        return float(np.max(self.values))

    @property
    def true_min(self) -> float:
        return float(np.min(self.values))

    def _remaining(self, excluded: set[Hashable]) -> NDArray[np.int64]:
        if not excluded:
            return np.arange(len(self.values), dtype=np.int64)
        mask = np.ones(len(self.values), dtype=bool)
        integer_keys = [int(key) for key in excluded]
        mask[integer_keys] = False
        return np.flatnonzero(mask)

    def sample(
        self,
        rng: np.random.Generator,
        proposal: str,
        excluded: set[Hashable],
    ) -> Candidate:
        remaining = self._remaining(excluded)
        if remaining.size == 0:
            raise RuntimeError("finite oracle is exhausted")
        if proposal in {"uniform", "uniform_candidates"}:
            index = int(rng.choice(remaining))
        elif proposal in {"support_first", "structured"}:
            available_supports = []
            available_indices = []
            excluded_int = {int(key) for key in excluded}
            for support, indices in self._by_support.items():
                keep = np.array([i for i in indices if int(i) not in excluded_int], dtype=np.int64)
                if keep.size:
                    available_supports.append(support)
                    available_indices.append(keep)
            if proposal == "structured" and self.support_weights is not None:
                weights = np.array(
                    [max(0.0, self.support_weights.get(support, 0.0)) for support in available_supports],
                    dtype=float,
                )
                if weights.sum() == 0:
                    weights = np.ones(len(available_supports), dtype=float)
                weights /= weights.sum()
                support_pos = int(rng.choice(len(available_supports), p=weights))
            else:
                support_pos = int(rng.integers(len(available_supports)))
            index = int(rng.choice(available_indices[support_pos]))
        else:
            raise ValueError(f"unknown proposal: {proposal}")
        return Candidate(
            x=self.features[index].copy(),
            key=index,
            support=self.support_ids[index],
        )

    def candidate_pool(
        self,
        rng: np.random.Generator,
        count: int,
        excluded: set[Hashable],
    ) -> list[Candidate]:
        remaining = self._remaining(excluded)
        if remaining.size > count:
            remaining = rng.choice(remaining, size=count, replace=False)
        return [
            Candidate(
                x=self.features[int(i)].copy(),
                key=int(i),
                support=self.support_ids[int(i)],
            )
            for i in remaining
        ]

    def evaluate(self, candidate: Candidate) -> float:
        if candidate.key is None:
            raise ValueError("finite candidates require an integer key")
        return float(self.values[int(candidate.key)])
