# -*- coding: utf-8 -*-
"""

"""


from __future__ import annotations

__all__ = ["Diagonal", "Banded"]

from abc import ABCMeta, abstractmethod
from typing import Any, Tuple, Union

import jax.numpy as jnp
import numpy as np

from tinygp.helpers import JAXArray, dataclass
from tinygp.solvers.quasisep import core


class Noise(metaclass=ABCMeta):
    __array_priority__ = 2000

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def __add__(self, other: JAXArray) -> JAXArray:
        raise NotImplementedError

    @abstractmethod
    def __radd__(self, other: JAXArray) -> JAXArray:
        raise NotImplementedError

    @abstractmethod
    def __matmul__(self, other: JAXArray) -> JAXArray:
        raise NotImplementedError

    @abstractmethod
    def to_qsm(self) -> Union[core.SymmQSM, core.DiagQSM]:
        raise NotImplementedError


@dataclass
class Diagonal(Noise):
    diag: JAXArray

    def __post_init__(self) -> None:
        if jnp.ndim(self.diag) != 1:
            raise ValueError(
                "The diagonal for the noise model be the same shape as the data; "
                "if passing a constant, it should be broadcasted first"
            )

    def _add(self, other: JAXArray) -> JAXArray:
        return (
            jnp.asarray(other)
            .at[jnp.diag_indices(other.shape[0])]
            .add(self.diag)
        )

    def __add__(self, other: JAXArray) -> JAXArray:
        return self._add(other)

    def __radd__(self, other: JAXArray) -> JAXArray:
        return self._add(other)

    def __matmul__(self, other: JAXArray) -> JAXArray:
        if jnp.ndim(other) == 1:
            return self.diag * other
        else:
            return self.diag[:, None] * other

    def to_qsm(self) -> core.DiagQSM:
        return core.DiagQSM(d=self.diag)


@dataclass
class Banded(Noise):
    diag: JAXArray
    off_diags: JAXArray

    def _indices(
        self,
    ) -> Tuple[Tuple[JAXArray, JAXArray], Tuple[JAXArray, JAXArray]]:
        N, J = jnp.shape(self.off_diags)
        sparse_idx_1 = []
        sparse_idx_2 = []
        dense_idx_1 = []
        dense_idx_2 = []
        for j in range(J):
            sparse_idx_1.append(np.arange(N - j - 1))
            sparse_idx_2.append(np.full(N - j - 1, j, dtype=int))
            dense_idx_1.append(np.arange(0, N - j - 1))
            dense_idx_2.append(np.arange(j + 1, N))

        return (
            (np.concatenate(sparse_idx_1), np.concatenate(sparse_idx_2)),
            (np.concatenate(dense_idx_1), np.concatenate(dense_idx_2)),
        )

    def _add(self, other: JAXArray) -> JAXArray:
        sparse_idx, dense_idx = self._indices()

        # Start by adding the diagonal
        result = (
            jnp.asarray(other)
            .at[jnp.diag_indices(other.shape[0])]
            .add(self.diag)
        )

        # Then the off diagonals, assuming symmetric
        return result.at[
            (
                np.append(dense_idx[0], dense_idx[1]),
                np.append(dense_idx[1], dense_idx[0]),
            )
        ].add(
            self.off_diags[
                (
                    np.append(sparse_idx[0], sparse_idx[0]),
                    np.append(sparse_idx[1], sparse_idx[1]),
                )
            ]
        )

    def __add__(self, other: JAXArray) -> JAXArray:
        return self._add(other)

    def __radd__(self, other: JAXArray) -> JAXArray:
        return self._add(other)

    def __matmul__(self, other: JAXArray) -> JAXArray:
        return self.to_qsm() @ other

    def to_qsm(self) -> core.SymmQSM:
        N, J = jnp.shape(self.off_diags)
        p = jnp.repeat(jnp.eye(1, J), N, axis=0)
        q = self.off_diags
        a = jnp.repeat(jnp.eye(J, k=1)[None], N, axis=0)
        return core.SymmQSM(
            diag=core.DiagQSM(d=self.diag),
            lower=core.StrictLowerTriQSM(p=p, q=q, a=a),
        )


@dataclass
class Dense(Noise):
    value: JAXArray

    def __add__(self, other: JAXArray) -> JAXArray:
        return self.value + other

    def __radd__(self, other: JAXArray) -> JAXArray:
        return other + self.value

    def __matmul__(self, other: JAXArray) -> JAXArray:
        return self.value @ other

    def to_qsm(self) -> Union[core.SymmQSM, core.DiagQSM]:
        raise NotImplementedError
