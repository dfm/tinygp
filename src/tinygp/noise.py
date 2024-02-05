"""
This subpackage provides the tools needed to build expressive observation
processes for ``tinygp`` Gaussian process models. The most commonly used noise
model is :class:`Diagonal`, which adds a constant diagonal matrix to the process
covariance to represent per-observation noise. This subpackage also includes a
:class:`Dense` model for adding a full rank observation model, and
:class:`Banded` to capture noise that can be represented by a banded matrix.
"""

from __future__ import annotations

__all__ = ["Diagonal", "Dense", "Banded"]

from abc import abstractmethod
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from tinygp.helpers import JAXArray

if TYPE_CHECKING:
    from tinygp.solvers.quasisep.core import DiagQSM, SymmQSM


class Noise(eqx.Module):
    """An abstract base class defining the noise model protocol"""

    __array_priority__ = 2001

    @abstractmethod
    def diagonal(self) -> JAXArray:
        """The diagonal elements of the noise model as an array"""
        raise NotImplementedError

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
    def to_qsm(self) -> SymmQSM | DiagQSM:
        """This noise model represented as a quasiseparable matrix"""
        raise NotImplementedError


class Diagonal(Noise):
    """A diagonal observation noise model

    This represents the observation model using per-observation measurement
    variances.

    Args:
        diag: The diagonal elements of the noise model.
    """

    diag: JAXArray

    def __check_init__(self) -> None:
        if jnp.ndim(self.diag) != 1:
            raise ValueError(
                "The diagonal for the noise model be the same shape as the data; "
                "if passing a constant, it should be broadcasted first"
            )

    def diagonal(self) -> JAXArray:
        return self.diag

    def _add(self, other: JAXArray) -> JAXArray:
        return jnp.asarray(other).at[jnp.diag_indices(other.shape[0])].add(self.diag)

    def __add__(self, other: JAXArray) -> JAXArray:
        return self._add(other)

    def __radd__(self, other: JAXArray) -> JAXArray:
        return self._add(other)

    def __matmul__(self, other: JAXArray) -> JAXArray:
        if jnp.ndim(other) == 1:
            return self.diag * other
        else:
            return self.diag[:, None] * other

    def to_qsm(self) -> DiagQSM:
        from tinygp.solvers.quasisep.core import DiagQSM

        return DiagQSM(d=self.diag)


class Dense(Noise):
    """A full rank observation noise model

    .. warning:: This model cannot be used in conjunction with the
        :class:`tinygp.solvers.QuasisepSolver` for scalable computations.

    Args:
        value: The N-by-N full rank observation model.
    """

    value: JAXArray

    def diagonal(self) -> JAXArray:
        return jnp.diag(self.value)

    def __add__(self, other: JAXArray) -> JAXArray:
        return self.value + other

    def __radd__(self, other: JAXArray) -> JAXArray:
        return other + self.value

    def __matmul__(self, other: JAXArray) -> JAXArray:
        return self.value @ other

    def to_qsm(self) -> SymmQSM | DiagQSM:
        """This cannot be compactly represented as a quasiseparable matrix"""
        raise NotImplementedError


class Banded(Noise):
    r"""A banded observation noise model

    This model captures noise that can be represented by a small number of
    off-diagonal elements in the observation matrix. One practical example of
    such an observation model is discussed by `Delisle et al. (2020)
    <https://arxiv.org/abs/2004.10678>`_. This matrix is defined by two arrays:
    ``diag`` and ``off_diags``, with shapes ``(N,)`` and ``(N, J)``
    respectively, where ``N`` is the number of data points and ``J`` is the
    number of non-zero off-diagonals required.

    For example, the following matrix has ``N = 4`` and ``J = 2``:

    .. math::

        N = \left(\begin{array}{cccc}
            n_{11} & n_{12} & n_{13} & 0      \\
            n_{12} & n_{22} & n_{23} & n_{24} \\
            n_{13} & n_{23} & n_{33} & n_{34} \\
            0      & n_{24} & n_{34} & n_{44}
        \end{array}\right)

    and it would be represented by the following arrays:

    .. code-block:: python

        diag = [n11, n22, n33, n44]

    and

    .. code-block:: python

        off_diags = [
            [n12, n13],
            [n23, n24],
            [n34,  * ],
            [ *,   * ],
        ]


    Where ``*`` represents an element that can have any arbitrary value, since it
    won't ever be accessed.
    """

    diag: JAXArray
    off_diags: JAXArray

    def diagonal(self) -> JAXArray:
        return self.diag

    def _indices(
        self,
    ) -> tuple[tuple[JAXArray, JAXArray], tuple[JAXArray, JAXArray]]:
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
        result = jnp.asarray(other).at[jnp.diag_indices(other.shape[0])].add(self.diag)

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

    def to_qsm(self) -> SymmQSM:
        from tinygp.solvers.quasisep import core

        N, J = jnp.shape(self.off_diags)
        p = jnp.repeat(jnp.eye(1, J), N, axis=0)
        q = self.off_diags
        a = jnp.repeat(jnp.eye(J, k=1)[None], N, axis=0)
        return core.SymmQSM(
            diag=core.DiagQSM(d=self.diag),
            lower=core.StrictLowerTriQSM(p=p, q=q, a=a),
        )
