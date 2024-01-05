"""
In ``tinygp``, "transforms" are a powerful and relatively safe way to build
extremely expressive kernels without resorting to writing a fully fledged custom
kernel. More details can be found in the :ref:`transforms` tutorial.
"""

from __future__ import annotations

__all__ = ["Transform", "Linear", "Cholesky", "Subspace"]

from collections.abc import Sequence
from functools import partial
from typing import Any, Callable

import equinox as eqx
import jax.numpy as jnp
from jax.scipy import linalg

from tinygp.helpers import JAXArray
from tinygp.kernels.base import Kernel


class Transform(Kernel):
    """Apply a transformation to the input coordinates of the kernel

    Args:
        transform: (Callable): A callable object that accepts coordinates as
            inputs and returns transformed coordinates.
        kernel (Kernel): The kernel to use in the transformed space.
    """

    transform: Callable[[Any], Any] = eqx.field(static=True)
    kernel: Kernel

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        return self.kernel.evaluate(self.transform(X1), self.transform(X2))


class Linear(Kernel):
    """Apply a linear transformation to the input coordinates of the kernel

    For example, the following transformed kernels are all equivalent, but the
    second supports more flexible transformations:

    .. code-block:: python

        >>> import numpy as np
        >>> from tinygp import kernels, transforms
        >>> kernel0 = kernels.Matern32(4.5)
        >>> kernel1 = transforms.Linear(1.0 / 4.5, kernels.Matern32())
        >>> np.testing.assert_allclose(
        ...     kernel0.evaluate(0.5, 0.1), kernel1.evaluate(0.5, 0.1)
        ... )

    Args:
        scale (JAXArray): A 0-, 1-, or 2-dimensional array specifying the
            scale of this transform.
        kernel (Kernel): The kernel to use in the transformed space.
    """

    scale: JAXArray
    kernel: Kernel

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        if jnp.ndim(self.scale) < 2:
            transform = partial(jnp.multiply, self.scale)
        elif jnp.ndim(self.scale) == 2:
            transform = partial(jnp.dot, self.scale)
        else:
            raise ValueError("'scale' must be 0-, 1-, or 2-dimensional")
        return self.kernel.evaluate(transform(X1), transform(X2))


class Cholesky(Kernel):
    """Apply a Cholesky transformation to the input coordinates of the kernel

    For example, the following transformed kernels are all equivalent, but the
    second supports more flexible transformations:

    .. code-block:: python

        >>> import numpy as np
        >>> from tinygp import kernels, transforms
        >>> kernel0 = kernels.Matern32(4.5)
        >>> kernel1 = transforms.Cholesky(4.5, kernels.Matern32())
        >>> np.testing.assert_allclose(
        ...     kernel0.evaluate(0.5, 0.1), kernel1.evaluate(0.5, 0.1)
        ... )

    Args:
        factor (JAXArray): A 0-, 1-, or 2-dimensional array specifying the
            Cholesky factor. If 2-dimensional, this must be a lower
            triangular matrix, but this is not checked.
        kernel (Kernel): The kernel to use in the transformed space.
    """

    factor: JAXArray
    kernel: Kernel

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        if jnp.ndim(self.factor) < 2:
            transform = partial(jnp.multiply, 1.0 / self.factor)
        elif jnp.ndim(self.factor) == 2:
            transform = partial(linalg.solve_triangular, self.factor, lower=True)
        else:
            raise ValueError("'scale' must be 0-, 1-, or 2-dimensional")
        return self.kernel.evaluate(transform(X1), transform(X2))

    @classmethod
    def from_parameters(
        cls, diagonal: JAXArray, off_diagonal: JAXArray, kernel: Kernel
    ) -> Cholesky:
        """Build a Cholesky transform with a sensible parameterization

        Args:
            diagonal (JAXArray): An ``(ndim,)`` array with the diagonal
                elements of ``factor``. These must be positive, but this
                is not checked.
            off_diagonal (JAXArray): An ``((ndim - 1) * ndim,)`` array
                with the off-diagonal elements of ``factor``.
            kernel (Kernel): The kernel to use in the transformed space.
        """
        ndim = diagonal.size
        if off_diagonal.size != ((ndim - 1) * ndim) // 2:
            raise ValueError(
                "Dimension mismatch: expected "
                f"(ndim-1)*ndim/2 = {((ndim - 1) * ndim) // 2} elements in "
                f"'off_diagonal'; got {off_diagonal.size}"
            )
        factor = jnp.zeros((ndim, ndim))
        factor = factor.at[jnp.diag_indices(ndim)].add(diagonal)
        factor = factor.at[jnp.tril_indices(ndim, -1)].add(off_diagonal)
        return cls(factor, kernel)


class Subspace(Kernel):
    """A kernel transform that selects a subset of the input dimensions

    For example, the following kernel only depends on the coordinates in the
    second (`1`-th) dimension:

    .. code-block:: python

        >>> import numpy as np
        >>> from tinygp import kernels, transforms
        >>> kernel = transforms.Subspace(1, kernels.Matern32())
        >>> np.testing.assert_allclose(
        ...     kernel.evaluate(np.array([0.5, 0.1]), np.array([-0.4, 0.7])),
        ...     kernel.evaluate(np.array([100.5, 0.1]), np.array([-70.4, 0.7])),
        ... )

    Args:
        axis: (Axis, optional): An integer or tuple of integers specifying the
            axes to select.
        kernel (Kernel): The kernel to use in the transformed space.
    """

    axis: Sequence[int] | int = eqx.field(static=True)
    kernel: Kernel

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        return self.kernel.evaluate(X1[self.axis], X2[self.axis])
