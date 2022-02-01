# -*- coding: utf-8 -*-

from __future__ import annotations

__all__ = ["Transform", "Linear", "Cholesky", "Subspace"]

from functools import partial
from typing import Any, Callable, Sequence, Union

import jax.numpy as jnp
from jax.scipy import linalg

from tinygp.kernels import Kernel
from tinygp.types import JAXArray


class Transform(Kernel):
    """Apply a transformation to the input coordinates of the kernel

    Args:
        transform: (Callable): A callable object that accepts coordinates as
            inputs and returns transformed coordinates.
        kernel (Kernel): The kernel to use in the transformed space.
    """

    # This type signature is a hack for Sphinx sphinx-doc/sphinx#9736
    def __init__(self, transform: Callable[[Any], Any], kernel: Kernel):
        self.transform = transform
        self.kernel = kernel

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        return self.kernel.evaluate(self.transform(X1), self.transform(X2))


class Linear(Transform):
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

    def __init__(self, scale: JAXArray, kernel: Kernel):
        self.scale = scale
        if jnp.ndim(scale) < 2:
            self.transform = partial(jnp.multiply, scale)
        elif jnp.ndim(scale) == 2:
            self.transform = partial(jnp.dot, scale)
        else:
            raise ValueError("'scale' must be 0-, 1-, or 2-dimensional")
        self.kernel = kernel


class Cholesky(Transform):
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
            Cholesky factor. If 2-dimensional, this must be a lower or
            upper triangular matrix as specified by ``lower``, but this is
            not checked.
        kernel (Kernel): The kernel to use in the transformed space.
        lower: (bool, optional): Is ``factor`` lower (vs upper) triangular.
    """

    def __init__(
        self, factor: JAXArray, kernel: Kernel, *, lower: bool = True
    ):
        self.factor = factor
        if jnp.ndim(factor) < 2:
            self.transform = partial(jnp.multiply, 1.0 / factor)
        elif jnp.ndim(factor) == 2:
            self.transform = partial(
                linalg.solve_triangular, factor, lower=lower
            )
        else:
            raise ValueError("'scale' must be 0-, 1-, or 2-dimensional")
        self.kernel = kernel

    @classmethod
    def from_parameters(
        cls, diagonal: JAXArray, off_diagonal: JAXArray, kernel: Kernel
    ) -> "Cholesky":
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
        return cls(factor, kernel, lower=True)


class Subspace(Transform):
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

    def __init__(self, axis: Union[Sequence[int], int], kernel: Kernel):
        self.transform = lambda X: X[axis]
        self.kernel = kernel
