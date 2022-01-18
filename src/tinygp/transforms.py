# -*- coding: utf-8 -*-

from __future__ import annotations

__all__ = ["Transform", "Affine", "Subspace"]

from functools import partial
from typing import Any, Callable, Sequence, Union

import jax.numpy as jnp
from jax.scipy import linalg

from .kernels import Kernel
from .types import JAXArray


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


class Affine(Transform):
    """Apply an affine transformation to the input coordinates of the kernel

    For example, the following transformed kernels are all equivalent:

    .. code-block:: python

        >>> import numpy as np
        >>> from tinygp import kernels, transforms
        >>> kernel0 = kernels.Matern32(4.5)
        >>> kernel1 = transforms.Affine(4.5, kernels.Matern32())
        >>> kernel2 = transforms.Affine(4.5 ** 2, kernels.Matern32(), variance=True)
        >>> np.testing.assert_allclose(
        ...     kernel0.evaluate(0.5, 0.1), kernel1.evaluate(0.5, 0.1)
        ... )
        >>> np.testing.assert_allclose(
        ...     kernel0.evaluate(0.5, 0.1), kernel2.evaluate(0.5, 0.1)
        ... )

    Args:
        scale (JAXArray): A 0-, 1-, or 2- dimensional array specifying the
            variance or covariance of the input dimensions.
        kernel (Kernel): The kernel to use in the transformed space.
        variance: (bool, optional): If ``True``, take the square root of
            ``scale`` before applying its inverse.
    """

    def __init__(
        self, scale: JAXArray, kernel: Kernel, *, variance: bool = False
    ):
        scale = jnp.asarray(scale)
        if scale.ndim < 2:
            if variance:
                self.transform = partial(jnp.multiply, 1.0 / jnp.sqrt(scale))
            else:
                self.transform = partial(jnp.multiply, 1.0 / scale)
        elif scale.ndim == 2:
            if variance:
                chol = linalg.cholesky(scale, lower=True)
                self.transform = partial(
                    linalg.solve_triangular, chol, lower=True
                )
            else:
                self.transform = partial(linalg.solve, scale)
        else:
            raise ValueError("'scale' must be 0-, 1-, or 2-dimensional")
        self.kernel = kernel


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
