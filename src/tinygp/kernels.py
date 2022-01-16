# -*- coding: utf-8 -*-

from __future__ import annotations

__all__ = [
    "Kernel",
    "Custom",
    "AffineTransform",
    "SubspaceTransform",
    "Sum",
    "Product",
    "Constant",
    "DotProduct",
    "Polynomial",
    "Linear",
    "Exp",
    "ExpSquared",
    "Matern32",
    "Matern52",
    "Cosine",
    "ExpSineSquared",
    "RationalQuadratic",
]

from typing import Callable, Optional, Sequence, Union

import jax
import jax.numpy as jnp

from .metrics import Metric, diagonal_metric, unit_metric
from .types import JAXArray

Axis = Union[int, Sequence[int], JAXArray]


class Kernel:
    """The base class for all kernel implementations

    This subclass provides default implementations to add and multiply kernels.
    Subclasses should accept parameters in their ``__init__`` and then override
    :func:`Kernel.evaluate` with custom behavior.
    """

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        """Evaluate the kernel at a pair of input coordinates

        This should be overridden be subclasses to return the kernel-specific
        value. Two things to note:

        1. Users should never directly call :func:`Kernel.evaluate`. Instead,
           always "call" the kernel instance directly; for example, you can
           evaluate the Matern-3/2 kernel using ``Matern32(1.5)(x1, x2)``, for
           arrays of input coordinates ``x1`` and ``x2``.
        2. When implementing a custom kernel, this method should treat ``X1``
           and ``X2`` as single datapoints. In other words, these inputs will
           typically either be scalars of have shape ``n_dim``, where ``n_dim``
           is the number of input dimensions, rather than ``n_data`` or
           ``(n_data, n_dim)``, and you should let the :class:`Kernel` ``vmap``
           magic handle all the broadcasting for you.
        """
        raise NotImplementedError()

    def __call__(
        self, X1: JAXArray, X2: Optional[JAXArray] = None
    ) -> JAXArray:
        if X2 is None:
            return jax.vmap(self.evaluate, in_axes=(0, 0))(X1, X1)
        return jax.vmap(
            jax.vmap(self.evaluate, in_axes=(None, 0)), in_axes=(0, None)
        )(X1, X2)

    def __add__(self, other: Union["Kernel", JAXArray]) -> "Kernel":
        if isinstance(other, Kernel):
            return Sum(self, other)
        return Sum(self, Constant(other))

    def __radd__(self, other: Union["Kernel", JAXArray]) -> "Kernel":
        if isinstance(other, Kernel):
            return Sum(other, self)
        return Sum(Constant(other), self)

    def __mul__(self, other: Union["Kernel", JAXArray]) -> "Kernel":
        if isinstance(other, Kernel):
            return Product(self, other)
        return Product(self, Constant(other))

    def __rmul__(self, other: Union["Kernel", JAXArray]) -> "Kernel":
        if isinstance(other, Kernel):
            return Product(other, self)
        return Product(Constant(other), self)


class Custom(Kernel):
    """A custom kernel class implemented as a callable

    Args:
        function: A callable with a signature and behavior that matches
        :func:`Kernel.evaluate`.
    """

    def __init__(self, function: Callable[[JAXArray, JAXArray], JAXArray]):
        self.function = function

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        return self.function(X1, X2)


class AffineTransform(Kernel):
    """Apply a linear transformation to the input coordinates of the kernel

    For example

    .. code-block:: python

        kernel = tinygp.kernels.AffineTransformation(
            tinygp.kernels.Matern32() 4.5
        )

    is equivalent to

    .. code-block:: python

        kernel = tinygp.kernels.Matern32(4.5)

    but the former allows for more flexible treatment of multivariate inputs.

    Args:
        kernel (Kernel): The
    """

    def __init__(
        self,
        kernel: Kernel,
        metric: Optional[Union[Metric, JAXArray]] = None,
    ):
        self.kernel = kernel
        if metric is None:
            self.metric = unit_metric
        elif callable(metric):
            self.metric = metric  # type: ignore
        else:
            self.metric = diagonal_metric(metric)  # type: ignore

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        return self.kernel.evaluate(self.metric(X1), self.metric(X2))


class SubspaceTransform(Kernel):
    def __init__(self, kernel: Kernel, axis: Optional[Axis] = None):
        self.kernel = kernel
        self.axis = axis

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        if self.axis is None:
            return self.kernel.evaluate(X1, X2)
        return self.kernel.evaluate(X1[self.axis], X2[self.axis])


class Sum(Kernel):
    def __init__(self, kernel1: Kernel, kernel2: Kernel):
        self.kernel1 = kernel1
        self.kernel2 = kernel2

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        return self.kernel1.evaluate(X1, X2) + self.kernel2.evaluate(X1, X2)


class Product(Kernel):
    def __init__(self, kernel1: Kernel, kernel2: Kernel):
        self.kernel1 = kernel1
        self.kernel2 = kernel2

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        return self.kernel1.evaluate(X1, X2) * self.kernel2.evaluate(X1, X2)


class Constant(Kernel):
    def __init__(self, value: JAXArray):
        self.value = jnp.asarray(value)

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        return self.value


class DotProduct(Kernel):
    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        return X1 @ X2


class Polynomial(Kernel):
    def __init__(self, *, order: JAXArray, sigma: JAXArray):
        self.order = order
        self.sigma2 = jnp.square(sigma)

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        return (X1 @ X2 + self.sigma2) ** self.order


class Linear(Kernel):
    def __init__(self, *, order: JAXArray, sigma: JAXArray):
        self.order = order
        self.sigma2 = jnp.square(sigma)

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        return (X1 @ X2 / self.sigma2) ** self.order


class Exp(Kernel):
    def __init__(self, scale: JAXArray = jnp.ones(())):
        self.scale = scale

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        return jnp.exp(-jnp.sum(jnp.abs((X1 - X2) / self.scale)))


class ExpSquared(Kernel):
    def __init__(self, scale: JAXArray = jnp.ones(())):
        self.scale = scale

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        return jnp.exp(-0.5 * jnp.sum(jnp.square((X1 - X2) / self.scale)))


class Matern32(Kernel):
    def __init__(self, scale: JAXArray = jnp.ones(())):
        self.scale = scale

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        r = jnp.sum(jnp.abs((X1 - X2) / self.scale))
        arg = jnp.sqrt(3.0) * r
        return (1.0 + arg) * jnp.exp(-arg)


class Matern52(Kernel):
    def __init__(self, scale: JAXArray = jnp.ones(())):
        self.scale = scale

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        r = jnp.sum(jnp.abs((X1 - X2) / self.scale))
        arg = jnp.sqrt(5.0) * r
        return (1.0 + arg + jnp.square(arg) / 3.0) * jnp.exp(-arg)


class Cosine(Kernel):
    def __init__(self, *, period: JAXArray):
        self.period = period

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        r = jnp.sum(jnp.abs((X1 - X2) / self.period))
        return jnp.cos(2 * jnp.pi * r)


class ExpSineSquared(Kernel):
    def __init__(self, *, period: JAXArray, gamma: JAXArray):
        self.period = period
        self.gamma = gamma

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        r = jnp.sum(jnp.abs((X1 - X2) / self.period))
        return jnp.exp(-self.gamma * jnp.square(jnp.sin(jnp.pi * r)))


class RationalQuadratic(Kernel):
    def __init__(self, alpha: JAXArray):
        self.alpha = alpha

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        r2 = jnp.sum(jnp.square(X1 - X2))
        return (1.0 + 0.5 * r2 / self.alpha) ** -self.alpha
