# -*- coding: utf-8 -*-

from __future__ import annotations

__all__ = [
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
    "RationalQuadratic",
]

from typing import Callable, Optional, Tuple, Union

import jax
import jax.numpy as jnp

from .metrics import Metric, diagonal_metric, unit_metric
from .types import JAXArray

Axis = Union[int, Tuple[int], JAXArray]


class Kernel:
    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
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
    def __init__(self, function: Callable[[JAXArray, JAXArray], JAXArray]):
        self.function = function

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        return self.function(X1, X2)


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


class SubspaceKernel(Kernel):
    def __init__(self, kernel: Kernel, axis: Optional[Axis] = None):
        self.kernel = kernel
        self.axis = axis

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        if self.axis is None:
            return self.kernel.evaluate(X1, X2)
        return self.kernel.evaluate(X1[self.axis], X2[self.axis])


class MetricKernel(Kernel):
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
        arg = jnp.sqrt(3.0 * jnp.sum(jnp.abs((X1 - X2) / self.scale)))
        return (1.0 + arg) * jnp.exp(-arg)


class Matern52(Kernel):
    def __init__(self, scale: JAXArray = jnp.ones(())):
        self.scale = scale

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        r = jnp.sum(jnp.abs((X1 - X2) / self.scale))
        arg1 = 5.0 * r ** 2
        arg2 = jnp.sqrt(5.0) * r
        return (1.0 + arg2 + arg1 / 3.0) * jnp.exp(-arg2)


class Cosine(Kernel):
    def __init__(self, *, period: JAXArray):
        self.period = period

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        return jnp.cos(2 * jnp.pi * jnp.abs((X1 - X2) / self.period))


class ExpSineSquared(Kernel):
    def __init__(self, *, period: JAXArray, gamma: JAXArray):
        self.period = period
        self.gamma = gamma

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        x = jnp.abs((X1 - X2) / self.period)
        return jnp.exp(-self.gamma * jnp.square(jnp.sin(jnp.pi * x)))


class RationalQuadratic(Kernel):
    def __init__(self, alpha: JAXArray):
        self.alpha = alpha

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        r2 = jnp.sum(jnp.square(X1 - X2))
        return (1.0 + 0.5 * r2 / self.alpha) ** -self.alpha
