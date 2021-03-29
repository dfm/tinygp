# -*- coding: utf-8 -*-

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

from typing import Tuple, Union

import jax
import jax.numpy as jnp

from .metrics import Metric, diagonal_metric
from .types import JAXArray

Axis = Union[int, Tuple[int], JAXArray]


class Kernel:
    def __call__(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        raise NotImplementedError()

    def evaluate_diag(self, X: JAXArray) -> JAXArray:
        return jax.vmap(self)(X, X)

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        return jax.vmap(lambda _X1: jax.vmap(lambda _X2: self(_X1, _X2))(X2))(
            X1
        )

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


class Sum(Kernel):
    def __init__(self, kernel1: Kernel, kernel2: Kernel):
        self.kernel1 = kernel1
        self.kernel2 = kernel2

    def __call__(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        return self.kernel1(X1, X2) + self.kernel2(X1, X2)


class Product(Kernel):
    def __init__(self, kernel1: Kernel, kernel2: Kernel):
        self.kernel1 = kernel1
        self.kernel2 = kernel2

    def __call__(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        return self.kernel1(X1, X2) * self.kernel2(X1, X2)


class Constant(Kernel):
    def __init__(self, value: JAXArray):
        self.value = jnp.asarray(value)

    def __call__(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        return self.value


class SubspaceKernel(Kernel):
    def __init__(self, *, axis: Axis = None):
        self.axis = axis

    def evaluate_subspace(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        raise NotImplementedError()

    def __call__(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        if self.axis is None:
            return self.evaluate_subspace(X1, X2)
        return self.evaluate_subspace(X1[self.axis], X2[self.axis])


class DotProduct(SubspaceKernel):
    def evaluate_subspace(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        return X1 @ X2


class Polynomial(SubspaceKernel):
    def __init__(self, *, order: int, sigma: JAXArray, axis: Axis = None):
        self.order = float(order)
        self.sigma2 = jnp.asarray(sigma) ** 2
        super().__init__(axis=axis)

    def evaluate_subspace(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        return (X1 @ X2 + self.sigma2) ** self.order


class Linear(SubspaceKernel):
    def __init__(self, *, order: int, sigma: JAXArray, axis: Axis = None):
        self.order = float(order)
        self.sigma2 = jnp.asarray(sigma) ** 2
        super().__init__(axis=axis)

    def evaluate_subspace(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        return (X1 @ X2 / self.sigma2) ** self.order


class MetricKernel(Kernel):
    def __init__(self, metric: Union[Metric, JAXArray]):
        if callable(metric):
            self.metric = metric
        else:
            self.metric = diagonal_metric(metric)

    def evaluate_radial(self, r2: JAXArray) -> JAXArray:
        raise NotImplementedError()

    def __call__(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        r2 = self.metric(X1 - X2)
        return self.evaluate_radial(jnp.where(jnp.isclose(r2, 0.0), 0.0, r2))


class Exp(MetricKernel):
    def evaluate_radial(self, r2: JAXArray) -> JAXArray:
        return jnp.exp(-jnp.sqrt(r2))


class ExpSquared(MetricKernel):
    def evaluate_radial(self, r2: JAXArray) -> JAXArray:
        return jnp.exp(-0.5 * r2)


class Matern32(MetricKernel):
    def evaluate_radial(self, r2: JAXArray) -> JAXArray:
        arg = jnp.sqrt(3.0 * r2)
        return (1.0 + arg) * jnp.exp(-arg)


class Matern52(MetricKernel):
    def evaluate_radial(self, r2: JAXArray) -> JAXArray:
        arg1 = 5.0 * r2
        arg2 = jnp.sqrt(arg1)
        return (1.0 + arg2 + arg1 / 3.0) * jnp.exp(-arg2)


class Cosine(MetricKernel):
    def evaluate_radial(self, r2: JAXArray) -> JAXArray:
        return jnp.cos(2 * jnp.pi * jnp.sqrt(r2))


class ExpSineSquared(MetricKernel):
    def __init__(self, metric: Metric, *, gamma: JAXArray):
        self.gamma = jnp.asarray(gamma)
        super().__init__(metric)

    def evaluate_radial(self, r2: JAXArray) -> JAXArray:
        return jnp.exp(
            -self.gamma * jnp.square(jnp.sin(jnp.pi * jnp.sqrt(r2)))
        )


class RationalQuadratic(MetricKernel):
    def __init__(self, metric: Metric, *, alpha: JAXArray):
        self.alpha = jnp.asarray(alpha)
        super().__init__(metric)

    def evaluate_radial(self, r2: JAXArray) -> JAXArray:
        return (1.0 + 0.5 * r2 / self.alpha) ** -self.alpha
