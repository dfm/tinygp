# -*- coding: utf-8 -*-
"""
The kernels implemented in this submodule 
"""

from __future__ import annotations

__all__ = ["Quasisep", "Celerite", "SHO", "Exp", "Matern32", "Matern52"]

from abc import ABCMeta, abstractmethod
from typing import Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import block_diag

from tinygp.helpers import JAXArray, dataclass, field
from tinygp.kernels.base import Kernel
from tinygp.solvers.quasisep.core import DiagQSM, StrictLowerTriQSM, SymmQSM
from tinygp.solvers.quasisep.general import GeneralQSM


class Quasisep(Kernel, metaclass=ABCMeta):
    @abstractmethod
    def Pinf(self) -> JAXArray:
        raise NotImplementedError

    @abstractmethod
    def h(self, X: JAXArray) -> JAXArray:
        raise NotImplementedError

    @abstractmethod
    def A(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        raise NotImplementedError

    def to_symm_qsm(self, X: JAXArray) -> SymmQSM:
        Pinf = self.Pinf()
        a = jax.vmap(self.A)(
            jax.tree_util.tree_map(lambda y: jnp.append(y[0], y[:-1]), X), X
        )
        h = jax.vmap(self.h)(X)
        q = h
        p = h @ Pinf
        d = jnp.sum(p * q, axis=1)
        p = jax.vmap(jnp.dot)(p, a)
        return SymmQSM(
            diag=DiagQSM(d=d), lower=StrictLowerTriQSM(p=p, q=q, a=a)
        )

    def to_general_qsm(self, X1: JAXArray, X2: JAXArray) -> GeneralQSM:
        sortable = jax.vmap(self.coord_to_sortable)
        idx = jnp.searchsorted(sortable(X2), sortable(X1), side="right") - 1

        Xs = jax.tree_util.tree_map(lambda x: np.append(x[0], x[:-1]), X2)
        Pinf = self.Pinf()
        a = jax.vmap(self.A)(Xs, X2)
        h1 = jax.vmap(self.h)(X1)
        h2 = jax.vmap(self.h)(X2)

        ql = h2
        pl = h1 @ Pinf
        qu = h1
        pu = h2 @ Pinf

        i = jnp.clip(idx, 0, ql.shape[0] - 1)
        Xi = jax.tree_map(lambda x: jnp.asarray(x)[i], X2)
        pl = jax.vmap(jnp.dot)(pl, jax.vmap(self.A)(Xi, X1))

        i = jnp.clip(idx + 1, 0, pu.shape[0] - 1)
        Xi = jax.tree_map(lambda x: jnp.asarray(x)[i], X2)
        qu = jax.vmap(jnp.dot)(jax.vmap(self.A)(X1, Xi), qu)

        return GeneralQSM(pl=pl, ql=ql, pu=pu, qu=qu, a=a, idx=idx)

    def matmul(
        self,
        X1: JAXArray,
        X2: Optional[JAXArray] = None,
        y: Optional[JAXArray] = None,
    ) -> JAXArray:
        if y is None:
            assert X2 is not None
            y = X2
            X2 = None

        if X2 is None:
            return self.to_symm_qsm(X1) @ y

        else:
            return self.to_general_qsm(X1, X2) @ y

    def coord_to_sortable(self, X: JAXArray) -> JAXArray:
        return X

    def __add__(self, other: Union["Kernel", JAXArray]) -> "Kernel":
        if not isinstance(other, Quasisep):
            raise ValueError(
                "Quasisep kernels can only be added to other Quasisep kernels"
            )
        return Sum(self, other)

    def __radd__(self, other: Union["Kernel", JAXArray]) -> "Kernel":
        if not isinstance(other, Quasisep):
            raise ValueError(
                "Quasisep kernels can only be added to other Quasisep kernels"
            )
        return Sum(other, self)

    def __mul__(self, other: Union["Kernel", JAXArray]) -> "Kernel":
        if isinstance(other, Quasisep):
            return Product(self, other)
        if isinstance(other, Kernel) or jnp.ndim(other) != 0:
            raise ValueError(
                "Quasisep kernels can only be multiplied by scalars and other "
                "Quasisep kernels"
            )
        return Scale(other, self)

    def __rmul__(self, other: Union["Kernel", JAXArray]) -> "Kernel":
        if isinstance(other, Quasisep):
            return Product(other, self)
        if isinstance(other, Kernel) or jnp.ndim(other) != 0:
            raise ValueError(
                "Quasisep kernels can only be multiplied by scalars and other "
                "Quasisep kernels"
            )
        return Scale(other, self)

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        Pinf = self.Pinf()
        h1 = self.h(X1)
        h2 = self.h(X2)
        return jnp.where(
            X1 < X2,
            h1 @ Pinf @ self.A(X1, X2) @ h2,
            h2 @ Pinf @ self.A(X2, X1) @ h1,
        )


class Sum(Quasisep):
    def __init__(self, kernel1: Quasisep, kernel2: Quasisep):
        self.kernel1 = kernel1
        self.kernel2 = kernel2

    def Pinf(self) -> JAXArray:
        return block_diag(self.kernel1.Pinf(), self.kernel2.Pinf())

    def h(self, X: JAXArray) -> JAXArray:
        return jnp.concatenate((self.kernel1.h(X), self.kernel2.h(X)))

    def A(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        return block_diag(self.kernel1.A(X1, X2), self.kernel2.A(X1, X2))


def _prod_helper(a1: JAXArray, a2: JAXArray) -> JAXArray:
    i, j = np.meshgrid(np.arange(a1.shape[0]), np.arange(a2.shape[0]))
    i = i.flatten()
    j = j.flatten()
    if a1.ndim == 1:
        return a1[i] * a2[j]
    elif a1.ndim == 2:
        return a1[i[:, None], i[None, :]] * a2[j[:, None], j[None, :]]
    else:
        raise NotImplementedError


class Product(Quasisep):
    def __init__(self, kernel1: Quasisep, kernel2: Quasisep):
        self.kernel1 = kernel1
        self.kernel2 = kernel2

    def Pinf(self) -> JAXArray:
        return _prod_helper(self.kernel1.Pinf(), self.kernel2.Pinf())

    def h(self, X: JAXArray) -> JAXArray:
        return _prod_helper(self.kernel1.h(X), self.kernel2.h(X))

    def A(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        return _prod_helper(self.kernel1.A(X1, X2), self.kernel2.A(X1, X2))


class Scale(Quasisep):
    def __init__(self, scale: JAXArray, kernel: Quasisep):
        self.scale = scale
        self.kernel = kernel

    def Pinf(self) -> JAXArray:
        return self.scale * self.kernel.Pinf()

    def h(self, X: JAXArray) -> JAXArray:
        return self.kernel.h(X)

    def A(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        return self.kernel.A(X1, X2)


@dataclass
class Celerite(Quasisep):
    a: JAXArray
    b: JAXArray
    c: JAXArray
    d: JAXArray

    def Pinf(self) -> JAXArray:
        a = self.a
        b = self.b
        c = self.c
        d = self.d
        diff = jnp.square(c) - jnp.square(d)
        return jnp.array(
            [
                [a, b],
                [
                    b * diff + 2 * a * c * d,
                    -self.a * diff + 2 * b * c * d,
                ],
            ]
        )

    def h(self, X: JAXArray) -> JAXArray:
        return jnp.array([1.0, 0.0])

    def A(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        dt = X2 - X1
        cos = jnp.cos(self.d * dt)
        sin = jnp.sin(self.d * dt)
        return jnp.exp(-self.c * dt) * jnp.array([[cos, -sin], [sin, cos]])


@dataclass
class SHO(Quasisep):
    omega: JAXArray
    quality: JAXArray
    sigma: JAXArray = field(default_factory=lambda: jnp.ones(()))

    def Pinf(self) -> JAXArray:
        return jnp.diag(jnp.array([1, jnp.square(self.omega)]))

    def h(self, X: JAXArray) -> JAXArray:
        return jnp.array([self.sigma, 0])

    def A(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        dt = X2 - X1
        w = self.omega
        q = self.quality

        def critical(dt: JAXArray) -> JAXArray:
            return jnp.exp(-w * dt) * jnp.array(
                [[1 + w * dt, dt], [-jnp.square(w) * dt, 1 - w * dt]]
            )

        def underdamped(dt: JAXArray) -> JAXArray:
            f = jnp.sqrt(jnp.maximum(4 * jnp.square(q) - 1, 0))
            arg = 0.5 * f * w * dt / q
            sin = jnp.sin(arg)
            cos = jnp.cos(arg)
            return jnp.exp(-0.5 * w * dt / q) * jnp.array(
                [
                    [cos + sin / f, 2 * q * sin / (w * f)],
                    [-2 * q * w * sin / f, cos - sin / f],
                ]
            )

        def overdamped(dt: JAXArray) -> JAXArray:
            f = jnp.sqrt(jnp.maximum(1 - 4 * jnp.square(q), 0))
            arg = 0.5 * f * w * dt / q
            sinh = jnp.sinh(arg)
            cosh = jnp.cosh(arg)
            return jnp.exp(-0.5 * w * dt / q) * jnp.array(
                [
                    [cosh + sinh / f, 2 * q * sinh / (w * f)],
                    [-2 * q * w * sinh / f, cosh - sinh / f],
                ]
            )

        return jax.lax.cond(
            jnp.allclose(q, 0.5),
            critical,
            lambda dt: jax.lax.cond(q > 0.5, underdamped, overdamped, dt),
            dt,
        )


@dataclass
class Exp(Quasisep):
    scale: JAXArray
    sigma: JAXArray = field(default_factory=lambda: jnp.ones(()))

    def Pinf(self) -> JAXArray:
        return jnp.ones((1, 1))

    def h(self, X: JAXArray) -> JAXArray:
        return jnp.array([self.sigma])

    def A(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        dt = X2 - X1
        return jnp.exp(-dt[None, None] / self.scale)

    # def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
    #     tau = jnp.abs(X1 - X2)
    #     return jnp.square(self.sigma) * jnp.exp(-tau / self.scale)


@dataclass
class Matern32(Quasisep):
    scale: JAXArray
    sigma: JAXArray = field(default_factory=lambda: jnp.ones(()))

    def Pinf(self) -> JAXArray:
        return jnp.diag(jnp.array([1, 3 / jnp.square(self.scale)]))

    def h(self, X: JAXArray) -> JAXArray:
        return jnp.array([self.sigma, 0])

    def A(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        dt = X2 - X1
        f = np.sqrt(3) / self.scale
        return jnp.exp(-f * dt) * jnp.array(
            [[1 + f * dt, -dt], [jnp.square(f) * dt, 1 - f * dt]]
        )


@dataclass
class Matern52(Quasisep):
    scale: JAXArray
    sigma: JAXArray = field(default_factory=lambda: jnp.ones(()))

    def Pinf(self) -> JAXArray:
        f = np.sqrt(5) / self.scale
        f2 = jnp.square(f)
        f2o3 = f2 / 3
        return jnp.array(
            [[1, 0, -f2o3], [0, f2o3, 0], [-f2o3, 0, jnp.square(f2)]]
        )

    def h(self, X: JAXArray) -> JAXArray:
        return jnp.array([self.sigma, 0, 0])

    def A(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        dt = X2 - X1
        f = np.sqrt(5) / self.scale
        f2 = jnp.square(f)
        d2 = jnp.square(dt)
        return jnp.exp(-f * dt) * jnp.array(
            [
                [
                    0.5 * f2 * d2 + f * dt + 1,
                    -0.5 * f * f2 * d2,
                    0.5 * f2 * f * dt * (f * dt - 2),
                ],
                [
                    dt * (f * dt + 1),
                    -f2 * d2 + f * dt + 1,
                    f2 * dt * (f * dt - 3),
                ],
                [
                    0.5 * d2,
                    0.5 * dt * (2 - f * dt),
                    0.5 * f2 * d2 - 2 * f * dt + 1,
                ],
            ]
        )
