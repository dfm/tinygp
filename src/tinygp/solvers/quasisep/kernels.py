# -*- coding: utf-8 -*-

from __future__ import annotations

__all__ = ["Quasisep", "Celerite", "SHO", "Exp", "Matern32", "Matern52"]

from abc import ABCMeta, abstractmethod
from typing import Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import block_diag

from tinygp.helpers import JAXArray, dataclass
from tinygp.kernels.base import Kernel
from tinygp.solvers.quasisep.core import DiagQSM, StrictLowerTriQSM, SymmQSM
from tinygp.solvers.quasisep.general import GeneralQSM


class Quasisep(Kernel, metaclass=ABCMeta):
    @abstractmethod
    def p(self, X: JAXArray) -> JAXArray:
        raise NotImplementedError

    @abstractmethod
    def q(self, X: JAXArray) -> JAXArray:
        raise NotImplementedError

    @abstractmethod
    def A(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        raise NotImplementedError

    def to_symm_qsm(self, X: JAXArray) -> SymmQSM:
        a = jax.vmap(self.A)(
            jax.tree_util.tree_map(lambda y: jnp.append(y[0], y[:-1]), X), X
        )
        q = jax.vmap(self.q)(X)
        p = jax.vmap(self.p)(X)
        d = jnp.sum(p * q, axis=1)
        p = jax.vmap(jnp.dot)(p, a)
        return SymmQSM(
            diag=DiagQSM(d=d), lower=StrictLowerTriQSM(p=p, q=q, a=a)
        )

    def to_general_qsm(self, X1: JAXArray, X2: JAXArray) -> GeneralQSM:
        sortable = jax.vmap(self.coord_to_sortable)
        idx = jnp.searchsorted(sortable(X2), sortable(X1), side="right") - 1

        Xs = jax.tree_util.tree_map(lambda x: np.append(x[0], x[:-1]), X2)
        a = jax.vmap(self.A)(Xs, X2)
        ql = jax.vmap(self.q)(X2)
        pl = jax.vmap(self.p)(X1)
        qu = jax.vmap(self.q)(X1)
        pu = jax.vmap(self.p)(X2)

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


class Sum(Quasisep):
    def __init__(self, kernel1: Quasisep, kernel2: Quasisep):
        self.kernel1 = kernel1
        self.kernel2 = kernel2

    def p(self, X: JAXArray) -> JAXArray:
        return jnp.concatenate((self.kernel1.p(X), self.kernel2.p(X)))

    def q(self, X: JAXArray) -> JAXArray:
        return jnp.concatenate((self.kernel1.q(X), self.kernel2.q(X)))

    def A(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        return block_diag(self.kernel1.A(X1, X2), self.kernel2.A(X1, X2))

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        return self.kernel1.evaluate(X1, X2) + self.kernel2.evaluate(X1, X2)


class Product(Quasisep):
    def __init__(self, kernel1: Quasisep, kernel2: Quasisep):
        self.kernel1 = kernel1
        self.kernel2 = kernel2

    def p(self, X: JAXArray) -> JAXArray:
        p1 = self.kernel1.p(X)
        p2 = self.kernel2.p(X)
        i, j = np.meshgrid(np.arange(p1.shape[0]), np.arange(p2.shape[0]))
        i = i.flatten()
        j = j.flatten()
        return p1[i] * p2[j]

    def q(self, X: JAXArray) -> JAXArray:
        q1 = self.kernel1.q(X)
        q2 = self.kernel2.q(X)
        i, j = np.meshgrid(np.arange(q1.shape[0]), np.arange(q2.shape[0]))
        i = i.flatten()
        j = j.flatten()
        return q1[i] * q2[j]

    def A(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        a1 = self.kernel1.A(X1, X2)
        a2 = self.kernel2.A(X1, X2)
        i, j = np.meshgrid(np.arange(a1.shape[0]), np.arange(a2.shape[0]))
        i = i.flatten()
        j = j.flatten()
        return a1[i[:, None], i[None, :]] * a2[j[:, None], j[None, :]]

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        return self.kernel1.evaluate(X1, X2) * self.kernel2.evaluate(X1, X2)


class Scale(Quasisep):
    def __init__(self, scale: JAXArray, kernel: Quasisep):
        self.scale = scale
        self.kernel = kernel

    def p(self, X: JAXArray) -> JAXArray:
        return self.scale * self.kernel.p(X)

    def q(self, X: JAXArray) -> JAXArray:
        return self.kernel.q(X)

    def A(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        return self.kernel.A(X1, X2)

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        return self.kernel.evaluate(X1, X2) * self.scale


@dataclass
class Celerite(Quasisep):
    a: JAXArray
    b: JAXArray
    c: JAXArray
    d: JAXArray

    def p(self, X: JAXArray) -> JAXArray:
        return jnp.array([self.a, self.b])

    def q(self, X: JAXArray) -> JAXArray:
        return jnp.array([1.0, 0.0])

    def A(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        dt = X2 - X1
        cos = jnp.cos(self.d * dt)
        sin = jnp.sin(self.d * dt)
        return jnp.exp(-self.c * dt) * jnp.array([[cos, -sin], [sin, cos]])

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        tau = jnp.abs(X1 - X2)
        return jnp.exp(-self.c * tau) * (
            self.a * jnp.cos(self.d * tau) + self.b * jnp.sin(self.d * tau)
        )


@dataclass
class SHO(Quasisep):
    omega: JAXArray
    quality: JAXArray
    sigma: JAXArray = jnp.ones(())

    def p(self, X: JAXArray) -> JAXArray:
        return jnp.array([self.sigma, 0])

    def q(self, X: JAXArray) -> JAXArray:
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

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        def underdamped(tau: JAXArray) -> JAXArray:
            f = jnp.sqrt(jnp.maximum(4 * jnp.square(q) - 1, 0))
            arg = 0.5 * f * w * tau / q
            return jnp.cos(arg) + jnp.sin(arg) / f

        def overdamped(tau: JAXArray) -> JAXArray:
            f = jnp.sqrt(jnp.maximum(1 - 4 * jnp.square(q), 0))
            arg = 0.5 * f * w * tau / q
            return jnp.cosh(arg) + jnp.sinh(arg) / f

        s2 = jnp.square(self.sigma)
        w = self.omega
        q = self.quality
        tau = jnp.abs(X1 - X2)
        return (
            s2
            * jnp.exp(-0.5 * w * tau / q)
            * jax.lax.cond(
                jnp.allclose(q, 0.5),
                lambda tau: 1 + w * tau,
                lambda tau: jax.lax.cond(
                    q > 0.5, underdamped, overdamped, tau
                ),
                tau,
            )
        )


@dataclass
class Exp(Quasisep):
    scale: JAXArray
    sigma: JAXArray = jnp.ones(())

    def p(self, X: JAXArray) -> JAXArray:
        return jnp.array([self.sigma])

    def q(self, X: JAXArray) -> JAXArray:
        return jnp.array([self.sigma])

    def A(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        dt = X2 - X1
        return jnp.exp(-dt[None, None] / self.scale)

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        tau = jnp.abs(X1 - X2)
        return jnp.square(self.sigma) * jnp.exp(-tau / self.scale)


# Notes for deriving relevant matrices:
# See also: https://github.com/SheffieldML/GPy/blob/bb1bc5088671f9316bc92a46d356734e34c2d5c0/GPy/kern/src/sde_matern.py
#
# => For Matern-3/2:
#
# import sympy as sm
#
# f, s, t = sm.symbols("f, s, t", real=True)
# F = sm.Matrix([[0, 1], [-f**2, -2*f]])
# L = sm.Matrix([[0], [1]])
# phi = sm.simplify((F * t).exp())
# P = sm.Matrix([
#     sm.symbols("p1, p2"),
#     sm.symbols("p3, p4"),
# ])
# Pinf = sm.solve(F @ P + P @ F.T + 4 * f**3 * s * L @ L.T, P)
#
# => For Matern-5/2:
#
# F = sm.Matrix([[0, 1, 0], [0, 0, 1], [-f**3, -3*f**2, -3*f]])
# and
# Pinf = [[sigma, 0, -sigma * f^2 / 3], ...]


@dataclass
class Matern32(Quasisep):
    scale: JAXArray
    sigma: JAXArray = jnp.ones(())

    def p(self, X: JAXArray) -> JAXArray:
        return jnp.array([self.sigma, 0])

    def q(self, X: JAXArray) -> JAXArray:
        return jnp.array([self.sigma, 0])

    def A(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        dt = X2 - X1
        f = np.sqrt(3) / self.scale
        return jnp.exp(-f * dt) * jnp.array(
            [[1 + f * dt, dt], [-jnp.square(f) * dt, 1 - f * dt]]
        )

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        r = jnp.abs(X1 - X2) / self.scale
        arg = np.sqrt(3) * r
        return jnp.square(self.sigma) * (1 + arg) * jnp.exp(-arg)


@dataclass
class Matern52(Quasisep):
    scale: JAXArray
    sigma: JAXArray = jnp.ones(())

    def p(self, X: JAXArray) -> JAXArray:
        return jnp.array(
            [self.sigma, 0, -5 * self.sigma / (3 * jnp.square(self.scale))]
        )

    def q(self, X: JAXArray) -> JAXArray:
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

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        r = jnp.abs(X1 - X2) / self.scale
        arg = np.sqrt(5) * r
        return (
            jnp.square(self.sigma)
            * (1 + arg + jnp.square(arg) / 3)
            * jnp.exp(-arg)
        )
