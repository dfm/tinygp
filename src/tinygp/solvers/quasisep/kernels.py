# -*- coding: utf-8 -*-

from __future__ import annotations

__all__ = ["Quasisep"]

from abc import ABCMeta, abstractmethod
from typing import Union

import jax
import jax.numpy as jnp
import numpy as np

from tinygp.helpers import JAXArray, dataclass
from tinygp.kernels.base import Kernel
from tinygp.solvers.quasisep.core import DiagQSM, StrictLowerTriQSM, SymmQSM


class Quasisep(Kernel, metaclass=ABCMeta):
    @abstractmethod
    def to_qsm(self, X: JAXArray) -> SymmQSM:
        raise NotImplementedError

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

    def to_qsm(self, X: JAXArray) -> SymmQSM:
        return self.kernel1.to_qsm(X) + self.kernel2.to_qsm(X)

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        return self.kernel1.evaluate(X1, X2) + self.kernel2.evaluate(X1, X2)


class Product(Quasisep):
    def __init__(self, kernel1: Quasisep, kernel2: Quasisep):
        self.kernel1 = kernel1
        self.kernel2 = kernel2

    def to_qsm(self, X: JAXArray) -> SymmQSM:
        return self.kernel1.to_qsm(X) * self.kernel2.to_qsm(X)

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        return self.kernel1.evaluate(X1, X2) * self.kernel2.evaluate(X1, X2)


class Scale(Quasisep):
    def __init__(self, scale: JAXArray, kernel: Quasisep):
        self.scale = scale
        self.kernel = kernel

    def to_qsm(self, X: JAXArray) -> SymmQSM:
        return self.kernel.to_qsm(X) * self.scale

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        return self.kernel.evaluate(X1, X2) * self.scale


@dataclass
class Matern32(Quasisep):
    scale: JAXArray

    def to_qsm(self, X: JAXArray) -> SymmQSM:
        if jnp.ndim(X) != 1:
            raise ValueError("Only 1D inputs are supported")

        f = np.sqrt(3) / self.scale
        dt = jnp.append(0, jnp.diff(X))
        a = jax.vmap(
            lambda dt: jnp.exp(-f * dt)
            * jnp.array(  # type: ignore
                [[1 + f * dt, dt], [-jnp.square(f) * dt, 1 - f * dt]]
            )
        )(dt)
        p = a[:, 0, :]
        q = jnp.stack((jnp.ones_like(dt), jnp.zeros_like(dt)), axis=-1)

        return SymmQSM(
            diag=DiagQSM(d=jnp.ones_like(dt)),
            lower=StrictLowerTriQSM(p=p, q=q, a=a),
        )

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        r = jnp.abs(X1 - X2) / self.scale
        arg = np.sqrt(3) * r
        return (1 + arg) * jnp.exp(-arg)


@dataclass
class Matern52(Quasisep):
    scale: JAXArray

    def to_qsm(self, X: JAXArray) -> SymmQSM:
        if jnp.ndim(X) != 1:
            raise ValueError("Only 1D inputs are supported")

        @jax.vmap
        def impl(dt):  # type: ignore
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

        f = np.sqrt(5) / self.scale
        dt = jnp.append(0, jnp.diff(X))
        a = impl(dt)  # type: ignore

        # In SDE notation, p_k = (1, 0, 0) @ P_inf @ phi_k
        p = a[:, 0, :] - jnp.square(f) * a[:, 2, :] / 3
        q = jnp.stack(
            (jnp.ones_like(dt), jnp.zeros_like(dt), jnp.zeros_like(dt)),
            axis=-1,
        )

        return SymmQSM(
            diag=DiagQSM(d=jnp.ones_like(dt)),
            lower=StrictLowerTriQSM(p=p, q=q, a=a),
        )

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        r = jnp.abs(X1 - X2) / self.scale
        arg = np.sqrt(5) * r
        return (1 + arg + jnp.square(arg) / 3) * jnp.exp(-arg)


@dataclass
class Celerite(Quasisep):
    a: JAXArray
    b: JAXArray
    c: JAXArray
    d: JAXArray

    def to_qsm(self, X: JAXArray) -> SymmQSM:
        if jnp.ndim(X) != 1:
            raise ValueError("Only 1D inputs are supported")

        @jax.vmap
        def impl(dt):  # type: ignore
            cos = jnp.cos(self.d * dt)
            sin = jnp.sin(self.d * dt)
            return jnp.exp(-self.c * dt) * jnp.array([[cos, -sin], [sin, cos]])

        dt = jnp.append(0, jnp.diff(X))
        a = impl(dt)  # type: ignore
        p = self.a * a[:, 0, :] + self.b * a[:, 1, :]
        q = jnp.stack((jnp.ones_like(dt), jnp.zeros_like(dt)), axis=-1)

        return SymmQSM(
            diag=DiagQSM(d=jnp.full_like(dt, self.a)),
            lower=StrictLowerTriQSM(p=p, q=q, a=a),
        )

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        tau = jnp.abs(X1 - X2)
        return jnp.exp(-self.c * tau) * (
            self.a * jnp.cos(self.d * tau) + self.b * jnp.sin(self.d * tau)
        )
