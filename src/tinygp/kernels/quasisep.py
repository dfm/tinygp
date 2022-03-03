# -*- coding: utf-8 -*-
"""
The kernels implemented in this subpackage are used with the
:class:`tinygp.solvers.QuasisepSolver` to allow scalable GP computations by
exploting quasiseparable structure in the relevant matrices (see
:ref:`api-solvers-quasisep` for more technical details). For now, these methods
are experimental, so you may find the documentation patchy in places. You are
encouraged to `open issues or pull requests
<https://github.com/dfm/tinygp/issues>`_ as you find gaps.
"""

from __future__ import annotations

__all__ = [
    "Quasisep",
    "Sum",
    "Product",
    "Scale",
    "Celerite",
    "SHO",
    "Exp",
    "Matern32",
    "Matern52",
]

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
    """The base class for all quasiseparable kernels

    Instead of directly implementing the ``p``, ``q``, and ``a`` elements of the
    :class:`tinygp.solvers.quasisep.core.StrictLowerQSM`, this class implements
    ``h``, ``Pinf``, and ``A``, where:

    - ``q = h``,
    - ``p = h.T @ Pinf @ A``, and
    - ``a = A``.

    This notation follows the notation from state space models for stochastic
    differential equations, and so far it seems like a good way to specify these
    models, but these details are subject to change in future versions of
    ``tinygp``.
    """

    @abstractmethod
    def Pinf(self) -> JAXArray:
        """The stationary covariance of the process"""
        raise NotImplementedError

    @abstractmethod
    def h(self, X: JAXArray) -> JAXArray:
        """The 'observation model' for the process"""
        raise NotImplementedError

    @abstractmethod
    def A(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        """The transition matrix between two neighboring coordinates"""
        raise NotImplementedError

    def coord_to_sortable(self, X: JAXArray) -> JAXArray:
        """A helper function used to convert coordinates to sortable 1-D values

        By default, this is the identity, but in cases where ``X`` is structured
        (e.g. multivariate inputs), this can be used to appropriately unwrap
        that structure.
        """
        return X

    def to_symm_qsm(self, X: JAXArray) -> SymmQSM:
        """The symmetric quasiseparable representation of this kernel"""
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
        """The generalized quasiseparable representation of this kernel"""
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
        """The kernel evaluated via the quasiseparable representation"""
        Pinf = self.Pinf()
        h1 = self.h(X1)
        h2 = self.h(X2)
        return jnp.where(
            X1 < X2,
            h1 @ Pinf @ self.A(X1, X2) @ h2,
            h2 @ Pinf @ self.A(X2, X1) @ h1,
        )

    def evaluate_diag(self, X: JAXArray) -> JAXArray:
        """For quasiseparable kernels, the variance is simple to compute"""
        h = self.h(X)
        return h @ self.Pinf() @ h


class Sum(Quasisep):
    """A helper to represent the sum of two quasiseparable kernels"""

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
    """A helper to represent the product of two quasiseparable kernels"""

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
    """The product of a scalar and a quasiseparable kernel"""

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
    r"""The baseline kernel from the ``celerite`` package

    This form of the kernel was introduced by `Foreman-Mackey et al. (2017)
    <https://arxiv.org/abs/1703.09710>`_, and implemented in the `celerite
    <https://celerite.readthedocs.io>`_ package. It shouldn't generally be used
    on its own, and other kernels described in this subpackage should generally
    be preferred.

    This kernel takes the form:

    .. math::

        k(\tau)=\exp(-c\,\tau)\,\left[a\,\cos(d\,\tau)+b\,\sin(d\,\tau)\right]

    for :math:`\tau = |x_i - x_j|`.
    """
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
    r"""The damped, driven simple harmonic oscillator kernel

    This form of the kernel was introduced by `Foreman-Mackey et al. (2017)
    <https://arxiv.org/abs/1703.09710>`_, and it takes the form:

    .. math::

        k(\tau) = \sigma^2\,\exp\left(-\frac{\omega\,\tau}{2\,Q}\right)
        \left\{\begin{array}{ll}
            1 + \omega\,\tau & \mbox{for } Q = 1/2 \\
            \cosh(f\,\omega\,\tau/2\,Q) + \sinh(f\,\omega\,\tau/2\,Q)/f
                & \mbox{for } Q < 1/2 \\
            \cos(g\,\omega\,\tau/2\,Q) + \sin(g\,\omega\,\tau/2\,Q)/g
                & \mbox{for } Q > 1/2
        \end{array}\right.

    for :math:`\tau = |x_i - x_j|`, :math:`f = \sqrt{1 - 4\,Q^2}`, and
    :math:`g = \sqrt{4\,Q^2 - 1}`.

    Args:
        omega: The parameter :math:`\omega`.
        quality: The parameter :math:`Q`.
        sigma: The parameter :math:`\sigma`.
    """

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
    r"""A scalable implementation of :class:`tinygp.kernels.stationary.Exp`

    This kernel takes the form:

    .. math::

        k(\tau)=\sigma^2\,\exp\left(-\frac{\tau}{\ell}\right)

    for :math:`\tau = |x_i - x_j|`.

    Args:
        scale: The parameter :math:`\ell`.
        sigma: The parameter :math:`\sigma`.
    """

    scale: JAXArray
    sigma: JAXArray = field(default_factory=lambda: jnp.ones(()))

    def Pinf(self) -> JAXArray:
        return jnp.ones((1, 1))

    def h(self, X: JAXArray) -> JAXArray:
        return jnp.array([self.sigma])

    def A(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        dt = X2 - X1
        return jnp.exp(-dt[None, None] / self.scale)


@dataclass
class Matern32(Quasisep):
    r"""A scalable implementation of :class:`tinygp.kernels.stationary.Matern32`

    This kernel takes the form:

    .. math::

        k(\tau)=\sigma^2\,\left(1+f\,\tau\right)\,\exp(-f\,\tau)

    for :math:`\tau = |x_i - x_j|` and :math:`f = \sqrt{3} / \ell`.

    Args:
        scale: The parameter :math:`\ell`.
        sigma: The parameter :math:`\sigma`.
    """
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
    r"""A scalable implementation of :class:`tinygp.kernels.stationary.Matern52`

    This kernel takes the form:

    .. math::

        k(\tau)=\sigma^2\,\left(1+f\,\tau + \frac{f^2\,\tau^2}{3}\right)
            \,\exp(-f\,\tau)

    for :math:`\tau = |x_i - x_j|` and :math:`f = \sqrt{5} / \ell`.

    Args:
        scale: The parameter :math:`\ell`.
        sigma: The parameter :math:`\sigma`.
    """

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
