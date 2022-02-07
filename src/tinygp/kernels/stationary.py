# -*- coding: utf-8 -*-

from __future__ import annotations

__all__ = [
    "Distance",
    "L1Distance",
    "L2Distance",
    "Stationary",
    "Exp",
    "ExpSquared",
    "Matern32",
    "Matern52",
    "Cosine",
    "ExpSineSquared",
    "RationalQuadratic",
]

from typing import Optional

import jax.numpy as jnp

from tinygp.kernels import Kernel
from tinygp.types import JAXArray


class Distance:
    """An abstract base class defining a distance metric interface"""

    def distance(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        """Compute the distance between two coordinates under this metric"""
        raise NotImplementedError()

    def squared_distance(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        """Compute the squared distance between two coordinates

        By default this returns the squared result of
        :func:`tinygp.kernels.stationary.Distance.distance`, but some metrics
        can take advantage of these separate implementations to avoid
        unnecessary square roots.
        """
        return jnp.square(self.distance(X1, X2))


class L1Distance(Distance):
    """The L1 or Manhattan distance between two coordinates"""

    def distance(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        return jnp.sum(jnp.abs(X1 - X2))


class L2Distance(Distance):
    """The L2 or Euclidean distance bettwen two coordaintes"""

    def distance(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        return jnp.sqrt(self.squared_distance(X1, X2))

    def squared_distance(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        return jnp.sum(jnp.square(X1 - X2))


class Stationary(Kernel):
    """A stationary kernel is defined with respect to a distance metric

    Note that a stationary kernel is *always* isotropic. If you need more
    non-isotropic length scales, wrap your kernel in a transform using
    :class:`tinygp.transforms.Linear` or :class:`tinygp.transforms.Cholesky`.

    Args:
        scale: The length scale, in the same units as ``distance`` for the
            kernel. This must be a scalar.
        distance: An object that implements ``distance`` and
            ``squared_distance`` methods. Typically a subclass of
            :class:`tinygp.kernels.stationary.Distance`. Each stationary kernel
            also has a ``default_distance`` property that is used when
            ``distance`` isn't provided.
    """

    default_distance: Distance = L1Distance()

    def __init__(
        self,
        scale: JAXArray = jnp.ones(()),
        *,
        distance: Optional[Distance] = None,
    ):
        if jnp.ndim(scale):
            raise ValueError(
                "Only scalar scales are permitted for stationary kernels; use"
                "transforms.Linear or transforms.Cholesky for more flexiblity"
            )
        self.scale = scale
        self.distance = self.default_distance if distance is None else distance


class Exp(Stationary):
    r"""The exponential kernel

    .. math::

        k(\mathbf{x}_i,\,\mathbf{x}_j) = \exp(-r)

    where, by default,

    .. math::

        r = ||(\mathbf{x}_i - \mathbf{x}_j) / \ell||_1

    Args:
        scale: The parameter :math:`\ell`.
    """

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        return jnp.exp(-self.distance.distance(X1, X2) / self.scale)


class ExpSquared(Stationary):
    r"""The exponential squared or radial basis function kernel

    .. math::

        k(\mathbf{x}_i,\,\mathbf{x}_j) = \exp(-r^2 / 2)

    where, by default,

    .. math::

        r^2 = ||(\mathbf{x}_i - \mathbf{x}_j) / \ell||_2^2

    Args:
        scale: The parameter :math:`\ell`.
    """
    default_distance: Distance = L2Distance()

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        r2 = self.distance.squared_distance(X1, X2) / jnp.square(self.scale)
        return jnp.exp(-0.5 * r2)


class Matern32(Stationary):
    r"""The Matern-3/2 kernel

    .. math::

        k(\mathbf{x}_i,\,\mathbf{x}_j) = (1 + \sqrt{3}\,r)\,\exp(-\sqrt{3}\,r)

    where, by default,

    .. math::

        r = ||(\mathbf{x}_i - \mathbf{x}_j) / \ell||_1

    Args:
        scale: The parameter :math:`\ell`.
    """

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        r = self.distance.distance(X1, X2) / self.scale
        arg = jnp.sqrt(3.0) * r
        return (1.0 + arg) * jnp.exp(-arg)


class Matern52(Stationary):
    r"""The Matern-5/2 kernel

    .. math::

        k(\mathbf{x}_i,\,\mathbf{x}_j) = (1 + \sqrt{5}\,r +
            5\,r^2/\sqrt{3})\,\exp(-\sqrt{5}\,r)

    where, by default,

    .. math::

        r = ||(\mathbf{x}_i - \mathbf{x}_j) / \ell||_1

    Args:
        scale: The parameter :math:`\ell`.
    """

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        r = self.distance.distance(X1, X2) / self.scale
        arg = jnp.sqrt(5.0) * r
        return (1.0 + arg + jnp.square(arg) / 3.0) * jnp.exp(-arg)


class Cosine(Stationary):
    r"""The cosine kernel

    .. math::

        k(\mathbf{x}_i,\,\mathbf{x}_j) = \cos(2\,\pi\,r)

    where, by default,

    .. math::

        r = ||(\mathbf{x}_i - \mathbf{x}_j) / P||_1

    Args:
        period: The parameter :math:`P`.
    """

    def __init__(
        self, period: JAXArray, *, distance: Optional[Distance] = None
    ):
        super().__init__(scale=period, distance=distance)
        self.period = self.scale

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        r = self.distance.distance(X1, X2) / self.period
        return jnp.cos(2 * jnp.pi * r)


class ExpSineSquared(Stationary):
    r"""The exponential sine squared or quasiperiodic kernel

    .. math::

        k(\mathbf{x}_i,\,\mathbf{x}_j) = \exp(-\Gamma\,\sin^2 \pi r)

    where, by default,

    .. math::

        r = ||(\mathbf{x}_i - \mathbf{x}_j) / P||_1

    Args:
        period: The parameter :math:`P`.
        gamma: The parameter :math:`\Gamma`.
    """

    def __init__(
        self,
        *,
        period: JAXArray,
        gamma: JAXArray,
        distance: Optional[Distance] = None,
    ):
        super().__init__(scale=period, distance=distance)
        self.period = self.scale
        self.gamma = gamma

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        r = self.distance.distance(X1, X2) / self.period
        return jnp.exp(-self.gamma * jnp.square(jnp.sin(jnp.pi * r)))


class RationalQuadratic(Stationary):
    r"""The rational quadratic

    .. math::

        k(\mathbf{x}_i,\,\mathbf{x}_j) = (1 + r^2 / 2\,\alpha)^{-\alpha}

    where, by default,

    .. math::

        r^2 = ||(\mathbf{x}_i - \mathbf{x}_j) / \ell||_2^2

    Args:
        scale: The parameter :math:`\ell`.
        alpha: The parameter :math:`\alpha`.
    """

    def __init__(
        self,
        *,
        alpha: JAXArray,
        scale: JAXArray = jnp.ones(()),
        distance: Optional[Distance] = None,
    ):
        super().__init__(scale=scale, distance=distance)
        self.scale = scale
        self.alpha = alpha

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        r2 = self.distance.squared_distance(X1, X2) / jnp.square(self.scale)
        return (1.0 + 0.5 * r2 / self.alpha) ** -self.alpha
