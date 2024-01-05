"""
This submodule defines a set of distance metrics that can be used when working
with multivariate data. By default, all
:class:`tinygp.kernels.stationary.Stationary` kernels will use either an
:class:`L1Distance` or :class:`L2Distance`, when applied in multiple dimensions,
but it is possible to define custom metrics, as discussed in the :ref:`geometry`
tutorial.
"""

from __future__ import annotations

__all__ = ["Distance", "L1Distance", "L2Distance"]

from abc import abstractmethod

import equinox as eqx
import jax.numpy as jnp

from tinygp.helpers import JAXArray


class Distance(eqx.Module):
    """An abstract base class defining a distance metric interface"""

    @abstractmethod
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
    """The L2 or Euclidean distance between two coordinates"""

    def distance(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        r1 = L1Distance().distance(X1, X2)
        r2 = self.squared_distance(X1, X2)
        zeros = jnp.equal(r2, 0)
        r2 = jnp.where(zeros, jnp.ones_like(r2), r2)
        return jnp.where(zeros, r1, jnp.sqrt(r2))

    def squared_distance(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        return jnp.sum(jnp.square(X1 - X2))
