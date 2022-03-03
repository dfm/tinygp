# -*- coding: utf-8 -*-
"""
This submodule defines a set of distance metrics that can be used when working
with multivariate data. By default, all
:class:`tinygp.kernels.stationary.Stationary` kernels will use either an
:class:`L1Distance` or :class:`L2Distance`, when applied in multiple dimensions,
but it is possible to define custom metrics, as dicussed in the :ref:`geometry`
tutorial.
"""

from __future__ import annotations

__all__ = ["Distance", "L1Distance", "L2Distance"]

from abc import ABCMeta, abstractmethod

import jax.numpy as jnp

from tinygp.helpers import JAXArray, dataclass


class Distance(metaclass=ABCMeta):
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


@dataclass
class L1Distance(Distance):
    """The L1 or Manhattan distance between two coordinates"""

    def distance(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        return jnp.sum(jnp.abs(X1 - X2))


@dataclass
class L2Distance(Distance):
    """The L2 or Euclidean distance bettwen two coordaintes"""

    def distance(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        return jnp.sqrt(self.squared_distance(X1, X2))

    def squared_distance(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        return jnp.sum(jnp.square(X1 - X2))
