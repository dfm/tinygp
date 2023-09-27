# mypy: ignore-errors

from __future__ import annotations

__all__ = ["TinyDistribution"]

import jax.numpy as jnp
from numpyro.distributions import constraints
from numpyro.distributions.distribution import Distribution
from numpyro.distributions.util import (
    is_prng_key,
    lazy_property,
    validate_sample,
)


class TinyDistribution(Distribution):
    """Blah"""

    support = constraints.real_vector

    def __init__(self, gp, validate_args=None):
        self.gp = gp
        self.loc = gp.loc
        batch_shape = ()
        event_shape = jnp.shape(self.loc)
        super().__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        return self.gp.sample(key, shape=sample_shape)

    @validate_sample
    def log_prob(self, value):
        return self.gp.log_probability(value)

    @lazy_property
    def covariance_matrix(self):
        return self.gp.covariance

    @lazy_property
    def precision_matrix(self):
        return self.gp.solver.solve_triangular(
            self.gp.solver.solve_triangular(jnp.eye(self.loc.shape[0])),
            transpose=True,
        )

    @property
    def mean(self):
        return self.loc

    @property
    def variance(self):
        return self.gp.variance

    def tree_flatten(self):
        return self.gp, None

    @classmethod
    def tree_unflatten(cls, aux_data, params):
        return cls(params)

    @staticmethod
    def infer_shapes(*args):
        raise NotImplementedError
