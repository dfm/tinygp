# -*- coding: utf-8 -*-
# mypy: ignore-errors

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tinygp import GaussianProcess, kernels


@pytest.fixture
def random():
    return np.random.default_rng(1058390)


@pytest.fixture
def data(random):
    X = random.uniform(-3, 3, (50, 5))
    y = random.normal(len(X))
    return X, y


def test_sample(data):
    X, _ = data

    with jax.experimental.enable_x64(True):
        gp = GaussianProcess(
            kernels.Matern32(1.5), X, diag=0.01, mean=lambda x: jnp.sum(x)
        )
        y = gp.sample(jax.random.PRNGKey(543))
        assert y.shape == (len(X),)

        y = gp.sample(jax.random.PRNGKey(543), shape=(7, 3))
        assert y.shape == (7, 3, len(X))

        y = gp.sample(jax.random.PRNGKey(543), shape=(100_000,))
        assert y.shape == (100_000, len(X))
        np.testing.assert_allclose(
            jnp.mean(y, axis=0), jnp.sum(X, axis=1), atol=0.015
        )
        np.testing.assert_allclose(
            jnp.cov(y, rowvar=0), gp.covariance, atol=0.015
        )


def test_means(data):
    X, y = data

    gp1 = GaussianProcess(
        kernels.Matern32(1.5), X, diag=0.01, mean=lambda x: 0.0
    )
    gp2 = GaussianProcess(kernels.Matern32(1.5), X, diag=0.01, mean=0.0)
    gp3 = GaussianProcess(kernels.Matern32(1.5), X, diag=0.01)

    np.testing.assert_allclose(gp1.mean, gp2.mean)
    np.testing.assert_allclose(gp1.mean, gp3.mean)
    np.testing.assert_allclose(gp1.log_probability(y), gp2.log_probability(y))
    np.testing.assert_allclose(gp1.log_probability(y), gp3.log_probability(y))
