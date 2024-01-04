# mypy: ignore-errors

import jax
import jax.numpy as jnp
import pytest
from numpy import random as np_random

from tinygp import GaussianProcess, kernels
from tinygp.test_utils import assert_allclose


@pytest.fixture
def random():
    return np_random.default_rng(1058390)


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
        assert_allclose(jnp.mean(y, axis=0), jnp.sum(X, axis=1), atol=0.015)
        assert_allclose(jnp.cov(y, rowvar=False), gp.covariance, atol=0.015)


def test_means(data):
    X, y = data

    gp1 = GaussianProcess(kernels.Matern32(1.5), X, diag=0.01, mean=lambda x: 0.0)
    gp2 = GaussianProcess(kernels.Matern32(1.5), X, diag=0.01, mean=0.0)
    gp3 = GaussianProcess(kernels.Matern32(1.5), X, diag=0.01)

    assert_allclose(gp1.mean, gp2.mean)
    assert_allclose(gp1.mean, gp3.mean)
    assert_allclose(gp1.log_probability(y), gp2.log_probability(y))
    assert_allclose(gp1.log_probability(y), gp3.log_probability(y))


@pytest.mark.parametrize("tree", [True, False])
def test_condition_shape_error(data, tree):
    if tree:

        class CustomDistance(kernels.Distance):
            def distance(self, X1, X2):
                return kernels.L2Distance().distance(X1["x"], X2["x"])

        distance = CustomDistance()
    else:
        distance = kernels.L2Distance()

    X, y = data
    kernel = kernels.ExpSquared(distance=distance)
    gp = GaussianProcess(kernel, {"x": X} if tree else X, diag=0.1)
    gp.condition(y, {"x": X[0][None]} if tree else X[0][None])

    with pytest.raises(ValueError):
        gp.condition(y, X[0])

    if tree:
        with pytest.raises(ValueError):
            gp.condition(y, {"x": X[0]})
