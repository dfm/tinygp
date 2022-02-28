# -*- coding: utf-8 -*-
# mypy: ignore-errors

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tinygp import kernels
from tinygp.solvers import DirectSolver, solver


@pytest.fixture
def random():
    return np.random.default_rng(1058390)


@pytest.fixture
def data(random):
    x1 = random.uniform(-3, 3, (50, 5))
    x2 = random.uniform(-5, 5, (50, 5))
    return x1, x2


def test_constant(data):
    x1, x2 = data

    # Check for dimension issues, either when instantiated...
    with pytest.raises(ValueError):
        kernels.Constant(jnp.ones(3))

    # ... or when multiplied
    with pytest.raises(ValueError):
        jnp.ones(3) * kernels.Matern32(1.5)

    # Check that multiplication has the expected behavior
    factor = 2.5
    k1 = kernels.Matern32(2.5)
    np.testing.assert_allclose(factor * k1(x1, x2), (factor * k1)(x1, x2))


def test_custom(data):
    x1, x2 = data

    # Check that known kernels work as expected
    scale = 1.5
    k1 = kernels.Custom(
        lambda X1, X2: jnp.exp(-0.5 * jnp.sum(jnp.square((X1 - X2) / scale)))
    )
    k2 = kernels.ExpSquared(scale)
    np.testing.assert_allclose(k1(x1, x2), k2(x1, x2))

    # Check that an invalid kernel raises as expected
    kernel = kernels.Custom(
        lambda X1, X2: jnp.exp(-0.5 * jnp.square((X1 - X2) / scale))
    )
    with pytest.raises(ValueError):
        kernel(x1, x2)


def test_ops(data):
    x1, x2 = data

    k1 = 1.5 * kernels.Matern32(2.5)
    k2 = 0.9 * kernels.ExpSineSquared(scale=1.5, gamma=0.3)

    np.testing.assert_allclose(k1(x1, x2) + k2(x1, x2), (k1 + k2)(x1, x2))
    np.testing.assert_allclose(k1(x1, x2) * k2(x1, x2), (k1 * k2)(x1, x2))


def test_conditioned(data):
    x1, x2 = data
    with jax.experimental.enable_x64():
        k1 = 1.5 * kernels.Matern32(2.5)
        k2 = 0.9 * kernels.ExpSineSquared(scale=1.5, gamma=0.3)
        K = k1(x1, x1) + 0.1 * jnp.eye(x1.shape[0])
        solver = DirectSolver.init(k1, x1, 0.1)
        cond = kernels.Conditioned(x1, solver, k2)
        np.testing.assert_allclose(
            cond(x1, x2),
            k2(x1, x2) - k2(x1, x1) @ jnp.linalg.solve(K, k2(x1, x2)),
        )
