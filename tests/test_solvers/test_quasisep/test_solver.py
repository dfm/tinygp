# -*- coding: utf-8 -*-
# mypy: ignore-errors

import jax
import numpy as np
import pytest

from tinygp import GaussianProcess, kernels
from tinygp.kernels import quasisep
from tinygp.solvers import DirectSolver, QuasisepSolver


@pytest.fixture
def random():
    return np.random.default_rng(84930)


@pytest.fixture
def data(random):
    x = np.sort(random.uniform(-3, 3, 50))
    y = np.sin(x)
    t = np.sort(random.uniform(-3, 3, 10))
    return x, y, t


@pytest.fixture(
    params=[
        (
            quasisep.Matern32(sigma=1.8, scale=1.5),
            1.8**2 * kernels.Matern32(1.5),
        ),
        (
            1.8**2 * quasisep.Matern32(1.5),
            1.8**2 * kernels.Matern32(1.5),
        ),
        (
            quasisep.Matern52(sigma=1.8, scale=1.5),
            1.8**2 * kernels.Matern52(1.5),
        ),
        (
            quasisep.Exp(sigma=1.8, scale=1.5),
            1.8**2 * kernels.Exp(1.5),
        ),
    ]
)
def kernel_pair(request):
    return request.param


def test_consistent_with_direct(kernel_pair, data):
    kernel0 = quasisep.Matern32(sigma=3.8, scale=4.5)
    kernel1, kernel2 = kernel_pair
    x, y, t = data
    gp1 = GaussianProcess(kernel1, x, diag=0.1, solver=QuasisepSolver)
    gp2 = GaussianProcess(kernel2, x, diag=0.1, solver=DirectSolver)

    np.testing.assert_allclose(gp1.covariance, gp2.covariance)
    np.testing.assert_allclose(
        gp1.solver.normalization(), gp2.solver.normalization()
    )
    np.testing.assert_allclose(gp1.log_probability(y), gp2.log_probability(y))
    np.testing.assert_allclose(
        gp1.sample(jax.random.PRNGKey(0)), gp2.sample(jax.random.PRNGKey(0))
    )
    np.testing.assert_allclose(
        gp1.sample(jax.random.PRNGKey(0), shape=(5, 7)),
        gp2.sample(jax.random.PRNGKey(0), shape=(5, 7)),
    )

    gp1p = gp1.condition(y)
    gp2p = gp2.condition(y)
    assert isinstance(gp1p.gp.solver, QuasisepSolver)
    np.testing.assert_allclose(gp1p.log_probability, gp2p.log_probability)
    np.testing.assert_allclose(gp1p.gp.loc, gp2p.gp.loc)
    np.testing.assert_allclose(gp1p.gp.variance, gp2p.gp.variance)
    np.testing.assert_allclose(
        gp1p.gp.covariance, gp2p.gp.covariance, atol=1e-7
    )

    gp1p = gp1.condition(y, kernel=kernel0)
    gp2p = gp2.condition(y, kernel=kernel0)
    assert isinstance(gp1p.gp.solver, QuasisepSolver)
    np.testing.assert_allclose(gp1p.log_probability, gp2p.log_probability)
    np.testing.assert_allclose(gp1p.gp.loc, gp2p.gp.loc)
    np.testing.assert_allclose(gp1p.gp.variance, gp2p.gp.variance)
    np.testing.assert_allclose(
        gp1p.gp.covariance, gp2p.gp.covariance, atol=1e-7
    )

    gp1p = gp1.condition(y, X_test=t, kernel=kernel0)
    gp2p = gp2.condition(y, X_test=t, kernel=kernel0)
    assert not isinstance(gp1p.gp.solver, QuasisepSolver)
    np.testing.assert_allclose(gp1p.log_probability, gp2p.log_probability)
    np.testing.assert_allclose(gp1p.gp.loc, gp2p.gp.loc)
    np.testing.assert_allclose(gp1p.gp.variance, gp2p.gp.variance)
    np.testing.assert_allclose(
        gp1p.gp.covariance, gp2p.gp.covariance, atol=1e-7
    )
