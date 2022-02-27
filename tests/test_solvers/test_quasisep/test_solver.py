# -*- coding: utf-8 -*-
# mypy: ignore-errors

import jax
import numpy as np
import pytest

from tinygp import GaussianProcess
from tinygp import kernels as base_kernels
from tinygp.solvers import DirectSolver, QuasisepSolver
from tinygp.solvers.quasisep import kernels


@pytest.fixture
def random():
    return np.random.default_rng(84930)


@pytest.fixture
def data(random):
    x = np.sort(random.uniform(-3, 3, 50))
    y = np.sin(x)
    return x, y


@pytest.fixture(
    params=[
        (
            kernels.Matern32(sigma=1.8, scale=1.5),
            1.8**2 * base_kernels.Matern32(1.5),
        ),
        (
            1.8**2 * kernels.Matern32(1.5),
            1.8**2 * base_kernels.Matern32(1.5),
        ),
        (
            kernels.Matern52(sigma=1.8, scale=1.5),
            1.8**2 * base_kernels.Matern52(1.5),
        ),
        (
            kernels.Exp(sigma=1.8, scale=1.5),
            1.8**2 * base_kernels.Exp(1.5),
        ),
    ]
)
def kernel_pair(request):
    return request.param


def test_consistent_with_direct(kernel_pair, data):
    kernel1, kernel2 = kernel_pair
    x, y = data
    gp1 = GaussianProcess(kernel1, x, diag=0.1, solver=QuasisepSolver)
    gp2 = GaussianProcess(kernel2, x, diag=0.1, solver=DirectSolver)

    np.testing.assert_allclose(gp1.covariance.to_dense(), gp2.covariance)
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
