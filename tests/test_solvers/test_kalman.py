# mypy: ignore-errors

import jax
import jax.numpy as jnp
import pytest
from numpy import random as np_random

from tinygp import GaussianProcess
from tinygp.kernels import quasisep
from tinygp.solvers import QuasisepSolver
from tinygp.solvers.kalman import KalmanSolver, kalman_filter, kalman_gains
from tinygp.test_utils import assert_allclose


@pytest.fixture
def random():
    return np_random.default_rng(84930)


@pytest.fixture
def data(random):
    x = jnp.sort(random.uniform(-3, 3, 50))
    y = jnp.sin(x)
    return x, y


@pytest.fixture(
    params=[
        quasisep.Matern32(sigma=1.8, scale=1.5),
        1.8**2 * quasisep.Matern32(1.5),
        quasisep.Matern52(sigma=1.8, scale=1.5),
        quasisep.Exp(sigma=1.8, scale=1.5),
        quasisep.Cosine(sigma=1.8, scale=1.5),
        quasisep.SHO(sigma=1.8, omega=1.5, quality=3.0),
        quasisep.SHO(sigma=1.8, omega=1.5, quality=0.2),
        quasisep.Celerite(1.1, 0.8, 0.9, 0.1),
        1.5 * quasisep.Matern52(1.5) + 0.3 * quasisep.Exp(1.5),
        quasisep.Matern52(1.5) * quasisep.SHO(omega=1.5, quality=0.1),
        1.5 * quasisep.Matern52(1.5) * quasisep.Celerite(1.1, 0.8, 0.9, 0.1),
        quasisep.CARMA(alpha=jnp.array([1.4, 2.3, 1.5]), beta=jnp.array([0.1, 0.5])),
    ]
)
def kernel(request):
    return request.param


def test_filter(kernel, data):
    x, y = data
    diag = jnp.full_like(x, 0.1)

    logp0 = GaussianProcess(kernel, x, diag=diag).log_probability(y)

    Pinf = kernel.stationary_covariance()
    A = jax.vmap(kernel.transition_matrix)(jnp.append(x[0], x[:-1]), x)
    H = jax.vmap(kernel.observation_model)(x)
    s, K = kalman_gains(Pinf, A, H, diag)
    v = kalman_filter(A, H, K, y)
    logp = -0.5 * jnp.sum(jnp.square(v) / s + jnp.log(2 * jnp.pi * s))

    assert_allclose(logp, logp0)


def test_consistent_with_direct(kernel, data):
    x, y = data
    gp1 = GaussianProcess(kernel, x, diag=0.1, solver=KalmanSolver)
    gp2 = GaussianProcess(kernel, x, diag=0.1, solver=QuasisepSolver)

    assert_allclose(gp1.log_probability(y), gp2.log_probability(y))
    assert_allclose(gp1.solver.normalization(), gp2.solver.normalization())
