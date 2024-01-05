# mypy: ignore-errors

import jax
import jax.numpy as jnp
import pytest
from numpy import random as np_random

from tinygp import GaussianProcess, kernels
from tinygp.kernels import quasisep
from tinygp.solvers import DirectSolver, QuasisepSolver
from tinygp.test_utils import assert_allclose


@pytest.fixture
def random():
    return np_random.default_rng(84930)


@pytest.fixture
def data(random):
    x = jnp.sort(random.uniform(-3, 3, 50))
    y = jnp.sin(x)
    t = jnp.sort(random.uniform(-3, 3, 10))
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
        (
            quasisep.Cosine(sigma=1.8, scale=1.5),
            1.8**2 * kernels.Cosine(1.5),
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

    assert_allclose(gp1.covariance, gp2.covariance)
    assert_allclose(gp1.solver.normalization(), gp2.solver.normalization())
    assert_allclose(gp1.log_probability(y), gp2.log_probability(y))
    assert_allclose(
        gp1.sample(jax.random.PRNGKey(0)), gp2.sample(jax.random.PRNGKey(0))
    )
    assert_allclose(
        gp1.sample(jax.random.PRNGKey(0), shape=(5, 7)),
        gp2.sample(jax.random.PRNGKey(0), shape=(5, 7)),
    )

    gp1p = gp1.condition(y)
    gp2p = gp2.condition(y)
    assert isinstance(gp1p.gp.solver, QuasisepSolver)
    assert_allclose(gp1p.log_probability, gp2p.log_probability)
    assert_allclose(gp1p.gp.loc, gp2p.gp.loc)
    assert_allclose(gp1p.gp.variance, gp2p.gp.variance)
    assert_allclose(gp1p.gp.covariance, gp2p.gp.covariance)

    gp1p = gp1.condition(y, kernel=kernel0)
    gp2p = gp2.condition(y, kernel=kernel0)
    assert isinstance(gp1p.gp.solver, QuasisepSolver)
    assert_allclose(gp1p.log_probability, gp2p.log_probability)
    assert_allclose(gp1p.gp.loc, gp2p.gp.loc)
    assert_allclose(gp1p.gp.variance, gp2p.gp.variance)
    assert_allclose(gp1p.gp.covariance, gp2p.gp.covariance)

    gp1p = gp1.condition(y, X_test=t, kernel=kernel0)
    gp2p = gp2.condition(y, X_test=t, kernel=kernel0)
    assert not isinstance(gp1p.gp.solver, QuasisepSolver)
    assert_allclose(gp1p.log_probability, gp2p.log_probability)
    assert_allclose(gp1p.gp.loc, gp2p.gp.loc)
    assert_allclose(gp1p.gp.variance, gp2p.gp.variance)
    assert_allclose(gp1p.gp.covariance, gp2p.gp.covariance)


def test_celerite(data):
    celerite = pytest.importorskip("celerite")

    x, y, _ = data
    yerr = 0.1

    a, b, c, d = 1.1, 0.8, 0.9, 0.1
    celerite_kernel = celerite.terms.ComplexTerm(
        jnp.log(a), jnp.log(b), jnp.log(c), jnp.log(d)
    )
    celerite_gp = celerite.GP(celerite_kernel)
    celerite_gp.compute(x, yerr)
    expected = celerite_gp.log_likelihood(y)

    kernel = quasisep.Celerite(a, b, c, d)
    gp = GaussianProcess(kernel, x, diag=yerr**2)
    calc = gp.log_probability(y)

    assert_allclose(calc, expected)


def test_unsorted(data):
    random = np_random.default_rng(0)
    inds = random.permutation(len(data[0]))
    x_ = data[0][inds]
    y_ = data[1][inds]

    kernel = quasisep.Matern32(sigma=1.8, scale=1.5)
    with pytest.raises(ValueError):
        GaussianProcess(kernel, x_, diag=0.1)

    @jax.jit
    def impl(X, y):
        return GaussianProcess(kernel, X, diag=0.1).log_probability(y)

    with pytest.raises(jax.lib.xla_extension.XlaRuntimeError) as exc_info:
        impl(x_, y_).block_until_ready()
    assert exc_info.match(r"Input coordinates must be sorted")
