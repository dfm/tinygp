# mypy: ignore-errors

import jax
import jax.numpy as jnp
import pytest
from numpy import random as np_random

from tinygp import GaussianProcess, kernels
from tinygp.kernels import quasisep
from tinygp.noise import Banded
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
        (
            quasisep.Matern32(sigma=1.8, scale=1.5)
            + quasisep.Matern52(sigma=0.9, scale=0.7),
            1.8**2 * kernels.Matern32(1.5) + 0.9**2 * kernels.Matern52(0.7),
        ),
    ]
)
def kernel_pair(request):
    return request.param


@pytest.mark.parametrize("parallel", [False, True], ids=["sequential", "parallel"])
def test_consistent_with_direct(kernel_pair, data, parallel):
    kernel0 = quasisep.Matern32(sigma=3.8, scale=4.5)
    kernel1, kernel2 = kernel_pair
    x, y, t = data
    gp1 = GaussianProcess(
        kernel1, x, diag=0.1, solver=QuasisepSolver, parallel=parallel
    )
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
    assert gp1p.gp.solver.parallel == parallel
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

    with pytest.raises(jax.errors.JaxRuntimeError) as exc_info:
        impl(x_, y_).block_until_ready()
    assert exc_info.match(r"Input coordinates must be sorted")


@pytest.mark.parametrize(
    "kernel",
    [
        quasisep.Matern32(sigma=1.8, scale=1.5),
        quasisep.Cosine(sigma=1.2, scale=0.8),
        quasisep.Matern32(sigma=1.8, scale=1.5)
        + quasisep.SHO(omega=1.5, quality=3.0, sigma=1.2),
    ],
    ids=["matern32", "cosine", "matern32+sho"],
)
@pytest.mark.parametrize("parallel", [False, True], ids=["sequential", "parallel"])
def test_conditioned_gp_operations(kernel, random, parallel):
    # Conditioning at the training points with the default jitter gives a
    # nearly singular covariance. This checks that the conditioned GP's own
    # factorization (which used to be numerically fragile, especially with
    # the parallel algorithms) is accurate enough for downstream operations.
    with jax.enable_x64(True):
        N = 100
        x = jnp.sort(random.uniform(-3, 3, N))
        y1 = random.normal(size=N)
        y2 = random.normal(size=N)
        diag = 0.1 + 0.05 * random.uniform(size=N)

        gp1 = GaussianProcess(
            kernel, x, diag=diag, solver=QuasisepSolver, parallel=parallel
        )
        gp2 = GaussianProcess(kernel, x, diag=diag, solver=DirectSolver)
        cond1 = gp1.condition(y1)
        cond2 = gp2.condition(y1)

        # The conditioned covariance should have the same quasiseparable rank as
        # the prior, since we're conditioning with the same kernel
        assert isinstance(cond1.gp.solver, QuasisepSolver)
        assert cond1.gp.solver.parallel == parallel
        assert cond1.gp.solver.matrix.lower.p.shape == gp1.solver.matrix.lower.p.shape

        assert_allclose(cond1.gp.covariance, cond2.gp.covariance)
        assert_allclose(cond1.gp.variance, cond2.gp.variance)
        assert_allclose(cond1.gp.log_probability(y2), cond2.gp.log_probability(y2))
        assert jnp.isfinite(cond1.gp.sample(jax.random.PRNGKey(0))).all()

        # predict(return_var=True) at the training points uses condition_diag
        mu1, var1 = gp1.predict(y1, return_var=True)
        mu2, var2 = gp2.predict(y1, return_var=True)
        assert_allclose(mu1, mu2)
        assert_allclose(var1, var2)

        # Chained conditioning should also be well-behaved
        cond1b = cond1.gp.condition(y2)
        cond2b = cond2.gp.condition(y2)
        assert_allclose(cond1b.log_probability, cond2b.log_probability)
        assert_allclose(cond1b.gp.loc, cond2b.gp.loc)
        assert_allclose(cond1b.gp.variance, cond2b.gp.variance)


@pytest.mark.parametrize("parallel", [False, True], ids=["sequential", "parallel"])
def test_condition_banded_noise(random, parallel):
    N = 50
    x = jnp.sort(random.uniform(-3, 3, N))
    y = random.normal(size=N)
    noise = Banded(
        diag=0.1 + 0.05 * random.uniform(size=N),
        off_diags=0.01 * jnp.ones((N, 1)),
    )
    kernel = quasisep.Matern32(sigma=1.8, scale=1.5)
    gp1 = GaussianProcess(
        kernel, x, noise=noise, solver=QuasisepSolver, parallel=parallel
    )
    gp2 = GaussianProcess(kernel, x, noise=noise, solver=DirectSolver)
    cond1 = gp1.condition(y)
    cond2 = gp2.condition(y)
    assert isinstance(cond1.gp.solver, QuasisepSolver)
    assert_allclose(cond1.gp.covariance, cond2.gp.covariance)
    assert_allclose(cond1.gp.variance, cond2.gp.variance)
    assert_allclose(
        gp1.predict(y, return_var=True)[1], gp2.predict(y, return_var=True)[1]
    )
