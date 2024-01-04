# mypy: ignore-errors

import warnings

import jax.numpy as jnp
import pytest
from numpy import random as np_random

from tinygp import GaussianProcess, kernels
from tinygp.test_utils import assert_allclose

george = pytest.importorskip("george")


@pytest.fixture
def random():
    return np_random.default_rng(1058390)


@pytest.fixture(
    scope="module",
    params=["Constant", "DotProduct", "Polynomial"],
)
def kernel(request):
    return {
        "Constant": (
            kernels.Constant(value=1.5),
            george.kernels.ConstantKernel(log_constant=jnp.log(1.5 / 5), ndim=5),
        ),
        "DotProduct": (
            kernels.DotProduct(),
            george.kernels.DotProductKernel(ndim=3),
        ),
        "Polynomial": (
            kernels.Polynomial(order=2.5, sigma=2.3),
            george.kernels.PolynomialKernel(
                order=2.5, log_sigma2=2 * jnp.log(2.3), ndim=1
            ),
        ),
    }[request.param]


@pytest.fixture(scope="module", params=["Cosine", "ExpSineSquared"])
def periodic_kernel(request):
    return {
        "Cosine": (
            kernels.Cosine(scale=2.3),
            george.kernels.CosineKernel(log_period=jnp.log(2.3)),
        ),
        "ExpSineSquared": (
            kernels.ExpSineSquared(scale=2.3, gamma=1.3),
            george.kernels.ExpSine2Kernel(gamma=1.3, log_period=jnp.log(2.3)),
        ),
    }[request.param]


@pytest.fixture(
    scope="module",
    params=["Exp", "ExpSquared", "Matern32", "Matern52", "RationalQuadratic"],
)
def stationary_kernel(request):
    scale = 1.5
    return {
        "Exp": (
            kernels.Exp(scale),
            george.kernels.ExpKernel(scale**2),
        ),
        "ExpSquared": (
            kernels.ExpSquared(scale),
            george.kernels.ExpSquaredKernel(scale**2),
        ),
        "Matern32": (
            kernels.Matern32(scale),
            george.kernels.Matern32Kernel(scale**2),
        ),
        "Matern52": (
            kernels.Matern52(scale),
            george.kernels.Matern52Kernel(scale**2),
        ),
        "RationalQuadratic": (
            kernels.RationalQuadratic(alpha=1.5),
            george.kernels.RationalQuadraticKernel(metric=1.0, log_alpha=jnp.log(1.5)),
        ),
    }[request.param]


def compare_kernel_value(random, tiny_kernel, george_kernel):
    x1 = jnp.sort(random.uniform(0, 10, (50, george_kernel.ndim)))
    x2 = jnp.sort(random.uniform(0, 10, (45, george_kernel.ndim)))
    assert_allclose(
        tiny_kernel(x1, x2),
        george_kernel.get_value(x1, x2),
    )
    assert_allclose(
        tiny_kernel(x1, x1),
        george_kernel.get_value(x1),
    )
    assert_allclose(
        tiny_kernel(x1),
        george_kernel.get_value(x1, diag=True),
    )


def compare_gps(random, tiny_kernel, george_kernel):
    x = jnp.sort(random.uniform(0, 10, (50, george_kernel.ndim)))
    t = jnp.sort(random.uniform(0, 10, (12, george_kernel.ndim)))
    y = jnp.sin(x[:, 0])
    diag = random.uniform(0.1, 0.2, 50)

    # Set up the GPs
    george_gp = george.GP(george_kernel)
    george_gp.compute(x, jnp.sqrt(diag))
    tiny_gp = GaussianProcess(tiny_kernel, x, diag=diag)

    # Likelihood
    assert_allclose(tiny_gp.log_probability(y), george_gp.log_likelihood(y))

    # Filtering
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)

        assert_allclose(
            tiny_gp.predict(y),
            george_gp.predict(y, x, return_var=False, return_cov=False),
        )

        # Filtering with explicit value
        assert_allclose(
            tiny_gp.predict(y, x),
            george_gp.predict(y, x, return_var=False, return_cov=False),
        )
        assert_allclose(
            tiny_gp.predict(y, t),
            george_gp.predict(y, t, return_var=False, return_cov=False),
        )

        # Variance
        assert_allclose(
            tiny_gp.predict(y, return_var=True)[1],
            george_gp.predict(y, x, return_var=True, return_cov=False)[1],
        )
        assert_allclose(
            tiny_gp.predict(y, t, return_var=True)[1],
            george_gp.predict(y, t, return_var=True, return_cov=False)[1],
        )

        # Covariance
        assert_allclose(
            tiny_gp.predict(y, return_cov=True)[1],
            george_gp.predict(y, x, return_var=False, return_cov=True)[1],
        )
        assert_allclose(
            tiny_gp.predict(y, t, return_cov=True)[1],
            george_gp.predict(y, t, return_var=False, return_cov=True)[1],
        )


def test_kernel_value(random, kernel):
    tiny_kernel, george_kernel = kernel
    compare_kernel_value(random, tiny_kernel, george_kernel)
    tiny_kernel *= 0.3
    george_kernel *= 0.3
    compare_kernel_value(random, tiny_kernel, george_kernel)


def test_periodic_kernel_value(random, periodic_kernel):
    tiny_kernel, george_kernel = periodic_kernel
    compare_kernel_value(random, tiny_kernel, george_kernel)
    tiny_kernel *= 0.3
    george_kernel *= 0.3
    compare_kernel_value(random, tiny_kernel, george_kernel)


def test_metric_kernel_value(random, stationary_kernel):
    tiny_kernel, george_kernel = stationary_kernel
    compare_kernel_value(random, tiny_kernel, george_kernel)
    tiny_kernel *= 0.3
    george_kernel *= 0.3
    compare_kernel_value(random, tiny_kernel, george_kernel)


def test_gp(random, kernel):
    tiny_kernel, george_kernel = kernel
    if isinstance(tiny_kernel, kernels.Polynomial):
        pytest.xfail("The Polynomial kernel is numerically unstable")
    compare_gps(random, tiny_kernel, george_kernel)


def test_periodic_gp(random, periodic_kernel):
    tiny_kernel, george_kernel = periodic_kernel
    compare_gps(random, tiny_kernel, george_kernel)


def test_metric_gp(random, stationary_kernel):
    tiny_kernel, george_kernel = stationary_kernel
    compare_gps(random, tiny_kernel, george_kernel)
