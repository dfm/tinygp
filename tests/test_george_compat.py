# -*- coding: utf-8 -*-
# mypy: ignore-errors

from functools import partial

import numpy as np
import pytest

from tinygp import GaussianProcess, kernels

george = pytest.importorskip("george")


@pytest.fixture
def random():
    return np.random.default_rng(1058390)


@pytest.fixture(
    scope="module",
    params=["Constant", "DotProduct", "Polynomial"],
)
def kernel(request):
    return {
        "Constant": (
            kernels.Constant(value=1.5),
            george.kernels.ConstantKernel(
                log_constant=np.log(1.5 / 5), ndim=5
            ),
        ),
        "DotProduct": (
            kernels.DotProduct(),
            george.kernels.DotProductKernel(ndim=3),
        ),
        "Polynomial": (
            kernels.Polynomial(order=2.5, sigma=1.3),
            george.kernels.PolynomialKernel(
                order=2.5, log_sigma2=2 * np.log(1.3), ndim=1
            ),
        ),
    }[request.param]


@pytest.fixture(scope="module", params=["Cosine", "ExpSineSquared"])
def periodic_kernel(request):
    return {
        "Cosine": (
            kernels.Cosine(period=2.3),
            george.kernels.CosineKernel(log_period=np.log(2.3)),
        ),
        "ExpSineSquared": (
            kernels.ExpSineSquared(period=2.3, gamma=1.3),
            george.kernels.ExpSine2Kernel(gamma=1.3, log_period=np.log(2.3)),
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
            george.kernels.ExpKernel(scale ** 2),
        ),
        "ExpSquared": (
            kernels.ExpSquared(scale),
            george.kernels.ExpSquaredKernel(scale ** 2),
        ),
        "Matern32": (
            kernels.Matern32(scale),
            george.kernels.Matern32Kernel(scale ** 2),
        ),
        "Matern52": (
            kernels.Matern52(scale),
            george.kernels.Matern52Kernel(scale ** 2),
        ),
        "RationalQuadratic": (
            kernels.RationalQuadratic(alpha=1.5),
            george.kernels.RationalQuadraticKernel(
                metric=1.0, log_alpha=np.log(1.5)
            ),
        ),
    }[request.param]


def compare_kernel_value(random, tiny_kernel, george_kernel):
    x1 = np.sort(random.uniform(0, 10, (50, george_kernel.ndim)))
    x2 = np.sort(random.uniform(0, 10, (45, george_kernel.ndim)))
    np.testing.assert_allclose(
        tiny_kernel(x1, x2),
        george_kernel.get_value(x1, x2),
    )
    np.testing.assert_allclose(
        tiny_kernel(x1, x1),
        george_kernel.get_value(x1),
    )
    np.testing.assert_allclose(
        tiny_kernel(x1),
        george_kernel.get_value(x1, diag=True),
    )


def compare_gps(random, tiny_kernel, george_kernel):
    x = np.sort(random.uniform(0, 10, (50, george_kernel.ndim)))
    t = np.sort(random.uniform(0, 10, (12, george_kernel.ndim)))
    y = np.sin(x[:, 0])
    diag = random.uniform(0.01, 0.1, 50)

    # Set up the GPs
    george_gp = george.GP(george_kernel)
    george_gp.compute(x, np.sqrt(diag))
    tiny_gp = GaussianProcess(tiny_kernel, x, diag=diag)

    # Likelihood
    np.testing.assert_allclose(
        tiny_gp.condition(y), george_gp.log_likelihood(y)
    )

    # Filtering
    np.testing.assert_allclose(
        tiny_gp.predict(y),
        george_gp.predict(y, x, return_var=False, return_cov=False),
    )

    # Filtering with explicit value
    np.testing.assert_allclose(
        tiny_gp.predict(y, x),
        george_gp.predict(y, x, return_var=False, return_cov=False),
    )
    np.testing.assert_allclose(
        tiny_gp.predict(y, t),
        george_gp.predict(y, t, return_var=False, return_cov=False),
    )

    # Variance
    np.testing.assert_allclose(
        tiny_gp.predict(y, return_var=True)[1],
        george_gp.predict(y, x, return_var=True, return_cov=False)[1],
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        tiny_gp.predict(y, t, return_var=True)[1],
        george_gp.predict(y, t, return_var=True, return_cov=False)[1],
        rtol=1e-5,
    )

    # Covariance
    np.testing.assert_allclose(
        tiny_gp.predict(y, return_cov=True)[1],
        george_gp.predict(y, x, return_var=False, return_cov=True)[1],
        atol=1e-5,
    )
    np.testing.assert_allclose(
        tiny_gp.predict(y, t, return_cov=True)[1],
        george_gp.predict(y, t, return_var=False, return_cov=True)[1],
        atol=1e-5,
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
    compare_gps(random, tiny_kernel, george_kernel)


def test_periodic_gp(random, periodic_kernel):
    tiny_kernel, george_kernel = periodic_kernel
    compare_gps(random, tiny_kernel, george_kernel)


def test_metric_gp(random, stationary_kernel):
    tiny_kernel, george_kernel = stationary_kernel
    compare_gps(random, tiny_kernel, george_kernel)
