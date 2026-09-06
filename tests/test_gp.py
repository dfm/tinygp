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

    with jax.enable_x64(True):
        gp = GaussianProcess(kernels.Matern32(1.5), X, diag=0.01, mean=jnp.sum)
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
        with pytest.raises(ValueError):
            gp.predict(y, {"x": X[0]})
    else:
        with pytest.raises(ValueError):
            gp.predict(y, X[0])


class LatentKernel(kernels.Kernel):
    kernel: kernels.Kernel
    coeff_prim: jax.Array
    coeff_deriv: jax.Array

    def __init__(self, kernel, coeff_prim, coeff_deriv):
        self.kernel = kernel
        self.coeff_prim, self.coeff_deriv = jnp.broadcast_arrays(
            jnp.asarray(coeff_prim), jnp.asarray(coeff_deriv)
        )

    def evaluate(self, X1, X2):
        t1, label1 = X1
        t2, label2 = X2
        Kp = jax.grad(self.kernel.evaluate, argnums=0)
        Kpp = jax.grad(Kp, argnums=1)
        K = self.kernel.evaluate(t1, t2)
        d2K_dx1dx2 = Kpp(t1, t2)
        dK_dx2 = jax.grad(self.kernel.evaluate, argnums=1)(t1, t2)
        dK_dx1 = Kp(t1, t2)
        a1 = self.coeff_prim[label1]
        a2 = self.coeff_prim[label2]
        b1 = self.coeff_deriv[label1]
        b2 = self.coeff_deriv[label2]
        return a1 * a2 * K + a1 * b2 * dK_dx2 + b1 * a2 * dK_dx1 + b1 * b2 * d2K_dx1dx2


@pytest.mark.parametrize(
    "kernel_type",
    ["stationary", "quasisep", "pytree"],
)
def test_predict_equivalence(random, kernel_type):
    with jax.enable_x64(True):
        if kernel_type == "stationary":
            X = jnp.sort(random.uniform(0, 10, (40, 2)), axis=0)
            y = jnp.sin(X[:, 0]) + 0.1 * random.normal(size=len(X))
            kernel = kernels.Matern32(1.5)
            gp = GaussianProcess(kernel, X, diag=0.05, mean=jnp.sum)
            X_test = jnp.sort(random.uniform(-2, 12, (60, 2)), axis=0)
        elif kernel_type == "quasisep":
            X = jnp.sort(random.uniform(0, 10, 50))
            y = jnp.sin(X) + 0.1 * random.normal(size=len(X))
            kernel = kernels.quasisep.SHO(omega=1.2, quality=2.5)
            gp = GaussianProcess(kernel, X, diag=0.05)
            X_test = jnp.sort(random.uniform(-2, 12, 70))
        elif kernel_type == "pytree":
            t_train = jnp.sort(random.uniform(0, 10, 30))
            label_train = random.integers(0, 2, size=30)
            X = (t_train, label_train)
            y = jnp.sin(t_train) + 0.1 * random.normal(size=len(t_train))
            base_k = kernels.ExpSquared(2.0)
            kernel = LatentKernel(base_k, [1.0, 0.5], [-0.1, 0.3])
            gp = GaussianProcess(kernel, X, diag=0.01)
            t_test = jnp.sort(random.uniform(-1, 11, 45))
            label_test = random.integers(0, 2, size=45)
            X_test = (t_test, label_test)

        _, cond = gp.condition(y, X_test)
        mu_pred = gp.predict(y, X_test)
        mu_var, var_pred = gp.predict(y, X_test, return_var=True)
        mu_cov, cov_pred = gp.predict(y, X_test, return_cov=True)

        assert_allclose(mu_pred, cond.loc)
        assert_allclose(mu_var, cond.loc)
        assert_allclose(mu_cov, cond.loc)
        assert_allclose(var_pred, cond.variance)
        assert_allclose(cov_pred, cond.covariance)


def test_predict_edge_cases(random):
    with jax.enable_x64(True):
        X = jnp.sort(random.uniform(0, 10, 40))
        y = jnp.sin(X)
        kernel = kernels.Matern32(1.5)
        gp = GaussianProcess(kernel, X, diag=0.05, mean=jnp.sin)

        # X_test is None
        _, cond_none = gp.condition(y)
        mu_none, var_none = gp.predict(y, return_var=True)
        assert_allclose(mu_none, cond_none.loc)
        assert_allclose(var_none, cond_none.variance)

        # X_test is exactly X
        _, cond_x = gp.condition(y, X)
        mu_x, var_x = gp.predict(y, X, return_var=True)
        assert_allclose(mu_x, cond_x.loc)
        assert_allclose(var_x, cond_x.variance)

        # N_test = 1
        X_1 = jnp.array([5.2])
        _, cond_1 = gp.condition(y, X_1)
        mu_1, var_1 = gp.predict(y, X_1, return_var=True)
        assert_allclose(mu_1, cond_1.loc)
        assert_allclose(var_1, cond_1.variance)

        # include_mean=False
        _, cond_nomean = gp.condition(y, X_1, include_mean=False)
        mu_nomean, var_nomean = gp.predict(y, X_1, include_mean=False, return_var=True)
        assert_allclose(mu_nomean, cond_nomean.loc)
        assert_allclose(var_nomean, cond_nomean.variance)

        # Custom cross kernel
        cross_kernel = kernels.Exp(1.2)
        _, cond_cross = gp.condition(y, X_1, kernel=cross_kernel)
        mu_cross, var_cross = gp.predict(y, X_1, kernel=cross_kernel, return_var=True)
        assert_allclose(mu_cross, cond_cross.loc)
        assert_allclose(var_cross, cond_cross.variance)


def test_predict_return_var_takes_priority_over_return_cov(random):
    # Regression test: per the `predict` docstring, if `return_var` is True,
    # `return_cov` must be ignored -- the second return value must be the
    # 1-D variance, not the 2-D covariance, even when both flags are set.
    with jax.enable_x64(True):
        X = jnp.sort(random.uniform(0, 10, 20))
        y = jnp.sin(X)
        gp = GaussianProcess(kernels.Matern32(1.5), X, diag=0.05)
        X_test = jnp.sort(random.uniform(-1, 11, 6))

        _, cond = gp.condition(y, X_test)
        mu, out = gp.predict(y, X_test, return_var=True, return_cov=True)
        assert out.shape == (6,)
        assert_allclose(mu, cond.loc)
        assert_allclose(out, cond.variance)
