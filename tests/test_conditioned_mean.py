import jax.numpy as jnp
import pytest

from tinygp import GaussianProcess, kernels
from tinygp.test_utils import assert_allclose


@pytest.mark.parametrize("include_mean", [False, True])
@pytest.mark.parametrize("method", ["predict", "condition"])
def test_conditioned_gp_can_be_updated_again(include_mean, method):
    x = jnp.linspace(0, 1, 5)
    y = 0.2 + jnp.sin(x)
    x_new = jnp.array([1.1, 1.5, 1.8])
    y_new = jnp.array([0.7, 0.9, 0.4])
    x_test = jnp.array([-0.2, 0.6, 1.3, 2.1])
    kernel = kernels.ExpSquared(0.7)
    gp = GaussianProcess(kernel, x, diag=0.1, mean=0.3)
    conditioned = gp.condition(y, x_new, diag=0.2, include_mean=include_mean).gp

    train_cov = kernel(x, x) + 0.1 * jnp.eye(len(x))
    alpha = jnp.linalg.solve(train_cov, y - 0.3)

    def posterior_mean(points):
        return kernel(points, x) @ alpha + (0.3 if include_mean else 0.0)

    def posterior_cov(first, second):
        return kernel(first, second) - kernel(first, x) @ jnp.linalg.solve(
            train_cov, kernel(x, second)
        )

    assert_allclose(conditioned.loc, posterior_mean(x_new))
    new_cov = posterior_cov(x_new, x_new) + 0.2 * jnp.eye(len(x_new))
    cross_cov = posterior_cov(x_test, x_new)
    expected_mean = posterior_mean(x_test) + cross_cov @ jnp.linalg.solve(
        new_cov, y_new - posterior_mean(x_new)
    )
    expected_cov = posterior_cov(x_test, x_test) - cross_cov @ jnp.linalg.solve(
        new_cov, cross_cov.T
    )

    if method == "predict":
        mean, variance = conditioned.predict(y_new, x_test, return_var=True)
        jitter = jnp.sqrt(jnp.finfo(expected_mean.dtype).eps)
        assert_allclose(variance, jnp.diag(expected_cov) + jitter)
    else:
        result = conditioned.condition(y_new, x_test, diag=0.05).gp
        mean = result.loc
        assert_allclose(result.covariance, expected_cov + 0.05 * jnp.eye(len(x_test)))
    assert_allclose(mean, expected_mean)
