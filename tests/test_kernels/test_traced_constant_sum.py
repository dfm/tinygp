import jax
import jax.numpy as jnp
import pytest

from tinygp import GaussianProcess, kernels
from tinygp.test_utils import assert_allclose


@pytest.mark.parametrize("constant", [0.0, 0.5])
def test_left_constant_sum_under_jax_transforms(constant):
    x = jnp.linspace(-1, 1, 5)
    kernel = kernels.ExpSquared(0.7)

    def evaluate(value):
        return (value + kernel)(x, x)

    assert_allclose(jax.jit(evaluate)(constant), kernel(x, x) + constant)
    derivative = jax.jit(jax.jacrev(evaluate))(constant)
    assert_allclose(derivative, jnp.ones((len(x), len(x))))
    constants = jnp.array([0.0, 0.5, 1.0])
    expected = kernel(x, x)[None] + constants[:, None, None]
    assert_allclose(jax.vmap(evaluate)(constants), expected)


def test_left_constant_sum_gp_likelihood_gradient():
    x = jnp.linspace(-1, 1, 5)
    y = jnp.sin(x)
    kernel = kernels.ExpSquared(0.7)

    def left(value):
        return GaussianProcess(value + kernel, x, diag=0.1).log_probability(y)

    def dense_reference(value):
        covariance = kernel(x, x) + value + 0.1 * jnp.eye(len(x))
        return jax.scipy.stats.multivariate_normal.logpdf(
            y, jnp.zeros_like(y), covariance
        )

    actual = jax.jit(jax.value_and_grad(left))(0.5)
    expected = jax.jit(jax.value_and_grad(dense_reference))(0.5)
    assert_allclose(actual[0], expected[0])
    assert_allclose(actual[1], expected[1])


def test_builtin_zero_sum_keeps_kernel_identity():
    kernel = kernels.ExpSquared(0.7)
    assert 0 + kernel is kernel
    assert 0.0 + kernel is kernel
    assert sum([kernel]) is kernel
