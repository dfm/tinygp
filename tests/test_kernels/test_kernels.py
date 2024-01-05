import jax
import jax.numpy as jnp
import pytest
from numpy import random as np_random

from tinygp import kernels, noise
from tinygp.solvers import DirectSolver
from tinygp.test_utils import assert_allclose


@pytest.fixture
def random():
    return np_random.default_rng(1058390)


@pytest.fixture
def data(random):
    x1 = random.uniform(-3, 3, (50, 5))
    x2 = random.uniform(-5, 5, (50, 5))
    return x1, x2


def test_constant(data):
    x1, x2 = data

    # Check for dimension issues when evaluated
    v = jnp.ones(3)
    with pytest.raises(ValueError):
        k1 = kernels.Constant(jnp.ones(3))
        k1.evaluate(v, v)

    # Check for dimension issues when multiplied and evaluated.
    k = jnp.ones(3) * kernels.Matern32(1.5)
    with pytest.raises(ValueError):
        k.evaluate(v, v)  # type: ignore

    # Check that multiplication has the expected behavior
    factor = 2.5
    k1 = kernels.Matern32(2.5)
    assert_allclose(factor * k1(x1, x2), (factor * k1)(x1, x2))


def test_custom(data):
    x1, x2 = data

    # Check that known kernels work as expected
    scale = 1.5
    k1 = kernels.Custom(
        lambda X1, X2: jnp.exp(-0.5 * jnp.sum(jnp.square((X1 - X2) / scale)))
    )
    k2 = kernels.ExpSquared(scale)
    assert_allclose(k1(x1, x2), k2(x1, x2))

    # Check that an invalid kernel raises as expected
    kernel = kernels.Custom(
        lambda X1, X2: jnp.exp(-0.5 * jnp.square((X1 - X2) / scale))
    )
    with pytest.raises(ValueError):
        kernel(x1, x2)


def test_ops(data):
    x1, x2 = data

    k1 = 1.5 * kernels.Matern32(2.5)
    k2 = 0.9 * kernels.ExpSineSquared(scale=1.5, gamma=0.3)

    assert_allclose(k1(x1, x2) + k2(x1, x2), (k1 + k2)(x1, x2))
    assert_allclose(k1(x1, x2) * k2(x1, x2), (k1 * k2)(x1, x2))


def test_conditioned(data):
    x1, x2 = data
    with jax.experimental.enable_x64():  # type: ignore
        k1 = 1.5 * kernels.Matern32(2.5)
        k2 = 0.9 * kernels.ExpSineSquared(scale=1.5, gamma=0.3)
        K = k1(x1, x1) + 0.1 * jnp.eye(x1.shape[0])
        solver = DirectSolver.init(k1, x1, noise.Diagonal(jnp.full(x1.shape[0], 0.1)))
        cond = kernels.Conditioned(x1, solver, k2)
        assert_allclose(
            cond(x1, x2),
            k2(x1, x2) - k2(x1, x1) @ jnp.linalg.solve(K, k2(x1, x2)),
        )


def test_dot_product(data):
    x1, x2 = data
    kernel = kernels.DotProduct()
    assert_allclose(kernel(x1, x2), jnp.dot(x1, x2.T))
    assert_allclose(kernel(x1[:, 0], x2[:, 0]), x1[:, 0][:, None] * x2[:, 0][None])


@pytest.mark.parametrize(
    "kernel",
    [
        kernels.Custom(lambda x, y: jnp.exp(-0.5 * jnp.sum(jnp.square(x - y)))),
        kernels.Constant(0.5),
        kernels.DotProduct(),
        kernels.Polynomial(order=1.5, scale=0.5, sigma=1.3),
        kernels.Exp(0.5),
        kernels.ExpSquared(0.5),
        kernels.Matern32(0.5),
        kernels.Matern52(0.5),
        kernels.Cosine(0.5),
        kernels.ExpSineSquared(0.5, gamma=1.5),
        kernels.RationalQuadratic(0.5, alpha=1.5),
    ],
)
def test_kernel_as_pytree(data, kernel):
    x1, x2 = data

    def check_roundtrip(kernel):
        expect = jax.jit(lambda kernel_: kernel_(x1, x2))(kernel)
        flat, spec = jax.tree_util.tree_flatten(kernel)
        calc = jax.tree_util.tree_unflatten(spec, flat)(x1, x2)
        assert_allclose(calc, expect)

    check_roundtrip(kernel)
    check_roundtrip(0.5 * kernel)
    check_roundtrip(kernel + kernel)
    check_roundtrip(kernel * kernel)


@pytest.mark.parametrize(
    "kernel",
    [
        kernels.ExpSineSquared,
        kernels.RationalQuadratic,
    ],
)
def test_required_parameters(kernel):
    with pytest.raises(ValueError):
        kernel(0.5)
