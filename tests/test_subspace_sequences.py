import jax
import jax.numpy as jnp
import pytest

from tinygp import GaussianProcess, kernels, transforms
from tinygp.test_utils import assert_allclose


@pytest.mark.parametrize("axes", [(0, 2), (2, 0), (-1, 0), (0,)])
@pytest.mark.parametrize("as_list", [False, True])
def test_subspace_sequence_selects_features(axes, as_list):
    x = jnp.arange(15.0).reshape(5, 3) / 10
    scale = jnp.arange(1, len(axes) + 1, dtype=float)
    kernel = transforms.Subspace(
        list(axes) if as_list else axes,
        transforms.Linear(scale, kernels.ExpSquared()),
    )

    def reference(values):
        selected = jnp.stack([values[:, axis] for axis in axes], axis=-1) * scale
        differences = selected[:, None] - selected[None]
        return jnp.exp(-0.5 * jnp.sum(differences**2, axis=-1))

    assert_allclose(jax.jit(lambda values: kernel(values, values))(x), reference(x))
    assert_allclose(kernel(x), jnp.ones(len(x)))
    actual_grad = jax.jit(jax.grad(lambda values: kernel(values, values).sum()))(x)
    expected_grad = jax.jit(jax.grad(lambda values: reference(values).sum()))(x)
    assert_allclose(actual_grad, expected_grad)


def test_subspace_sequence_in_gp_likelihood():
    x = jnp.arange(15.0).reshape(5, 3) / 10
    y = jnp.sin(x[:, 0])
    kernel = transforms.Subspace((0, 2), kernels.ExpSquared(0.7))
    selected = jnp.stack([x[:, 0], x[:, 2]], axis=-1)
    covariance = kernels.ExpSquared(0.7)(selected, selected) + 0.1 * jnp.eye(len(x))
    expected = jax.scipy.stats.multivariate_normal.logpdf(
        y, jnp.zeros_like(y), covariance
    )
    assert_allclose(GaussianProcess(kernel, x, diag=0.1).log_probability(y), expected)
