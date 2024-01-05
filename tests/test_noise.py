# mypy: ignore-errors

import jax.numpy as jnp
from numpy import random as np_random

import tinygp
from tinygp.test_utils import assert_allclose


def check_noise_model(noise, dense_rep):
    random = np_random.default_rng(6675)

    assert_allclose(noise.diagonal(), jnp.diag(dense_rep))
    assert_allclose(noise + jnp.zeros_like(dense_rep), dense_rep)

    y1 = random.normal(size=dense_rep.shape)
    assert_allclose(noise + y1, dense_rep + y1)
    assert_allclose(y1 + noise, y1 + dense_rep)
    assert_allclose(noise @ y1, dense_rep @ y1)

    y2 = random.normal(size=(dense_rep.shape[1], 3))
    assert_allclose(noise @ y2, dense_rep @ y2)

    y3 = random.normal(size=dense_rep.shape[1])
    assert_allclose(noise @ y3, dense_rep @ y3)

    try:
        qsm = noise.to_qsm()
    except NotImplementedError:
        pass
    else:
        assert_allclose(qsm @ y1, dense_rep @ y1)
        assert_allclose(qsm @ y2, dense_rep @ y2)
        assert_allclose(qsm @ y3, dense_rep @ y3)


def test_diagonal():
    N = 50
    random = np_random.default_rng(9432)
    diag = random.normal(size=N)
    noise = tinygp.noise.Diagonal(diag=diag)
    check_noise_model(noise, jnp.diag(diag))


def test_banded():
    N, J = 50, 5
    random = np_random.default_rng(9432)

    # Create a random symmetric banded matrix
    R = random.normal(size=(N, N))
    R[jnp.triu_indices(N, J + 1)] = 0
    R[jnp.tril_indices(N)] = R.T[jnp.tril_indices(N)]

    # Extract the diagonal and off-diagonal elements
    diag = jnp.diag(R)
    off_diags = jnp.zeros((N, J))
    for j in range(J):
        off_diags = off_diags.at[: N - j - 1, j].set(
            R[(jnp.arange(0, N - j - 1), jnp.arange(j + 1, N))]
        )

    noise = tinygp.noise.Banded(diag=diag, off_diags=off_diags)
    check_noise_model(noise, R)


def test_dense():
    N = 50
    random = np_random.default_rng(9432)
    M = random.normal(size=(N, N))
    noise = tinygp.noise.Dense(value=M)
    check_noise_model(noise, M)
