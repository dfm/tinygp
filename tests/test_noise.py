# -*- coding: utf-8 -*-
# mypy: ignore-errors

import jax.numpy as jnp
import numpy as np

import tinygp


def check_noise_model(noise, dense_rep):
    random = np.random.default_rng(6675)

    np.testing.assert_allclose(noise.diagonal(), jnp.diag(dense_rep))
    np.testing.assert_allclose(noise + np.zeros_like(dense_rep), dense_rep)

    y1 = random.normal(size=dense_rep.shape)
    np.testing.assert_allclose(noise + y1, dense_rep + y1)
    np.testing.assert_allclose(y1 + noise, y1 + dense_rep)
    np.testing.assert_allclose(noise @ y1, dense_rep @ y1)

    y2 = random.normal(size=(dense_rep.shape[1], 3))
    np.testing.assert_allclose(noise @ y2, dense_rep @ y2)

    y3 = random.normal(size=dense_rep.shape[1])
    np.testing.assert_allclose(noise @ y3, dense_rep @ y3)

    try:
        qsm = noise.to_qsm()
    except NotImplementedError:
        pass
    else:
        np.testing.assert_allclose(qsm @ y1, dense_rep @ y1)
        np.testing.assert_allclose(qsm @ y2, dense_rep @ y2)
        np.testing.assert_allclose(qsm @ y3, dense_rep @ y3)


def test_diagonal():
    N = 50
    random = np.random.default_rng(9432)
    diag = random.normal(size=N)
    noise = tinygp.noise.Diagonal(diag=diag)
    check_noise_model(noise, np.diag(diag))


def test_banded():
    N, J = 50, 5
    random = np.random.default_rng(9432)

    # Create a random symmetric banded matrix
    R = random.normal(size=(N, N))
    R[np.triu_indices(N, J + 1)] = 0
    R[np.tril_indices(N)] = R.T[np.tril_indices(N)]

    # Extract the diagonal and off-diagonal elements
    diag = np.diag(R)
    off_diags = np.zeros((N, J))
    for j in range(J):
        off_diags[: N - j - 1, j] = R[
            (np.arange(0, N - j - 1), np.arange(j + 1, N))
        ]

    noise = tinygp.noise.Banded(diag=diag, off_diags=off_diags)
    check_noise_model(noise, R)


def test_dense():
    N = 50
    random = np.random.default_rng(9432)
    M = random.normal(size=(N, N))
    noise = tinygp.noise.Dense(value=M)
    check_noise_model(noise, M)
