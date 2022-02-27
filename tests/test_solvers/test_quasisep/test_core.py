# -*- coding: utf-8 -*-
# mypy: ignore-errors

from itertools import combinations

import jax.numpy as jnp
import numpy as np
import pytest

from tinygp.solvers.quasisep.core import (
    DiagQSM,
    LowerTriQSM,
    SquareQSM,
    StrictLowerTriQSM,
    StrictUpperTriQSM,
    SymmQSM,
)


@pytest.fixture(params=["random", "celerite"])
def name(request):
    return request.param


@pytest.fixture
def matrices(name):
    return get_matrices(name)


@pytest.fixture
def some_nice_matrices():
    diag1, p1, q1, a1, _, _, _, _ = get_matrices("celerite")
    diag2, p2, q2, a2, _, _, _, _ = get_matrices("random")
    mat1 = LowerTriQSM(
        diag=DiagQSM(diag1),
        lower=StrictLowerTriQSM(p=p1, q=q1, a=a1),
    )
    mat2 = SquareQSM(
        diag=DiagQSM(diag2),
        lower=StrictLowerTriQSM(p=p2, q=q2, a=a2),
        upper=StrictUpperTriQSM(p=p2, q=q2, a=a2),
    )
    mat3 = SquareQSM(
        diag=DiagQSM(diag1),
        lower=StrictLowerTriQSM(p=p1, q=q1, a=a1),
        upper=StrictUpperTriQSM(
            p=jnp.zeros_like(p2), q=jnp.zeros_like(q2), a=a2
        ),
    )
    mat4 = SquareQSM(
        diag=DiagQSM(diag1),
        lower=StrictLowerTriQSM(p=p1, q=q1, a=a1),
        upper=StrictUpperTriQSM(p=p2, q=q2, a=a2),
    )
    return mat1, mat2, mat3, mat4


def get_matrices(name):
    N = 100
    random = np.random.default_rng(1234)
    diag = np.exp(random.normal(size=N))

    if name == "random":
        J = 5
        p = random.normal(size=(N, J))
        q = random.normal(size=(N, J))
        a = np.repeat(np.eye(J)[None, :, :], N, axis=0)
        l = np.tril(p @ q.T, -1)
        u = np.triu(q @ p.T, 1)
        diag += np.sum(p * q, axis=1)

    elif name == "celerite":
        t = np.sort(random.uniform(0, 10, N))

        a = np.array([1.0, 2.5])
        b = np.array([0.5, 1.5])
        c = np.array([1.2, 0.5])
        d = np.array([0.5, 0.1])

        tau = np.abs(t[:, None] - t[None, :])[:, :, None]
        K = np.sum(
            np.exp(-c[None, None] * tau)
            * (
                a[None, None] * np.cos(d[None, None] * tau)
                + b[None, None] * np.sin(d[None, None] * tau)
            ),
            axis=-1,
        )
        K[np.diag_indices_from(K)] += diag
        diag = np.diag(K)
        l = np.tril(K, -1)
        u = np.triu(K, 1)

        cos = np.cos(d[None] * t[:, None])
        sin = np.sin(d[None] * t[:, None])
        p = np.concatenate(
            (
                a[None] * cos + b[None] * sin,
                a[None] * sin - b[None] * cos,
            ),
            axis=1,
        )
        q = np.concatenate((cos, sin), axis=1)
        c = np.append(c, c)
        dt = np.append(0, np.diff(t))
        a = np.stack(
            [np.diag(v) for v in np.exp(-c[None] * dt[:, None])], axis=0
        )
        p = np.einsum("ni,nij->nj", p, a)

    else:
        assert False

    v = random.normal(size=N)
    m = random.normal(size=(N, 4))
    return diag, p, q, a, v, m, l, u


def test_quasisep_def():
    random = np.random.default_rng(2022)
    n = 17
    m1 = 3
    m2 = 5
    d = random.normal(size=n)
    p = random.normal(size=(n, m1))
    q = random.normal(size=(n, m1))
    a = random.normal(size=(n, m1, m1))
    g = random.normal(size=(n, m2))
    h = random.normal(size=(n, m2))
    b = random.normal(size=(n, m2, m2))
    m = SquareQSM(
        diag=DiagQSM(d=d),
        lower=StrictLowerTriQSM(p=p, q=q, a=a),
        upper=StrictUpperTriQSM(p=g, q=h, a=b),
    ).to_dense()

    def get_value(i, j):
        if i == j:
            return d[i]
        if j < i:
            tmp = np.copy(q[j])
            for k in range(j + 1, i):
                tmp = a[k] @ tmp
            return p[i] @ tmp
        if j > i:
            tmp = np.copy(h[i])
            for k in range(i + 1, j):
                tmp = tmp @ b[k].T
            return tmp @ g[j]

    for i in range(n):
        for j in range(n):
            np.testing.assert_allclose(get_value(i, j), m[i, j])


def test_strict_tri_matmul(matrices):
    _, p, q, a, v, m, l, u = matrices
    mat = StrictLowerTriQSM(p=p, q=q, a=a)

    # Check multiplication into identity / to dense
    np.testing.assert_allclose(mat.to_dense(), l)
    np.testing.assert_allclose(mat.T.to_dense(), u)

    # Check matvec
    np.testing.assert_allclose(mat @ v, l @ v)
    np.testing.assert_allclose(mat.T @ v, u @ v)

    # Check matmat
    np.testing.assert_allclose(mat @ m, l @ m)
    np.testing.assert_allclose(mat.T @ m, u @ m)


def test_tri_matmul(matrices):
    diag, p, q, a, v, m, l, _ = matrices
    mat = LowerTriQSM(
        diag=DiagQSM(diag), lower=StrictLowerTriQSM(p=p, q=q, a=a)
    )
    dense = l + np.diag(diag)

    # Check multiplication into identity / to dense
    np.testing.assert_allclose(mat.to_dense(), dense)
    np.testing.assert_allclose(mat.T.to_dense(), dense.T)

    # Check matvec
    np.testing.assert_allclose(mat @ v, dense @ v)
    np.testing.assert_allclose(mat.T @ v, dense.T @ v)

    # Check matmat
    np.testing.assert_allclose(mat @ m, dense @ m)
    np.testing.assert_allclose(mat.T @ m, dense.T @ m)


@pytest.mark.parametrize("symm", [True, False])
def test_square_matmul(symm, matrices):
    diag, p, q, a, v, m, l, u = matrices
    if symm:
        mat = SymmQSM(
            diag=DiagQSM(diag), lower=StrictLowerTriQSM(p=p, q=q, a=a)
        )
    else:
        mat = SquareQSM(
            diag=DiagQSM(diag),
            lower=StrictLowerTriQSM(p=p, q=q, a=a),
            upper=StrictUpperTriQSM(p=p, q=q, a=a),
        )

    # Create and double check the dense reconstruction
    dense = mat.to_dense()
    np.testing.assert_allclose(np.tril(dense, -1), l)
    np.testing.assert_allclose(np.triu(dense, 1), u)
    np.testing.assert_allclose(np.diag(dense), diag)

    # Test matmuls
    np.testing.assert_allclose(mat @ v, dense @ v)
    np.testing.assert_allclose(mat @ m, dense @ m)
    np.testing.assert_allclose(v.T @ mat, v.T @ dense)
    np.testing.assert_allclose(m.T @ mat, m.T @ dense)


@pytest.mark.parametrize("name", ["celerite"])
def test_tri_inv(matrices):
    diag, p, q, a, _, _, _, _ = matrices
    mat = LowerTriQSM(
        diag=DiagQSM(diag), lower=StrictLowerTriQSM(p=p, q=q, a=a)
    )
    dense = mat.to_dense()
    minv = mat.inv()
    np.testing.assert_allclose(minv.to_dense(), jnp.linalg.inv(dense))
    np.testing.assert_allclose(
        minv.matmul(dense), np.eye(len(diag)), atol=1e-12
    )


@pytest.mark.parametrize("name", ["celerite"])
def test_tri_solve(matrices):
    diag, p, q, a, v, m, _, _ = matrices
    mat = LowerTriQSM(
        diag=DiagQSM(diag), lower=StrictLowerTriQSM(p=p, q=q, a=a)
    )
    dense = mat.to_dense()
    np.testing.assert_allclose(mat.solve(v), np.linalg.solve(dense, v))
    np.testing.assert_allclose(mat.solve(m), np.linalg.solve(dense, m))

    np.testing.assert_allclose(mat.T.solve(v), np.linalg.solve(dense.T, v))
    np.testing.assert_allclose(mat.T.solve(m), np.linalg.solve(dense.T, m))

    np.testing.assert_allclose(mat.inv().solve(v), dense @ v)
    np.testing.assert_allclose(mat.inv().solve(m), dense @ m)
    np.testing.assert_allclose(mat.T.inv().solve(v), dense.T @ v)
    np.testing.assert_allclose(mat.T.inv().solve(m), dense.T @ m)


@pytest.mark.parametrize("symm", [True, False])
def test_square_inv(symm, matrices):
    diag, p, q, a, _, _, l, u = matrices
    if symm:
        mat = SymmQSM(
            diag=DiagQSM(diag), lower=StrictLowerTriQSM(p=p, q=q, a=a)
        )
    else:
        mat = SquareQSM(
            diag=DiagQSM(diag),
            lower=StrictLowerTriQSM(p=p, q=q, a=a),
            upper=StrictUpperTriQSM(p=p, q=q, a=a),
        )

    # Create and double check the dense reconstruction
    dense = mat.to_dense()
    np.testing.assert_allclose(np.tril(dense, -1), l)
    np.testing.assert_allclose(np.triu(dense, 1), u)
    np.testing.assert_allclose(np.diag(dense), diag)

    # Invert the QS matrix
    minv = mat.inv()
    np.testing.assert_allclose(
        minv.to_dense(), jnp.linalg.inv(dense), rtol=2e-6
    )
    np.testing.assert_allclose(
        minv.matmul(dense), np.eye(len(diag)), atol=1e-12
    )

    # In this case, we know our matrix to be symmetric - so should its inverse be!
    # This may change in the future as we expand test cases
    if not symm:
        np.testing.assert_allclose(minv.lower.p, minv.upper.p)
        np.testing.assert_allclose(minv.lower.q, minv.upper.q)
        np.testing.assert_allclose(minv.lower.a, minv.upper.a)

    # The inverse of the inverse should be itself... don't actually do this!
    # Note: we can't actually directly compare the generators because there's
    # enough degrees of freedom that they won't necessarily round trip. It's
    # good enough to check that it produces the correct dense reconstruction.
    mat2 = minv.inv()
    np.testing.assert_allclose(mat2.to_dense(), dense, rtol=1e-4)


def test_gram(matrices):
    diag, p, q, a, _, _, _, _ = matrices
    mat = SquareQSM(
        diag=DiagQSM(diag),
        lower=StrictLowerTriQSM(p=p, q=q, a=a),
        upper=StrictUpperTriQSM(p=p, q=q, a=a),
    )
    dense = mat.to_dense()
    np.testing.assert_allclose(mat.gram().to_dense(), dense.T @ dense)

    mat = mat.inv()
    dense = mat.to_dense()
    np.testing.assert_allclose(mat.gram().to_dense(), dense.T @ dense)

    mat = SquareQSM(
        diag=DiagQSM(diag),
        lower=StrictLowerTriQSM(p=p, q=q, a=a),
        upper=StrictUpperTriQSM(
            p=jnp.zeros_like(p), q=jnp.zeros_like(q), a=jnp.zeros_like(a)
        ),
    )
    dense = mat.to_dense()
    np.testing.assert_allclose(mat.gram().to_dense(), dense.T @ dense)


@pytest.mark.parametrize("name", ["celerite"])
def test_cholesky(matrices):
    diag, p, q, a, v, m, _, _ = matrices
    mat = SymmQSM(diag=DiagQSM(diag), lower=StrictLowerTriQSM(p=p, q=q, a=a))
    dense = mat.to_dense()
    chol = mat.cholesky()
    np.testing.assert_allclose(chol.to_dense(), np.linalg.cholesky(dense))

    mat = mat.inv()
    dense = mat.to_dense()
    chol = mat.cholesky()
    np.testing.assert_allclose(chol.to_dense(), np.linalg.cholesky(dense))

    np.testing.assert_allclose(
        chol.solve(v), np.linalg.solve(chol.to_dense(), v)
    )
    np.testing.assert_allclose(
        chol.solve(m), np.linalg.solve(chol.to_dense(), m)
    )


def test_tri_qsmul(some_nice_matrices):
    mat1, mat2, mat3, mat4 = some_nice_matrices

    def check(mat1, mat2):
        mat = mat1 @ mat2
        a = mat.to_dense()
        b = mat1.to_dense() @ mat2.to_dense()
        np.testing.assert_allclose(np.diag(a), np.diag(b), atol=1e-12)
        np.testing.assert_allclose(np.tril(a, -1), np.tril(b, -1), atol=1e-12)
        np.testing.assert_allclose(np.triu(a, 1), np.triu(b, 1), atol=1e-12)

    minv = mat1.inv()
    mTinv = mat1.T.inv()
    for m in [mat2, mat3, mat4, mat2.inv()]:
        check(mat1, m)
        check(minv, m)
        check(mat1.T, m)
        check(mTinv, m)


def test_square_qsmul(some_nice_matrices):
    mat1, mat2, mat3, mat4 = some_nice_matrices
    mat1 += mat1.lower.transpose()

    def check(mat1, mat2):
        mat = mat1 @ mat2
        a = mat.to_dense()
        b = mat1.to_dense() @ mat2.to_dense()
        np.testing.assert_allclose(np.diag(a), np.diag(b), atol=1e-12)
        np.testing.assert_allclose(np.tril(a, -1), np.tril(b, -1), atol=1e-12)
        np.testing.assert_allclose(np.triu(a, 1), np.triu(b, 1), atol=1e-12)

    for m1, m2 in combinations(
        [mat1, mat2, mat3, mat4, mat1.inv(), mat2.inv()], 2
    ):
        check(m1, m2)


def test_ops(some_nice_matrices):
    mat1, mat2, mat3, mat4 = some_nice_matrices

    def check(mat1, mat2):
        for m1, m2 in combinations([mat1, mat2, mat1.lower, mat2.lower], 2):
            a = m1.to_dense()
            b = m2.to_dense()
            np.testing.assert_allclose((-m1).to_dense(), -a, atol=1e-12)
            np.testing.assert_allclose((m1 + m2).to_dense(), a + b, atol=1e-12)
            np.testing.assert_allclose((m1 - m2).to_dense(), a - b, atol=1e-12)
            np.testing.assert_allclose((m1 * m2).to_dense(), a * b, atol=1e-12)
            np.testing.assert_allclose(
                (2.5 * m1).to_dense(), 2.5 * a, atol=1e-12
            )

    for m1, m2 in combinations(
        [mat1, mat2, mat3, mat4, mat1.inv(), mat2.inv()], 2
    ):
        check(m1, m2)
