# mypy: ignore-errors

from itertools import combinations

import jax.numpy as jnp
import pytest
from numpy import random as np_random

from tinygp.solvers.quasisep.core import (
    DiagQSM,
    LowerTriQSM,
    SquareQSM,
    StrictLowerTriQSM,
    StrictUpperTriQSM,
    SymmQSM,
)
from tinygp.test_utils import assert_allclose


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
        upper=StrictUpperTriQSM(p=jnp.zeros_like(p2), q=jnp.zeros_like(q2), a=a2),
    )
    mat4 = SquareQSM(
        diag=DiagQSM(diag1),
        lower=StrictLowerTriQSM(p=p1, q=q1, a=a1),
        upper=StrictUpperTriQSM(p=p2, q=q2, a=a2),
    )
    return mat1, mat2, mat3, mat4


def get_matrices(name):
    N = 100
    random = np_random.default_rng(1234)
    diag = jnp.exp(random.normal(size=N))

    if name == "random":
        J = 5
        p = random.normal(size=(N, J))
        q = random.normal(size=(N, J))
        a = jnp.repeat(jnp.eye(J)[None, :, :], N, axis=0)
        l = jnp.tril(p @ q.T, -1)
        u = jnp.triu(q @ p.T, 1)
        diag += jnp.sum(p * q, axis=1)

    elif name == "celerite":
        t = jnp.sort(random.uniform(0, 10, N))

        a = jnp.array([1.0, 2.5])
        b = jnp.array([0.5, 1.5])
        c = jnp.array([1.2, 0.5])
        d = jnp.array([0.5, 0.1])

        tau = jnp.abs(t[:, None] - t[None, :])[:, :, None]
        K = jnp.sum(
            jnp.exp(-c[None, None] * tau)
            * (
                a[None, None] * jnp.cos(d[None, None] * tau)
                + b[None, None] * jnp.sin(d[None, None] * tau)
            ),
            axis=-1,
        )
        K += jnp.diag(diag)
        diag = jnp.diag(K)
        l = jnp.tril(K, -1)
        u = jnp.triu(K, 1)

        cos = jnp.cos(d[None] * t[:, None])
        sin = jnp.sin(d[None] * t[:, None])
        p = jnp.concatenate(
            (
                a[None] * cos + b[None] * sin,
                a[None] * sin - b[None] * cos,
            ),
            axis=1,
        )
        q = jnp.concatenate((cos, sin), axis=1)
        c = jnp.append(c, c)
        dt = jnp.append(0, jnp.diff(t))
        a = jnp.stack([jnp.diag(v) for v in jnp.exp(-c[None] * dt[:, None])], axis=0)
        p = jnp.einsum("ni,nij->nj", p, a)

    else:
        raise AssertionError()

    v = random.normal(size=N)
    m = random.normal(size=(N, 4))
    return diag, p, q, a, v, m, l, u


def test_quasisep_def():
    random = np_random.default_rng(2022)
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
            tmp = jnp.copy(q[j])
            for k in range(j + 1, i):
                tmp = a[k] @ tmp
            return p[i] @ tmp
        if j > i:
            tmp = jnp.copy(h[i])
            for k in range(i + 1, j):
                tmp = tmp @ b[k].T
            return tmp @ g[j]

    for i in range(n):
        for j in range(n):
            assert_allclose(get_value(i, j), m[i, j])


def test_strict_tri_matmul(matrices):
    _, p, q, a, v, m, l, u = matrices
    mat = StrictLowerTriQSM(p=p, q=q, a=a)

    # Check multiplication into identity / to dense
    assert_allclose(mat.to_dense(), l)
    assert_allclose(mat.T.to_dense(), u)

    # Check matvec
    assert_allclose(mat @ v, l @ v)
    assert_allclose(mat.T @ v, u @ v)

    # Check matmat
    assert_allclose(mat @ m, l @ m)
    assert_allclose(mat.T @ m, u @ m)


def test_tri_matmul(matrices):
    diag, p, q, a, v, m, l, _ = matrices
    mat = LowerTriQSM(diag=DiagQSM(diag), lower=StrictLowerTriQSM(p=p, q=q, a=a))
    dense = l + jnp.diag(diag)

    # Check multiplication into identity / to dense
    assert_allclose(mat.to_dense(), dense)
    assert_allclose(mat.T.to_dense(), dense.T)

    # Check matvec
    assert_allclose(mat @ v, dense @ v)
    assert_allclose(mat.T @ v, dense.T @ v)

    # Check matmat
    assert_allclose(mat @ m, dense @ m)
    assert_allclose(mat.T @ m, dense.T @ m)


@pytest.mark.parametrize("symm", [True, False])
def test_square_matmul(symm, matrices):
    diag, p, q, a, v, m, l, u = matrices
    if symm:
        mat = SymmQSM(diag=DiagQSM(diag), lower=StrictLowerTriQSM(p=p, q=q, a=a))
    else:
        mat = SquareQSM(
            diag=DiagQSM(diag),
            lower=StrictLowerTriQSM(p=p, q=q, a=a),
            upper=StrictUpperTriQSM(p=p, q=q, a=a),
        )

    # Create and double check the dense reconstruction
    dense = mat.to_dense()
    assert_allclose(jnp.tril(dense, -1), l)
    assert_allclose(jnp.triu(dense, 1), u)
    assert_allclose(jnp.diag(dense), diag)

    # Test matmuls
    assert_allclose(mat @ v, dense @ v)
    assert_allclose(mat @ m, dense @ m)
    assert_allclose(v.T @ mat, v.T @ dense)
    assert_allclose(m.T @ mat, m.T @ dense)


@pytest.mark.parametrize("name", ["celerite"])
def test_tri_inv(matrices):
    diag, p, q, a, _, _, _, _ = matrices
    mat = LowerTriQSM(diag=DiagQSM(diag), lower=StrictLowerTriQSM(p=p, q=q, a=a))
    dense = mat.to_dense()
    minv = mat.inv()
    assert_allclose(minv.to_dense(), jnp.linalg.inv(dense))
    assert_allclose(minv.matmul(dense), jnp.eye(len(diag)))


@pytest.mark.parametrize("name", ["celerite"])
def test_tri_solve(matrices):
    diag, p, q, a, v, m, _, _ = matrices
    mat = LowerTriQSM(diag=DiagQSM(diag), lower=StrictLowerTriQSM(p=p, q=q, a=a))
    dense = mat.to_dense()
    assert_allclose(mat.solve(v), jnp.linalg.solve(dense, v))
    assert_allclose(mat.solve(m), jnp.linalg.solve(dense, m))

    assert_allclose(mat.T.solve(v), jnp.linalg.solve(dense.T, v))
    assert_allclose(mat.T.solve(m), jnp.linalg.solve(dense.T, m))

    assert_allclose(mat.inv().solve(v), dense @ v)
    assert_allclose(mat.inv().solve(m), dense @ m)
    assert_allclose(mat.T.inv().solve(v), dense.T @ v)
    assert_allclose(mat.T.inv().solve(m), dense.T @ m)


@pytest.mark.parametrize("symm", [True, False])
def test_square_inv(symm, matrices):
    diag, p, q, a, _, _, l, u = matrices
    if symm:
        mat = SymmQSM(diag=DiagQSM(diag), lower=StrictLowerTriQSM(p=p, q=q, a=a))
    else:
        mat = SquareQSM(
            diag=DiagQSM(diag),
            lower=StrictLowerTriQSM(p=p, q=q, a=a),
            upper=StrictUpperTriQSM(p=p, q=q, a=a),
        )

    # Create and double check the dense reconstruction
    dense = mat.to_dense()
    assert_allclose(jnp.tril(dense, -1), l)
    assert_allclose(jnp.triu(dense, 1), u)
    assert_allclose(jnp.diag(dense), diag)

    # Invert the QS matrix
    minv = mat.inv()
    assert_allclose(minv.to_dense(), jnp.linalg.inv(dense))
    assert_allclose(minv.matmul(dense), jnp.eye(len(diag)))

    # In this case, we know our matrix to be symmetric - so should its inverse be!
    # This may change in the future as we expand test cases
    if not symm:
        assert_allclose(minv.lower.p, minv.upper.p)
        assert_allclose(minv.lower.q, minv.upper.q)
        assert_allclose(minv.lower.a, minv.upper.a)

    # The inverse of the inverse should be itself... don't actually do this!
    # Note: we can't actually directly compare the generators because there's
    # enough degrees of freedom that they won't necessarily round trip. It's
    # good enough to check that it produces the correct dense reconstruction.
    if dense.dtype == "float64":
        mat2 = minv.inv()
        assert_allclose(mat2.to_dense(), dense, rtol=1e-4)


def test_gram(matrices):
    diag, p, q, a, _, _, _, _ = matrices
    mat = SquareQSM(
        diag=DiagQSM(diag),
        lower=StrictLowerTriQSM(p=p, q=q, a=a),
        upper=StrictUpperTriQSM(p=p, q=q, a=a),
    )
    dense = mat.to_dense()
    assert_allclose(mat.gram().to_dense(), dense.T @ dense)

    mat = mat.inv()
    dense = mat.to_dense()
    assert_allclose(mat.gram().to_dense(), dense.T @ dense)

    mat = SquareQSM(
        diag=DiagQSM(diag),
        lower=StrictLowerTriQSM(p=p, q=q, a=a),
        upper=StrictUpperTriQSM(
            p=jnp.zeros_like(p), q=jnp.zeros_like(q), a=jnp.zeros_like(a)
        ),
    )
    dense = mat.to_dense()
    assert_allclose(mat.gram().to_dense(), dense.T @ dense)


@pytest.mark.parametrize("name", ["celerite"])
def test_cholesky(matrices):
    diag, p, q, a, v, m, _, _ = matrices
    mat = SymmQSM(diag=DiagQSM(diag), lower=StrictLowerTriQSM(p=p, q=q, a=a))
    dense = mat.to_dense()
    chol = mat.cholesky()
    assert_allclose(chol.to_dense(), jnp.linalg.cholesky(dense))

    mat = mat.inv()
    dense = mat.to_dense()
    chol = mat.cholesky()
    assert_allclose(chol.to_dense(), jnp.linalg.cholesky(dense))

    assert_allclose(chol.solve(v), jnp.linalg.solve(chol.to_dense(), v))
    assert_allclose(chol.solve(m), jnp.linalg.solve(chol.to_dense(), m))


def test_tri_qsmul(some_nice_matrices):
    mat1, mat2, mat3, mat4 = some_nice_matrices

    def check(mat1, mat2):
        mat = mat1 @ mat2
        a = mat.to_dense()
        b = mat1.to_dense() @ mat2.to_dense()
        assert_allclose(jnp.diag(a), jnp.diag(b))
        assert_allclose(jnp.tril(a, -1), jnp.tril(b, -1))
        assert_allclose(jnp.triu(a, 1), jnp.triu(b, 1))

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
        assert_allclose(jnp.diag(a), jnp.diag(b))
        assert_allclose(jnp.tril(a, -1), jnp.tril(b, -1))
        assert_allclose(jnp.triu(a, 1), jnp.triu(b, 1))

    for m1, m2 in combinations([mat1, mat2, mat3, mat4, mat1.inv(), mat2.inv()], 2):
        check(m1, m2)


def test_ops(some_nice_matrices):
    mat1, mat2, mat3, mat4 = some_nice_matrices

    def check(mat1, mat2):
        for m1, m2 in combinations([mat1, mat2, mat1.lower, mat2.lower], 2):
            a = m1.to_dense()
            b = m2.to_dense()
            assert_allclose((-m1).to_dense(), -a)
            assert_allclose((m1 + m2).to_dense(), a + b)
            assert_allclose((m1 - m2).to_dense(), a - b)
            assert_allclose((m1 * m2).to_dense(), a * b)
            assert_allclose((2.5 * m1).to_dense(), 2.5 * a)

    for m1, m2 in combinations([mat1, mat2, mat3, mat4, mat1.inv(), mat2.inv()], 2):
        check(m1, m2)
