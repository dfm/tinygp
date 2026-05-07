# mypy: ignore-errors

import jax.numpy as jnp
import pytest
from numpy import random as np_random

from tinygp.kernels.quasisep import Matern32, Matern52
from tinygp.solvers.quasisep.core import DiagQSM
from tinygp.solvers.quasisep.ops import (
    cholesky,
    cholesky_parallel,
    lower_matmul,
    lower_matmul_parallel,
    lower_solve,
    lower_solve_parallel,
    symm_inv,
    symm_inv_parallel,
    upper_matmul,
    upper_matmul_parallel,
    upper_solve,
    upper_solve_parallel,
)
from tinygp.test_utils import assert_allclose


@pytest.fixture(params=[Matern32, Matern52])
def data(request):
    N = 100
    random = np_random.default_rng(1234)
    t = jnp.sort(jnp.asarray(random.uniform(0, 10, N)))
    kernel = request.param(scale=1.3)
    qsm = kernel.to_symm_qsm(t) + DiagQSM(jnp.full(N, 0.1))
    (d,) = qsm.diag
    p, q, a = qsm.lower
    x = jnp.asarray(random.normal(size=(N, 3)))
    return d, p, q, a, x


def test_lower_matmul_parallel(data):
    _, p, q, a, x = data
    assert_allclose(lower_matmul_parallel(p, q, a, x), lower_matmul(p, q, a, x))


def test_upper_matmul_parallel(data):
    _, p, q, a, x = data
    assert_allclose(upper_matmul_parallel(p, q, a, x), upper_matmul(p, q, a, x))


def test_cholesky_parallel(data):
    d, p, q, a, _ = data
    c_seq, w_seq = cholesky(d, p, q, a)
    c_par, w_par = cholesky_parallel(d, p, q, a)
    assert_allclose(c_par, c_seq)
    assert_allclose(w_par, w_seq)


def test_lower_solve_parallel(data):
    d, p, q, a, x = data
    c, w = cholesky(d, p, q, a)
    assert_allclose(lower_solve_parallel(c, p, w, a, x), lower_solve(c, p, w, a, x))


def test_upper_solve_parallel(data):
    d, p, q, a, x = data
    c, w = cholesky(d, p, q, a)
    assert_allclose(upper_solve_parallel(c, p, w, a, x), upper_solve(c, p, w, a, x))


def test_symm_inv_parallel(data):
    d, p, q, a, _ = data
    lam_s, t_s, s_s, ell_s = symm_inv(d, p, q, a)
    lam_p, t_p, s_p, ell_p = symm_inv_parallel(d, p, q, a)
    assert_allclose(lam_p, lam_s)
    assert_allclose(t_p, t_s)
    assert_allclose(s_p, s_s)
    assert_allclose(ell_p, ell_s)
