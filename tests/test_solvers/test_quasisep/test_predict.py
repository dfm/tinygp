# mypy: ignore-errors

import jax
import jax.numpy as jnp
import pytest
from numpy import random as np_random

from tinygp import GaussianProcess
from tinygp.kernels import quasisep as qk
from tinygp.noise import Dense
from tinygp.solvers.quasisep import predict
from tinygp.solvers.quasisep.solver import QuasisepSolver
from tinygp.test_utils import assert_allclose


@pytest.fixture
def data():
    rng = np_random.default_rng(42)
    N, M = 30, 50
    X_train = jnp.sort(jnp.asarray(rng.uniform(0, 10, N)))
    y = jnp.asarray(rng.normal(size=N))
    # include extrapolation on both sides
    X_test = jnp.sort(jnp.asarray(rng.uniform(-2, 12, M)))
    return X_train, y, X_test


@pytest.fixture(params=[False, True], ids=["sequential", "parallel"])
def parallel(request):
    return request.param


@pytest.fixture(
    params=[
        qk.Matern32(scale=1.5),
        qk.Matern52(scale=1.5),
        qk.SHO(omega=2.0, quality=3.0),
        qk.Cosine(scale=1.2),
        qk.Matern32(scale=1.0) + qk.Matern52(scale=2.0),
        2.5 * qk.Matern32(scale=1.5),
    ],
    ids=["Matern32", "Matern52", "SHO", "Cosine", "sum", "scaled"],
)
def kernel(request):
    return request.param


def test_predict_mean_and_var(data, kernel, parallel):
    X_train, y, X_test = data
    diag = 0.1
    N = X_train.shape[0]

    gp = GaussianProcess(
        kernel, X_train, diag=diag, solver=QuasisepSolver, parallel=parallel
    )
    solver = gp.solver

    # dense reference
    K = kernel(X_train, X_train) + diag * jnp.eye(N)
    Ks = kernel(X_train, X_test)
    Kss = jax.vmap(kernel.evaluate_diag)(X_test)
    L = jnp.linalg.cholesky(K)
    beta_ref = jnp.linalg.solve(K, y)
    mu_ref = Ks.T @ beta_ref
    A = jax.scipy.linalg.solve_triangular(L, Ks, lower=True)
    var_ref = Kss - jnp.sum(A**2, axis=0)

    alpha = solver.solve_triangular(y)
    beta = solver.solve_triangular(alpha, transpose=True)
    state = predict.precompute(solver, beta, parallel=parallel)

    mu = jax.vmap(lambda x: predict.predict_mean(kernel, state, x))(X_test)
    var = jax.vmap(lambda x: predict.predict_var(kernel, state, x))(X_test)

    assert_allclose(mu, mu_ref)
    assert_allclose(var, var_ref)


def test_condition_end_to_end(data, kernel, parallel):
    X_train, y, X_test = data
    diag = 0.1
    N = X_train.shape[0]

    gp = GaussianProcess(
        kernel, X_train, diag=diag, solver=QuasisepSolver, parallel=parallel
    )
    cond = gp.condition(y, X_test).gp

    K = kernel(X_train, X_train) + diag * jnp.eye(N)
    Ks = kernel(X_train, X_test)
    Kss = jax.vmap(kernel.evaluate_diag)(X_test)
    L = jnp.linalg.cholesky(K)
    mu_ref = Ks.T @ jnp.linalg.solve(K, y)
    A = jax.scipy.linalg.solve_triangular(L, Ks, lower=True)
    var_ref = Kss - jnp.sum(A**2, axis=0)

    assert_allclose(cond.mean, mu_ref)
    assert_allclose(cond.variance, var_ref)

    # Re-evaluation at fresh points goes through the Conditioned mean/kernel.
    rng = np_random.default_rng(7)
    X_new = jnp.asarray(rng.uniform(-1, 11, 5))
    Ks_new = kernel(X_train, X_new)
    Kss_new = jax.vmap(kernel.evaluate_diag)(X_new)
    mu_new = Ks_new.T @ jnp.linalg.solve(K, y)
    A_new = jax.scipy.linalg.solve_triangular(L, Ks_new, lower=True)
    var_new = Kss_new - jnp.sum(A_new**2, axis=0)

    assert_allclose(jax.vmap(cond.mean_function)(X_new), mu_new)
    assert_allclose(jax.vmap(cond.kernel.evaluate_diag)(X_new), var_new)


def _dense_reference(kernel, X_train, y, X_test, diag):
    N = X_train.shape[0]
    K = kernel(X_train, X_train) + diag * jnp.eye(N)
    Ks = kernel(X_train, X_test)
    Kss = jax.vmap(kernel.evaluate_diag)(X_test)
    mu = Ks.T @ jnp.linalg.solve(K, y)
    L = jnp.linalg.cholesky(K)
    A = jax.scipy.linalg.solve_triangular(L, Ks, lower=True)
    var = Kss - jnp.sum(A**2, axis=0)
    return mu, var


# Geometries where an anchoring off-by-one would bite: test points coincident
# with training points, duplicated training times, tiny datasets, and
# extrapolation past both ends of the data.
GEOMETRIES = {
    "coincident": (
        jnp.array([0.0, 1.0, 2.5, 4.0]),
        jnp.array([-1.0, 1.0, 2.5, 4.0, 7.0]),
    ),
    "duplicates": (
        jnp.array([0.0, 1.0, 1.0, 1.0, 3.0]),
        jnp.array([0.5, 1.0, 2.0]),
    ),
    "single": (jnp.array([1.0]), jnp.array([0.0, 1.0, 2.0])),
    "pair": (jnp.array([1.0, 2.0]), jnp.array([0.0, 1.5, 3.0])),
    "extrapolation": (
        jnp.array([0.0, 1.0, 2.0]),
        jnp.array([-10.0, -0.5, 2.5, 10.0]),
    ),
}


@pytest.mark.parametrize("geometry", sorted(GEOMETRIES))
def test_edge_case_geometries(kernel, parallel, geometry):
    X_train, X_test = GEOMETRIES[geometry]
    rng = np_random.default_rng(99)
    y = jnp.asarray(rng.normal(size=X_train.shape[0]))
    diag = 0.1

    gp = GaussianProcess(
        kernel, X_train, diag=diag, solver=QuasisepSolver, parallel=parallel
    )
    cond = gp.condition(y, X_test).gp
    mu_ref, var_ref = _dense_reference(kernel, X_train, y, X_test, diag)
    assert_allclose(cond.mean, mu_ref)
    assert_allclose(cond.variance, var_ref)


def test_far_extrapolation_gradient_finite(parallel):
    # Regression test: the anchors were computed unconditionally and only masked
    # with a single jnp.where, so for a test point far past the data the
    # transition over the (negative) extrapolation gap overflowed to inf in
    # Matern-type kernels and the VJP turned the masked inf into a NaN gradient.
    rng = np_random.default_rng(0)
    X_train = jnp.sort(jnp.asarray(rng.uniform(0, 10, 20)))
    y = jnp.asarray(rng.normal(size=20))
    X_test = jnp.array([-5e4, 5.0, 5e4])

    @jax.grad
    def objective(scale):
        gp = GaussianProcess(
            qk.Matern32(scale=scale),
            X_train,
            diag=0.1,
            solver=QuasisepSolver,
            parallel=parallel,
        )
        cond = gp.condition(y, X_test).gp
        return jnp.sum(cond.mean) + jnp.sum(cond.variance)

    assert jnp.isfinite(objective(1.5))


def test_conditioned_covariance_includes_full_noise(parallel):
    # Regression test: ConditionedSolver.covariance() previously added
    # jnp.diag(noise.diagonal()), silently dropping the off-diagonal part of a
    # Dense test-noise model that the other conditioning paths keep.
    rng = np_random.default_rng(5)
    N, M = 8, 5
    X_train = jnp.sort(jnp.asarray(rng.uniform(0, 10, N)))
    y = jnp.asarray(rng.normal(size=N))
    X_test = jnp.sort(jnp.asarray(rng.uniform(0, 10, M)))
    R = rng.normal(size=(M, M))
    R = jnp.asarray(R @ R.T + M * jnp.eye(M))

    kernel = qk.Matern32(scale=1.5)
    diag = 0.1
    gp = GaussianProcess(
        kernel, X_train, diag=diag, solver=QuasisepSolver, parallel=parallel
    )
    cond = gp.condition(y, X_test, noise=Dense(value=R)).gp

    K = kernel(X_train, X_train) + diag * jnp.eye(N)
    Ks = kernel(X_train, X_test)
    cov_ref = kernel(X_test, X_test) - Ks.T @ jnp.linalg.solve(K, Ks) + R
    assert_allclose(cond.covariance, cov_ref)


def test_fast_path_survives_jit(data):
    # Regression: the fast path must fire when the GP crosses a jit/pytree
    # boundary. Gating it on object identity (``kernel is self.kernel``) silently
    # disabled it under jit, because flatten/unflatten produces distinct objects.
    X_train, y, X_test = data
    gp = GaussianProcess(
        qk.Matern32(scale=1.5), X_train, diag=0.1, solver=QuasisepSolver
    )

    # Round-tripping the pytree is exactly what jit does to ``self``.
    leaves, treedef = jax.tree_util.tree_flatten(gp)
    gp_rt = jax.tree_util.tree_unflatten(treedef, leaves)
    cond = gp_rt.condition(y, X_test).gp
    assert isinstance(cond.kernel, predict.ConditionedKernel)
    assert isinstance(cond.solver, predict.ConditionedSolver)

    # And the jitted entry point agrees with the eager fast path to precision.
    mu_eager, var_eager = gp.predict(y, X_test, return_var=True)
    mu_jit, var_jit = jax.jit(lambda g, yy, xx: g.predict(yy, xx, return_var=True))(
        gp, y, X_test
    )
    assert_allclose(mu_jit, mu_eager)
    assert_allclose(var_jit, var_eager)
