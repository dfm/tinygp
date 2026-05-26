"""O(J^2)-per-test-point predictive mean and variance from the QSM Cholesky.

After the Cholesky factorization at N sorted training points, four train-only
affine scans give a state from which the predictive mean and (diagonal)
variance at any test point follow from a binary search plus an O(J^2)
contraction. This module also provides the conditioned mean, kernel, and lazy
solver objects that hold that state for use in a conditioned
:class:`~tinygp.GaussianProcess`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp

from tinygp.helpers import JAXArray
from tinygp.kernels.base import Conditioned, Kernel
from tinygp.means import MeanBase
from tinygp.noise import Noise
from tinygp.solvers.direct import DirectSolver
from tinygp.solvers.quasisep.block import ensure_dense
from tinygp.solvers.solver import ConditionedComponents, Solver

if TYPE_CHECKING:
    from tinygp.kernels.quasisep import Quasisep
    from tinygp.solvers.quasisep.solver import QuasisepSolver


class PredictState(eqx.Module):
    """Train-only state for fast prediction; built once by :func:`precompute`."""

    X_train: JAXArray  # the training inputs (any pytree)
    t_train: JAXArray  # (N,) sortable coordinates, for the per-test binary search
    a_inv: JAXArray  # (N, J, J), inverse training transitions (right-anchor pull-back)
    h_fwd: JAXArray  # (N+1, J), forward mean accumulator (depends on y)
    h_bwd: JAXArray  # (N+1, J), backward mean accumulator (depends on y)
    f: JAXArray  # (N+1, J, J), the (inclusive) Cholesky carry
    P: JAXArray  # (N+1, J, J), backward congruence accumulator


def precompute(
    solver: QuasisepSolver, beta: JAXArray, *, parallel: bool
) -> PredictState:
    """Run the four train-only scans and bundle them into a :class:`PredictState`."""
    a = jax.vmap(ensure_dense)(solver.matrix.lower.a)
    # The right-anchor pull-back needs a[iR]^{-1} per test point. Precompute the
    # inverse training transitions once here, so the per-test step is a matmul
    # rather than a batched J x J linalg.solve -- the latter lowers to a general
    # per-matrix LU that is far slower and erratic (non-monotonic in J) for tiny
    # matrices, which otherwise dominates the per-test cost at large M. The
    # inverse is analytic: a[k] = transition(t_{k-1}, t_k), so
    # a[k]^{-1} = transition(t_k, t_{k-1}), avoiding an LU entirely.
    kernel = solver.kernel
    X_prev = jax.tree_util.tree_map(
        lambda v: jnp.concatenate([v[:1], v[:-1]], axis=0), solver.X
    )
    a_inv = jax.vmap(lambda xc, xp: ensure_dense(kernel.transition_matrix(xc, xp)))(
        solver.X, X_prev
    )
    t_train = jax.vmap(kernel.coord_to_sortable)(solver.X)
    h_fwd, h_bwd = _precompute_mean(solver, a, beta, parallel=parallel)
    f, P = _precompute_variance(solver, a, parallel=parallel)
    return PredictState(
        X_train=solver.X,
        t_train=t_train,
        a_inv=a_inv,
        h_fwd=h_fwd,
        h_bwd=h_bwd,
        f=f,
        P=P,
    )


def _pad_fwd(x: JAXArray) -> JAXArray:
    return jnp.concatenate([jnp.zeros_like(x[:1]), x], axis=0)


def _pad_bwd(x: JAXArray) -> JAXArray:
    return jnp.concatenate([x, jnp.zeros_like(x[-1:])], axis=0)


def _precompute_mean(
    solver: QuasisepSolver, a: JAXArray, beta: JAXArray, *, parallel: bool
) -> tuple[JAXArray, JAXArray]:
    """Two affine matmul scans over the training data; depends on y via beta."""
    p, q, _ = solver.matrix.lower
    h_fwd = _scan_h(a, q * beta[:, None], reverse=False, parallel=parallel)
    h_bwd = _scan_h(a, p * beta[:, None], reverse=True, parallel=parallel)
    return _pad_fwd(h_fwd), _pad_bwd(h_bwd)


def _scan_h(a: JAXArray, b: JAXArray, *, reverse: bool, parallel: bool) -> JAXArray:
    """Inclusive prefix of ``h_k = A_k h_{k-1} + b_k`` (``A_k = a_k^T`` if reverse)."""
    A = jnp.swapaxes(a, -1, -2) if reverse else a
    if parallel:

        def combine(left, right):
            (Al, bl), (Ar, br) = left, right
            return Ar @ Al, jnp.einsum("...ij,...j->...i", Ar, bl) + br

        _, h = jax.lax.associative_scan(combine, (A, b), reverse=reverse)
        return h

    def impl(h, data):
        Ak, bk = data
        hk = Ak @ h + bk
        return hk, hk

    _, h = jax.lax.scan(impl, jnp.zeros_like(b[0]), (A, b), reverse=reverse)
    return h


def _precompute_variance(
    solver: QuasisepSolver, a: JAXArray, *, parallel: bool
) -> tuple[JAXArray, JAXArray]:
    """Forward variance term reuses the carry; one backward congruence scan.

    ``cholesky_carry`` is the *exclusive* carry (the value entering each step,
    ``f_excl[k] = f_{k-1}``), so the *inclusive* carry ``f_incl[k] = f_k`` is the
    same array shifted by one, ``f_incl[k] = f_excl[k+1]``, with a single
    boundary element ``f_{N-1} = a_{N-1} f_excl[N-1] a_{N-1}^T + w_{N-1}
    w_{N-1}^T`` for the last point. The padded array returned here (read at anchor
    ``idx = i+1`` in ``predict_var``) is therefore just
    ``[0, f_0, ..., f_{N-1}] = concat(f_excl, f_{N-1})``, avoiding the O(N J^3)
    einsum that would otherwise recompute values already in ``f_excl``.
    """
    c = solver.factor.diag.d
    p, w, _ = solver.factor.lower
    f_excl = solver.cholesky_carry
    last = a[-1] @ f_excl[-1] @ a[-1].T + jnp.outer(w[-1], w[-1])
    f = jnp.concatenate([f_excl, last[None]], axis=0)
    inv_c = 1.0 / c
    A = a - jnp.einsum("n,nj,nk->njk", inv_c, w, p)
    B = jnp.einsum("n,nj,nk->njk", inv_c**2, p, p)
    P_incl = _congruence_inclusive(
        jnp.swapaxes(A, -1, -2), B, reverse=True, parallel=parallel
    )
    return f, _pad_bwd(P_incl)


def _congruence_inclusive(
    A: JAXArray, B: JAXArray, *, reverse: bool, parallel: bool
) -> JAXArray:
    if parallel:

        def combine(left, right):
            (Al, Bl), (Ar, Br) = left, right
            return Ar @ Al, Ar @ Bl @ jnp.swapaxes(Ar, -1, -2) + Br

        _, z = jax.lax.associative_scan(combine, (A, B), reverse=reverse)
        return z

    def impl(z, data):
        Ak, Bk = data
        zk = Ak @ z @ Ak.T + Bk
        return zk, zk

    init = jnp.zeros_like(B[-1] if reverse else B[0])
    _, z = jax.lax.scan(impl, init, (A, B), reverse=reverse)
    return z


def _anchor(
    kernel: Quasisep, state: PredictState, x_star: JAXArray
) -> tuple[JAXArray, JAXArray, JAXArray]:
    """Binary-search a test point and build its two cross-covariance anchors."""
    t_train = state.t_train
    N = t_train.shape[0]
    ts = kernel.coord_to_sortable(x_star)
    idx = jnp.searchsorted(t_train, ts, side="right")
    iL = jnp.clip(idx - 1, 0, N - 1)
    iR = jnp.clip(idx, 0, N - 1)

    h_star = kernel.observation_model(x_star)
    Pinf = kernel.stationary_covariance()

    xL = jax.tree_util.tree_map(lambda v: v[iL], state.X_train)
    xR = jax.tree_util.tree_map(lambda v: v[iR], state.X_train)

    # Left anchor: propagate from t_iL to t_*.
    phi_L = kernel.transition_matrix(xL, x_star)
    xi = phi_L.T @ (Pinf @ h_star)
    xi = jnp.where(idx > 0, xi, jnp.zeros_like(xi))

    # Right anchor: propagate from t_* to t_iR, then pull back across the
    # training gap a[iR] = transition(t_iL, t_iR) so the contraction lines up
    # with h_bwd's accumulation in p (which already absorbs one a and Pinf). The
    # pull-back a[iR]^{-1} uses the precomputed inverse (see precompute) so this
    # is a matmul rather than a per-test linalg.solve.
    phi_R = kernel.transition_matrix(x_star, xR)
    zeta = state.a_inv[iR] @ (phi_R @ h_star)
    zeta = jnp.where(idx < N, zeta, jnp.zeros_like(zeta))

    return idx, xi, zeta


def predict_mean_and_var(
    kernel: Quasisep, state: PredictState, x_star: JAXArray
) -> tuple[JAXArray, JAXArray]:
    """Predictive mean and diagonal variance at one test point (shared anchor)."""
    idx, xi, zeta = _anchor(kernel, state, x_star)
    mean = xi @ state.h_fwd[idx] + zeta @ state.h_bwd[idx]
    f_i = state.f[idx]
    delta = zeta - f_i @ xi
    var = kernel.evaluate_diag(x_star) - xi @ f_i @ xi - delta @ state.P[idx] @ delta
    return mean, var


def predict_mean(kernel: Quasisep, state: PredictState, x_star: JAXArray) -> JAXArray:
    idx, xi, zeta = _anchor(kernel, state, x_star)
    return xi @ state.h_fwd[idx] + zeta @ state.h_bwd[idx]


def predict_var(kernel: Quasisep, state: PredictState, x_star: JAXArray) -> JAXArray:
    idx, xi, zeta = _anchor(kernel, state, x_star)
    f_i = state.f[idx]
    delta = zeta - f_i @ xi
    return kernel.evaluate_diag(x_star) - xi @ f_i @ xi - delta @ state.P[idx] @ delta


class ConditionedCovariance(eqx.Module):
    """Marker passed as ``covariance_value`` to select the lazy conditioned solver."""

    train_solver: QuasisepSolver
    train_kernel: Quasisep


class ConditionedMean(MeanBase):
    state: PredictState
    kernel: Quasisep
    include_mean: bool = eqx.field(static=True)
    mean_function: MeanBase | None = None

    def __call__(self, X: JAXArray) -> JAXArray:
        mu = predict_mean(self.kernel, self.state, X)
        if self.include_mean and self.mean_function is not None:
            mu += self.mean_function(X)
        return mu


class ConditionedKernel(Conditioned):
    """Conditioned kernel with the fast O(J^2) diagonal variance.

    Inherits the dense off-diagonal :meth:`evaluate` from
    :class:`tinygp.kernels.Conditioned` (each pairwise call costs an O(N)
    triangular solve, so building a full block is O(M^2 N); prefer
    :meth:`evaluate_diag` for variances) and overrides only the diagonal.
    """

    state: PredictState

    def evaluate_diag(self, X: JAXArray) -> JAXArray:
        return predict_var(self.kernel, self.state, X)


class ConditionedSolver(Solver):
    """Lazy solver for a GP conditioned via the quasiseparable Cholesky.

    The conditioned GP stores its mean and variance directly (precomputed via
    the fast path), so ``cond.loc`` and ``cond.variance`` never reach this
    solver. It is only hit by operations that need the *full* conditional
    covariance: ``cond.covariance`` builds the dense matrix with a fast
    quasiseparable solve (no factorization), while ``log_probability``,
    ``sample``, and the triangular solves additionally Cholesky-factor it via a
    :class:`~tinygp.solvers.direct.DirectSolver`. That factorization is rebuilt
    on each call and not cached, so those operations are O(M^3) every time --
    cheap to call by accident, expensive at large M. (Under ``jax.jit`` repeated
    factorizations in one call are de-duplicated by common-subexpression
    elimination.)
    """

    X: JAXArray
    noise: Noise
    train_kernel: Quasisep
    train_solver: QuasisepSolver

    def __init__(
        self,
        kernel: Kernel,
        X: JAXArray,
        noise: Noise,
        *,
        covariance: Any | None = None,
        **_: Any,
    ):
        del kernel
        if TYPE_CHECKING:
            assert isinstance(covariance, ConditionedCovariance)
        self.X = X
        self.noise = noise
        self.train_kernel = covariance.train_kernel
        self.train_solver = covariance.train_solver

    def covariance(self) -> JAXArray:
        Ks = self.train_kernel(self.train_solver.X, self.X)
        A = self.train_solver.solve_triangular(Ks)
        Kss = self.train_kernel(self.X, self.X)
        return Kss - A.T @ A + jnp.diag(self.noise.diagonal())

    def variance(self) -> JAXArray:
        return jnp.diagonal(self.covariance())

    def _dense(self) -> Solver:
        return DirectSolver(
            self.train_kernel, self.X, self.noise, covariance=self.covariance()
        )

    def normalization(self) -> JAXArray:
        return self._dense().normalization()

    def solve_triangular(self, y: JAXArray, *, transpose: bool = False) -> JAXArray:
        return self._dense().solve_triangular(y, transpose=transpose)

    def dot_triangular(self, y: JAXArray) -> JAXArray:
        return self._dense().dot_triangular(y)

    def condition(
        self,
        kernel: Kernel | None,
        X_train: JAXArray,
        X_test: JAXArray | None,
        noise: Noise,
        alpha: JAXArray,
        *,
        include_mean: bool,
        mean_function: MeanBase,
    ) -> ConditionedComponents:
        return self._dense().condition(
            kernel,
            X_train,
            X_test,
            noise,
            alpha,
            include_mean=include_mean,
            mean_function=mean_function,
        )
