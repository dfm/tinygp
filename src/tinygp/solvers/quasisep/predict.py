"""O(J^2)-per-test-point predictive variance from the QSM Cholesky.

After the Cholesky factorization at N sorted training points, two train-only
scans (the Cholesky carry itself and one backward congruence) give a state from
which the predictive variance at any test point follows from a binary search
plus an O(J^2) contraction, using the same cross-covariance row generators
(:meth:`~tinygp.kernels.quasisep.Quasisep.anchor`) as the rectangular
quasiseparable product. The predictive *mean* needs none of this: it is that
product, ``kernel.matmul(X_test, X_train, alpha)``.

With the Cholesky factor ``L = (c; p, w, a)`` and ``v = L^{-1} k_*``, the
variance is ``k** - v^T v``. With ``(idx, pl, qu)`` the anchors of the test
point, ``iL = clip(idx)`` and ``iR = clip(idx + 1)``, the entries split into a
head, ``sum_{n <= iL} v_n^2 = pl^T f_iL pl`` with ``f`` the inclusive Cholesky
carry, and a tail ``sum_{n >= iR} v_n^2 = s^T P_iR s`` where
``s = qu - a_iR f_iL pl`` and ``P`` is the backward congruence
``P_k = A_k^T P_{k+1} A_k + h_k h_k^T / c_k^2`` with
``A_k = a_{k+1} (I - w_k h_k^T / c_k)``. Every propagation in this form runs
forward in time, so nothing here amplifies like ``exp(+gap / scale)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp

from tinygp.helpers import JAXArray
from tinygp.kernels.base import Conditioned
from tinygp.solvers.quasisep import ops
from tinygp.solvers.quasisep.block import ensure_dense

if TYPE_CHECKING:
    from tinygp.kernels.quasisep import Quasisep
    from tinygp.solvers.quasisep.solver import QuasisepSolver


class PredictState(eqx.Module):
    """Train-only state for fast prediction; built once by :func:`precompute`."""

    f: JAXArray  # (N, J, J), the inclusive Cholesky carry
    P: JAXArray  # (N, J, J), the backward congruence accumulator


def precompute(solver: QuasisepSolver) -> PredictState:
    """Run the two train-only scans and bundle them into a :class:`PredictState`.

    The Cholesky carry is recomputed here (one extra scan) rather than stored
    on the solver, so likelihood-only workflows never hold the N x J^2 array.
    """
    (d,) = solver.matrix.diag
    p, q, a = solver.matrix.lower
    c = solver.factor.diag.d
    w = solver.factor.lower.q
    impl = ops.cholesky_parallel if solver.parallel else ops.cholesky
    f = impl(d, p, q, a)[2]

    h = jax.vmap(solver.kernel.observation_model)(solver.X)
    a_next = jax.vmap(ensure_dense)(
        jax.tree_util.tree_map(
            lambda v: jnp.concatenate([v[1:], jnp.eye(v.shape[-1])[None]]), a
        )
    )
    inv_c = 1.0 / c
    A = a_next @ (jnp.eye(h.shape[1]) - jnp.einsum("n,nj,nk->njk", inv_c, w, h))
    B = jnp.einsum("n,nj,nk->njk", inv_c**2, h, h)
    P = ops.congruence_scan(
        jnp.swapaxes(A, -1, -2), B, reverse=True, parallel=solver.parallel
    )
    return PredictState(f=f, P=P)


def predict_var(
    kernel: Quasisep, solver: QuasisepSolver, state: PredictState, x_star: JAXArray
) -> JAXArray:
    """Predictive (noise-free) variance at one test point."""
    idx, pl, qu = kernel.anchor(x_star, solver.X)
    N = state.f.shape[0]
    iL = jnp.clip(idx, 0, N - 1)
    iR = jnp.clip(idx + 1, 0, N - 1)
    f = state.f[iL]  # pl == 0 when idx < 0, so no boundary element is needed
    a_R = ensure_dense(jax.tree_util.tree_map(lambda v: v[iR], solver.matrix.lower.a))
    s = qu - a_R @ (f @ pl)
    tail = jnp.where(idx < N - 1, s @ state.P[iR] @ s, 0.0)
    return kernel.evaluate_diag(x_star) - pl @ f @ pl - tail


class ConditionedKernel(Conditioned):
    """Conditioned kernel with the fast O(J^2) diagonal variance.

    Inherits the dense off-diagonal :meth:`evaluate` and block ``__call__``
    from :class:`tinygp.kernels.Conditioned` (building a full ``M x M`` block
    materializes a dense ``N x M`` cross covariance and triangular solve;
    prefer :meth:`evaluate_diag` for variances) and overrides only the
    diagonal.
    """

    state: PredictState

    def evaluate_diag(self, X: JAXArray) -> JAXArray:
        return predict_var(self.kernel, self.solver, self.state, X)
