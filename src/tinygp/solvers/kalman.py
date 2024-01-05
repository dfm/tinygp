from __future__ import annotations

__all__ = ["kalman_filter", "KalmanSolver"]

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tinygp.helpers import JAXArray
from tinygp.kernels.base import Kernel
from tinygp.noise import Diagonal, Noise
from tinygp.solvers.solver import Solver


class KalmanSolver(Solver):
    """A scalable solver that uses Kalman filtering

    This implementation is very limited and it is meant primarily

    You generally won't instantiate this object directly but, if you do, you'll
    probably want to use the :func:`KalmanSolver.init` method instead of the
    usual constructor.
    """

    X: JAXArray
    A: JAXArray
    H: JAXArray
    s: JAXArray
    K: JAXArray

    def __init__(
        self,
        kernel: Kernel,
        X: JAXArray,
        noise: Noise,
        *,
        covariance: Any | None = None,
    ):
        """Build a :class:`KalmanSolver` for a given kernel and coordinates

        Args:
            kernel: The kernel function. This must be an instance of a subclass
                of :class:`tinygp.kernels.quasisep.Quasisep`.
            X: The input coordinates.
            noise: The noise model for the process. This must be diagonal for
                this solver.
            covariance: Not yet supported by this solver.
        """
        from tinygp.kernels.quasisep import Quasisep

        assert isinstance(kernel, Quasisep)
        assert isinstance(noise, Diagonal)
        assert covariance is None

        self.X = X
        Pinf = kernel.stationary_covariance()
        self.A = jax.vmap(kernel.transition_matrix)(
            jax.tree_util.tree_map(lambda y: jnp.append(y[0], y[:-1]), X), X
        )
        self.H = jax.vmap(kernel.observation_model)(X)
        self.s, self.K = kalman_gains(Pinf, self.A, self.H, noise.diag)

    def variance(self) -> JAXArray:
        raise NotImplementedError

    def covariance(self) -> JAXArray:
        raise NotImplementedError

    def normalization(self) -> JAXArray:
        return 0.5 * jnp.sum(jnp.log(2 * np.pi * self.s))

    def solve_triangular(self, y: JAXArray, *, transpose: bool = False) -> JAXArray:
        assert not transpose
        return kalman_filter(self.A, self.H, self.K, y) / jnp.sqrt(self.s)

    def dot_triangular(self, y: JAXArray) -> JAXArray:
        del y
        raise NotImplementedError

    def condition(self, kernel: Kernel, X_test: JAXArray | None, noise: Noise) -> Any:
        del kernel, X_test, noise
        raise NotImplementedError


@jax.jit
def kalman_gains(
    Pinf: JAXArray, A: JAXArray, H: JAXArray, diag: JAXArray
) -> tuple[JAXArray, JAXArray]:
    def step(carry, data):  # type: ignore
        Pp = carry
        Ak, hk, dk = data

        Pn = Pinf + Ak.transpose() @ (Pp - Pinf) @ Ak
        tmp = Pn @ hk
        sk = hk @ tmp + dk
        Kk = tmp / sk
        Pk = Pn - sk * jnp.outer(Kk, Kk)

        return Pk, (sk, Kk)

    init = Pinf
    return jax.lax.scan(step, init, (A, H, diag))[1]


@jax.jit
def kalman_filter(A: JAXArray, H: JAXArray, K: JAXArray, y: JAXArray) -> JAXArray:
    def step(carry, data):  # type: ignore
        mp = carry
        Ak, hk, Kk, yk = data

        mn = Ak.transpose() @ mp
        vk = yk - hk @ mn
        mk = mn + Kk * vk

        return mk, vk

    init = jnp.zeros_like(H[0])
    return jax.lax.scan(step, init, (A, H, K, y))[1]
