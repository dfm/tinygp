import jax
import jax.numpy as jnp
import jax.scipy as jsp

from tinygp import GaussianProcess
from tinygp.kernels import quasisep
from tinygp.solvers import DirectSolver, QuasisepSolver
from tinygp.solvers.kalman import KalmanSolver
from tinygp.test_utils import assert_allclose


class CausalFilter(quasisep.Quasisep):
    """A minimal non-reversible two-state process: driver -> response."""

    def design_matrix(self):
        return jnp.array([[-1.0, 0.0], [0.8, -2.0]])

    def stationary_covariance(self):
        return jnp.array([[0.5, 2.0 / 15.0], [2.0 / 15.0, 91.0 / 300.0]])

    def observation_model(self, X):
        _time, channel = X
        return jnp.eye(2)[channel]

    def coord_to_sortable(self, X):
        time, _channel = X
        return time

    def transition_matrix(self, X1, X2):
        time1, _channel1 = X1
        time2, _channel2 = X2
        return jsp.linalg.expm(self.design_matrix().T * (time2 - time1))


def direct_covariance(kernel, X1, X2):
    def evaluate(x1, x2):
        t1, _ = x1
        t2, _ = x2
        h1 = kernel.observation_model(x1)
        h2 = kernel.observation_model(x2)
        Pinf = kernel.stationary_covariance()
        return jax.lax.cond(
            t1 < t2,
            lambda: h2 @ kernel.transition_matrix(x1, x2).T @ Pinf @ h1,
            lambda: h1 @ kernel.transition_matrix(x2, x1).T @ Pinf @ h2,
        )

    return jax.vmap(lambda x1: jax.vmap(lambda x2: evaluate(x1, x2))(X2))(X1)


def test_nonreversible_covariance_and_cross_matmul():
    kernel = CausalFilter()
    time = jnp.array([0.0, 0.3, 0.8, 1.4, 2.2, 3.1])
    channel = jnp.array([0, 1, 0, 1, 1, 0])
    X = (time, channel)
    X_test = (jnp.array([0.1, 0.9, 1.8, 2.8]), jnp.array([1, 0, 1, 0]))

    expected = direct_covariance(kernel, X, X)
    assert_allclose(kernel(X, X), expected)
    assert_allclose(kernel.to_symm_qsm(X).to_dense(), expected)

    cross = direct_covariance(kernel, X_test, X)
    y = jnp.linspace(-0.5, 0.7, time.size)
    assert_allclose(kernel.matmul(X_test, X, y), cross @ y)

    # The response after a driver impulse differs from the reverse ordering;
    # reversible test kernels cannot expose this orientation requirement.
    driver_then_response = direct_covariance(
        kernel, (jnp.array([1.0]), jnp.array([1])), (jnp.array([0.0]), jnp.array([0]))
    )[0, 0]
    response_then_driver = direct_covariance(
        kernel, (jnp.array([1.0]), jnp.array([0])), (jnp.array([0.0]), jnp.array([1]))
    )[0, 0]
    assert not jnp.isclose(driver_then_response, response_then_driver)


def test_nonreversible_solvers_and_conditioning_agree():
    kernel = CausalFilter()
    time = jnp.array([0.0, 0.3, 0.8, 1.4, 2.2, 3.1])
    channel = jnp.array([0, 1, 0, 1, 1, 0])
    X = (time, channel)
    y = jnp.array([0.2, -0.1, 0.3, 0.15, -0.2, 0.05])
    diag = jnp.full(time.shape, 0.05)

    gp_direct = GaussianProcess(kernel, X, diag=diag, solver=DirectSolver)
    gp_quasisep = GaussianProcess(kernel, X, diag=diag, solver=QuasisepSolver)
    gp_kalman = GaussianProcess(kernel, X, diag=diag, solver=KalmanSolver)

    assert_allclose(gp_quasisep.covariance, gp_direct.covariance)
    assert_allclose(gp_quasisep.log_probability(y), gp_direct.log_probability(y))
    assert_allclose(gp_kalman.log_probability(y), gp_direct.log_probability(y))

    conditioned_direct = gp_direct.condition(y)
    conditioned_quasisep = gp_quasisep.condition(y)
    assert_allclose(conditioned_quasisep.gp.loc, conditioned_direct.gp.loc)
    assert_allclose(
        conditioned_quasisep.gp.covariance, conditioned_direct.gp.covariance
    )

    X_test = (jnp.array([0.1, 0.9, 1.8, 2.8]), jnp.array([1, 0, 1, 0]))
    conditioned_direct = gp_direct.condition(y, X_test=X_test)
    conditioned_quasisep = gp_quasisep.condition(y, X_test=X_test)
    assert_allclose(conditioned_quasisep.gp.loc, conditioned_direct.gp.loc)
    assert_allclose(
        conditioned_quasisep.gp.covariance, conditioned_direct.gp.covariance
    )
