import numpy as np
import numpy.testing as npt
import casadi as cas

from socp import ArmReaching, DirectMultipleShooting, NoiseDiscretization, MeanAndCovariance
from socp.transcriptions.unscented_transform import UnscentedTransform
from socp.transcriptions.variables_abstract import VariablesAbstract

MOTOR_NOISE_MAGNITUDE = np.array([0.05, 0.03])
SENSORY_NOISE_MAGNITUDE = np.array([0.01, 0.02, 0.015, 0.008])
SIGMA = np.concatenate([MOTOR_NOISE_MAGNITUDE, SENSORY_NOISE_MAGNITUDE])


def _declare_noises(discretization_cls):
    ocp_example = ArmReaching()
    dynamics_transcription = DirectMultipleShooting()
    discretization_method = discretization_cls(dynamics_transcription)
    noises_vector = discretization_method.declare_noises(
        ocp_example=ocp_example,
        dynamics_transcription=dynamics_transcription,
        n_shooting=3,
        nb_random=1,
        motor_noise_magnitude=MOTOR_NOISE_MAGNITUDE,
        sensory_noise_magnitude=SENSORY_NOISE_MAGNITUDE,
        seed=0,
    )
    return noises_vector


def test_noise_magnitude_matrix_is_std_not_variance():
    """
    noise_magnitude_matrix must store sigma (std), not sigma^2, for both NoiseDiscretization
    and MeanAndCovariance: NoiseDiscretization samples with it directly (np.random.normal's
    `scale` is a std), and get_sigma_states consumes it as a Cholesky-style "1 std" offset.
    """
    for discretization_cls in [NoiseDiscretization, MeanAndCovariance]:
        noises_vector = _declare_noises(discretization_cls)
        noise_magnitude_matrix = np.array(noises_vector.noise_magnitude_matrix)
        npt.assert_array_almost_equal(np.diag(noise_magnitude_matrix), SIGMA)
        off_diagonal = noise_magnitude_matrix - np.diag(np.diag(noise_magnitude_matrix))
        npt.assert_array_almost_equal(off_diagonal, np.zeros_like(off_diagonal))


def test_mean_and_covariance_sigma_ww_is_variance():
    """
    MeanAndCovariance's (and DirectMultipleShooting/DirectCollocation*/Variational*'s shared)
    sigma_ww = get_noise_matrix(node).T @ get_noise_matrix(node) pattern must recover sigma^2 when
    evaluated at the true noise magnitude, since it plugs directly into the Lyapunov covariance
    propagation formula P_{k+1} = dFdx @ P @ dFdx.T + dFdw @ sigma_ww @ dFdw.T.
    """
    noises_vector = _declare_noises(MeanAndCovariance)

    noise_single = noises_vector.get_noise_single(0)
    noise_matrix = noises_vector.get_noise_matrix(0)
    sigma_ww_func = cas.Function("sigma_ww", [noise_single], [noise_matrix.T @ noise_matrix])

    numeric_noise = np.array(noises_vector.get_one_vector_numerical(0)).flatten()
    sigma_ww = np.array(sigma_ww_func(numeric_noise))

    npt.assert_array_almost_equal(np.diag(sigma_ww), SIGMA**2)


def test_get_sigma_states_reconstructs_exact_covariance():
    """
    Regression test for the get_sigma_states scaling bug found while writing this suite:
    the +-offsets must be scaled by sqrt(n) (n = augmented state+noise dimension) so that the
    unweighted sample-covariance recovery formula used everywhere downstream
    (diff @ diff.T / (nb_sigma_points - 1)) reconstructs the true covariance exactly, not
    true_covariance / n.
    """
    nb_q = 2
    nb_states = 2 * nb_q
    nb_noises = 2
    n_augmented = nb_states + nb_noises
    nb_sigma_points = 1 + 2 * n_augmented

    state_indices = {"q": range(0, nb_q), "qdot": range(nb_q, 2 * nb_q)}
    variables = UnscentedTransform.Variables(
        n_shooting=1,
        nb_collocation_points=1,
        nb_m_points=1,
        state_indices=state_indices,
        control_indices={},
        ref_indices=range(0, 0),
        nb_random=1,
        nb_sigma_points=nb_sigma_points,
    )
    variables.add_time(0.0)

    q_mean = np.array([1.0, 2.0])
    qdot_mean = np.array([0.5, -0.5])
    for node in [0, 1]:
        for i_sigma in range(nb_sigma_points):
            variables.add_state("q", node, i_sigma, q_mean)
            variables.add_state("qdot", node, i_sigma, qdot_mean)

    rng = np.random.RandomState(0)
    random_matrix = rng.rand(nb_states, nb_states)
    target_state_covariance = random_matrix @ random_matrix.T + nb_states * np.eye(nb_states)
    chol_cov = np.linalg.cholesky(target_state_covariance)
    chol_cov_vector = VariablesAbstract.reshape_cholesky_matrix_to_vector(cas.DM(chol_cov))
    variables.add_chol_cov(0, chol_cov_vector)

    noise_sigma = np.array([0.1, 0.2])
    noise_matrix = cas.diag(cas.DM(noise_sigma))

    sigma_states = np.array(variables.get_sigma_states(0, noise_matrix))
    mean = sigma_states.mean(axis=1, keepdims=True)
    diff = sigma_states - mean
    recovered_covariance = (diff @ diff.T) / (sigma_states.shape[1] - 1)

    target_covariance = np.zeros((n_augmented, n_augmented))
    target_covariance[:nb_states, :nb_states] = target_state_covariance
    target_covariance[nb_states:, nb_states:] = np.diag(noise_sigma**2)

    npt.assert_array_almost_equal(recovered_covariance, target_covariance)
    npt.assert_array_almost_equal(mean[:nb_states].flatten(), np.hstack([q_mean, qdot_mean]))
