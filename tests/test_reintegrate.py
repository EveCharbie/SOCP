import numpy as np
import numpy.testing as npt

from socp import VertebrateArm
from socp.analysis.reintegrate_solution import reintegrate, reintegrate_transcription_study


def _build_fake_ocp(n_shooting: int):
    ocp_example = VertebrateArm(nb_random=1)
    n_simulations = 64

    time_vector = np.linspace(0, 0.1, n_shooting + 1)
    states_opt_mean = np.tile(np.array([[1.5], [1.5], [0.0], [0.0]]), (1, n_shooting + 1))
    states_opt_array = np.zeros_like(states_opt_mean)
    controls_opt_array = np.zeros((6, n_shooting + 1))
    ref_opt_array = np.zeros((ocp_example.model.nb_references, n_shooting + 1))

    ocp = {
        "n_shooting": n_shooting,
        "ocp_example": ocp_example,
        "motor_noise_magnitude": np.array([0.01, 0.01]),
        "sensory_noise_magnitude": np.array([0.01, 0.01]),
    }
    return time_vector, states_opt_mean, states_opt_array, controls_opt_array, ref_opt_array, ocp, n_simulations


def _check_initial_state_exactness(x_simulated, states_opt_mean, ocp_example, n_simulations):
    initial_states = x_simulated[:, 0, :]
    initial_mean = states_opt_mean[:, 0]
    initial_covariance = np.diag(np.array(ocp_example.initial_state_variability) ** 2)

    npt.assert_array_almost_equal(np.mean(initial_states, axis=1), initial_mean)
    npt.assert_array_almost_equal(np.median(initial_states, axis=1), initial_mean)

    centered = initial_states - initial_mean[:, np.newaxis]
    recovered_covariance = (centered @ centered.T) / n_simulations
    npt.assert_array_almost_equal(recovered_covariance, initial_covariance)


def test_reintegrate_initial_state_is_exact():
    time_vector, states_opt_mean, states_opt_array, controls_opt_array, ref_opt_array, ocp, n_simulations = _build_fake_ocp(n_shooting=1)

    x_simulated = reintegrate(
        time_vector, states_opt_mean, states_opt_array, controls_opt_array, ref_opt_array,
        ocp, n_simulations, "dummy.pkl", plot_flag=False,
    )

    _check_initial_state_exactness(x_simulated, states_opt_mean, ocp["ocp_example"], n_simulations)


def test_reintegrate_transcription_study_initial_state_is_exact():
    time_vector, states_opt_mean, states_opt_array, controls_opt_array, ref_opt_array, ocp, n_simulations = _build_fake_ocp(n_shooting=1)

    x_simulated = reintegrate_transcription_study(
        time_vector, states_opt_mean, states_opt_array, controls_opt_array, ref_opt_array,
        ocp, n_simulations, "dummy.pkl", plot_flag=False,
    )

    _check_initial_state_exactness(x_simulated, states_opt_mean, ocp["ocp_example"], n_simulations)


def test_reintegrate_runs_end_to_end():
    """
    Smoke test with a couple of real shooting nodes (dynamics actually get integrated),
    catching signature/import breakage rather than checking specific numerical values.
    """
    time_vector, states_opt_mean, states_opt_array, controls_opt_array, ref_opt_array, ocp, n_simulations = _build_fake_ocp(n_shooting=2)

    x_simulated = reintegrate(
        time_vector, states_opt_mean, states_opt_array, controls_opt_array, ref_opt_array,
        ocp, n_simulations, "dummy.pkl", plot_flag=False,
    )

    assert x_simulated.shape == (ocp["ocp_example"].model.nb_states, 3, n_simulations)
    assert np.all(np.isfinite(x_simulated))
