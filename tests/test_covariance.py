import numpy as np
import numpy.testing as npt

from socp import VertebrateArm, DirectMultipleShooting, NoiseDiscretization, prepare_ocp
from socp.analysis.estimate_covariance import estimate_covariance


def test_covariance_noise_sampling():

    ocp_example = VertebrateArm(nb_random=300)
    dynamics_transcription = DirectMultipleShooting()
    discretization_method = NoiseDiscretization(dynamics_transcription)

    ocp = prepare_ocp(
        ocp_example=ocp_example,
        dynamics_transcription=dynamics_transcription,
        discretization_method=discretization_method,
    )

    # Get the initial state covariance
    expected_covariance = ocp_example.initial_covariance
    states = ocp["variable_init"].get_states_matrix(node=0)
    mean_states = np.mean(states, axis=1)
    covariance = estimate_covariance(mean_states[:, np.newaxis], np.array(states)[:, np.newaxis, :])

    npt.assert_almost_equal(expected_covariance, covariance[:, :, 0], decimal=3)
