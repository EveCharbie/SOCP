import numpy as np
import numpy.testing as npt

from socp import ArmReaching
from socp.examples.arm_reaching import HAND_INITIAL_TARGET, HAND_FINAL_TARGET


def test_target_position():

    expected_initial_shoulder = 20 * np.pi / 180
    expected_final_shoulder = 55 * np.pi / 180

    ocp_example = ArmReaching()
    initial_q = ocp_example.model.inverse_kinematics_target(target_pos=HAND_INITIAL_TARGET)
    final_q = ocp_example.model.inverse_kinematics_target(target_pos=HAND_FINAL_TARGET)

    npt.assert_almost_equal(initial_q[0], expected_initial_shoulder, decimal=3)
    npt.assert_almost_equal(final_q[0], expected_final_shoulder, decimal=3)
