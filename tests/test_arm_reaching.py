import numpy as np
import numpy.testing as npt

from socp import ArmReaching
from socp.examples.arm_reaching import HAND_INITIAL_TARGET, HAND_FINAL_TARGET


def test_target_position():

    # Expected angles from Van Wouwe
    expected_initial_shoulder = 20 * np.pi / 180
    expected_final_shoulder = 55 * np.pi / 180

    ocp_example = ArmReaching()
    initial_q = ocp_example.model.inverse_kinematics_target(target_pos=HAND_INITIAL_TARGET)
    final_q = ocp_example.model.inverse_kinematics_target(target_pos=HAND_FINAL_TARGET)

    # 0.3491724903013795, 2.2456972964685837
    npt.assert_almost_equal(initial_q[0], expected_initial_shoulder, decimal=3)
    # 0.9599310885954864, 1.1593948518836004
    npt.assert_almost_equal(final_q[0], expected_final_shoulder, decimal=3)

    # Test that the marker function gives the correct hand position
    hand_pos_initial = ocp_example.model.end_effector_position(initial_q)
    hand_pos_final = ocp_example.model.end_effector_position(final_q)

    npt.assert_almost_equal(hand_pos_initial, HAND_INITIAL_TARGET, decimal=3)
    npt.assert_almost_equal(hand_pos_final, HAND_FINAL_TARGET, decimal=3)
