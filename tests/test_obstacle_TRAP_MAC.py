import numpy as np
import numpy.testing as npt
import casadi as cas

from socp import (
    ObstacleAvoidance,
    DirectCollocationTrapezoidal,
    MeanAndCovariance,
    prepare_ocp,
    solve_ocp,
)
from socp.utils import is_semi_definite_positive
from socp.transcriptions.variables_abstract import VariablesAbstract


def test_solve_DC_MAC():

    dynamics_transcription = DirectCollocationTrapezoidal()
    discretization_method = MeanAndCovariance(dynamics_transcription, with_helper_matrix=True)

    # --- Run the problem a first time without robustification of the constraint --- #
    ocp_example = ObstacleAvoidance(is_robustified=False, with_lbq_bound=False)

    # Prepare the problem
    ocp = prepare_ocp(
        ocp_example=ocp_example,
        dynamics_transcription=dynamics_transcription,
        discretization_method=discretization_method,
    )

    # Solve the problem
    w_opt, solver, grad_f_func, grad_g_func, save_path = solve_ocp(
        ocp,
        ocp_example=ocp_example,
        hessian_approximation="exact",  # or "limited-memory",
        linear_solver="ma57",
        pre_optim_plot=False,
        show_online_optim=False,
        plot_solution=False,
    )

    # --- Run the problem a second time robustified --- #
    ocp_example = ObstacleAvoidance(is_robustified=True, with_lbq_bound=False)

    # Prepare the problem
    ocp = prepare_ocp(
        ocp_example=ocp_example,
        dynamics_transcription=dynamics_transcription,
        discretization_method=discretization_method,
    )
    # Warm-start
    ocp["w0"] = cas.DM(w_opt)

    # Solve the problem
    w_opt, solver, grad_f_func, grad_g_func, save_path = solve_ocp(
        ocp,
        ocp_example=ocp_example,
        hessian_approximation="exact",  # or "limited-memory",
        linear_solver="ma57",
        pre_optim_plot=False,
        show_online_optim=False,
        plot_solution=False,
    )

    # --- Test the results --- #
    nb_variables = ocp["w"].shape[0]
    assert nb_variables == 5701
    nb_constraints = ocp["g"].shape[0]
    assert nb_constraints == 5705
    nb_non_zero_elem_in_grad_f = grad_f_func(ocp["w"]).nnz()
    assert nb_non_zero_elem_in_grad_f == 5701
    nb_non_zero_elem_in_grad_g = grad_g_func(ocp["w"]).nnz()
    assert nb_non_zero_elem_in_grad_g == 67431
    nb_iterations = solver.stats()["iter_count"]
    npt.assert_equal(nb_iterations, 270)

    cost_function = cas.Function("cost_function", [ocp["w"]], [ocp["j"]])
    optimal_cost = float(cost_function(w_opt))
    npt.assert_almost_equal(optimal_cost, 8.33022993497717, decimal=5)

    npt.assert_almost_equal(np.sum(w_opt), 386.97682860321925, decimal=5)

    variable_opt = ocp["discretization_method"].Variables(
        ocp["ocp_example"].n_shooting,
        ocp["dynamics_transcription"].nb_collocation_points,
        ocp["dynamics_transcription"].nb_m_points,
        ocp["ocp_example"].model.state_indices,
        ocp["ocp_example"].model.control_indices,
        ocp["ocp_example"].model.nb_random,
        ocp["discretization_method"].with_helper_matrix,
    )
    variable_opt.set_from_vector(w_opt, only_has_symbolics=True, qdot_variables_skipped=False)

    for i_node in range(ocp["n_shooting"]):
        cov_matrix_0 = variable_opt.get_cov_matrix(i_node)
        cov_matrix_1 = variable_opt.get_cov_matrix(i_node + 1)
        m_matrix = variable_opt.get_m_matrix(i_node)
        noises_numerical = np.diag(np.array([1, 1]))
        dGdx, dGdz, dGdw, dFdz = ocp["dynamics_transcription"].jacobian_funcs(
            variable_opt.get_time(),
            variable_opt.get_states(i_node),
            variable_opt.get_collocation_points(i_node),
            variable_opt.get_controls(i_node),
            np.array([1, 1]),
        )

        assert is_semi_definite_positive(cov_matrix_0)
        assert is_semi_definite_positive(cov_matrix_1)

        cov_integrated = m_matrix @ (dGdx @ cov_matrix_0 @ dGdx.T + dGdw @ noises_numerical @ dGdw.T) @ m_matrix.T
        npt.assert_array_less(np.max(np.abs(cov_matrix_1 - cov_integrated)), 1e-5)

        m_constraint = dFdz.T - dGdz.T @ m_matrix.T
        npt.assert_array_less(np.max(np.abs(m_constraint)), 1e-5)


def test_jacobian_functions():
    """
    This is the way to fall back on the Van Wouwe implementation from the 2 z implementation
    (which is closer to Gillis' implementation).
    """

    def get_m_matrix(nb_points: int, nb_states: int):
        m_matrix = None
        offset = 0
        for i_collocation in range(nb_points):
            m_vector = cas.SX.sym(f"M_{i_collocation}", nb_states, nb_states)
            m_matrix_i = VariablesAbstract.reshape_vector_to_matrix(
                m_vector,
                (nb_states, nb_states),
            )
            if m_matrix is None:
                m_matrix = m_matrix_i
            else:
                m_matrix = cas.horzcat(m_matrix, m_matrix_i)
            offset += nb_states * nb_states
        return m_matrix

    # Setup the problem
    dynamics_transcription = DirectCollocationTrapezoidal()
    discretization_method = MeanAndCovariance(dynamics_transcription, with_helper_matrix=True)
    ocp_example = ObstacleAvoidance(is_robustified=True, with_lbq_bound=False)
    nb_states = 4
    dt = 0.1

    # Set the variables
    x_1 = cas.SX.sym("x", 4)
    x_2 = cas.SX.sym("x", 4)
    u_1 = cas.SX.sym("u", 2)
    u_2 = cas.SX.sym("u", 2)
    w_1 = cas.SX.sym("w", 2)
    w_2 = cas.SX.sym("w", 2)
    cov_pre = cas.SX.sym("cov_pre", nb_states, nb_states)
    time = cas.SX.sym("time")
    sigma_ww = cas.diag(w_1)

    xdot_pre_x = discretization_method.state_dynamics(
        ocp_example,
        x_1,
        u_1,
        w_1,
    )
    xdot_post_x = discretization_method.state_dynamics(
        ocp_example,
        x_2,
        u_2,
        w_2,
    )

    # --- Charbie version --- We consider z = [x_k, x_{i+1}] temporarily
    z = cas.SX.sym("z", nb_states, 2)
    # State dynamics
    xdot_pre_z = discretization_method.state_dynamics(
        ocp_example,
        z[:, 0],
        u_1,
        w_1,
    )
    xdot_post_z = discretization_method.state_dynamics(
        ocp_example,
        z[:, 1],
        u_2,
        w_2,
    )
    F = z[:, 1]
    G_z = [z[:, 0] - x_1]
    G_z += [(z[:, 1] - z[:, 0]) - (xdot_pre_z + xdot_post_z) * dt / 2]
    G_x = [x_1 - x_1]
    # G_x = [z[:, 0] - x_1]
    G_x += [(x_2 - x_1) - (xdot_pre_x + xdot_post_x) * dt / 2]
    m_matrix_charbie = get_m_matrix(2, nb_states)

    dFdz_charbie = cas.jacobian(F, z)
    dGdz_charbie = cas.jacobian(cas.horzcat(*G_z), z)

    dGdx_charbie = cas.jacobian(cas.horzcat(*G_x), x_1)

    dFdw_charbie = cas.jacobian(F, w_1)
    dGdw_charbie = cas.jacobian(cas.horzcat(*G_z), w_1)

    cov_integrated_charbie_fcn = cas.Function(
        "cov_integrated_charbie",
    [
            time,
            x_1,
            x_2,
            z,
            u_1,
            u_2,
            w_1,
            w_2,
            m_matrix_charbie,
            cov_pre,
    ],
            [m_matrix_charbie @ (dGdx_charbie @ cov_pre @ dGdx_charbie.T + dGdw_charbie @ sigma_ww @ dGdw_charbie.T) @ m_matrix_charbie.T],
    )
    constraint_charbie_fcn = cas.Function(
        "constraint_charbie",
        [
            time,
            x_1,
            x_2,
            z,
            u_1,
            u_2,
            w_1,
            w_2,
            m_matrix_charbie,
        ],
        [dFdz_charbie.T - dGdz_charbie.T @ m_matrix_charbie.T],
    )
    jacobian_charbie_func = cas.Function(
        "jacobian_funcs",
        [
            time,
            x_1,
            x_2,
            z,
            u_1,
            u_2,
            w_1,
            w_2,
        ],
        [dGdx_charbie, dFdz_charbie, dGdz_charbie, dFdw_charbie, dGdw_charbie],
    )

    # -- Van Wouwe version --- We consider z = x_{i+1}
    m_matrix_van_wouwe = get_m_matrix(1, nb_states)
    dGdz_van_wouwe = cas.SX.eye(nb_states) - cas.jacobian(xdot_post_x, x_2) * dt / 2
    dGdx_van_wouwe = -cas.SX.eye(nb_states) - cas.jacobian(xdot_pre_x, x_1) * dt / 2
    dGdw_van_wouwe = - cas.jacobian(xdot_pre_x, w_1) * dt / 2

    cov_integrated_van_wouwe_fcn = cas.Function(
        "cov_integrated_van_wouwe",
        [
            time,
            x_1,
            x_2,
            z,
            u_1,
            u_2,
            w_1,
            w_2,
            m_matrix_van_wouwe,
            cov_pre,
        ],
        [m_matrix_van_wouwe @ (dGdx_van_wouwe @ cov_pre @ dGdx_van_wouwe.T + dGdw_van_wouwe @ sigma_ww @ dGdw_van_wouwe.T) @ m_matrix_van_wouwe.T],
    )
    constraint_van_wouwe_fcn = cas.Function(
        "constraint_charbie",
        [
            time,
            x_1,
            x_2,
            z,
            u_1,
            u_2,
            w_1,
            w_2,
            m_matrix_van_wouwe,
        ],
        [m_matrix_van_wouwe @ dGdz_van_wouwe - cas.SX.eye(nb_states)],
    )
    jacobian_van_wouwe_func = cas.Function(
        "jacobian_funcs",
        [
            time,
            x_1,
            x_2,
            u_1,
            u_2,
            w_1,
            w_2,
        ],
        [dGdx_van_wouwe, dGdz_van_wouwe, dGdw_van_wouwe],
    )

    # Comparison
    time_value = 0.0
    x_1_value = np.random.rand(nb_states)
    x_2_value = np.random.rand(nb_states)
    u_1_value = np.random.rand(2)
    u_2_value = np.random.rand(2)
    w_1_value = np.random.rand(2)
    w_2_value = np.random.rand(2)
    m_matrix_value = np.random.rand(nb_states, nb_states * 2)
    m_matrix_value_with_zeros = np.zeros((nb_states, nb_states * 2))
    m_matrix_value_with_zeros[:, nb_states:] = m_matrix_value[:, nb_states:]

    # Create a symmetric positive definite matrix of shape (4, 4)
    np.random.seed(0)  # For reproducibility
    random_matrix = np.random.rand(nb_states, nb_states)
    cov_pre_value = np.dot(random_matrix, random_matrix.T) + nb_states * np.eye(nb_states)


    constraint_charbie = constraint_charbie_fcn(
        time_value,
        x_1_value,
        x_2_value,
        np.vstack((x_1_value, x_2_value)).T,
        u_1_value,
        u_2_value,
        w_1_value,
        w_2_value,
        m_matrix_value_with_zeros,
    )
    constraint_van_wouwe = constraint_van_wouwe_fcn(
        time_value,
        x_1_value,
        x_2_value,
        x_2_value,
        u_1_value,
        u_2_value,
        w_1_value,
        w_2_value,
        m_matrix_value[:, nb_states:],
    )
    npt.assert_almost_equal(np.array(constraint_charbie[nb_states:, :]).reshape(-1, ), np.array(-constraint_van_wouwe.T).reshape(-1, ), decimal=5)

    # Now that the constraint is equivalent, we can find the values of M that satisfy each constraint version
    dGdx_charbie, dFdz_charbie, dGdz_charbie, dFdw_charbie, dGdw_charbie = jacobian_charbie_func(
        time_value,
        x_1_value,
        x_2_value,
        np.vstack((x_1_value, x_2_value)).T,
        u_1_value,
        u_2_value,
        w_1_value,
        w_2_value,
    )
    m_charbie = np.linalg.solve(dGdz_charbie.T, dFdz_charbie.T).T
    npt.assert_almost_equal(dFdz_charbie.T - dGdz_charbie.T @ m_charbie.T, np.zeros((nb_states*2, nb_states)), decimal=5)

    cov_charbie = cov_integrated_charbie_fcn(
        time_value,
        x_1_value,
        x_2_value,
        np.vstack((x_1_value, x_2_value)).T,
        u_1_value,
        u_2_value,
        w_1_value,
        w_2_value,
        m_charbie,
        cov_pre_value,
    )

    dGdx_van_wouwe, dGdz_van_wouwe, dGdw_van_wouwe = jacobian_van_wouwe_func(
        time_value,
        x_1_value,
        x_2_value,
        u_1_value,
        u_2_value,
        w_1_value,
        w_2_value,
    )
    m_van_wouwe = np.eye(nb_states) @ np.linalg.inv(dGdz_van_wouwe)
    npt.assert_almost_equal(m_van_wouwe @ dGdz_van_wouwe - np.eye(nb_states), np.zeros((nb_states, nb_states)), decimal=5)

    cov_van_wouwe = cov_integrated_van_wouwe_fcn(
        time_value,
        x_1_value,
        x_2_value,
        x_2_value,
        u_1_value,
        u_2_value,
        w_1_value,
        w_2_value,
        m_van_wouwe,
        cov_pre_value,
    )

    npt.assert_almost_equal(np.array(dGdx_charbie[nb_states:, :]), np.array(dGdx_van_wouwe), decimal=5)
    npt.assert_almost_equal(np.array(dGdw_charbie[nb_states:, :]), np.array(dGdw_van_wouwe), decimal=5)
    npt.assert_almost_equal(np.array(cov_charbie), np.array(cov_van_wouwe), decimal=5)