"""
This script aims to reposition a torque actuated arm.
"""
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
import casadi as cas

from socp import (
    VertebrateArm,
    VariationalPolynomial,
    UnscentedTransform,
    prepare_ocp,
    solve_ocp,
    save_results,
)


def run_vertebrate(
    dynamics_transcription,
    discretization_method,
    nb_random: int = 15,
    seed: int = 0,
):

    ocp_example = VertebrateArm(nb_random=nb_random, seed=seed)

    # Prepare the problem
    ocp = prepare_ocp(
        ocp_example=ocp_example,
        dynamics_transcription=dynamics_transcription,
        discretization_method=discretization_method,
    )

    # Solve the problem
    w_opt, solver, grad_f_func, grad_g_func, save_path, g_without_bounds_at_init = solve_ocp(
        ocp,
        ocp_example=ocp_example,
        hessian_approximation="exact",  # or "limited-memory",
        linear_solver="ma57",  # TODO: change back to ma57
        pre_optim_plot=False,
        show_online_optim=False,
        save_path_suffix="",
        plot_solution=True,
    )

    data_saved = save_results(
        w_opt,
        ocp,
        g_without_bounds_at_init,
        save_path,
        ocp_example.n_simulations,
        solver,
        grad_f_func,
        grad_g_func,
        reintegration_type="reintegrate_transcription_study",
    )
    print(f"Results saved in {save_path}")

    # # --- Use saved data TODO: remove --- #
    # import pickle
    # save_path = "/home/charbie/Documents/Programmation/SOCP/tests/results/VertebrateArm_VariationalPolynomial_UnscentedTransform_CVG_1p0e-08_2026-05-30-11-18_.pkl"
    # with open(save_path, "rb") as f:
    #     data_saved = pickle.load(f)
    # w_opt = data_saved["w_opt"]
    # # ----------------------------------- #

    ocp_example.specific_plot_results(ocp, data_saved, save_path.replace(".pkl", "_specific.png"))


    # Get optimization variables
    qdot_variables_skipped = True
    (
        states_lower_bounds,
        states_upper_bounds,
        states_initial_guesses,
        controls_lower_bounds,
        controls_upper_bounds,
        controls_initial_guesses,
        collocation_points_initial_guesses,
    ) = ocp_example.get_bounds_and_init(
        ocp_example.n_shooting,
        dynamics_transcription.nb_collocation_points,
    )
    variable_opt = discretization_method.declare_variables(
        ocp_example=ocp_example,
        dynamics_transcription=dynamics_transcription,
        states_lower_bounds=states_lower_bounds,
        controls_lower_bounds=controls_lower_bounds,
    )
    variable_opt.set_from_vector(w_opt, only_has_symbolics=True, qdot_variables_skipped=qdot_variables_skipped)
    nb_sigma_points = variable_opt.nb_sigma_points

    motor_noise_magnitude, sensory_noise_magnitude = ocp_example.get_noises_magnitude()
    noises_vector = discretization_method.declare_noises(
        ocp_example=ocp_example,
        dynamics_transcription=dynamics_transcription,
        n_shooting=ocp_example.n_shooting,
        nb_random=1,
        motor_noise_magnitude=motor_noise_magnitude,
        sensory_noise_magnitude=sensory_noise_magnitude,
        seed=seed,
    )

    variables_vector = discretization_method.declare_variables(
        ocp_example=ocp_example,
        dynamics_transcription=dynamics_transcription,
        states_lower_bounds=states_lower_bounds,
        controls_lower_bounds=controls_lower_bounds,
    )
    variables_vector.set_from_vector(ocp["w"], only_has_symbolics=True, qdot_variables_skipped=qdot_variables_skipped)

    # Plots
    fig, axs = plt.subplots(2, 1)
    time_vector = np.linspace(0, variable_opt.get_time(), ocp_example.n_shooting + 1)
    dt = variable_opt.get_time() / ocp_example.n_shooting
    sub_time_vector = np.linspace(0, dt, dynamics_transcription.nb_collocation_points + 1)
    for i_shooting in range(ocp_example.n_shooting + 1):

        # Mean states
        q = variable_opt.get_state("q", i_shooting)
        axs[0].plot(time_vector[i_shooting], q[0], "og")
        axs[1].plot(time_vector[i_shooting], q[1], "og")

        # Collocation points
        qz = np.zeros((dynamics_transcription.nb_collocation_points, variable_opt.nb_q, variable_opt.nb_sigma_points))
        for i_collocation in range(dynamics_transcription.nb_collocation_points):
            for i_sigma in range(variable_opt.nb_sigma_points):
                qz[i_collocation, :, i_sigma] = variable_opt.get_specific_collocation_point(name="q", node=i_shooting, sigma_point=i_sigma, point=i_collocation)
            axs[0].plot(np.ones((nb_sigma_points, )) * time_vector[i_shooting] + sub_time_vector[i_collocation], qz[i_collocation, 0, :], ".r")
            axs[0].plot(time_vector[i_shooting] + sub_time_vector[i_collocation], np.mean(qz[i_collocation, 0, :]), "xr")
            axs[1].plot(np.ones((nb_sigma_points, )) * time_vector[i_shooting] + sub_time_vector[i_collocation], qz[i_collocation, 1, :], ".r")
            axs[1].plot(time_vector[i_shooting] + sub_time_vector[i_collocation], np.mean(qz[i_collocation, 1, :]), "xr")

        # Sigma points
        sigma_ww = cas.diag(cas.vertcat(motor_noise_magnitude, sensory_noise_magnitude))
        sigma_points = variable_opt.get_sigma_states(i_shooting, sigma_ww)

        axs[0].plot(np.ones((nb_sigma_points, )) * time_vector[i_shooting], np.array(sigma_points[0, :]).reshape(-1), ".c")
        axs[0].plot(time_vector[i_shooting], np.mean(sigma_points[0, :]), "xc")
        axs[1].plot(np.ones((nb_sigma_points, )) * time_vector[i_shooting], np.array(sigma_points[1, :]).reshape(-1), ".c")
        axs[1].plot(time_vector[i_shooting], np.mean(sigma_points[1, :]), "xc")

        # Initial mean(x_k) = mean(z_k^init) -> OK
        npt.assert_almost_equal(q, np.mean(qz[0, :, :], axis=1))

        # Initial mean(x_k) = mean(sigma_points) -> OK
        npt.assert_almost_equal(q, np.mean(sigma_points[:variable_opt.nb_q], axis=1))

        # Constraint mean(x_k+1) = mean(z_k^end) -> OK
        if i_shooting < ocp_example.n_shooting:
            q_next = variable_opt.get_state("q", i_shooting + 1)
            npt.assert_almost_equal(q_next, np.mean(qz[-1, :, :], axis=1))

        # Constraint Cov(z_k^end) = L_k+1.T @ L_k+1
        if i_shooting < ocp_example.n_shooting:
            diff = qz[-1, :, :] - np.repeat(np.mean(qz[-1, :, :], axis=1)[:, np.newaxis], variable_opt.nb_sigma_points, axis=1)
            integrated_cov = (diff @ diff.T) / (variable_opt.nb_sigma_points - 1)
            cholesky_cov = discretization_method.get_covariance(variable_opt, node=i_shooting+1, is_matrix=True)
            npt.assert_almost_equal(integrated_cov, cholesky_cov)

        # Constraint
        if i_shooting < ocp_example.n_shooting:
            qz_pre = np.zeros(
                (variable_opt.nb_q * variable_opt.nb_sigma_points, dynamics_transcription.nb_collocation_points))
            qz_post = np.zeros(
                (variable_opt.nb_q * variable_opt.nb_sigma_points, dynamics_transcription.nb_collocation_points))
            for i_collocation in range(dynamics_transcription.nb_collocation_points):
                for i_sigma in range(variable_opt.nb_sigma_points):
                    qz_pre[variable_opt.nb_q * (i_sigma): variable_opt.nb_q * (i_sigma + 1), i_collocation] = variable_opt.get_specific_collocation_point(name="q",
                                                                                                node=i_shooting,
                                                                                                sigma_point=i_sigma,
                                                                                                point=i_collocation)
                    qz_post[variable_opt.nb_q * (i_sigma): variable_opt.nb_q * (i_sigma + 1), i_collocation] = variable_opt.get_specific_collocation_point(name="q",
                                                                                                node=i_shooting+1,
                                                                                                sigma_point=i_sigma,
                                                                                                point=i_collocation)

            transition_defect, slope_defects = variational_polynomial_defects(
                dynamics_transcription,
                discretization_method,
                ocp_example,
                variables_vector,
                variable_opt,
                noises_vector,
                qz_pre=qz_pre,
                qz_post=qz_post,
            )
            npt.assert_almost_equal(transition_defect, np.zeros_like(transition_defect))
            for i_slope in range(len(slope_defects)):
                npt.assert_almost_equal(slope_defects[i_slope], np.zeros_like(slope_defects[i_slope]))

        # Constraint z_k^init = sigma point projection
        npt.assert_almost_equal(qz[0, :, :], sigma_points[:variable_opt.nb_q, :])

    plt.savefig("UnscentedTransform_test.png")
    plt.show()
    # plt.close()


def variational_polynomial_defects(
        dynamics_transcription,
        discretization_method,
        ocp_example,
        variables_vector,
        variable_opt,
        noises_vector,
        qz_pre,
        qz_post,
    ):

    nb_total_q = ocp_example.model.nb_q * variable_opt.nb_sigma_points
    dt = variable_opt.get_time() / ocp_example.n_shooting

    # Declare some useful functions
    lagrangian_func = discretization_method.get_lagrangian(
        ocp_example=ocp_example,
        q=variables_vector.get_state_list(name="q", node=0)[0],
        qdot=variables_vector.get_state_list(name="qdot", node=0)[0],
        u=variables_vector.get_controls(node=0),
    )
    DqL_func = cas.Function(
        "DqL_func",
        [
            variables_vector.get_state_list(name="q", node=0)[0],
            variables_vector.get_state_list(name="qdot", node=0)[0],
            variables_vector.get_controls(node=0),
        ],
        [
            discretization_method.get_lagrangian_jacobian_q(
                ocp_example,
                lagrangian_func(
                    q=variables_vector.get_state_list(name="q", node=0)[0],
                    qdot=variables_vector.get_state_list(name="qdot", node=0)[0],
                    u=variables_vector.get_controls(node=0),
                )["L"],
                q=variables_vector.get_state_list(name="q", node=0),
                qdot=variables_vector.get_state_list(name="qdot", node=0),
            )(
                variables_vector.get_state_list(name="q", node=0)[0],
                variables_vector.get_state_list(name="qdot", node=0)[0],
            )
        ],
    )
    DvL_func = cas.Function(
        "DvL_func",
        [
            variables_vector.get_state_list(name="q", node=0)[0],
            variables_vector.get_state_list(name="qdot", node=0)[0],
            variables_vector.get_controls(node=0),
        ],
        [
            discretization_method.get_lagrangian_jacobian_qdot(
                ocp_example,
                lagrangian_func(
                    q=variables_vector.get_state_list(name="q", node=0)[0],
                    qdot=variables_vector.get_state_list(name="qdot", node=0)[0],
                    u=variables_vector.get_controls(node=0),
                )["L"],
                q=variables_vector.get_state_list(name="q", node=0),
                qdot=variables_vector.get_state_list(name="qdot", node=0),
            )(
                variables_vector.get_state_list(name="q", node=0)[0],
                variables_vector.get_state_list(name="qdot", node=0)[0],
            )
        ],
    )

    # Transition defect
    p_previous = dynamics_transcription.get_fd(
        ocp_example=ocp_example,
        variables_vector=variable_opt,
        noises_vector=noises_vector,
        nb_total_q=nb_total_q,
        dt=dt,
        z_matrix=qz_pre,
        DqL_func=DqL_func,
        DvL_func=DvL_func,
        i_collocation=dynamics_transcription.nb_collocation_points - 1,
        node=0,
    )

    transition_defect = p_previous + dynamics_transcription.get_fd(
        ocp_example=ocp_example,
        variables_vector=variable_opt,
        noises_vector=noises_vector,
        nb_total_q=nb_total_q,
        dt=dt,
        z_matrix=qz_post,
        DqL_func=DqL_func,
        DvL_func=DvL_func,
        i_collocation=0,
        node=1,
    )

    slope_defects = []
    for i_collocation in range(1, dynamics_transcription.nb_collocation_points - 1):
        slope_defects += [
            variables_vector.reshape_matrix_to_vector(dynamics_transcription.get_fd(
                ocp_example=ocp_example,
                variables_vector=variable_opt,
                noises_vector=noises_vector,
                nb_total_q=nb_total_q,
                dt=dt,
                z_matrix=qz_post,
                DqL_func=DqL_func,
                DvL_func=DvL_func,
                i_collocation=i_collocation,
                node=1,
            ))
        ]

    return transition_defect, slope_defects


if __name__ == "__main__":

    dynamics_transcription = VariationalPolynomial(order=5)
    discretization_method = UnscentedTransform(dynamics_transcription)
    run_vertebrate(dynamics_transcription, discretization_method)
