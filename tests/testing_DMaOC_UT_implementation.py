"""
This script aims to reposition a torque actuated arm.
"""
import matplotlib.pyplot as plt
import numpy as np
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

    ocp_example.specific_plot_results(ocp, data_saved, save_path.replace(".pkl", "_specific.png"))


    # Get optimization variables
    qdot_variables_skipped = True
    variable_opt = discretization_method.Variables(
        n_shooting=ocp_example.n_shooting,
        nb_collocation_points=dynamics_transcription.nb_collocation_points,
        nb_m_points=dynamics_transcription.nb_m_points,
        state_indices=ocp_example.model.state_indices,
        control_indices=ocp_example.model.control_indices,
        nb_random=ocp_example.model.nb_random,
        nb_sigma_points=ocp_example.model.nb_sigma_points(q_only=qdot_variables_skipped),
    )
    variable_opt.set_from_vector(w_opt, only_has_symbolics=True, qdot_variables_skipped=qdot_variables_skipped)
    nb_sigma_points = variable_opt.nb_sigma_points

    motor_noise_magnitude, sensory_noise_magnitude = ocp_example.get_noises_magnitude()
    noises_vector = discretization_method.declare_noises(
        ocp_example=ocp_example,
        n_shooting=ocp_example.n_shooting,
        nb_random=1,
        motor_noise_magnitude=motor_noise_magnitude,
        sensory_noise_magnitude=sensory_noise_magnitude,
        seed=seed,
    )

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
        for i_collocation in range(dynamics_transcription.nb_collocation_points):
            qz = np.zeros((variable_opt.nb_q, variable_opt.nb_sigma_points))
            for i_sigma in range(variable_opt.nb_sigma_points):
                qz[:, i_sigma] = variable_opt.get_specific_collocation_point(name="q", node=i_shooting, sigma_point=i_sigma, point=i_collocation)
            axs[0].plot(np.ones((nb_sigma_points, )) * time_vector[i_shooting] + sub_time_vector[i_collocation], qz[0, :], ".r")
            axs[0].plot(time_vector[i_shooting] + sub_time_vector[i_collocation], np.mean(qz[0, :]), "xr")
            axs[1].plot(np.ones((nb_sigma_points, )) * time_vector[i_shooting] + sub_time_vector[i_collocation], qz[1, :], ".r")
            axs[1].plot(time_vector[i_shooting] + sub_time_vector[i_collocation], np.mean(qz[1, :]), "xr")

        # Sigma points
        sigma_ww = cas.diag(noises_vector.get_one_vector_numerical(i_shooting))
        sigma_points = variable_opt.get_sigma_states(i_shooting, sigma_ww)

        # # To remove
        # import casadi as cas
        # skipped_qdot = True
        # x_mean = None
        # for state_name in variable_opt.state_names:
        #     this_state = variable_opt.x_list[i_shooting][state_name]
        #     if this_state is not None and (not skipped_qdot or state_name != "qdot"):
        #         if x_mean is None:
        #             x_mean = this_state
        #         else:
        #             x_mean = cas.vertcat(x_mean, this_state)
        # x_mean = cas.vertcat(x_mean, cas.diag(sigma_ww))
        #
        # # Get the +- one STD sigma points (Eq. 4 from D'Hondt et al. 2026 preprint)
        # l_matrix = variable_opt.get_chol_cov_matrix(node=i_shooting)
        # augmented_l_matrix = cas.vertcat(
        #     cas.horzcat(l_matrix, cas.DM.zeros(l_matrix.shape[0], sigma_ww.shape[0])),
        #     cas.horzcat(cas.DM.zeros(sigma_ww.shape[0], l_matrix.shape[0]), sigma_ww),
        # )
        # sigma_minus = cas.DM.zeros(augmented_l_matrix.shape[0], augmented_l_matrix.shape[1])
        # sigma_plus = cas.DM.zeros(augmented_l_matrix.shape[0], augmented_l_matrix.shape[1])
        # for i_col in range(augmented_l_matrix.shape[1]):
        #     sigma_minus[:, i_col] = x_mean - augmented_l_matrix[:, i_col]
        #     sigma_plus[:, i_col] = x_mean + augmented_l_matrix[:, i_col]
        #
        # sigma_states = cas.horzcat(
        #     x_mean,
        #     sigma_plus,
        #     sigma_minus,
        # )
        # sigma_points = sigma_states

        axs[0].plot(np.ones((nb_sigma_points, )) * time_vector[i_shooting], np.array(sigma_points[0, :]).reshape(-1), ".c")
        axs[0].plot(time_vector[i_shooting], np.mean(sigma_points[0, :]), "xc")
        axs[1].plot(np.ones((nb_sigma_points, )) * time_vector[i_shooting], np.array(sigma_points[1, :]).reshape(-1), ".c")
        axs[1].plot(time_vector[i_shooting], np.mean(sigma_points[1, :]), "xc")

    plt.savefig("UnscentedTransform_test.png")
    plt.show()
    # plt.close()



if __name__ == "__main__":

    dynamics_transcription = VariationalPolynomial(order=5)
    discretization_method = UnscentedTransform(dynamics_transcription)
    run_vertebrate(dynamics_transcription, discretization_method)
