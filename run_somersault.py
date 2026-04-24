"""
This script aims to generate a planar backward somersault.
The model is torque driven and has 7 degrees of freedom: 3 unactuated (2 root translations and one rotation) and
4 actuated (knee, hip, shoulder, neck).
There is a sensory feedback on the joint angles and velocity and head angle and velocity (vestibular).
The feedback is directly added to the torque actuation, without delay.
This example reproduced the one from Charbonneau & al. 2026.
"""
import matplotlib.pyplot as plt
import pickle

from socp import (
    Somersault,
    DirectMultipleShooting,
    DirectCollocationTrapezoidal,
    DirectCollocationPolynomial,
    Variational,
    VariationalPolynomial,
    Deterministic,
    NoiseDiscretization,
    MeanAndCovariance,
    prepare_ocp,
    solve_ocp,
    save_results,
    get_the_save_path,
)


def run_somersault(
    dynamics_transcription,
    discretization_method,
    nb_random: int = 10,
):


    # --- First run : Deterministic --- #
    ocp_example = Somersault(nb_random=1)

    # # Prepare the problem
    # ocp = prepare_ocp(
    #     ocp_example=ocp_example,
    #     dynamics_transcription=dynamics_transcription,
    #     discretization_method=Deterministic(dynamics_transcription),
    # )
    #
    # # Solve the problem
    # w_opt, solver, grad_f_func, grad_g_func, save_path, g_without_bounds_at_init = solve_ocp(
    #     ocp,
    #     ocp_example=ocp_example,
    #     hessian_approximation="exact",  # or "limited-memory",
    #     linear_solver="ma57",  # TODO: change back to ma57
    #     pre_optim_plot=False,
    #     show_online_optim=False,  # Cannot plot the deterministic, because I cannot delete the OnlineCallback
    #     save_path_suffix="",
    # )
    #
    # data_saved = save_results(w_opt, ocp, g_without_bounds_at_init, save_path, ocp_example.n_simulations, solver, grad_f_func, grad_g_func)
    # print(f"Results saved in {save_path}")
    #
    # plt.close("all")

    # --- Use saved data TODO: remove --- #
    save_path = "results/Somersault_DirectMultipleShooting_Deterministic_CVG_1p0e-06_2026-04-09-17-27_.pkl"
    with open(save_path, "rb") as f:
        data_saved = pickle.load(f)
    w_opt = data_saved["w_opt"]
    # ----------------------------------- #


    # --- Second run : Stochastic, but cold started with deterministic solution --- #
    socp_example = Somersault(nb_random=nb_random)

    # Prepare the problem
    socp = prepare_ocp(
        ocp_example=socp_example,
        dynamics_transcription=dynamics_transcription,
        discretization_method=discretization_method,
    )
    (
        states_lower_bounds,
        states_upper_bounds,
        states_initial_guesses,
        controls_lower_bounds,
        controls_upper_bounds,
        controls_initial_guesses,
        collocation_points_initial_guesses,
    ) = socp_example.get_bounds_and_init(n_shooting=socp_example.n_shooting,
                                     nb_collocation_points=dynamics_transcription.nb_collocation_points)

    # Cold start
    if isinstance(dynamics_transcription, (Variational, VariationalPolynomial)):
        qdot_variables_skipped = True
    else:
        qdot_variables_skipped = False

    deterministic_opt = Deterministic(dynamics_transcription).Variables(
        n_shooting=ocp_example.n_shooting,
        nb_collocation_points=dynamics_transcription.nb_collocation_points,
        state_indices=ocp_example.model.state_indices,
        control_indices=ocp_example.model.control_indices,
    )
    deterministic_opt.set_from_vector(w_opt, only_has_symbolics=True, qdot_variables_skipped=qdot_variables_skipped)

    stochastic_w0 = discretization_method.Variables(
        n_shooting=socp_example.n_shooting,
        nb_collocation_points=dynamics_transcription.nb_collocation_points,
        state_indices=socp_example.model.state_indices,
        control_indices=socp_example.model.control_indices,
        nb_m_points=dynamics_transcription.nb_m_points,
        nb_random=socp_example.model.nb_random,
    )
    stochastic_w0.add_time(deterministic_opt.get_time())
    for i_node in range(socp_example.n_shooting + 1):
        if discretization_method.name == "MeanAndCovariance":
            # X
            stochastic_w0.add_state("q", node=i_node, value=deterministic_opt.get_state("q", node=i_node))
            if i_node == 0 or i_node == socp_example.n_shooting or dynamics_transcription.name not in ["Variational", "VariationalPolynomial"]:
                stochastic_w0.add_state("qdot", node=i_node, value=deterministic_opt.get_state("qdot", node=i_node))

            # Z
            if dynamics_transcription.name in ["DirectCollocationPolynomial", "VariationalPolynomial"]:
                for i_point in range(dynamics_transcription.nb_collocation_points):
                    stochastic_w0.add_collocation_point(
                        "q",
                        node=i_node,
                        point=i_point,
                        value=deterministic_opt.get_collocation_point(
                            "q",
                            node=i_node,
                        ),
                    )
                    if i_node == 0 or i_node == socp_example.n_shooting or dynamics_transcription.name not in ["Variational", "VariationalPolynomial"]:
                        stochastic_w0.add_collocation_point(
                            "qdot",
                            node=i_node,
                            point=i_point,
                            value=deterministic_opt.get_collocation_point(
                                "q",
                                node=i_node,
                            ),
                        )

        else:
            for i_random in range(socp_example.model.nb_random):
                # X
                stochastic_w0.add_state("q", node=i_node, random=i_random, value=deterministic_opt.get_state("q", node=i_node))
                if i_node == 0 or i_node == socp_example.n_shooting or dynamics_transcription.name not in ["Variational",
                                                                                                          "VariationalPolynomial"]:
                    stochastic_w0.add_state("qdot", node=i_node, random=i_random, value=deterministic_opt.get_state("qdot", node=i_node))

                # Z
                if dynamics_transcription.name in ["DirectCollocationPolynomial", "VariationalPolynomial"]:
                    for i_point in range(dynamics_transcription.nb_collocation_points):
                        stochastic_w0.add_collocation_point(
                            "q",
                            node=i_node,
                            random=i_random,
                            point=i_point,
                            value=deterministic_opt.get_specific_collocation_point(
                                "q",
                                node=i_node,
                            random=i_random,
                            point=i_point,
                            ),
                        )
                        if i_node == 0 or i_node == socp_example.n_shooting or dynamics_transcription.name not in [
                            "Variational", "VariationalPolynomial"]:
                            stochastic_w0.add_collocation_point(
                                "qdot",
                                node=i_node,
                                random=i_random,
                                point=i_point,
                                value=deterministic_opt.get_specific_collocation_point(
                                    "qdot",
                                    node=i_node,
                                    random=i_random,
                                    point=i_point,
                                ),
                            )

        # U
        stochastic_w0.add_control("tau", node=i_node, value=deterministic_opt.get_control("tau", node=i_node))
        stochastic_w0.add_control("k", node=i_node, value=controls_initial_guesses["k"][:, i_node])

    socp["w0"] = stochastic_w0.get_full_vector(keep_only_symbolic=True, skip_qdot_variables=qdot_variables_skipped)

    # Solve the problem
    w_opt, solver, grad_f_func, grad_g_func, save_path, g_without_bounds_at_init = solve_ocp(
        socp,
        ocp_example=socp_example,
        hessian_approximation="exact",  # or "limited-memory",
        linear_solver="ma57",  # TODO: change back to ma57
        pre_optim_plot=False,
        show_online_optim=True,
        save_path_suffix="",
    )

    data_saved = save_results(w_opt, socp, g_without_bounds_at_init, save_path, socp_example.n_simulations, solver, grad_f_func, grad_g_func)
    print(f"Results saved in {save_path}")

    socp_example.specific_plot_results(socp, data_saved, save_path.replace(".pkl", "_specific.png"))


if __name__ == "__main__":

    # # DirectCollocationPolynomial - NoiseDiscretization ->
    # dynamics_transcription = DirectCollocationPolynomial()
    # discretization_method = NoiseDiscretization(dynamics_transcription)
    # run_somersault(dynamics_transcription, discretization_method)
    #
    # # DirectCollocationPolynomial - MeanAndCovariance ->
    # dynamics_transcription = DirectCollocationPolynomial()
    # discretization_method = MeanAndCovariance(dynamics_transcription)
    # run_somersault(dynamics_transcription, discretization_method)

    # DirectMultipleShooting - NoiseDiscretization ->
    dynamics_transcription = DirectMultipleShooting()
    discretization_method = NoiseDiscretization(dynamics_transcription)
    run_somersault(dynamics_transcription, discretization_method, nb_random=10)

    # # DirectMultipleShooting - MeanAndCovariance ->
    # dynamics_transcription = DirectMultipleShooting()
    # discretization_method = MeanAndCovariance(dynamics_transcription)
    # run_somersault(dynamics_transcription, discretization_method)
    #
    # # DirectCollocationTrapezoidal - NoiseDiscretization ->
    # dynamics_transcription = DirectCollocationTrapezoidal()
    # discretization_method = NoiseDiscretization(dynamics_transcription)
    # run_somersault(dynamics_transcription, discretization_method)
    #
    # # DirectCollocationTrapezoidal - MeanAndCovariance ->
    # dynamics_transcription = DirectCollocationTrapezoidal()
    # discretization_method = MeanAndCovariance(dynamics_transcription)
    # run_somersault(dynamics_transcription, discretization_method, with_lbq_bound=True)
    #
    # # Variational - NoiseDiscretization ->
    # dynamics_transcription = Variational()
    # discretization_method = NoiseDiscretization(dynamics_transcription)
    # run_somersault(dynamics_transcription, discretization_method)
    #
    # # Variational - MeanAndCovariance ->
    # dynamics_transcription = Variational()
    # discretization_method = MeanAndCovariance(dynamics_transcription)
    # run_somersault(dynamics_transcription, discretization_method)
    #
    # # VariationalPolynomial - NoiseDiscretization ->
    # dynamics_transcription = VariationalPolynomial(order=5)
    # discretization_method = NoiseDiscretization(dynamics_transcription)
    # run_somersault(dynamics_transcription, discretization_method)
    #
    # # VariationalPolynomial - MeanAndCovariance ->
    # dynamics_transcription = VariationalPolynomial(order=5)
    # discretization_method = MeanAndCovariance(dynamics_transcription)
    # run_somersault(dynamics_transcription, discretization_method)
