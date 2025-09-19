import pickle
from datetime import datetime

import casadi as cas
import numpy as np
from bioptim import Solver, SolutionMerge, OnlineOptim

from SOCP import (
    prepare_ocp,
    save_ocp,
    plot_ocp,
    animate_ocp,
    prepare_basic_socp,
    save_basic_socp,
    ExampleType,
)


RUN_OCP = True
RUN_SOCP = True
print(RUN_OCP, RUN_SOCP)
print(datetime.now().strftime("%d-%m %H:%M:%S"))

PLOT_FLAG = True
ANIMATE_FLAG = False
example_type = ExampleType.CIRCLE
force_field_magnitude = 0


nb_random = 5
n_threads = 5

n_q = 2
dt = 0.01
final_time = 0.8
n_shooting = int(final_time / dt)
tol = 1e-6

hand_initial_position = np.array([0.0, 0.2742])  # Directly from Tom's version
hand_final_position = np.array([9.359873986980460e-12, 0.527332023564034])  # Directly from Tom's version

motor_noise_std = 0.1
wPq_std = 3e-4
wPqdot_std = 0.0024

# Solver parameters
# solver = Solver.IPOPT(show_online_optim=True, show_options=dict(show_bounds=True))
# solver = Solver.IPOPT(online_optim=OnlineOptim.SERVER, show_options=dict(show_bounds=True))
solver = Solver.IPOPT(show_online_optim=False)
solver.set_linear_solver("ma97")
solver.set_bound_frac(1e-8)
solver.set_bound_push(1e-8)
solver.set_maximum_iterations(50000)

# --- Run the deterministic --- #
save_path_ocp = f"results/ocp_forcefield{force_field_magnitude}_{example_type.value}.pkl"

if RUN_OCP:
    ocp = prepare_ocp(
        final_time=final_time,
        n_shooting=n_shooting,
        hand_final_position=hand_final_position,
        example_type=example_type,
        force_field_magnitude=force_field_magnitude,
    )
    ocp.add_plot_penalty()
    # ocp.add_plot_check_conditioning()
    # ocp.add_plot_ipopt_outputs()

    ocp_tol = 1e-8
    solver.set_tol(ocp_tol)
    sol_ocp = ocp.solve(solver=solver)

    sol_ocp.print_cost()

    save_ocp(sol_ocp, save_path_ocp, tol)

    if PLOT_FLAG:
        plot_ocp(
            sol_ocp,
            motor_noise_std,
            hand_initial_position,
            hand_final_position,
            force_field_magnitude,
            n_shooting,
            final_time,
            n_simulations=100,
        )

    if ANIMATE_FLAG:
        q_sol = sol_ocp.decision_states(to_merge=SolutionMerge.NODES)["q"]
        excitations_sol = sol_ocp.decision_controls(to_merge=SolutionMerge.NODES)["muscles"]
        animate_ocp(
            example_type,
            final_time,
            n_shooting,
            q_sol,
            excitations_sol,
        )


# --- Run the SOCP --- #
print_motor_noise_std = "{:1.1e}".format(motor_noise_std)
print_wPq_std = "{:1.1e}".format(wPq_std)
print_wPqdot_std = "{:1.1e}".format(wPqdot_std)
print_tol = "{:1.1e}".format(tol).replace(".", "p")
save_path = f"results/socp_forcefield{force_field_magnitude}_nbrandoms{nb_random}_{example_type.value}_{print_motor_noise_std}_{print_wPq_std}_{print_wPqdot_std}.pkl"

motor_noise_magnitude = cas.DM(np.array([motor_noise_std**2 / dt for _ in range(n_q)]))  # All DoFs except root
sensory_noise_magnitude = cas.DM(
    cas.vertcat(
        np.array([wPq_std**2 / dt for _ in range(n_q)]),
        np.array([wPqdot_std**2 / dt for _ in range(n_q)]),
    )
)  # since the head is fixed to the pelvis, the vestibular feedback is in the states ref

if RUN_SOCP:

    with open(save_path_ocp, "rb") as file:
        data = pickle.load(file)
        q_last = data["q_sol"]
        qdot_last = data["qdot_sol"]
        activations_last = data["activations_sol"]
        excitations_last = data["excitations_sol"]
        tau_last = data["tau_sol"]
        k_last = None
        ref_last = None

    motor_noise_numerical, sensory_noise_numerical, socp = prepare_basic_socp(
        final_time=final_time,
        n_shooting=n_shooting,
        hand_final_position=hand_final_position,
        motor_noise_magnitude=motor_noise_magnitude,
        sensory_noise_magnitude=sensory_noise_magnitude,
        force_field_magnitude=force_field_magnitude,
        example_type=example_type,
        q_last=q_last,
        qdot_last=qdot_last,
        activations_last=activations_last,
        excitations_last=excitations_last,
        tau_last=tau_last,
        k_last=k_last,
        ref_last=ref_last,
        nb_random=nb_random,
        n_threads=n_threads,
    )
    socp.add_plot_penalty()
    socp.add_plot_ipopt_outputs()
    # socp.check_conditioning()

    # date_time = datetime.now().strftime("%d-%m-%H-%M-%S")
    # path_to_temporary_results = f"temporary_results_{date_time}"
    # if path_to_temporary_results not in os.listdir("results/"):
    #     os.mkdir("results/" + path_to_temporary_results)
    # nb_iter_save = 10
    # socp.save_intermediary_ipopt_iterations("results/" + path_to_temporary_results, "SOCP", nb_iter_save)

    solver.set_tol(tol)
    sol_socp = socp.solve(solver)

    save_ocp(sol_ocp, save_path_ocp, tol)

    if PLOT_FLAG:
        print("Not iomplemented")
        # plot_basic_socp(
        #     sol_socp,
        #     motor_noise_std,
        #     hand_initial_position,
        #     hand_final_position,
        #     n_simulations=100,
        # )

    if ANIMATE_FLAG:
        print("Not iomplemented")
        # q_sol = sol_ocp.decision_states(to_merge=SolutionMerge.NODES)["q"]
        # excitations_sol = sol_ocp.decision_controls(to_merge=SolutionMerge.NODES)["muscles"]
        # animate_basic_socp(
        #     example_type,
        #     final_time,
        #     n_shooting,
        #     q_sol,
        #     excitations_sol,
        # )


#
# # --- Run the SOCP+ (variable noise) --- #
# if RUN_SOCP_VARIABLE:
#     save_path = save_path.replace(".pkl", "_VARIABLE.pkl")
#
#     motor_noise_magnitude = cas.DM(
#         np.array(
#             [
#                 motor_noise_std**2 / dt,
#                 motor_noise_std**2 / dt,
#                 motor_noise_std**2 / dt,
#                 motor_noise_std**2 / dt,
#             ]
#         )
#     )  # All DoFs except root
#     sensory_noise_magnitude = cas.DM(
#         np.array(
#             [
#                 wPq_std**2 / dt,  # Proprioceptive position
#                 wPq_std**2 / dt,
#                 wPq_std**2 / dt,
#                 wPq_std**2 / dt,
#                 wPqdot_std**2 / dt,  # Proprioceptive velocity
#                 wPqdot_std**2 / dt,
#                 wPqdot_std**2 / dt,
#                 wPqdot_std**2 / dt,
#                 wPq_std**2 / dt,  # Vestibular position
#                 wPq_std**2 / dt,  # Vestibular velocity
#             ]
#         )
#     )
#
#     path_to_results = f"results/{model_name}_ocp_DMS_CVG_1e-8.pkl"
#     with open(path_to_results, "rb") as file:
#         data = pickle.load(file)
#         q_roots_last = data["q_roots_sol"]
#         q_joints_last = data["q_joints_sol"]
#         qdot_roots_last = data["qdot_roots_sol"]
#         qdot_joints_last = data["qdot_joints_sol"]
#         tau_joints_last = data["tau_joints_sol"]
#         time_last = data["time_sol"]
#         k_last = None
#         ref_last = None
#
#     motor_noise_numerical, sensory_noise_numerical, socp, noised_states = prepare_socp_VARIABLE(
#         biorbd_model_path=biorbd_model_path,
#         time_last=time_last,
#         n_shooting=n_shooting,
#         motor_noise_magnitude=motor_noise_magnitude,
#         sensory_noise_magnitude=sensory_noise_magnitude,
#         q_roots_last=q_roots_last,
#         q_joints_last=q_joints_last,
#         qdot_roots_last=qdot_roots_last,
#         qdot_joints_last=qdot_joints_last,
#         tau_joints_last=tau_joints_last,
#         k_last=None,
#         ref_last=None,
#         nb_random=nb_random,
#     )
#
#     socp.add_plot_penalty()
#     # socp.add_plot_check_conditioning()
#     socp.add_plot_ipopt_outputs()
#
#     solver.set_tol(tol)
#     sol_socp = socp.solve(solver)
#
#     states = sol_socp.decision_states(to_merge=SolutionMerge.NODES)
#     controls = sol_socp.decision_controls(to_merge=SolutionMerge.NODES)
#
#     q_roots_sol, q_joints_sol, qdot_roots_sol, qdot_joints_sol = (
#         states["q_roots"],
#         states["q_joints"],
#         states["qdot_roots"],
#         states["qdot_joints"],
#     )
#     tau_joints_sol, k_sol, ref_sol = controls["tau_joints"], controls["k"], controls["ref"]
#     time_sol = sol_socp.decision_time()[-1]
#
#     data = {
#         "q_roots_sol": q_roots_sol,
#         "q_joints_sol": q_joints_sol,
#         "qdot_roots_sol": qdot_roots_sol,
#         "qdot_joints_sol": qdot_joints_sol,
#         "tau_joints_sol": tau_joints_sol,
#         "time_sol": time_sol,
#         "k_sol": k_sol,
#         "ref_sol": ref_sol,
#         "motor_noise_numerical": motor_noise_numerical,
#         "sensory_noise_numerical": sensory_noise_numerical,
#     }
#
#     save_path = save_path.replace(".", "p")
#     if sol_socp.status != 0:
#         save_path = save_path.replace("ppkl", f"_DMS_{nb_random}random_DVG_{print_tol}.pkl")
#     else:
#         save_path = save_path.replace("ppkl", f"_DMS_{nb_random}random_CVG_{print_tol}.pkl")
#
#     # --- Save the results --- #
#     with open(save_path, "wb") as file:
#         pickle.dump(data, file)
#
#     with open(save_path.replace(".pkl", f"_sol.pkl"), "wb") as file:
#         del sol_socp.ocp
#         pickle.dump(sol_socp, file)
#
#     print(save_path)
#     # import bioviz
#     # b = bioviz.Viz(model_path=biorbd_model_path_with_mesh)
#     # b.load_movement(np.vstack((q_roots_sol, q_joints_sol)))
#     # b.exec()
#
#
# # --- Run the SOCP+ (feedforward) --- #
# n_q += 1
#
# if RUN_SOCP_FEEDFORWARD:
#     save_path = save_path.replace(".pkl", "_FEEDFORWARD.pkl")
#
#     motor_noise_magnitude = cas.DM(
#         np.array(
#             [
#                 motor_noise_std**2 / dt,
#                 0.0,
#                 motor_noise_std**2 / dt,
#                 motor_noise_std**2 / dt,
#                 motor_noise_std**2 / dt,
#             ]
#         )
#     )  # All DoFs except root
#     sensory_noise_magnitude = cas.DM(
#         np.array(
#             [
#                 wPq_std**2 / dt,  # Proprioceptive position
#                 wPq_std**2 / dt,
#                 wPq_std**2 / dt,
#                 wPq_std**2 / dt,
#                 wPqdot_std**2 / dt,  # Proprioceptive velocity
#                 wPqdot_std**2 / dt,
#                 wPqdot_std**2 / dt,
#                 wPqdot_std**2 / dt,
#                 wPq_std**2 / dt,  # Vestibular position
#                 wPq_std**2 / dt,  # Vestibular velocity
#                 wPq_std**2 / dt,  # Visual
#             ]
#         )
#     )
#
#     path_to_results = f"results/{model_name}_ocp_DMS_CVG_1e-8.pkl"
#     with open(path_to_results, "rb") as file:
#         data = pickle.load(file)
#         q_roots_last = data["q_roots_sol"]
#         q_joints_last = data["q_joints_sol"]
#         qdot_roots_last = data["qdot_roots_sol"]
#         qdot_joints_last = data["qdot_joints_sol"]
#         tau_joints_last = data["tau_joints_sol"]
#         time_last = data["time_sol"]
#         k_last = None
#         ref_last = None
#
#     q_joints_last = np.vstack((q_joints_last[0, :], np.zeros((1, q_joints_last.shape[1])), q_joints_last[1:, :]))
#     q_joints_last[1, :5] = -0.5
#     q_joints_last[1, 5:-5] = np.linspace(-0.5, 0.3, n_shooting + 1 - 10)
#     q_joints_last[1, -5:] = 0.3
#
#     qdot_joints_last = np.vstack(
#         (qdot_joints_last[0, :], np.ones((1, qdot_joints_last.shape[1])) * 0.01, qdot_joints_last[1:, :])
#     )
#     tau_joints_last = np.vstack(
#         (tau_joints_last[0, :], np.ones((1, tau_joints_last.shape[1])) * 0.01, tau_joints_last[1:, :])
#     )
#
#     motor_noise_numerical, sensory_noise_numerical, socp, noised_states = prepare_socp_FEEDFORWARD(
#         biorbd_model_path=biorbd_model_path_vision,
#         time_last=time_last,
#         n_shooting=n_shooting,
#         motor_noise_magnitude=motor_noise_magnitude,
#         sensory_noise_magnitude=sensory_noise_magnitude,
#         q_roots_last=q_roots_last,
#         q_joints_last=q_joints_last,
#         qdot_roots_last=qdot_roots_last,
#         qdot_joints_last=qdot_joints_last,
#         tau_joints_last=tau_joints_last,
#         k_last=None,
#         ref_last=None,
#         nb_random=nb_random,
#     )
#
#     socp.add_plot_penalty()
#     socp.add_plot_ipopt_outputs()
#
#     solver.set_tol(tol)
#     sol_socp = socp.solve(solver)
#
#     states = sol_socp.decision_states(to_merge=SolutionMerge.NODES)
#     controls = sol_socp.decision_controls(to_merge=SolutionMerge.NODES)
#
#     q_roots_sol, q_joints_sol, qdot_roots_sol, qdot_joints_sol = (
#         states["q_roots"],
#         states["q_joints"],
#         states["qdot_roots"],
#         states["qdot_joints"],
#     )
#     tau_joints_sol, k_sol, ref_fb_sol = controls["tau_joints"], controls["k"], controls["ref"]
#     time_sol = sol_socp.decision_time()[-1]
#     ref_ff_sol = sol_socp.parameters["final_somersault"]
#
#     data = {
#         "q_roots_sol": q_roots_sol,
#         "q_joints_sol": q_joints_sol,
#         "qdot_roots_sol": qdot_roots_sol,
#         "qdot_joints_sol": qdot_joints_sol,
#         "tau_joints_sol": tau_joints_sol,
#         "time_sol": time_sol,
#         "k_sol": k_sol,
#         "ref_fb_sol": ref_fb_sol,
#         "ref_ff_sol": ref_ff_sol,  # final somersault
#         "motor_noise_numerical": motor_noise_numerical,
#         "sensory_noise_numerical": sensory_noise_numerical,
#     }
#
#     save_path = save_path.replace(".", "p")
#     if sol_socp.status != 0:
#         save_path = save_path.replace("ppkl", f"_DMS_{nb_random}random_DVG_{print_tol}.pkl")
#     else:
#         save_path = save_path.replace("ppkl", f"_DMS_{nb_random}random_CVG_{print_tol}.pkl")
#
#     # --- Save the results --- #
#     with open(save_path, "wb") as file:
#         pickle.dump(data, file)
#
#     with open(save_path.replace(".pkl", f"_sol.pkl"), "wb") as file:
#         del sol_socp.ocp
#         pickle.dump(sol_socp, file)
#
#     print(save_path)
#     # import bioviz
#     # b = bioviz.Viz(model_path=biorbd_model_path_vision_with_mesh)
#     # b.load_movement(np.vstack((q_roots_sol, q_joints_sol)))
#     # b.exec()
#
#
# # --- Run the SOCP+ (variable noise & feedforward) --- #
# save_path = save_path.replace(".pkl", "_VARIABLE_FEEDFORWARD.pkl")
# n_q += 1
#
# if RUN_SOCP_VARIABLE_FEEDFORWARD:
#
#     motor_noise_magnitude = cas.DM(
#         np.array(
#             [
#                 motor_noise_std**2 / dt,
#                 0.0,
#                 motor_noise_std**2 / dt,
#                 motor_noise_std**2 / dt,
#                 motor_noise_std**2 / dt,
#             ]
#         )
#     )  # All DoFs except root
#     sensory_noise_magnitude = cas.DM(
#         np.array(
#             [
#                 wPq_std**2 / dt,  # Proprioceptive position
#                 wPq_std**2 / dt,
#                 wPq_std**2 / dt,
#                 wPq_std**2 / dt,
#                 wPqdot_std**2 / dt,  # Proprioceptive velocity
#                 wPqdot_std**2 / dt,
#                 wPqdot_std**2 / dt,
#                 wPqdot_std**2 / dt,
#                 wPq_std**2 / dt,  # Vestibular position
#                 wPq_std**2 / dt,  # Vestibular velocity
#                 wPq_std**2 / dt,  # Visual
#             ]
#         )
#     )
#
#     path_to_results = f"results/{model_name}_ocp_DMS_CVG_1e-8.pkl"
#     with open(path_to_results, "rb") as file:
#         data = pickle.load(file)
#         q_roots_last = data["q_roots_sol"]
#         q_joints_last = data["q_joints_sol"]
#         qdot_roots_last = data["qdot_roots_sol"]
#         qdot_joints_last = data["qdot_joints_sol"]
#         tau_joints_last = data["tau_joints_sol"]
#         time_last = data["time_sol"]
#         k_last = None
#         ref_last = None
#         ref_ff_last = None
#     q_joints_last = np.vstack((q_joints_last[0, :], np.zeros((1, q_joints_last.shape[1])), q_joints_last[1:, :]))
#     q_joints_last[1, :5] = -0.5
#     q_joints_last[1, 5:-5] = np.linspace(-0.5, 0.3, n_shooting + 1 - 10)
#     q_joints_last[1, -5:] = 0.3
#
#     qdot_joints_last = np.vstack(
#         (qdot_joints_last[0, :], np.ones((1, qdot_joints_last.shape[1])) * 0.01, qdot_joints_last[1:, :])
#     )
#     tau_joints_last = np.vstack(
#         (tau_joints_last[0, :], np.ones((1, tau_joints_last.shape[1])) * 0.01, tau_joints_last[1:, :])
#     )
#     motor_noise_numerical, sensory_noise_numerical, socp, noised_states = prepare_socp_VARIABLE_FEEDFORWARD(
#         biorbd_model_path=biorbd_model_path_vision,
#         time_last=time_last,
#         n_shooting=n_shooting,
#         motor_noise_magnitude=motor_noise_magnitude,
#         sensory_noise_magnitude=sensory_noise_magnitude,
#         q_roots_last=q_roots_last,
#         q_joints_last=q_joints_last,
#         qdot_roots_last=qdot_roots_last,
#         qdot_joints_last=qdot_joints_last,
#         tau_joints_last=tau_joints_last,
#         k_last=k_last,
#         ref_last=ref_last,
#         ref_ff_last=ref_ff_last,
#         nb_random=nb_random,
#     )
#     socp.add_plot_penalty()
#     socp.add_plot_ipopt_outputs()
#
#     save_path = save_path.replace(".", "p")
#
#     date_time = datetime.now().strftime("%d-%m-%H-%M-%S")
#     path_to_temporary_results = f"temporary_results_{date_time}"
#     if path_to_temporary_results not in os.listdir("results/"):
#         os.mkdir("results/" + path_to_temporary_results)
#     nb_iter_save = 10
#     # sol_last.ocp.save_intermediary_ipopt_iterations(
#     #     "results/" + path_to_temporary_results, "Model2D_7Dof_0C_3M_socp_DMS_5p0e-01_5p0e-03_1p5e-02_VARIABLE_FEEDFORWARD", nb_iter_save
#     # )
#     socp.save_intermediary_ipopt_iterations(
#         "results/" + path_to_temporary_results,
#         "Model2D_7Dof_0C_3M_socp_DMS_5p0e-01_5p0e-03_1p5e-02_VARIABLE_FEEDFORWARD",
#         nb_iter_save,
#     )
#
#     solver.set_tol(tol)
#     sol_socp = socp.solve(solver)
#
#     states = sol_socp.decision_states(to_merge=SolutionMerge.NODES)
#     controls = sol_socp.decision_controls(to_merge=SolutionMerge.NODES)
#
#     q_roots_sol, q_joints_sol, qdot_roots_sol, qdot_joints_sol = (
#         states["q_roots"],
#         states["q_joints"],
#         states["qdot_roots"],
#         states["qdot_joints"],
#     )
#     tau_joints_sol, k_sol, ref_fb_sol = controls["tau_joints"], controls["k"], controls["ref"]
#     time_sol = sol_socp.decision_time()[-1]
#     ref_ff_sol = sol_socp.parameters["final_somersault"]
#
#     data = {
#         "q_roots_sol": q_roots_sol,
#         "q_joints_sol": q_joints_sol,
#         "qdot_roots_sol": qdot_roots_sol,
#         "qdot_joints_sol": qdot_joints_sol,
#         "tau_joints_sol": tau_joints_sol,
#         "time_sol": time_sol,
#         "k_sol": k_sol,
#         "ref_fb_sol": ref_fb_sol,
#         "ref_ff_sol": ref_ff_sol,
#         "motor_noise_numerical": motor_noise_numerical,
#         "sensory_noise_numerical": sensory_noise_numerical,
#     }
#
#     if sol_socp.status != 0:
#         save_path = save_path.replace("ppkl", f"_DMS_{nb_random}random_DVG_{print_tol}.pkl")
#     else:
#         save_path = save_path.replace("ppkl", f"_DMS_{nb_random}random_CVG_{print_tol}.pkl")
#
#     # --- Save the results --- #
#     with open(save_path, "wb") as file:
#         pickle.dump(data, file)
#
#     with open(save_path.replace(".pkl", f"_sol.pkl"), "wb") as file:
#         del sol_socp.ocp
#         pickle.dump(sol_socp, file)
#
#     print(save_path)
#     # import bioviz
#     # b = bioviz.Viz(model_path=biorbd_model_path_vision_with_mesh)
#     # b.load_movement(np.vstack((q_roots_sol, q_joints_sol)))
#     # b.exec()
