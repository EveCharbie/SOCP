import casadi as cas
import numpy as np

from socp import (
    ObstacleAvoidance,
    DirectCollocationTrapezoidal,
    DirectCollocationPolynomial,
    MeanAndCovariance,
    prepare_ocp,
    solve_ocp,
    save_results,
)
from socp.transcriptions.discretization_abstract import DiscretizationAbstract
from socp.transcriptions.noises_abstract import NoisesAbstract
from socp.transcriptions.transcription_abstract import TranscriptionAbstract
from socp.transcriptions.variables_abstract import VariablesAbstract
from socp.examples.example_abstract import ExampleAbstract
from socp.constraints import Constraints
from socp.analysis.covariance_integrator import CovarianceIntegrator


class DirectCollocationTrapezoidalVanWouwe(TranscriptionAbstract):

    def __init__(self) -> None:

        super().__init__()  # Does nothing

    @property
    def name(self) -> str:
        return "DirectCollocationTrapezoidalVanWouwe"

    @property
    def nb_collocation_points(self):
        return 0

    @property
    def nb_m_points(self):
        return 1  # Van Wouwe's version

    def initialize_dynamics_integrator(
        self,
        ocp_example: ExampleAbstract,
        discretization_method: DiscretizationAbstract,
        variables_vector: VariablesAbstract,
        noises_vector: NoisesAbstract,
    ) -> None:
        """
        Formulate discrete time dynamics integration using a trapezoidal collocation scheme.
        """

        # Note: The first and second x and u used to declare the casadi functions, but all nodes will be used during the evaluation of the functions
        self.discretization_method = discretization_method

        dt = variables_vector.get_time() / ocp_example.n_shooting
        nb_states = variables_vector.nb_states

        # State dynamics
        xdot_pre = discretization_method.state_dynamics(
            ocp_example,
            variables_vector.get_states(0),
            variables_vector.get_controls(0),
            noises_vector.get_noise_single(0),
        )
        xdot_post = discretization_method.state_dynamics(
            ocp_example,
            variables_vector.get_states(1),
            variables_vector.get_controls(1),
            noises_vector.get_noise_single(1),
        )
        self.dynamics_func = cas.Function(
            f"dynamics",
            [variables_vector.get_states(0), variables_vector.get_controls(0), noises_vector.get_noise_single(0)],
            [xdot_pre],
            ["x", "u", "noise"],
            ["xdot"],
        )
        # dynamics_func = dynamics_func.expand()

        cov_integrated_vector = cas.SX()
        if discretization_method.name == "MeanAndCovariance":
            # Covariance dynamics
            cov_pre = variables_vector.get_cov_matrix(0)

            # In Van Wouwe's version, We consider z = x_{i+1}
            dGdz = cas.SX.eye(variables_vector.nb_states) - cas.jacobian(xdot_post, variables_vector.get_states(1)) * dt / 2
            dGdx = -cas.SX.eye(variables_vector.nb_states) - cas.jacobian(xdot_pre, variables_vector.get_states(0)) * dt / 2
            dGdw = - cas.jacobian(xdot_pre, noises_vector.get_noise_single(0)) * dt / 2

            self.jacobian_funcs = cas.Function(
                "jacobian_funcs",
                [
                    variables_vector.get_time(),
                    variables_vector.get_states(0),
                    variables_vector.get_states(1),
                    variables_vector.get_controls(0),
                    variables_vector.get_controls(1),
                    noises_vector.get_noise_single(0),
                    noises_vector.get_noise_single(1),
                ],
                [dGdx, dGdz, dGdw],
            )

            sigma_ww = cas.diag(noises_vector.get_noise_single(0))
            m_matrix = variables_vector.get_m_matrix(0)
            cov_integrated = m_matrix @ (dGdx @ cov_pre @ dGdx.T + dGdw @ sigma_ww @ dGdw.T) @ m_matrix.T
            cov_integrated_vector = variables_vector.reshape_matrix_to_vector(cov_integrated)

        # Integrator
        states_integrated = variables_vector.get_states(0) + (xdot_pre + xdot_post) / 2 * dt
        x_next = cas.vertcat(states_integrated, cov_integrated_vector)
        self.integration_func = cas.Function(
            "F",
            [
                variables_vector.get_time(),
                variables_vector.get_states(0),
                variables_vector.get_states(1),
                variables_vector.get_cov(0),
                variables_vector.get_cov(1),
                variables_vector.get_ms(0),
                variables_vector.get_ms(1),
                variables_vector.get_controls(0),
                variables_vector.get_controls(1),
                noises_vector.get_noise_single(0),
                noises_vector.get_noise_single(1),
            ],
            [x_next],
        )
        return

    def add_other_internal_constraints(
        self,
        ocp_example: ExampleAbstract,
        discretization_method: DiscretizationAbstract,
        variables_vector: VariablesAbstract,
        noises_vector: NoisesAbstract,
        i_node: int,
        constraints: Constraints,
    ) -> None:

        nb_states = variables_vector.nb_states

        if discretization_method.with_helper_matrix:
            # Constrain M at all collocation points to follow df_integrated/dz.T - dg_integrated/dz @ m.T = 0
            m_matrix = variables_vector.get_m_matrix(i_node)

            _, dGdz, _ = self.jacobian_funcs(
                variables_vector.get_time(),
                variables_vector.get_states(i_node),
                variables_vector.get_states(i_node + 1),
                variables_vector.get_controls(i_node),
                variables_vector.get_controls(i_node + 1),
                cas.DM.zeros(ocp_example.model.nb_noises * variables_vector.nb_random),
                cas.DM.zeros(ocp_example.model.nb_noises * variables_vector.nb_random),
            )
            constraint = m_matrix @ dGdz - cas.SX.eye(variables_vector.nb_states)

            constraints.add(
                g=variables_vector.reshape_matrix_to_vector(constraint),
                lbg=[0] * (nb_states * nb_states),
                ubg=[0] * (nb_states * nb_states),
                g_names=[f"helper_matrix_defect"] * (nb_states * nb_states),
                node=i_node,
            )

        return

    def set_dynamics_constraints(
        self,
        ocp_example: ExampleAbstract,
        discretization_method: DiscretizationAbstract,
        variables_vector: VariablesAbstract,
        noises_vector: NoisesAbstract,
        constraints: Constraints,
        n_threads: int = 8,
    ) -> None:

        nb_states = variables_vector.nb_states
        nb_variables = ocp_example.model.nb_states * variables_vector.nb_random
        n_shooting = variables_vector.n_shooting

        # Multi-thread continuity constraint
        multi_threaded_integrator = self.integration_func.map(n_shooting, "thread", n_threads)
        x_integrated = multi_threaded_integrator(
            variables_vector.get_time(),
            cas.horzcat(*[variables_vector.get_states(i_node) for i_node in range(0, n_shooting)]),
            cas.horzcat(*[variables_vector.get_states(i_node) for i_node in range(1, n_shooting + 1)]),
            cas.horzcat(*[variables_vector.get_cov(i_node) for i_node in range(0, n_shooting)]),
            cas.horzcat(*[variables_vector.get_cov(i_node) for i_node in range(1, n_shooting + 1)]),
            cas.horzcat(*[variables_vector.get_ms(i_node) for i_node in range(0, n_shooting)]),
            cas.horzcat(*[variables_vector.get_ms(i_node) for i_node in range(1, n_shooting + 1)]),
            cas.horzcat(*[variables_vector.get_controls(i_node) for i_node in range(0, n_shooting)]),
            cas.horzcat(*[variables_vector.get_controls(i_node) for i_node in range(1, n_shooting + 1)]),
            cas.horzcat(*[noises_vector.get_one_vector_numerical(i_node) for i_node in range(0, n_shooting)]),
            cas.horzcat(*[noises_vector.get_one_vector_numerical(i_node) for i_node in range(1, n_shooting + 1)]),
        )

        if discretization_method.name == "MeanAndCovariance":
            states_next = cas.horzcat(*[variables_vector.get_states(i_node) for i_node in range(1, n_shooting + 1)])
            cov_next = cas.horzcat(*[variables_vector.get_cov(i_node) for i_node in range(1, n_shooting + 1)])
            x_next = cas.vertcat(states_next, cov_next)
            nb_cov_variables = nb_states * nb_states
        else:
            nb_cov_variables = 0
            x_next = cas.horzcat(*[variables_vector.get_states(i_node) for i_node in range(1, n_shooting + 1)])

        g_continuity = x_integrated - x_next
        for i_node in range(n_shooting):
            constraints.add(
                g=g_continuity[:, i_node],
                lbg=[0] * (nb_variables + nb_cov_variables),
                ubg=[0] * (nb_variables + nb_cov_variables),
                g_names=[f"dynamics_continuity_node_{i_node}"] * (nb_variables + nb_cov_variables),
                node=i_node,
            )

        # Add other constraints if any
        for i_node in range(n_shooting):
            self.add_other_internal_constraints(
                ocp_example,
                discretization_method,
                variables_vector,
                noises_vector,
                i_node,
                constraints,
            )


def van_wouwe_implementation_test():

    # Common
    ocp_example = ObstacleAvoidance(is_robustified=False, with_lbq_bound=False)
    dynamics_transcription = DirectCollocationTrapezoidal()
    discretization_method = MeanAndCovariance(dynamics_transcription, with_helper_matrix=True)

    motor_noise_magnitude, sensory_noise_magnitude = ocp_example.get_noises_magnitude()
    noises_vector = discretization_method.declare_noises(
        ocp_example.model,
        ocp_example.n_shooting,
        ocp_example.nb_random,
        motor_noise_magnitude,
        sensory_noise_magnitude,
        seed=ocp_example.seed,
    )
    noises_array = noises_vector.get_noises_array()

    # Set up covariance integrator
    def dynamics_function(x, u, w, t):
        return np.array(ocp_example.model.dynamics(x, u, None, w)).reshape(-1, )

    sigma = np.eye(2)
    cov_integrator = CovarianceIntegrator(dynamics_function, n_states=4, n_disturbances=2, sigma=sigma, epsilon=1e-7)


    # # Test my implementation
    # dynamics_transcription = DirectCollocationTrapezoidal()
    # discretization_method = MeanAndCovariance(dynamics_transcription, with_helper_matrix=True)
    # ocp = prepare_ocp(
    #     ocp_example=ocp_example,
    #     dynamics_transcription=dynamics_transcription,
    #     discretization_method=discretization_method,
    # )
    # w_opt_charbie, solver, grad_f_func, grad_g_func, save_path = solve_ocp(
    #     ocp,
    #     ocp_example=ocp_example,
    #     hessian_approximation="exact",  # or "limited-memory",
    #     linear_solver="mumps",
    #     pre_optim_plot=False,
    #     show_online_optim=False,
    #     plot_solution=False,
    # )
    #
    # # Save the results
    # status = "CVG" if solver.stats()["success"] else "DVG"
    # save_path = f"{ocp_example.name}_test_trapezoidal_Charbie_{status}.pkl"
    # data_saved = save_results(w_opt_charbie, ocp, save_path, 100, solver, grad_f_func, grad_g_func)
    # ocp_example.specific_plot_results(ocp, data_saved, save_path.replace(".pkl", "_specific.png"))


    # # Test Van Wouwe implementation
    # dynamics_transcription = DirectCollocationTrapezoidalVanWouwe()
    # discretization_method = MeanAndCovariance(dynamics_transcription, with_helper_matrix=True)
    # ocp = prepare_ocp(
    #     ocp_example=ocp_example,
    #     dynamics_transcription=dynamics_transcription,
    #     discretization_method=discretization_method,
    # )
    # w_opt_van_wouwe, solver, grad_f_func, grad_g_func, save_path = solve_ocp(
    #     ocp,
    #     ocp_example=ocp_example,
    #     hessian_approximation="exact",  # or "limited-memory",
    #     linear_solver="mumps",
    #     pre_optim_plot=False,
    #     show_online_optim=False,
    #     plot_solution=False,
    # )
    #
    # # Save the results
    # status = "CVG" if solver.stats()["success"] else "DVG"
    # save_path = f"{ocp_example.name}_test_trapezoidal_VanWouwe_{status}.pkl"
    # data_saved = save_results(w_opt_van_wouwe, ocp, save_path, 100, solver, grad_f_func, grad_g_func)
    # ocp_example.specific_plot_results(ocp, data_saved, save_path.replace(".pkl", "_specific.png"))
    #
    # time_vector = np.linspace(0, data_saved["variable_opt"].get_time(), ocp["n_shooting"] + 1)
    # cov_integrator.integrate_with_state(
    #     x_traj=data_saved["states_opt_array"],
    #     u=data_saved["controls_opt_array"],
    #     w_nominal=noises_array,
    #     P0=data_saved["cov_opt_array"][:, :, 0],
    #     time_vector=time_vector,
    # )

    # Test Gillis implementation with order=1
    dynamics_transcription = DirectCollocationPolynomial(order=1)
    discretization_method = MeanAndCovariance(dynamics_transcription, with_helper_matrix=True)
    ocp = prepare_ocp(
        ocp_example=ocp_example,
        dynamics_transcription=dynamics_transcription,
        discretization_method=discretization_method,
    )
    w_opt_gillis, solver, grad_f_func, grad_g_func, save_path = solve_ocp(
        ocp,
        ocp_example=ocp_example,
        hessian_approximation="exact",  # or "limited-memory",
        linear_solver="mumps",
        pre_optim_plot=False,
        show_online_optim=False,
        plot_solution=False,
    )

    # Save the results
    status = "CVG" if solver.stats()["success"] else "DVG"
    save_path = f"{ocp_example.name}_test_trapezoidal_Gillis_{status}.pkl"
    data_saved = save_results(w_opt_gillis, ocp, save_path, 100, solver, grad_f_func, grad_g_func)
    ocp_example.specific_plot_results(ocp, data_saved, save_path.replace(".pkl", "_specific.png"))

    time_vector = np.linspace(0, data_saved["variable_opt"].get_time(), ocp["n_shooting"] + 1)
    result = cov_integrator.integrate_with_state(
        x_traj=data_saved["states_opt_array"],
        u=data_saved["controls_opt_array"],
        w_nominal=noises_array,
        P0=data_saved["cov_opt_array"][:, :, 0],
        time_vector=time_vector,
    )
    P_history = result["P"]


if __name__ == "__main__":
    van_wouwe_implementation_test()