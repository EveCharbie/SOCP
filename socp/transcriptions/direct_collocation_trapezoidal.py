import casadi as cas
import numpy as np

from .discretization_abstract import DiscretizationAbstract
from .noises_abstract import NoisesAbstract
from .transcription_abstract import TranscriptionAbstract
from .variables_abstract import VariablesAbstract
from ..examples.example_abstract import ExampleAbstract
from ..constraints import Constraints


class DirectCollocationTrapezoidal(TranscriptionAbstract):

    def __init__(self) -> None:

        super().__init__()  # Does nothing

    def initialize_dynamics_integrator(
        self,
        ocp_example: ExampleAbstract,
        discretization_method: DiscretizationAbstract,
        variables_vector: VariablesAbstract,
        noises_vector: NoisesAbstract,
    ) -> None:

        # Note: The first and second x and u used to declare the casadi functions, but all nodes will be used during the evaluation of the functions
        self.discretization_method = discretization_method
        dynamics_func, integration_func = self.declare_dynamics_integrator(
            ocp_example,
            discretization_method,
            variables_vector,
            noises_vector,
        )
        self.dynamics_func = dynamics_func
        self.integration_func = integration_func

    @property
    def name(self) -> str:
        return "DirectCollocationTrapezoidal"

    @property
    def nb_collocation_points(self):
        return 1

    def declare_dynamics_integrator(
        self,
        ocp_example: ExampleAbstract,
        discretization_method: DiscretizationAbstract,
        variables_vector: VariablesAbstract,
        noises_vector: NoisesAbstract,
    ) -> tuple[cas.Function, cas.Function]:
        """
        Formulate discrete time dynamics integration using a trapezoidal collocation scheme.
        """
        nb_states = ocp_example.model.nb_states
        dt = variables_vector.get_time() / ocp_example.n_shooting

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
        dynamics_func = cas.Function(
            f"dynamics",
            [variables_vector.get_states(0),
            variables_vector.get_controls(0),
            noises_vector.get_noise_single(0)],
            [xdot_pre],
            ["x", "u", "noise"],
            ["xdot"]
        )
        # dynamics_func = dynamics_func.expand()

        cov_integrated_vector = cas.SX()
        if discretization_method.name == "MeanAndCovariance":
            # Covariance dynamics
            cov_pre = variables_vector.get_cov_matrix(0)

            if self.discretization_method.with_helper_matrix:
                m_matrix, df_dx, df_dw, sigma_w = discretization_method.covariance_dynamics(
                    ocp_example,
                    variables_vector.get_states(0),
                    variables_vector.get_controls(0),
                    noises_vector.get_noise_single(0)
                )

                # Trapezoidal integration of the covariance dynamics with helper matrix
                dg_dx = -(df_dx * dt / 2 + cas.SX.eye(df_dx.shape))
                dg_dw = -(df_dw * dt)

                cov_integrated = m_matrix @ (dg_dx @ cov_pre @ dg_dx.T + dg_dw @ sigma_w @ dg_dw.T) * m_matrix.T
                cov_integrated_vector = ocp_example.model.reshape_matrix_to_vector(cov_integrated)
            else:
                cov_dot_pre = discretization_method.covariance_dynamics(
                    ocp_example,
                    variables_vector.get_states(0),
                    variables_vector.get_controls(0),
                    noises_vector.get_noise_single(0)
                )
                cov_dot_post = discretization_method.covariance_dynamics(
                    ocp_example,
                    variables_vector.get_states(1),
                    variables_vector.get_controls(1),
                    noises_vector.get_noise_single(1)
                )
                cov_integrated = cov_pre + (cov_dot_pre + cov_dot_post) / 2 * dt
                cov_integrated_vector = ocp_example.model.reshape_matrix_to_vector(cov_integrated)

        # Integrator
        states_integrated = variables_vector.get_states(0) + (xdot_pre + xdot_post) / 2 * dt
        x_next = cas.vertcat(states_integrated, cov_integrated_vector)
        integration_func = cas.Function(
            "F",
            [variables_vector.get_states(0),
            variables_vector.get_states(1),
            variables_vector.get_controls(0),
            variables_vector.get_controls(1),
             noises_vector.get_noise_single(0),
             noises_vector.get_noise_single(1)],
            [x_next],
            ["x_pre", "x_post", "u_pre", "u_post", "noise_pre", "noise_post"],
            ["x_next"],
        )
        # integration_func = integration_func.expand()
        return dynamics_func, integration_func

    def set_dynamics_constraints(
        self,
        ocp_example: ExampleAbstract,
        discretization_method: DiscretizationAbstract,
        variables_vector: VariablesAbstract,
        noises_vector: NoisesAbstract,
        constraints: Constraints,
        n_threads: int = 8,
    ) -> None:

        nb_states = ocp_example.nb_states
        nb_variables = ocp_example.model.nb_states * variables_vector.nb_random
        n_shooting = variables_vector.n_shooting

        # Multi-thread continuity constraint
        multi_threaded_integrator = self.integration_func.map(n_shooting, "thread", n_threads)
        x_integrated = multi_threaded_integrator(
            variables_vector.get_time(),
            cas.horzcat(*[variables_vector.get_states(i_node) for i_node in range(0, n_shooting)]),
            cas.horzcat(*[variables_vector.get_states(i_node) for i_node in range(1, n_shooting+1)]),
            # cas.horzcat(*[variables_vector.get_cov(i_node) for i_node in range(0, n_shooting)]),
            # cas.horzcat(*[variables_vector.get_ms(i_node) for i_node in range(0, n_shooting)]),
            cas.horzcat(*[variables_vector.get_controls(i_node) for i_node in range(0, n_shooting)]),
            cas.horzcat(*[variables_vector.get_controls(i_node) for i_node in range(1, n_shooting+1)]),
            cas.horzcat(*[noises_vector.get_one_vector_numerical(i_node) for i_node in range(0, n_shooting)]),
            cas.horzcat(*[noises_vector.get_one_vector_numerical(i_node) for i_node in range(1, n_shooting+1)]),
        )


        if discretization_method.name == "MeanAndCovariance":
            if discretization_method.with_cholesky:
                nb_cov_variables = ocp_example.model.nb_cholesky_components(nb_states)
                x_next = None
                for i_node in range(n_shooting):
                    states_next_vector = variables_vector.get_states(i_node + 1)
                    cov_vector = variables_vector.get_cov(i_node + 1)
                    triangular_matrix = ocp_example.model.reshape_vector_to_cholesky_matrix(
                        cov_vector,
                        (nb_states, nb_states),
                    )
                    cov_matrix = triangular_matrix @ triangular_matrix.T
                    cov_next_vector = ocp_example.model.reshape_matrix_to_vector(cov_matrix)
                    if x_next is None:
                        x_next = cas.vertcat(states_next_vector, cov_next_vector)
                    else:
                        x_next = cas.horzcat(x_next, cas.vertcat(states_next_vector, cov_next_vector))
            else:
                nb_cov_variables = nb_states * nb_states
                states_next = cas.horzcat(*[variables_vector.get_states(i_node) for i_node in range(1, n_shooting + 1)])
                cov_next = cas.horzcat(*[variables_vector.get_cov(i_node) for i_node in range(1, n_shooting + 1)])
                x_next = cas.vertcat(states_next, cov_next)
        else:
            nb_cov_variables = 0
            x_next = cas.horzcat(*[variables_vector.get_states(i_node) for i_node in range(1, n_shooting + 1)])

        g_continuity = cas.reshape(x_integrated - x_next, (-1, 1))
        for i_node in range(n_shooting):
            constraints.add(
                g=g_continuity[
                    i_node * (nb_variables + nb_cov_variables) : (i_node + 1) * (nb_variables + nb_cov_variables)
                ],
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
