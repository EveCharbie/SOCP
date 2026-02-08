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
        dynamics_func, integration_func, jacobian_funcs = self.declare_dynamics_integrator(
            ocp_example,
            discretization_method,
            variables_vector,
            noises_vector,
        )
        self.dynamics_func = dynamics_func
        self.integration_func = integration_func
        self.jacobian_funcs = jacobian_funcs

    @property
    def name(self) -> str:
        return "DirectCollocationTrapezoidal"

    @property
    def nb_collocation_points(self):
        return 0

    @property
    def nb_m_points(self):
        # return 2
        return 1

    def declare_dynamics_integrator(
        self,
        ocp_example: ExampleAbstract,
        discretization_method: DiscretizationAbstract,
        variables_vector: VariablesAbstract,
        noises_vector: NoisesAbstract,
    ) -> tuple[cas.Function, cas.Function, cas.Function]:
        """
        Formulate discrete time dynamics integration using a trapezoidal collocation scheme.
        """
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
        dynamics_func = cas.Function(
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

            if self.discretization_method.with_helper_matrix:
                # # We consider z = [x_k, x_{i+1}] temporarily
                # z = cas.SX.sym("z", nb_states, 2)
                # F = z[:, 1]
                # G = [z[:, 0] - variables_vector.get_states(0)]
                # G += [(z[:, 1] - z[:, 0]) - (xdot_pre + xdot_post) * dt / 2]
                #
                # dFdz = cas.jacobian(F, z)
                # dGdz = cas.jacobian(cas.horzcat(*G), z)
                #
                # dGdx = cas.jacobian(cas.horzcat(*G), variables_vector.get_states(0))
                #
                # dFdw = cas.jacobian(F, noises_vector.get_noise_single(0))
                # dGdw = cas.jacobian(cas.horzcat(*G), noises_vector.get_noise_single(0))
                #
                # jacobian_funcs = cas.Function(
                #     "jacobian_funcs",
                #     [
                #         variables_vector.get_time(),
                #         variables_vector.get_states(0),
                #         variables_vector.get_states(1),
                #         z,
                #         variables_vector.get_controls(0),
                #         variables_vector.get_controls(1),
                #         noises_vector.get_noise_single(0),
                #         noises_vector.get_noise_single(1),
                #     ],
                #     [dGdx, dFdz, dGdz, dFdw, dGdw],
                # )

                # We consider z = x_{i+1}
                dGdz = cas.SX.eye(variables_vector.nb_states) - cas.jacobian(xdot_post, variables_vector.get_states(1)) * dt / 2
                dGdx = -cas.SX.eye(variables_vector.nb_states) - cas.jacobian(xdot_pre, variables_vector.get_states(0)) * dt / 2
                dGdw = - cas.jacobian(xdot_pre, noises_vector.get_noise_single(0)) * dt / 2

                jacobian_funcs = cas.Function(
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
            else:
                raise NotImplementedError(
                    "Covariance dynamics with helper matrix is the only supported method for now."
                )

        # Integrator
        states_integrated = variables_vector.get_states(0) + (xdot_pre + xdot_post) / 2 * dt
        x_next = cas.vertcat(states_integrated, cov_integrated_vector)
        integration_func = cas.Function(
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
        # integration_func = integration_func.expand()
        return dynamics_func, integration_func, jacobian_funcs

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

            # _, dFdz, dGdz, _, _ = self.jacobian_funcs(
            #     variables_vector.get_time(),
            #     variables_vector.get_states(i_node),
            #     variables_vector.get_states(i_node + 1),
            #     cas.horzcat(variables_vector.get_states(i_node), variables_vector.get_states(i_node + 1)),
            #     variables_vector.get_controls(i_node),
            #     variables_vector.get_controls(i_node + 1),
            #     cas.DM.zeros(ocp_example.model.nb_noises * variables_vector.nb_random),
            #     cas.DM.zeros(ocp_example.model.nb_noises * variables_vector.nb_random),
            # )
            # constraint = dFdz.T - dGdz.T @ m_matrix.T
            #
            # constraints.add(
            #     g=variables_vector.reshape_matrix_to_vector(constraint),
            #     lbg=[0] * (nb_states * nb_states * 2),
            #     ubg=[0] * (nb_states * nb_states * 2),
            #     g_names=[f"helper_matrix_defect"] * (nb_states * nb_states * 2),
            #     node=i_node,
            # )

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
