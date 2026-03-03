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

    @property
    def name(self) -> str:
        return "DirectCollocationTrapezoidal"

    @property
    def nb_collocation_points(self):
        return 0

    @property
    def nb_m_points(self):
        return 2  # My version adapted from Gillis
        # return 1  # Van Wouwe's version

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
        xdot_pre = self.discretization_method.state_dynamics(
            ocp_example,
            variables_vector.get_states(0),
            variables_vector.get_controls(0),
            noises_vector.get_noise_single(0),
        )
        xdot_post = self.discretization_method.state_dynamics(
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

        if self.discretization_method.name == "MeanAndCovariance":
            # Covariance dynamics
            cov_pre = variables_vector.get_cov_matrix(0)

            # We consider z = [x_k, x_{i+1}] temporarily
            if ocp_example.model.use_sx:
                z = cas.SX.sym("z", nb_states, 2)
            else:
                z = cas.MX.sym("z", nb_states, 2)

            xdot_pre_z = self.discretization_method.state_dynamics(
                ocp_example,
                z[:, 0],
                variables_vector.get_controls(0),
                noises_vector.get_noise_single(0),
            )
            xdot_post_z = self.discretization_method.state_dynamics(
                ocp_example,
                z[:, 1],
                variables_vector.get_controls(1),
                noises_vector.get_noise_single(1),
            )

            # --- Charbie version --- #
            F = z[:, 1]
            G = [z[:, 0] - variables_vector.get_states(0)]
            G += [(z[:, 1] - z[:, 0]) - (xdot_pre_z + xdot_post_z) * dt / 2]

            dFdz = cas.jacobian(F, z)
            dGdz = cas.jacobian(cas.horzcat(*G), z)

            dGdx = cas.jacobian(cas.horzcat(*G), variables_vector.get_states(0))

            dFdw = cas.jacobian(F, noises_vector.get_noise_single(0))
            dGdw = cas.jacobian(cas.horzcat(*G), noises_vector.get_noise_single(0))

            self.jacobian_funcs = cas.Function(
                "jacobian_funcs",
                [
                    variables_vector.get_time(),
                    variables_vector.get_states(0),
                    variables_vector.get_states(1),
                    z,
                    variables_vector.get_controls(0),
                    variables_vector.get_controls(1),
                    noises_vector.get_noise_single(0),
                    noises_vector.get_noise_single(1),
                ],
                [dGdx, dFdz, dGdz, dFdw, dGdw],
            )
            # --- Charbie version --- #

            # # --- Van Wouwe version --- #
            # # We consider z = x_{i+1}
            # CX = cas.SX if ocp_example.model.use_sx else cas.MX
            # dGdz = CX.eye(variables_vector.nb_states) - cas.jacobian(xdot_post, variables_vector.get_states(1)) * dt / 2
            # dGdx = -CX.eye(variables_vector.nb_states) - cas.jacobian(xdot_pre, variables_vector.get_states(0)) * dt / 2
            # dGdw = - cas.jacobian(xdot_pre, noises_vector.get_noise_single(0)) * dt / 2
            #
            # self.jacobian_funcs = cas.Function(
            #     "jacobian_funcs",
            #     [
            #         variables_vector.get_time(),
            #         variables_vector.get_states(0),
            #         variables_vector.get_states(1),
            #         variables_vector.get_controls(0),
            #         variables_vector.get_controls(1),
            #         noises_vector.get_noise_single(0),
            #         noises_vector.get_noise_single(1),
            #     ],
            #     [dGdx, dGdz, dGdw],
            # )
            # # --- Van Wouwe version --- #

            sigma_ww = cas.diag(noises_vector.get_noise_single(0))
            m_matrix = variables_vector.get_m_matrix(0)
            cov_integrated = m_matrix @ (dGdx @ cov_pre @ dGdx.T + dGdw @ sigma_ww @ dGdw.T) @ m_matrix.T
            cov_integrated_vector = variables_vector.reshape_matrix_to_vector(cov_integrated)

            # Cov integrator
            self.cov_integration_func = cas.Function(
                "F",
                [
                    variables_vector.get_time(),
                    variables_vector.get_states(0),
                    variables_vector.get_states(1),
                    z,
                    variables_vector.get_cov(0),
                    variables_vector.get_cov(1),
                    variables_vector.get_ms(0),
                    variables_vector.get_ms(1),
                    variables_vector.get_controls(0),
                    variables_vector.get_controls(1),
                    noises_vector.get_noise_single(0),
                    noises_vector.get_noise_single(1),
                ],
                [cov_integrated_vector],
            )

        # Integrator
        states_integrated = variables_vector.get_states(0) + (xdot_pre + xdot_post) / 2 * dt
        self.x_integration_func = cas.Function(
            "F",
            [
                variables_vector.get_time(),
                variables_vector.get_states(0),
                variables_vector.get_states(1),
                variables_vector.get_controls(0),
                variables_vector.get_controls(1),
                noises_vector.get_noise_single(0),
                noises_vector.get_noise_single(1),
            ],
            [states_integrated],
        )
        return

    def m_constraint(
            self,
            ocp_example: ExampleAbstract,
            variables_vector: VariablesAbstract,
    ) -> cas.Function:

        # --- Charbie version --- #
        m_matrix = variables_vector.get_m_matrix(0)

        _, dFdz, dGdz, _, _ = self.jacobian_funcs(
            variables_vector.get_time(),
            variables_vector.get_states(0),
            variables_vector.get_states(1),
            cas.horzcat(variables_vector.get_states(0), variables_vector.get_states(1)),
            variables_vector.get_controls(0),
            variables_vector.get_controls(1),
            cas.DM.zeros(ocp_example.model.nb_noises * variables_vector.nb_random),
            cas.DM.zeros(ocp_example.model.nb_noises * variables_vector.nb_random),
        )
        constraint = dFdz.T - dGdz.T @ m_matrix.T
        # --- Charbie version --- #

        # # --- Van Wouwe version --- #
        # CX = cas.SX if ocp_example.model.use_sx else cas.MX
        # _, dGdz, _ = self.jacobian_funcs(
        #     variables_vector.get_time(),
        #     variables_vector.get_states(i_node),
        #     variables_vector.get_states(i_node + 1),
        #     variables_vector.get_controls(i_node),
        #     variables_vector.get_controls(i_node + 1),
        #     cas.DM.zeros(ocp_example.model.nb_noises * variables_vector.nb_random),
        #     cas.DM.zeros(ocp_example.model.nb_noises * variables_vector.nb_random),
        # )
        # constraint = m_matrix @ dGdz - CX.eye(variables_vector.nb_states)
        # # --- Van Wouwe version --- #

        return cas.Function(
            "m_constraint",
            [
                variables_vector.get_time(),
                variables_vector.get_state("q", 0),
                variables_vector.get_state("q", 1),
                variables_vector.get_controls(0),
                variables_vector.get_controls(1),
                variables_vector.get_ms(0),
            ],
            [variables_vector.reshape_matrix_to_vector(constraint)
            ],
        )

    def set_dynamics_constraints(
        self,
        ocp_example: ExampleAbstract,
        variables_vector: VariablesAbstract,
        noises_vector: NoisesAbstract,
        constraints: Constraints,
        n_threads: int = 8,
    ) -> None:

        nb_states = variables_vector.nb_states
        nb_variables = ocp_example.model.nb_states * variables_vector.nb_random
        n_shooting = variables_vector.n_shooting

        # Multi-thread continuity constraint
        multi_threaded_integrator = self.x_integration_func.map(n_shooting, "thread", n_threads)
        x_integrated = multi_threaded_integrator(
            variables_vector.get_time(),
            cas.horzcat(*[variables_vector.get_states(i_node) for i_node in range(0, n_shooting)]),
            cas.horzcat(*[variables_vector.get_states(i_node) for i_node in range(1, n_shooting + 1)]),
            cas.horzcat(*[variables_vector.get_controls(i_node) for i_node in range(0, n_shooting)]),
            cas.horzcat(*[variables_vector.get_controls(i_node) for i_node in range(1, n_shooting + 1)]),
            cas.horzcat(*[noises_vector.get_one_vector_numerical(i_node) for i_node in range(0, n_shooting)]),
            cas.horzcat(*[noises_vector.get_one_vector_numerical(i_node) for i_node in range(1, n_shooting + 1)]),
        )
        x_next = cas.horzcat(*[variables_vector.get_states(i_node) for i_node in range(1, n_shooting + 1)])

        g_continuity = x_integrated - x_next
        for i_node in range(n_shooting):
            constraints.add(
                g=g_continuity[:, i_node],
                lbg=[0] * nb_variables,
                ubg=[0] * nb_variables,
                g_names=[f"dynamics_continuity_node_{i_node}"] * nb_variables,
                node=i_node,
            )

        if self.discretization_method.name == "MeanAndCovariance":
            nb_cov_variables = nb_states * nb_states

            multi_threaded_integrator = self.cov_integration_func.map(n_shooting, "thread", n_threads)
            cov_integrated = multi_threaded_integrator(
                variables_vector.get_time(),
                cas.horzcat(*[variables_vector.get_states(i_node) for i_node in range(0, n_shooting)]),
                cas.horzcat(*[variables_vector.get_states(i_node) for i_node in range(1, n_shooting + 1)]),
                cas.horzcat(*[cas.horzcat(
                    variables_vector.get_states(i_node),
                    variables_vector.get_states(i_node + 1)) for i_node in range(0, n_shooting)]),
                cas.horzcat(*[variables_vector.get_cov(i_node) for i_node in range(0, n_shooting)]),
                cas.horzcat(*[variables_vector.get_cov(i_node) for i_node in range(1, n_shooting + 1)]),
                cas.horzcat(*[variables_vector.get_ms(i_node) for i_node in range(0, n_shooting)]),
                cas.horzcat(*[variables_vector.get_ms(i_node) for i_node in range(1, n_shooting + 1)]),
                cas.horzcat(*[variables_vector.get_controls(i_node) for i_node in range(0, n_shooting)]),
                cas.horzcat(*[variables_vector.get_controls(i_node) for i_node in range(1, n_shooting + 1)]),
                cas.horzcat(*[noises_vector.get_one_vector_numerical(i_node) for i_node in range(0, n_shooting)]),
                cas.horzcat(*[noises_vector.get_one_vector_numerical(i_node) for i_node in range(1, n_shooting + 1)]),
            )
            cov_next = cas.horzcat(*[variables_vector.get_cov(i_node) for i_node in range(1, n_shooting + 1)])

            for i_node in range(n_shooting):
                constraints.add(
                    g=cov_next[:, i_node] - cov_integrated[:, i_node],
                    lbg=[0] * nb_cov_variables,
                    ubg=[0] * nb_cov_variables,
                    g_names=[f"cov_continuity"] * nb_cov_variables,
                    node=i_node,
                )

        # Multi-thread M_matrix constraint
        if self.discretization_method.name == "MeanAndCovariance":
            # Constrain M at all collocation points to follow df_integrated/dz.T - dg_integrated/dz @ m.T = 0
            multi_threaded_constraint = self.m_constraint(
                ocp_example=ocp_example,
                variables_vector=variables_vector,
            ).map(n_shooting, "thread", n_threads)
            m_constraint = multi_threaded_constraint(
                variables_vector.get_time(),
                cas.horzcat(*[variables_vector.get_state("q", i_node) for i_node in range(0, n_shooting)]),
                cas.horzcat(*[variables_vector.get_state("q", i_node) for i_node in range(1, n_shooting+1)]),
                cas.horzcat(*[variables_vector.get_controls(i_node) for i_node in range(0, n_shooting)]),
                cas.horzcat(*[variables_vector.get_controls(i_node) for i_node in range(1, n_shooting+1)]),
                cas.horzcat(*[variables_vector.get_ms(i_node) for i_node in range(0, n_shooting+1)]),
            )

            for i_node in range(n_shooting-1):
                nb_components = m_constraint[:, i_node].shape[0]
                constraints.add(
                    g=m_constraint[:, i_node],
                    lbg=[0] * nb_components,
                    ubg=[0] * nb_components,
                    g_names=[f"collocation_defect"] * nb_components,
                    node=i_node+1,
                )

