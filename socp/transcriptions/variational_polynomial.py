"""
Legendre polynomial but the collocation points are put on top of q (so the first collocation point is zero).
"""

import casadi as cas
import numpy as np

from .discretization_abstract import DiscretizationAbstract
from .lagrange_utils import LagrangePolynomial
from .lobatto_utils import LobattoPolynomial
from .noises_abstract import NoisesAbstract
from .transcription_abstract import TranscriptionAbstract
from .variables_abstract import VariablesAbstract
from ..examples.example_abstract import ExampleAbstract
from ..constraints import Constraints


class VariationalPolynomial(TranscriptionAbstract):

    def __init__(self, order: int = 5) -> None:

        super().__init__()  # Does nothing
        self.order = order
        self.lagrange_polynomial = LagrangePolynomial(order)

    @property
    def name(self) -> str:
        return "VariationalPolynomial"

    @property
    def nb_collocation_points(self):
        return self.order + 1

    @property
    def nb_m_points(self):
        return self.order + 1

    def get_slope(
        self,
        nb_total_q: int,
        lagrange_coefficients: np.ndarray,
        dt: cas.SX,
        z_matrix: cas.SX,
        j_collocation: int,
    ):
        # Equation (15) from Campos & al: Q_i = q_0 + h * sum_{j=1}^s a_{ij} * \dot{Q}_j
        Q = cas.SX.zeros(nb_total_q)
        for i_collocation in range(self.nb_collocation_points):
            Q += z_matrix[:, i_collocation] * lagrange_coefficients[i_collocation, j_collocation, 1]
        DP = Q / dt
        return DP

    def get_fd(
        self,
        ocp_example: ExampleAbstract,
        discretization_method: DiscretizationAbstract,
        nb_total_q: int,
        lagrange_coefficients: np.ndarray,
        dt: cas.SX,
        z_matrix: cas.SX,
        controls_0: cas.SX,
        controls_1: cas.SX,
        noises_0: cas.SX,
        noises_1: cas.SX,
        DqL_func: cas.Function,
        DvL_func: cas.Function,
        i_collocation: int,
    ):
        fd = 0
        for j_collocation in range(self.nb_collocation_points):

            DP = self.get_slope(
                nb_total_q=nb_total_q,
                lagrange_coefficients=lagrange_coefficients,
                dt=dt,
                z_matrix=z_matrix,
                j_collocation=j_collocation,
            )
            C = lagrange_coefficients[i_collocation, j_collocation, 0]
            DC = lagrange_coefficients[i_collocation, j_collocation, 1]

            controls = discretization_method.interpolate_between_nodes(
                var_pre=controls_0,
                var_post=controls_1,
                nb_points=2,
                current_point=self.lobatto.time_grid[j_collocation],
            )
            noises = discretization_method.interpolate_between_nodes(
                var_pre=noises_0,
                var_post=noises_1,
                nb_points=2,
                current_point=self.lobatto.time_grid[j_collocation],
            )

            DqL = DqL_func(
                z_matrix[:, j_collocation],
                DP,
                controls,
            )
            DvL = DvL_func(
                z_matrix[:, j_collocation],
                DP,
                controls,
            )
            force = discretization_method.get_non_conservative_forces(
                ocp_example,
                z_matrix[:, j_collocation],
                DP,
                controls,
                noises,
            )

            fd += self.lobatto.weights[j_collocation] * (dt * DqL * C + DvL * DC + dt * force * C)

        return fd

    def initialize_dynamics_integrator(
        self,
        ocp_example: ExampleAbstract,
        discretization_method: DiscretizationAbstract,
        variables_vector: VariablesAbstract,
        noises_vector: NoisesAbstract,
    ) -> None:
        """
        Formulate discrete Euler-Lagrange equations and set up a variational integrator.
        We consider that there are no holonomic constraints.
        The equations were "taken" from Wenger & al. 2017 (http://dx.doi.org/10.1063/1.4992494),
        Leyendecker & al. 2009 (https://doi.org/10.1002/oca.912), and
        Campos & al. 2015 (https://doi.org/10.48550/arXiv.1502.00325).
        Ober-Blobaum & Saake 2014 (https://doi.org/10.1007/s10444-014-9394-8)
        """

        # Note: The first and second x and u used to declare the casadi functions, but all nodes will be used during the evaluation of the functions
        self.discretization_method = discretization_method

        nb_total_q = ocp_example.model.nb_q * variables_vector.nb_random

        # Declare some coefficients
        self.lobatto = LobattoPolynomial(self.order)
        lagrange_coefficients = self.lobatto.get_lagrange_coefficients()

        # Declare some variables
        dt = variables_vector.get_time() / ocp_example.n_shooting
        q_0 = variables_vector.get_state("q", 0)
        q_1 = variables_vector.get_state("q", 1)
        z_matrix_0 = variables_vector.reshape_vector_to_matrix(
            variables_vector.get_collocation_point("q", 0),
            (nb_total_q, self.nb_collocation_points),
        )
        z_matrix_1 = variables_vector.reshape_vector_to_matrix(
            variables_vector.get_collocation_point("q", 1),
            (nb_total_q, self.nb_collocation_points),
        )
        z_matrix_penultimate = variables_vector.reshape_vector_to_matrix(
            variables_vector.get_collocation_point("q", variables_vector.n_shooting - 1),
            (nb_total_q, self.nb_collocation_points),
        )

        # Declare some useful functions
        lagrangian_func, variables = discretization_method.get_lagrangian_func(
            ocp_example=ocp_example,
            q_shape=nb_total_q,
            qdot_shape=nb_total_q,
            u_shape=ocp_example.model.nb_controls,
        )
        DqL_func = cas.Function(
            "DqL_func",
            [variables["q"], variables["qdot"], variables["u"]],
            [
                discretization_method.get_lagrangian_jacobian(
                    ocp_example,
                    lagrangian_func(variables["q"], variables["qdot"], variables["u"]),
                    variables["q"],
                )
            ],
        )
        DvL_func = cas.Function(
            "DvL_func",
            [variables["q"], variables["qdot"], variables["u"]],
            [
                discretization_method.get_lagrangian_jacobian(
                    ocp_example,
                    lagrangian_func(variables["q"], variables["qdot"], variables["u"]),
                    variables["qdot"],
                )
            ],
        )

        p_previous = self.get_fd(
            ocp_example=ocp_example,
            discretization_method=discretization_method,
            nb_total_q=nb_total_q,
            lagrange_coefficients=lagrange_coefficients,
            dt=dt,
            z_matrix=z_matrix_0,
            controls_0=variables_vector.get_controls(0),
            controls_1=variables_vector.get_controls(1),
            noises_0=noises_vector.get_noise_single(0),
            noises_1=noises_vector.get_noise_single(1),
            DqL_func=DqL_func,
            DvL_func=DvL_func,
            i_collocation=self.nb_collocation_points - 1,
        )

        transition_defect = p_previous + self.get_fd(
            ocp_example=ocp_example,
            discretization_method=discretization_method,
            nb_total_q=nb_total_q,
            lagrange_coefficients=lagrange_coefficients,
            dt=dt,
            z_matrix=z_matrix_1,
            controls_0=variables_vector.get_controls(1),
            controls_1=variables_vector.get_controls(2),
            noises_0=noises_vector.get_noise_single(1),
            noises_1=noises_vector.get_noise_single(2),
            DqL_func=DqL_func,
            DvL_func=DvL_func,
            i_collocation=0,
        )

        slope_defects = []
        for i_collocation in range(1, self.nb_collocation_points - 1):
            slope_defects += [
                self.get_fd(
                    ocp_example=ocp_example,
                    discretization_method=discretization_method,
                    nb_total_q=nb_total_q,
                    lagrange_coefficients=lagrange_coefficients,
                    dt=dt,
                    z_matrix=z_matrix_1,
                    controls_0=variables_vector.get_controls(1),
                    controls_1=variables_vector.get_controls(2),
                    noises_0=noises_vector.get_noise_single(1),
                    noises_1=noises_vector.get_noise_single(2),
                    DqL_func=DqL_func,
                    DvL_func=DvL_func,
                    i_collocation=i_collocation,
                )
            ]

        # Defects
        # First collocation state = x
        first_defect = [z_matrix_1[:, 0] - q_1]

        # Defect function
        defects = cas.vertcat(*first_defect, *slope_defects)
        self.defect_func = cas.Function(
            "defects",
            [
                variables_vector.get_time(),
                variables_vector.get_states(1),
                variables_vector.get_collocation_points(1),
                variables_vector.get_controls(1),
                variables_vector.get_controls(2),
                noises_vector.get_noise_single(1),
                noises_vector.get_noise_single(2),
            ],
            [defects],
        )
        # defect_func = defect_func.expand()

        # Defect function
        self.transition_defects_func = cas.Function(
            "defects",
            [
                variables_vector.get_time(),
                variables_vector.get_state("q", 0),
                variables_vector.get_state("q", 1),
                variables_vector.get_collocation_points(0),
                variables_vector.get_collocation_points(1),
                variables_vector.get_controls(0),
                variables_vector.get_controls(1),
                variables_vector.get_controls(2),
                noises_vector.get_noise_single(0),
                noises_vector.get_noise_single(1),
                noises_vector.get_noise_single(2),
            ],
            [transition_defect],
        )
        # transition_defects_func = transition_defects_func.expand()

        # Initial defect
        qdot_0 = variables_vector.get_state("qdot", 0)
        p0 = discretization_method.get_momentum(
            ocp_example=ocp_example,
            q=q_0,
            qdot=qdot_0,
            u=variables_vector.get_controls(0),
        )
        initial_defect = p0 + self.get_fd(
            ocp_example=ocp_example,
            discretization_method=discretization_method,
            nb_total_q=nb_total_q,
            lagrange_coefficients=lagrange_coefficients,
            dt=dt,
            z_matrix=z_matrix_0,
            controls_0=variables_vector.get_controls(0),
            controls_1=variables_vector.get_controls(1),
            noises_0=noises_vector.get_noise_single(0),
            noises_1=noises_vector.get_noise_single(1),
            DqL_func=DqL_func,
            DvL_func=DvL_func,
            i_collocation=0,
        )

        self.initial_defect_func = cas.Function(
            "defects",
            [
                variables_vector.get_time(),
                variables_vector.get_state("q", 0),
                variables_vector.get_state("qdot", 0),
                variables_vector.get_collocation_points(0),
                variables_vector.get_controls(0),
                variables_vector.get_controls(1),
                noises_vector.get_noise_single(0),
                noises_vector.get_noise_single(1),
            ],
            [initial_defect],
        )
        # initial_defect_func = initial_defect_func.expand()

        # Final defect
        q_N = variables_vector.get_state("q", variables_vector.n_shooting)
        qdot_N = variables_vector.get_state("qdot", variables_vector.n_shooting)
        pN = discretization_method.get_momentum(
            ocp_example=ocp_example,
            q=q_N,
            qdot=qdot_N,
            u=variables_vector.get_controls(variables_vector.n_shooting - 1),
        )

        p_penultimate = self.get_fd(
            ocp_example=ocp_example,
            discretization_method=discretization_method,
            nb_total_q=nb_total_q,
            lagrange_coefficients=lagrange_coefficients,
            dt=dt,
            z_matrix=z_matrix_penultimate,
            controls_0=variables_vector.get_controls(variables_vector.n_shooting - 1),
            controls_1=variables_vector.get_controls(variables_vector.n_shooting),
            noises_0=noises_vector.get_noise_single(variables_vector.n_shooting - 1),
            noises_1=noises_vector.get_noise_single(variables_vector.n_shooting),
            DqL_func=DqL_func,
            DvL_func=DvL_func,
            i_collocation=self.nb_collocation_points - 1,
        )
        final_defect = p_penultimate - pN

        self.final_defect_func = cas.Function(
            "defects",
            [
                variables_vector.get_time(),
                variables_vector.get_state("q", variables_vector.n_shooting),
                variables_vector.get_state("qdot", variables_vector.n_shooting),
                variables_vector.get_collocation_points(variables_vector.n_shooting - 1),
                variables_vector.get_controls(variables_vector.n_shooting - 1),
                variables_vector.get_controls(variables_vector.n_shooting),
                noises_vector.get_noise_single(variables_vector.n_shooting - 1),
                noises_vector.get_noise_single(variables_vector.n_shooting),
            ],
            [final_defect],
        )
        # final_defect_func = final_defect_func.expand()

        # Integrator
        # x_next = cas.vertcat(states_end, cov_integrated_vector)
        states_end = self.lagrange_polynomial.get_states_end(z_matrix_0)
        x_next = states_end
        self.integration_func = cas.Function(
            "F",
            [
                variables_vector.get_time(),
                variables_vector.get_state("q", 0),
                variables_vector.get_collocation_point("q", 0),
                # variables_vector.get_cov(0),
                # variables_vector.get_ms(0),
                variables_vector.get_controls(0),
                variables_vector.get_controls(1),
                noises_vector.get_noise_single(0),
                noises_vector.get_noise_single(1),
            ],
            [x_next],
        )
        # integration_func = integration_func.expand()

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

        nb_variables = ocp_example.model.nb_q * variables_vector.nb_random
        defects = self.defect_func(
            variables_vector.get_time(),
            variables_vector.get_states(i_node),
            variables_vector.get_collocation_points(i_node),
            variables_vector.get_controls(i_node),
            variables_vector.get_controls(i_node+1),
            cas.DM.zeros(ocp_example.model.nb_noises * variables_vector.nb_random),
            cas.DM.zeros(ocp_example.model.nb_noises * variables_vector.nb_random),
        )

        # First collocation state = x and Ld defects
        constraints.add(
            g=defects,
            lbg=[0] * (nb_variables * self.order),
            ubg=[0] * (nb_variables * self.order),
            g_names=[f"collocation_defect"] * nb_variables * self.order,
            node=i_node,
        )

        if discretization_method.with_helper_matrix:
            raise NotImplementedError("Helper matrix constraints not implemented yet for VariationalPolynomial.")
            # Constrain M at all collocation points to follow df_integrated/dz.T - dg_integrated/dz @ m.T = 0
            m_matrix = variables_vector.get_m_matrix(i_node)
            _, dGdz, _, dFdz = self.jacobian_funcs(
                variables_vector.get_time(),
                variables_vector.get_states(i_node),
                variables_vector.get_collocation_points(i_node),
                variables_vector.get_controls(i_node),
                cas.DM.zeros(ocp_example.model.nb_noises * variables_vector.nb_random),
            )

            constraint = dFdz.T - dGdz.T @ m_matrix.T
            constraints.add(
                g=variables_vector.reshape_matrix_to_vector(constraint),
                lbg=[0] * (dFdz.shape[1] * dFdz.shape[0]),
                ubg=[0] * (dFdz.shape[1] * dFdz.shape[0]),
                g_names=[f"helper_matrix_defect"] * (dFdz.shape[1] * dFdz.shape[0]),
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
        nb_variables = ocp_example.model.nb_q * variables_vector.nb_random
        n_shooting = variables_vector.n_shooting

        # Multi-thread continuity constraint
        multi_threaded_constraint = self.integration_func.map(n_shooting, "thread", n_threads)
        x_integrated = multi_threaded_constraint(
            variables_vector.get_time(),
            cas.horzcat(*[variables_vector.get_state("q", i_node) for i_node in range(0, n_shooting)]),
            cas.horzcat(*[variables_vector.get_collocation_point("q", i_node) for i_node in range(0, n_shooting)]),
            # cas.horzcat(*[variables_vector.get_cov(i_node) for i_node in range(0, n_shooting)]),
            # cas.horzcat(*[variables_vector.get_ms(i_node) for i_node in range(0, n_shooting)]),
            cas.horzcat(*[variables_vector.get_controls(i_node) for i_node in range(0, n_shooting)]),
            cas.horzcat(*[variables_vector.get_controls(i_node) for i_node in range(1, n_shooting+1)]),
            cas.horzcat(*[noises_vector.get_one_vector_numerical(i_node) for i_node in range(0, n_shooting)]),
            cas.horzcat(*[noises_vector.get_one_vector_numerical(i_node) for i_node in range(1, n_shooting+1)]),
        )

        if discretization_method.name == "MeanAndCovariance":
            nb_cov_variables = nb_states * nb_states
            states_next = cas.horzcat(*[variables_vector.get_state("q", i_node) for i_node in range(1, n_shooting + 1)])
            cov_next = cas.horzcat(*[variables_vector.get_cov(i_node) for i_node in range(1, n_shooting + 1)])
            x_next = cas.vertcat(states_next, cov_next)
        else:
            nb_cov_variables = 0
            x_next = cas.horzcat(*[variables_vector.get_state("q", i_node) for i_node in range(1, n_shooting + 1)])

        g_continuity = x_integrated - x_next
        for i_node in range(n_shooting - 1):
            constraints.add(
                g=g_continuity[:, i_node],
                lbg=[0] * (nb_variables + nb_cov_variables),
                ubg=[0] * (nb_variables + nb_cov_variables),
                g_names=[f"dynamics_continuity_node_{i_node+1}"] * (nb_variables + nb_cov_variables),
                node=i_node + 1,
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

        # Ld transition defect
        for i_node in range(1, n_shooting - 2):
            ld_transition_defect = self.transition_defects_func(
                variables_vector.get_time(),
                variables_vector.get_state("q", i_node),
                variables_vector.get_state("q", i_node + 1),
                variables_vector.get_collocation_points(i_node),
                variables_vector.get_collocation_points(i_node + 1),
                variables_vector.get_controls(i_node),
                variables_vector.get_controls(i_node + 1),
                variables_vector.get_controls(i_node + 2),
                noises_vector.get_one_vector_numerical(i_node),
                noises_vector.get_one_vector_numerical(i_node + 1),
                noises_vector.get_one_vector_numerical(i_node + 2),
            )
            constraints.add(
                g=ld_transition_defect,
                lbg=[0] * nb_variables,
                ubg=[0] * nb_variables,
                g_names=[f"Ld_continuity_node_{i_node+1}"] * nb_variables,
                node=i_node + 1,
            )

        # First node defect
        initial_defect = self.initial_defect_func(
            variables_vector.get_time(),
            variables_vector.get_state("q", 0),
            variables_vector.get_state("qdot", 0),
            variables_vector.get_collocation_points(0),
            variables_vector.get_controls(0),
            variables_vector.get_controls(1),
            noises_vector.get_one_vector_numerical(0),
            noises_vector.get_one_vector_numerical(1),
        )
        constraints.add(
            g=initial_defect,
            lbg=[0] * nb_variables,
            ubg=[0] * nb_variables,
            g_names=[f"dynamics_initial_defect"] * nb_variables,
            node=0,
        )

        # Last node defect
        final_defect = self.final_defect_func(
            variables_vector.get_time(),
            variables_vector.get_state("q", n_shooting),
            variables_vector.get_state("qdot", n_shooting),
            variables_vector.get_collocation_points(n_shooting - 1),
            variables_vector.get_controls(n_shooting - 1),
            variables_vector.get_controls(n_shooting),
            noises_vector.get_one_vector_numerical(n_shooting - 1),
            noises_vector.get_one_vector_numerical(n_shooting),
        )
        constraints.add(
            g=final_defect,
            lbg=[0] * nb_variables,
            ubg=[0] * nb_variables,
            g_names=[f"dynamics_final_defect"] * nb_variables,
            node=n_shooting,
        )
