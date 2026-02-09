"""
Legendre polynomial but the collocation points are put on top of q (so the first collocation point is zero).
"""

import casadi as cas
import numpy as np

from .discretization_abstract import DiscretizationAbstract
from .lagrange_utils import LagrangePolynomial
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
    def nb_collocation_points(self):
        return self.order + 1

    @property
    def nb_m_points(self):
        return self.order + 1

    def initialize_dynamics_integrator(
        self,
        ocp_example: ExampleAbstract,
        discretization_method: DiscretizationAbstract,
        variables_vector: VariablesAbstract,
        noises_vector: NoisesAbstract,
    ) -> None:

        # Note: The first and second x and u used to declare the casadi functions, but all nodes will be used during the evaluation of the functions
        self.discretization_method = discretization_method
        integration_func, transition_defects_func, defect_func, initial_defect_func, final_defect_func = (
            self.declare_dynamics_integrator(
                ocp_example,
                discretization_method,
                variables_vector,
                noises_vector,
            )
        )
        self.integration_func = integration_func
        self.transition_defects_func = transition_defects_func
        self.initial_defect_func = initial_defect_func
        self.defect_func = defect_func
        self.final_defect_func = final_defect_func

    @property
    def name(self) -> str:
        return "VariationalPolynomial"

    def declare_dynamics_integrator(
        self,
        ocp_example: ExampleAbstract,
        discretization_method: DiscretizationAbstract,
        variables_vector: VariablesAbstract,
        noises_vector: NoisesAbstract,
    ) -> tuple[cas.Function, cas.Function, cas.Function, cas.Function, cas.Function]:
        """
        Formulate discrete Euler-Lagrange equations and set up a variational integrator.
        We consider that there are no holonomic constraints.
        The equations were "taken" from Wenger & al. 2017 (http://dx.doi.org/10.1063/1.4992494),
        Leyendecker & al. 2009 (https://doi.org/10.1002/oca.912), and
        Campos & al. 2015 (https://doi.org/10.48550/arXiv.1502.00325).
        Ober-Blobaum & Saake 2014 (https://doi.org/10.1007/s10444-014-9394-8)
        """
        nb_total_q = ocp_example.model.nb_q * variables_vector.nb_random

        z_matrix_0 = variables_vector.reshape_vector_to_matrix(
            variables_vector.get_collocation_point("q", 0),
            (nb_total_q, self.nb_collocation_points),
        )
        z_matrix_1 = variables_vector.reshape_vector_to_matrix(
            variables_vector.get_collocation_point("q", 1),
            (nb_total_q, self.nb_collocation_points),
        )

        q_0 = variables_vector.get_state("q", 0)
        q_1 = variables_vector.get_state("q", 1)
        states_end = q_0 + self.lagrange_polynomial.get_states_end(z_matrix_0)
        dt = variables_vector.get_time() / ocp_example.n_shooting

        # Integrator
        # x_next = cas.vertcat(states_end, cov_integrated_vector)
        x_next = states_end
        integration_func = cas.Function(
            "F",
            [
                variables_vector.get_time(),
                variables_vector.get_state("q", 0),
                variables_vector.get_collocation_point("q", 0),
                # variables_vector.get_cov(0),
                # variables_vector.get_ms(0),
                variables_vector.get_controls(0),
                noises_vector.get_noise_single(0),
            ],
            [x_next],
        )
        # integration_func = integration_func.expand()

        # Defects
        # First collocation state = x
        first_defect = [z_matrix_0[:, 0]]

        # Collocation slopes
        Ld = 0
        fd_plus = 0
        b_i0 = self.lagrange_polynomial.lagrange_polynomial(
            j_collocation=0,
            time_control_interval=1,
        )
        f0 = [0]
        for j_collocation in range(1, self.nb_collocation_points):
            vertical_variation_0 = self.lagrange_polynomial.interpolate_first_derivative(
                z_matrix_0, self.lagrange_polynomial.time_grid[j_collocation]
            )
            slope_0 = vertical_variation_0 / dt
            b_j = self.lagrange_polynomial.lagrange_polynomial(
                j_collocation=j_collocation,
                time_control_interval=1,
            )
            a_j_i0 = self.lagrange_polynomial.first_part_of_lagrange_polynomial(
                j_collocation=1,
                c_i=j_collocation,
                time_control_interval=1,
            )
            Ld += (
                dt
                * b_j
                * discretization_method.get_lagrangian(
                    ocp_example=ocp_example,
                    q=q_0 + z_matrix_0[:, j_collocation],
                    qdot=slope_0,
                    u=variables_vector.get_controls(0),
                )
            )

            # Non-conservative forces (External forces, damping, etc.)
            f0 += [
                discretization_method.get_non_conservative_forces(
                    ocp_example,
                    q_0 + z_matrix_0[:, j_collocation],
                    slope_0,
                    variables_vector.get_controls(0),
                    noises_vector.get_noise_single(0),
                )
            ]

            # Equation (20+) from Campos & al. 2015
            fd_plus += dt * b_j * a_j_i0 / b_i0 * f0[j_collocation]

        p1 = (
            discretization_method.get_lagrangian_jacobian(
                ocp_example,
                Ld,
                z_matrix_0[:, 1],
            )
            / (dt * b_i0)
            + fd_plus
        )

        slope_defects = []
        for j_collocation in range(1, self.nb_collocation_points - 1):
            # Equation (21a) from Campos & al. 2015
            partial_force_term = 0
            for i_collocation in range(1, self.nb_collocation_points):
                b_i = self.lagrange_polynomial.lagrange_polynomial(
                    j_collocation=i_collocation,
                    time_control_interval=1,
                )
                a_i_j = self.lagrange_polynomial.first_part_of_lagrange_polynomial(
                    j_collocation=j_collocation,
                    c_i=i_collocation,
                    time_control_interval=1,
                )
                partial_force_term += dt**2 * b_i * a_i_j * f0[i_collocation]

            slope_defects += [
                discretization_method.get_lagrangian_jacobian(
                    ocp_example,
                    Ld,
                    z_matrix_0[:, j_collocation],
                )
                + partial_force_term
                - dt * b_j * p1
            ]

        # Defect function
        defects = cas.vertcat(*first_defect, *slope_defects)
        defect_func = cas.Function(
            "defects",
            [
                variables_vector.get_time(),
                variables_vector.get_states(0),
                variables_vector.get_collocation_points(0),
                variables_vector.get_controls(0),
                noises_vector.get_noise_single(0),
            ],
            [defects],
        )
        # defect_func = defect_func.expand()

        # Ld transition function Equation (21c) from Campos & al. 2015
        fd_minus = 0
        f1 = [0]
        for j_collocation in range(1, self.nb_collocation_points):
            vertical_variation_1 = self.lagrange_polynomial.interpolate_first_derivative(
                z_matrix_1, self.lagrange_polynomial.time_grid[j_collocation]
            )
            slope_1 = vertical_variation_1 / dt
            b_j = self.lagrange_polynomial.lagrange_polynomial(
                j_collocation=j_collocation,
                time_control_interval=1,
            )
            a_j_i0 = self.lagrange_polynomial.first_part_of_lagrange_polynomial(
                j_collocation=1,
                c_i=j_collocation,
                time_control_interval=1,
            )

            # Non-conservative forces (External forces, damping, etc.)
            f1 += [
                discretization_method.get_non_conservative_forces(
                    ocp_example,
                    q_1 + z_matrix_1[:, j_collocation],
                    slope_1,
                    variables_vector.get_controls(1),
                    noises_vector.get_noise_single(1),
                )
            ]

            # Equation (20+) from Campos & al. 2015
            fd_minus += dt * b_j * (1 - a_j_i0 / b_i0) * f1[j_collocation]

        partial_force_term = 0
        for i_collocation in range(1, self.nb_collocation_points):
            b_i = self.lagrange_polynomial.lagrange_polynomial(
                j_collocation=i_collocation,
                time_control_interval=1,
            )
            partial_force_term += dt * b_i * f1[i_collocation]
        transition_defect = fd_plus + fd_minus - partial_force_term

        # Defect function
        transition_defects_func = cas.Function(
            "defects",
            [
                variables_vector.get_time(),
                variables_vector.get_state("q", 0),
                variables_vector.get_state("q", 1),
                variables_vector.get_collocation_points(0),
                variables_vector.get_collocation_points(1),
                variables_vector.get_controls(0),
                variables_vector.get_controls(1),
                noises_vector.get_noise_single(0),
                noises_vector.get_noise_single(1),
            ],
            [transition_defect],
        )
        # transition_defects_func = transition_defects_func.expand()

        # Initial defect
        qdot_0 = variables_vector.get_state("qdot", 0)
        L_q0 = discretization_method.get_lagrangian(
            ocp_example=ocp_example,
            q=q_0,
            qdot=qdot_0,
            u=variables_vector.get_controls(0),
        )
        dL_dqdot0 = discretization_method.get_lagrangian_jacobian(
            ocp_example,
            L_q0,
            qdot_0,
        )
        L_qdot1 = discretization_method.get_lagrangian(
            ocp_example=ocp_example,
            q=q_1,
            qdot=slope_1,
            u=variables_vector.get_controls(0),
        )
        dL_dq1 = discretization_method.get_lagrangian_jacobian(
            ocp_example,
            L_qdot1,
            q_1,
        )
        initial_defect = dL_dqdot0 + dL_dq1 + fd_plus

        initial_defect_func = cas.Function(
            "defects",
            [
                variables_vector.get_time(),
                variables_vector.get_state("q", 0),
                variables_vector.get_state("q", 1),
                variables_vector.get_state("qdot", 0),
                variables_vector.get_collocation_points(0),
                variables_vector.get_controls(0),
                noises_vector.get_noise_single(0),
            ],
            [initial_defect],
        )
        # initial_defect_func = initial_defect_func.expand()

        # Final defect
        q_N = variables_vector.get_state("q", variables_vector.n_shooting)
        q_N_minus1 = variables_vector.get_state("q", variables_vector.n_shooting - 1)
        qdot_N = variables_vector.get_state("qdot", variables_vector.n_shooting)
        z_matrix_N_minus1 = variables_vector.reshape_vector_to_matrix(
            variables_vector.get_collocation_point("q", variables_vector.n_shooting - 1),
            (nb_total_q, self.nb_collocation_points),
        )
        fd_minusN_minus1 = 0
        for j_collocation in range(1, self.nb_collocation_points):
            vertical_variation_N_minus1 = self.lagrange_polynomial.interpolate_first_derivative(
                z_matrix_N_minus1, self.lagrange_polynomial.time_grid[j_collocation]
            )
            slope_N_minus1 = vertical_variation_N_minus1 / dt
            b_j = self.lagrange_polynomial.lagrange_polynomial(
                j_collocation=j_collocation,
                time_control_interval=1,
            )
            a_j_i0 = self.lagrange_polynomial.first_part_of_lagrange_polynomial(
                j_collocation=1,
                c_i=j_collocation,
                time_control_interval=1,
            )
            fN_minus1 = discretization_method.get_non_conservative_forces(
                ocp_example,
                q_N_minus1 + z_matrix_N_minus1[:, j_collocation],
                slope_N_minus1,
                variables_vector.get_controls(variables_vector.n_shooting - 1),
                noises_vector.get_noise_single(variables_vector.n_shooting - 1),
            )
            fd_minusN_minus1 += dt * b_j * (1 - a_j_i0 / b_i0) * fN_minus1

        vertical_variation_N_minus1 = self.lagrange_polynomial.interpolate_first_derivative(
            z_matrix_N_minus1, self.lagrange_polynomial.time_grid[0]
        )
        slope_N_minus1 = vertical_variation_N_minus1 / dt

        L_qN = discretization_method.get_lagrangian(
            ocp_example=ocp_example,
            q=q_N,
            qdot=qdot_N,
            u=variables_vector.get_controls(variables_vector.n_shooting - 1),
        )
        dL_dqdotN = discretization_method.get_lagrangian_jacobian(
            ocp_example,
            L_qN,
            qdot_N,
        )
        L_qdotN_minus1 = discretization_method.get_lagrangian(
            ocp_example=ocp_example,
            q=q_N_minus1,
            qdot=slope_N_minus1,
            u=variables_vector.get_controls(variables_vector.n_shooting - 1),
        )
        dL_dqN_minus1 = discretization_method.get_lagrangian_jacobian(
            ocp_example,
            L_qdotN_minus1,
            q_N_minus1,
        )

        final_defect = -dL_dqdotN + dL_dqN_minus1 + fd_minusN_minus1

        final_defect_func = cas.Function(
            "defects",
            [
                variables_vector.get_time(),
                variables_vector.get_state("q", variables_vector.n_shooting - 1),
                variables_vector.get_state("q", variables_vector.n_shooting),
                variables_vector.get_state("qdot", variables_vector.n_shooting),
                variables_vector.get_collocation_points(variables_vector.n_shooting - 1),
                variables_vector.get_controls(variables_vector.n_shooting - 1),
                noises_vector.get_noise_single(variables_vector.n_shooting - 1),
            ],
            [final_defect],
        )
        # initial_defect_func = initial_defect_func.expand()

        return integration_func, transition_defects_func, defect_func, initial_defect_func, final_defect_func

    def add_other_internal_constraints(
        self,
        ocp_example: ExampleAbstract,
        discretization_method: DiscretizationAbstract,
        variables_vector: VariablesAbstract,
        noises_vector: NoisesAbstract,
        i_node: int,
        constraints: Constraints,
    ) -> None:

        # import pickle
        # with open("w0_vector.pkl", "rb") as f:
        #     w0_vector = pickle.load(f)

        nb_variables = ocp_example.model.nb_q * variables_vector.nb_random
        defects = self.defect_func(
            variables_vector.get_time(),
            variables_vector.get_states(i_node),
            variables_vector.get_collocation_points(i_node),
            variables_vector.get_controls(i_node),
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
            cas.horzcat(*[noises_vector.get_one_vector_numerical(i_node) for i_node in range(0, n_shooting)]),
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
                noises_vector.get_one_vector_numerical(i_node),
                noises_vector.get_one_vector_numerical(i_node + 1),
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
            variables_vector.get_state("q", 1),
            variables_vector.get_state("qdot", 0),
            variables_vector.get_collocation_points(0),
            variables_vector.get_controls(0),
            noises_vector.get_one_vector_numerical(0),
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
            variables_vector.get_state("q", n_shooting - 1),
            variables_vector.get_state("q", n_shooting),
            variables_vector.get_state("qdot", n_shooting),
            variables_vector.get_collocation_points(n_shooting - 1),
            variables_vector.get_controls(n_shooting - 1),
            noises_vector.get_one_vector_numerical(n_shooting - 1),
        )
        constraints.add(
            g=final_defect,
            lbg=[0] * nb_variables,
            ubg=[0] * nb_variables,
            g_names=[f"dynamics_final_defect"] * nb_variables,
            node=n_shooting,
        )
