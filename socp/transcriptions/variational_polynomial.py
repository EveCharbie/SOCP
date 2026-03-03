"""
Variational integrator using Lobatto polynomials.
This implementation in based on Campos & al. 2015 (https://arxiv.org/abs/1502.00325 + https://github.com/cmcampos-xyz/paper-2013-hovi-ocms/blob/main/varInt.m).
"""

import casadi as cas
import numpy as np

from .discretization_abstract import DiscretizationAbstract
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
        self.lobatto = LobattoPolynomial(self.order)
        self.temporary_variables = None

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
        dt: cas.MX | cas.SX,
        z_matrix: cas.MX | cas.SX,
        j_collocation: int,
    ):
        # Equation (15) from Campos & al: Q_i = q_0 + h * sum_{j=1}^s a_{ij} * \dot{Q}_j
        Q = type(dt).zeros(nb_total_q)
        for i_collocation in range(self.nb_collocation_points):
            Q += z_matrix[:, i_collocation] * lagrange_coefficients[i_collocation, j_collocation, 1]
        DP = Q / dt
        return DP

    def get_fd(
        self,
        ocp_example: ExampleAbstract,
        nb_total_q: int,
        lagrange_coefficients: np.ndarray,
        dt: cas.MX | cas.SX,
        z_matrix: cas.MX | cas.SX,
        controls_0: cas.MX | cas.SX,
        controls_1: cas.MX | cas.SX,
        noises_0: cas.MX | cas.SX,
        noises_1: cas.MX | cas.SX,
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

            controls = self.discretization_method.interpolate_between_nodes(
                var_pre=controls_0,
                var_post=controls_1,
                time_ratio=self.lobatto.time_grid[j_collocation],
            )
            noises = self.discretization_method.interpolate_between_nodes(
                var_pre=noises_0,
                var_post=noises_1,
                time_ratio=self.lobatto.time_grid[j_collocation],
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
            force = self.discretization_method.get_non_conservative_forces(
                ocp_example,
                self.temporary_variables["q"],
                self.temporary_variables["qdot"],
                controls,
                noises,
            )(
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
        The equations were "taken" from Campos & al. 2015 (https://doi.org/10.48550/arXiv.1502.00325).
        But also inspired from Wenger & al. 2017 (http://dx.doi.org/10.1063/1.4992494),
        Leyendecker & al. 2009 (https://doi.org/10.1002/oca.912), and
        Ober-Blobaum & Saake 2014 (https://doi.org/10.1007/s10444-014-9394-8).
        """

        # Note: The first and second x and u used to declare the casadi functions, but all nodes will be used during the evaluation of the functions
        self.discretization_method = discretization_method

        nb_total_q = ocp_example.model.nb_q * variables_vector.nb_random
        nb_states = ocp_example.model.nb_q

        # Declare some coefficients
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
        self.temporary_variables = self.discretization_method.get_temporary_variables(
            ocp_example=ocp_example,
            nb_q=ocp_example.model.nb_q,
            nb_u=ocp_example.model.nb_controls,
        )
        lagrangian_func = self.discretization_method.get_lagrangian(
            ocp_example=ocp_example,
            q=self.temporary_variables["q"],
            qdot=self.temporary_variables["qdot"],
            u=self.temporary_variables["u"],
        )
        DqL_func = cas.Function(
            "DqL_func",
            [cas.vertcat(*self.temporary_variables["q"]),
             cas.vertcat(*self.temporary_variables["qdot"]),
             self.temporary_variables["u"],
             ],
            [
                self.discretization_method.get_lagrangian_jacobian_q(
                    ocp_example,
                    lagrangian_func(
                        q=cas.vertcat(*self.temporary_variables["q"]),
                        qdot=cas.vertcat(*self.temporary_variables["qdot"]),
                        u=self.temporary_variables["u"]),
                    q=self.temporary_variables["q"],
                    qdot=self.temporary_variables["qdot"],
                )["L"](
                    cas.vertcat(*self.temporary_variables["q"]),
                    cas.vertcat(*self.temporary_variables["qdot"]),
                )
            ],
        )
        DvL_func = cas.Function(
            "DvL_func",
            [cas.vertcat(*self.temporary_variables["q"]),
             cas.vertcat(*self.temporary_variables["qdot"]),
             self.temporary_variables["u"],
             ],
            [
                self.discretization_method.get_lagrangian_jacobian_qdot(
                    ocp_example,
                    lagrangian_func(
                        q=cas.vertcat(*self.temporary_variables["q"]),
                        qdot=cas.vertcat(*self.temporary_variables["qdot"]),
                        u=self.temporary_variables["u"]),
                    q=self.temporary_variables["q"],
                    qdot=self.temporary_variables["qdot"],
                )["L"](
                    cas.vertcat(*self.temporary_variables["q"]),
                    cas.vertcat(*self.temporary_variables["qdot"]),
                )
            ],
        )

        p_previous = self.get_fd(
            ocp_example=ocp_example,
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
                variables_vector.get_state("q", 1),
                variables_vector.get_collocation_point("q", 1),
                variables_vector.get_controls(1),
                variables_vector.get_controls(2),
                noises_vector.get_noise_single(1),
                noises_vector.get_noise_single(2),
            ],
            [defects],
        )

        # Defect function
        self.transition_defects_func = cas.Function(
            "transition_defects",
            [
                variables_vector.get_time(),
                variables_vector.get_state("q", 0),
                variables_vector.get_state("q", 1),
                variables_vector.get_collocation_point("q", 0),
                variables_vector.get_collocation_point("q", 1),
                variables_vector.get_controls(0),
                variables_vector.get_controls(1),
                variables_vector.get_controls(2),
                noises_vector.get_noise_single(0),
                noises_vector.get_noise_single(1),
                noises_vector.get_noise_single(2),
            ],
            [transition_defect],
        )

        # Initial defect
        qdot_0 = variables_vector.get_state("qdot", 0)
        p0 = self.discretization_method.get_momentum(
            ocp_example=ocp_example,
            q=self.temporary_variables["q"],
            qdot=self.temporary_variables["qdot"],
            u=self.temporary_variables["u"]
        )(
            q_0,
            qdot_0,
            variables_vector.get_controls(0),
        )
        initial_defect = p0 + self.get_fd(
            ocp_example=ocp_example,
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
            "initial_defects",
            [
                variables_vector.get_time(),
                variables_vector.get_state("q", 0),
                variables_vector.get_state("qdot", 0),
                variables_vector.get_collocation_point("q", 0),
                variables_vector.get_controls(0),
                variables_vector.get_controls(1),
                noises_vector.get_noise_single(0),
                noises_vector.get_noise_single(1),
            ],
            [initial_defect],
        )

        # Final defect
        q_N = variables_vector.get_state("q", variables_vector.n_shooting)
        qdot_N = variables_vector.get_state("qdot", variables_vector.n_shooting)
        pN = self.discretization_method.get_momentum(
            ocp_example=ocp_example,
            q=self.temporary_variables["q"],
            qdot=self.temporary_variables["qdot"],
            u=self.temporary_variables["u"],
        )(
            q_N,
            qdot_N,
            variables_vector.get_controls(variables_vector.n_shooting - 1),
        )

        p_penultimate = self.get_fd(
            ocp_example=ocp_example,
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
            "final_defects",
            [
                variables_vector.get_time(),
                variables_vector.get_state("q", variables_vector.n_shooting),
                variables_vector.get_state("qdot", variables_vector.n_shooting),
                variables_vector.get_collocation_point("q", variables_vector.n_shooting - 1),
                variables_vector.get_controls(variables_vector.n_shooting - 1),
                variables_vector.get_controls(variables_vector.n_shooting),
                noises_vector.get_noise_single(variables_vector.n_shooting - 1),
                noises_vector.get_noise_single(variables_vector.n_shooting),
            ],
            [final_defect],
        )

        self.jacobian_funcs = None
        if self.discretization_method.name == "MeanAndCovariance":
            m_matrix = variables_vector.get_m_matrix(1)

            sigma_ww = cas.diag(noises_vector.get_noise_single(1))

            # states_end = z_matrix_1[:, -1]
            states_end = z_matrix_1[:, 0]
            for j_collocation in range(self.nb_collocation_points):
                states_end += dt * self.lobatto.weights[j_collocation] * self.get_slope(
                nb_total_q=nb_total_q,
                lagrange_coefficients=lagrange_coefficients,
                dt=dt,
                z_matrix=z_matrix_1,
                j_collocation=j_collocation,
            )

            defects = cas.vertcat(*first_defect, *slope_defects)
            self.defect_func = cas.Function(
                "defects",
                [
                    variables_vector.get_time(),
                    variables_vector.get_state("q", 1),
                    variables_vector.get_collocation_point("q", 1),
                    variables_vector.get_controls(1),
                    variables_vector.get_controls(2),
                    noises_vector.get_noise_single(1),
                    noises_vector.get_noise_single(2),
                ],
                [defects],
            )

            all_defects = cas.vertcat(defects, transition_defect)

            dGdx = cas.jacobian(all_defects, variables_vector.get_state("q", 1))
            dGdz = cas.jacobian(all_defects, variables_vector.get_collocation_point("q", 1))
            dGdw = cas.jacobian(all_defects, noises_vector.get_noise_single(1))
            dFdz = cas.jacobian(states_end, variables_vector.get_collocation_point("q", 1))

            self.jacobian_funcs = cas.Function(
                "jacobian_func",
                [
                    variables_vector.get_time(),
                    variables_vector.get_state("q", 1),
                    variables_vector.get_collocation_point("q", 0),
                    variables_vector.get_collocation_point("q", 1),
                    variables_vector.get_controls(0),
                    variables_vector.get_controls(1),
                    variables_vector.get_controls(2),
                    noises_vector.get_noise_single(0),
                    noises_vector.get_noise_single(1),
                    noises_vector.get_noise_single(2),
                ],
                [dGdx, dGdz, dGdw, dFdz],
            )
            cov_matrix = variables_vector.get_cov_matrix(1)[:nb_states, :nb_states]
            cov_integrated = m_matrix @ (dGdx @ cov_matrix @ dGdx.T + dGdw @ sigma_ww @ dGdw.T) @ m_matrix.T

            cov_integrated_vector = variables_vector.reshape_matrix_to_vector(cov_integrated)

            # Cov integrator
            self.cov_integration_func = cas.Function(
                "F",
                [
                    variables_vector.get_time(),
                    variables_vector.get_state("q", 1),
                    variables_vector.get_collocation_point("q", 0),
                    variables_vector.get_collocation_point("q", 1),
                    variables_vector.get_cov(1),
                    variables_vector.get_ms(1),
                    variables_vector.get_controls(0),
                    variables_vector.get_controls(1),
                    variables_vector.get_controls(2),
                    noises_vector.get_noise_single(0),
                    noises_vector.get_noise_single(1),
                    noises_vector.get_noise_single(2),
                ],
                [cov_integrated_vector],
            )

            # First node cov integration
            m_matrix_first = variables_vector.get_m_matrix(0)
            sigma_ww_first = cas.diag(noises_vector.get_noise_single(0))

            states_end_first = z_matrix_0[:, 0]
            for j_collocation in range(self.nb_collocation_points):
                states_end_first += dt * self.lobatto.weights[j_collocation] * self.get_slope(
                nb_total_q=nb_total_q,
                lagrange_coefficients=lagrange_coefficients,
                dt=dt,
                z_matrix=z_matrix_0,
                j_collocation=j_collocation,
            )

            first_defect_first = [z_matrix_0[:, 0] - q_0]

            slope_defects_first = []
            for i_collocation in range(1, self.nb_collocation_points - 1):
                slope_defects_first += [
                    self.get_fd(
                        ocp_example=ocp_example,
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
                        i_collocation=i_collocation,
                    )
                ]

            defects_first = cas.vertcat(*first_defect_first, *slope_defects_first)
            all_defects_first = cas.vertcat(defects_first, initial_defect)

            dGdx_first = cas.jacobian(all_defects_first, variables_vector.get_state("q", 0))
            dGdz_first = cas.jacobian(all_defects_first, variables_vector.get_collocation_point("q", 0))
            dGdw_first = cas.jacobian(all_defects_first, noises_vector.get_noise_single(0))
            dFdz_first = cas.jacobian(states_end_first, variables_vector.get_collocation_point("q", 0))

            self.jacobian_funcs_first = cas.Function(
                "jacobian_func",
                [
                    variables_vector.get_time(),
                    variables_vector.get_state("q", 0),
                    variables_vector.get_state("qdot", 0),
                    variables_vector.get_collocation_point("q", 0),
                    variables_vector.get_controls(0),
                    variables_vector.get_controls(1),
                    noises_vector.get_noise_single(0),
                    noises_vector.get_noise_single(1),
                ],
                [dGdx_first, dGdz_first, dGdw_first, dFdz_first],
            )
            cov_matrix_first = variables_vector.get_cov_matrix(0)[:nb_states, :nb_states]
            cov_integrated_first = m_matrix_first @ (dGdx_first @ cov_matrix_first @ dGdx_first.T + dGdw_first @ sigma_ww_first @ dGdw_first.T) @ m_matrix_first.T
            cov_integrated_vector_first = variables_vector.reshape_matrix_to_vector(cov_integrated_first)


            self.cov_integration_func_first = cas.Function(
                "F",
                [
                    variables_vector.get_time(),
                    variables_vector.get_state("q", 0),
                    variables_vector.get_state("qdot", 0),
                    variables_vector.get_collocation_point("q", 0),
                    variables_vector.get_cov(0),
                    variables_vector.get_ms(0),
                    variables_vector.get_controls(0),
                    variables_vector.get_controls(1),
                    noises_vector.get_noise_single(0),
                    noises_vector.get_noise_single(1),
                ],
                [cov_integrated_vector_first],
            )

        # Integrator
        self.x_integration_func = cas.Function(
            "F",
            [
                variables_vector.get_collocation_point("q", 1),
            ],
            [z_matrix_1[:, -1]],
        )

        return

    def m_constraint(
            self,
            ocp_example: ExampleAbstract,
            variables_vector: VariablesAbstract,
    ) -> cas.Function:

        m_matrix = variables_vector.get_m_matrix(1)

        _, dGdz, _, dFdz = self.jacobian_funcs(
            variables_vector.get_time(),
            variables_vector.get_state("q", 0),
            variables_vector.get_collocation_point("q", 0),
            variables_vector.get_collocation_point("q", 1),
            variables_vector.get_controls(0),
            variables_vector.get_controls(1),
            variables_vector.get_controls(2),
            cas.DM.zeros(ocp_example.model.nb_noises * variables_vector.nb_random),
            cas.DM.zeros(ocp_example.model.nb_noises * variables_vector.nb_random),
            cas.DM.zeros(ocp_example.model.nb_noises * variables_vector.nb_random),
        )
        return cas.Function(
            "m_constraint",
            [
                variables_vector.get_time(),
                variables_vector.get_state("q", 0),
                variables_vector.get_collocation_point("q", 0),
                variables_vector.get_collocation_point("q", 1),
                variables_vector.get_controls(0),
                variables_vector.get_controls(1),
                variables_vector.get_controls(2),
                variables_vector.get_ms(1)
            ],
            [variables_vector.reshape_matrix_to_vector(
                dFdz.T - dGdz.T @ m_matrix.T
            )
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

        nb_states = ocp_example.model.nb_q
        nb_variables = ocp_example.model.nb_q * variables_vector.nb_random
        n_shooting = variables_vector.n_shooting

        # Multi-thread state continuity constraint
        multi_threaded_constraint = self.x_integration_func.map(n_shooting, "thread", n_threads)
        x_integrated = multi_threaded_constraint(
            cas.horzcat(*[variables_vector.get_collocation_point("q", i_node) for i_node in range(0, n_shooting)]),
        )
        states_next = cas.horzcat(*[variables_vector.get_state("q", i_node) for i_node in range(1, n_shooting + 1)])

        g_continuity = x_integrated - states_next
        for i_node in range(n_shooting):
            constraints.add(
                g=g_continuity[:, i_node],
                lbg=[0] * nb_variables,
                ubg=[0] * nb_variables,
                g_names=[f"dynamics_continuity_node_{i_node+1}"] * nb_variables,
                node=i_node + 1,
            )

        # Cov continuity constraint
        if self.discretization_method.name == "MeanAndCovariance":
            nb_cov_variables = nb_states * nb_states

            multi_threaded_constraint = self.cov_integration_func.map(n_shooting-1, "thread", n_threads)
            cov_integrated = multi_threaded_constraint(
                variables_vector.get_time(),
                cas.horzcat(*[variables_vector.get_state("q", i_node) for i_node in range(1, n_shooting)]),
                cas.horzcat(*[variables_vector.get_collocation_point("q", i_node) for i_node in range(0, n_shooting-1)]),
                cas.horzcat(*[variables_vector.get_collocation_point("q", i_node) for i_node in range(1, n_shooting)]),
                cas.horzcat(*[variables_vector.get_cov(i_node) for i_node in range(1, n_shooting)]),
                cas.horzcat(*[variables_vector.get_ms(i_node) for i_node in range(1, n_shooting)]),
                cas.horzcat(*[variables_vector.get_controls(i_node) for i_node in range(0, n_shooting-1)]),
                cas.horzcat(*[variables_vector.get_controls(i_node) for i_node in range(1, n_shooting)]),
                cas.horzcat(*[variables_vector.get_controls(i_node) for i_node in range(2, n_shooting + 1)]),
                cas.horzcat(*[noises_vector.get_one_vector_numerical(i_node) for i_node in range(0, n_shooting-1)]),
                cas.horzcat(*[noises_vector.get_one_vector_numerical(i_node) for i_node in range(1, n_shooting)]),
                cas.horzcat(*[noises_vector.get_one_vector_numerical(i_node) for i_node in range(2, n_shooting+1)]),
            )
            cov_next = cas.horzcat(*[variables_vector.get_cov(i_node) for i_node in range(2, n_shooting + 1)])

            for i_node in range(n_shooting-1):
                constraints.add(
                    g=cov_next[:, i_node] - cov_integrated[:, i_node],
                    lbg=[0] * nb_cov_variables,
                    ubg=[0] * nb_cov_variables,
                    g_names=[f"cov_continuity"] * nb_cov_variables,
                    node=i_node,
                )

            # First node cov continuity constraint
            g_continuity = self.cov_integration_func_first(
                variables_vector.get_time(),
                variables_vector.get_state("q", 0),
                variables_vector.get_state("qdot", 0),
                variables_vector.get_collocation_point("q", 0),
                variables_vector.get_cov(0),
                variables_vector.get_ms(0),
                variables_vector.get_controls(0),
                variables_vector.get_controls(1),
                noises_vector.get_one_vector_numerical(0),
                noises_vector.get_one_vector_numerical(1),
            )
            cov_next = variables_vector.get_cov(1)
            constraints.add(
                g=cov_next - g_continuity,
                lbg=[0] * nb_cov_variables,
                ubg=[0] * nb_cov_variables,
                g_names=[f"cov_continuity"] * nb_cov_variables,
                node=1,
            )


        # Multi-thread defect constraints
        multi_threaded_constraint = self.defect_func.map(n_shooting, "thread", n_threads)
        defects = multi_threaded_constraint(
            variables_vector.get_time(),
            cas.horzcat(*[variables_vector.get_state("q", i_node) for i_node in range(0, n_shooting)]),
            cas.horzcat(*[variables_vector.get_collocation_point("q", i_node) for i_node in range(0, n_shooting)]),
            cas.horzcat(*[variables_vector.get_controls(i_node) for i_node in range(0, n_shooting)]),
            cas.horzcat(*[variables_vector.get_controls(i_node) for i_node in range(1, n_shooting+1)]),
            cas.horzcat(*[cas.DM.zeros(ocp_example.model.nb_noises * variables_vector.nb_random) for i_node in range(0, n_shooting)]),
            cas.horzcat(*[cas.DM.zeros(ocp_example.model.nb_noises * variables_vector.nb_random) for i_node in range(0, n_shooting)]),
        )

        for i_node in range(n_shooting):
            constraints.add(
                g=defects[:, i_node],
                lbg=[0] * (nb_variables * self.order),
                ubg=[0] * (nb_variables * self.order),
                g_names=[f"collocation_defect"] * (nb_variables * self.order),
                node=i_node,
            )

        # Multi-thread M_matrix constraint
        if self.discretization_method.name == "MeanAndCovariance":
            # Constrain M at all collocation points to follow df_integrated/dz.T - dg_integrated/dz @ m.T = 0
            multi_threaded_constraint = self.m_constraint(
                ocp_example=ocp_example,
                variables_vector=variables_vector,
            ).map(n_shooting-1, "thread", n_threads)
            m_constraint = multi_threaded_constraint(
                variables_vector.get_time(),
                cas.horzcat(*[variables_vector.get_state("q", i_node) for i_node in range(0, n_shooting-1)]),
                cas.horzcat(*[variables_vector.get_collocation_point("q", i_node) for i_node in range(0, n_shooting-1)]),
                cas.horzcat(*[variables_vector.get_collocation_point("q", i_node) for i_node in range(1, n_shooting)]),
                cas.horzcat(*[variables_vector.get_controls(i_node) for i_node in range(0, n_shooting-1)]),
                cas.horzcat(*[variables_vector.get_controls(i_node) for i_node in range(1, n_shooting)]),
                cas.horzcat(*[variables_vector.get_controls(i_node) for i_node in range(2, n_shooting+1)]),
                cas.horzcat(*[variables_vector.get_ms(i_node) for i_node in range(1, n_shooting)]),
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

            # First node m constraint
            m_matrix_first = variables_vector.get_m_matrix(0)
            _, dGdz_first, _, dFdz_first = self.jacobian_funcs_first(
                variables_vector.get_time(),
                variables_vector.get_state("q", 0),
                variables_vector.get_state("qdot", 0),
                variables_vector.get_collocation_point("q", 0),
                variables_vector.get_controls(0),
                variables_vector.get_controls(1),
                cas.DM.zeros(ocp_example.model.nb_noises * variables_vector.nb_random),
                cas.DM.zeros(ocp_example.model.nb_noises * variables_vector.nb_random),
            )
            m_constraint_first = variables_vector.reshape_matrix_to_vector(
                dFdz_first.T - dGdz_first.T @ m_matrix_first.T
            )
            nb_components = m_constraint_first.shape[0]
            constraints.add(
                g=m_constraint_first,
                lbg=[0] * nb_components,
                ubg=[0] * nb_components,
                g_names=[f"m_constraint_first"] * nb_components,
                node=1,
            )

        # Ld transition defect
        multi_threaded_constraint = self.transition_defects_func.map(n_shooting-1, "thread", n_threads)

        ld_transition_defect = multi_threaded_constraint(
            variables_vector.get_time(),
            cas.horzcat(*[variables_vector.get_state("q", i_node) for i_node in range(0, n_shooting-1)]),
            cas.horzcat(*[variables_vector.get_state("q", i_node) for i_node in range(1, n_shooting)]),
            cas.horzcat(*[variables_vector.get_collocation_point("q", i_node) for i_node in range(0, n_shooting-1)]),
            cas.horzcat(*[variables_vector.get_collocation_point("q", i_node) for i_node in range(1, n_shooting)]),
            cas.horzcat(*[variables_vector.get_controls(i_node) for i_node in range(0, n_shooting-1)]),
            cas.horzcat(*[variables_vector.get_controls(i_node) for i_node in range(1, n_shooting)]),
            cas.horzcat(*[variables_vector.get_controls(i_node) for i_node in range(2, n_shooting+1)]),
            cas.horzcat(*[noises_vector.get_one_vector_numerical(i_node) for i_node in range(0, n_shooting-1)]),
            cas.horzcat(*[noises_vector.get_one_vector_numerical(i_node) for i_node in range(1, n_shooting)]),
            cas.horzcat(*[noises_vector.get_one_vector_numerical(i_node) for i_node in range(2, n_shooting+1)]),
        )

        for i_node in range(n_shooting - 1):
            constraints.add(
                g=ld_transition_defect[:, i_node],
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
            variables_vector.get_collocation_point("q", 0),
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
            variables_vector.get_collocation_point("q", n_shooting - 1),
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
