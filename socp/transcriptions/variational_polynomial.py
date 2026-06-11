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
        self.lagrange_coefficients = self.lobatto.get_lagrange_coefficients()

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
        nb_slopes: int,
        dt: cas.MX | cas.SX | cas.DM | np.ndarray,
        z_matrix: cas.MX | cas.SX | cas.DM | np.ndarray,
        j_collocation: int,
    ):
        # Equation (15) from Campos & al: Q_i = q_0 + h * sum_{j=1}^s a_{ij} * \dot{Q}_j
        if isinstance(dt, (np.ndarray, float)):
            Q = np.zeros((nb_slopes, ))
        else:
            Q = type(dt).zeros(nb_slopes)
        for i_collocation in range(self.nb_collocation_points):
            Q += z_matrix[:, i_collocation] * self.lagrange_coefficients[i_collocation, j_collocation, 1]
        DP = Q / dt
        return DP

    def get_fd(
        self,
        ocp_example: ExampleAbstract,
        variables_vector: VariablesAbstract,
        noises_vector: NoisesAbstract,
        nb_total_q: int,
        dt: cas.MX | cas.SX,
        z_matrix: cas.MX | cas.SX,
        DqL_func: cas.Function,
        DvL_func: cas.Function,
        i_collocation: int,
        node: int,
    ) -> cas.MX | cas.SX:

        controls_0 = variables_vector.get_controls(node)
        controls_1 = variables_vector.get_controls(node + 1)
        noises_0 = noises_vector.get_one_vector_numerical(node)
        noises_1 = noises_vector.get_one_vector_numerical(node + 1)

        if variables_vector.nb_sigma_points > 1:
            nb_slopes = ocp_example.model.nb_q
            noises_single = noises_vector.get_one_noise(node, sigma_point=0)
        else:
            nb_slopes = nb_total_q
            noises_single = noises_vector.get_noise_single(node)
        fd = variables_vector.cx.zeros(nb_slopes, variables_vector.nb_sigma_points)
        for j_collocation in range(self.nb_collocation_points):
            for i_sigma in range(variables_vector.nb_sigma_points):
                if variables_vector.nb_sigma_points > 1:
                    this_z_matrix = z_matrix[ocp_example.model.nb_q * i_sigma : ocp_example.model.nb_q * (i_sigma + 1), :]
                else:
                    this_z_matrix = z_matrix
                DP = self.get_slope(
                    nb_slopes=nb_slopes,
                    dt=dt,
                    z_matrix=this_z_matrix,
                    j_collocation=j_collocation,
                )
                C = self.lagrange_coefficients[i_collocation, j_collocation, 0]
                DC = self.lagrange_coefficients[i_collocation, j_collocation, 1]

                controls = self.discretization_method.interpolate_between_nodes(
                    var_pre=controls_0,
                    var_post=controls_1,
                    time_ratio=self.lobatto.time_grid[j_collocation],
                )

                noises = self.discretization_method.interpolate_between_nodes(
                    var_pre=noises_0[ocp_example.model.nb_noises * i_sigma : ocp_example.model.nb_noises * (i_sigma + 1)],
                    var_post=noises_1[ocp_example.model.nb_noises * i_sigma : ocp_example.model.nb_noises * (i_sigma + 1)],
                    time_ratio=self.lobatto.time_grid[j_collocation],
                )

                DqL = DqL_func(
                    this_z_matrix[:, j_collocation],
                    DP,
                    controls,
                )
                DvL = DvL_func(
                    this_z_matrix[:, j_collocation],
                    DP,
                    controls,
                )

                force = self.discretization_method.get_non_conservative_forces(
                    ocp_example=ocp_example,
                    q=variables_vector.get_state_list(name="q", node=0),
                    qdot=variables_vector.get_state_list(name="qdot", node=0),
                    padded_x=variables_vector.get_states_list(0),
                    u=controls,
                    noise=noises_single,
                )(
                    this_z_matrix[:, j_collocation],
                    DP,
                    cas.DM.zeros(cas.vertcat(*variables_vector.get_states_list(0)).shape[0]),  # TODO: see what to do in this case for not q and qdot states!
                    controls,
                    noises,
                )

                fd[:, i_sigma] += self.lobatto.weights[j_collocation] * (dt * DqL * C + DvL * DC + dt * force * C)

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

        if self.discretization_method.name in ["Deterministic", "MeanAndCovariance"]:
            nb_total_q = ocp_example.model.nb_q
        elif self.discretization_method.name == "NoiseDiscretization":
            nb_total_q = ocp_example.model.nb_q * variables_vector.nb_random
        elif self.discretization_method.name == "UnscentedTransform":
            nb_total_q = ocp_example.model.nb_q * variables_vector.nb_sigma_points
        else:
            raise NotImplementedError(f"Discretization method {self.discretization_method.name} not implemented.")

        nb_q = ocp_example.model.nb_q

        # Declare some variables
        dt = variables_vector.get_time() / ocp_example.n_shooting
        q_0 = variables_vector.get_state("q", 0)
        q_1 = variables_vector.get_state("q", 1)
        qz_matrix_0 = variables_vector.reshape_vector_to_matrix(
            variables_vector.get_collocation_point("q", 0),
            (nb_total_q, self.nb_collocation_points),
        )
        qz_matrix_1 = variables_vector.reshape_vector_to_matrix(
            variables_vector.get_collocation_point("q", 1),
            (nb_total_q, self.nb_collocation_points),
        )
        qz_matrix_penultimate = variables_vector.reshape_vector_to_matrix(
            variables_vector.get_collocation_point("q", variables_vector.n_shooting - 1),
            (nb_total_q, self.nb_collocation_points),
        )

        # Declare the noise matrix
        sigma_ww = noises_vector.get_noise_matrix(1)

        # Declare some useful functions
        lagrangian_func = self.discretization_method.get_lagrangian(
            ocp_example=ocp_example,
            q=variables_vector.get_state_list(name="q", node=0),
            qdot=variables_vector.get_state_list(name="qdot", node=0),
            u=variables_vector.get_controls(node=0),
        )
        DqL_func = cas.Function(
            "DqL_func",
            [
                cas.vertcat(*variables_vector.get_state_list(name="q", node=0)),
                cas.vertcat(*variables_vector.get_state_list(name="qdot", node=0)),
                variables_vector.get_controls(node=0),
            ],
            [
                self.discretization_method.get_lagrangian_jacobian_q(
                    ocp_example,
                    lagrangian_func(
                        q=cas.vertcat(*variables_vector.get_state_list(name="q", node=0)),
                        qdot=cas.vertcat(*variables_vector.get_state_list(name="qdot", node=0)),
                        u=variables_vector.get_controls(node=0),
                    )["L"],
                    q=variables_vector.get_state_list(name="q", node=0),
                    qdot=variables_vector.get_state_list(name="qdot", node=0),
                )(
                    cas.vertcat(*variables_vector.get_state_list(name="q", node=0)),
                    cas.vertcat(*variables_vector.get_state_list(name="qdot", node=0)),
                )
            ],
        )
        DvL_func = cas.Function(
            "DvL_func",
            [
                cas.vertcat(*variables_vector.get_state_list(name="q", node=0)),
                cas.vertcat(*variables_vector.get_state_list(name="qdot", node=0)),
                variables_vector.get_controls(node=0),
            ],
            [
                self.discretization_method.get_lagrangian_jacobian_qdot(
                    ocp_example,
                    lagrangian_func(
                        q=cas.vertcat(*variables_vector.get_state_list(name="q", node=0)),
                        qdot=cas.vertcat(*variables_vector.get_state_list(name="qdot", node=0)),
                        u=variables_vector.get_controls(node=0),
                    )["L"],
                    q=variables_vector.get_state_list(name="q", node=0),
                    qdot=variables_vector.get_state_list(name="qdot", node=0),
                )(
                    cas.vertcat(*variables_vector.get_state_list(name="q", node=0)),
                    cas.vertcat(*variables_vector.get_state_list(name="qdot", node=0)),
                )
            ],
        )

        # Integration
        if self.discretization_method.name == "UnscentedTransform":
            integrated_states = variables_vector.get_mean_sigma(qz_matrix_1[:, -1])
        elif self.discretization_method.name in ["Deterministic", "NoiseDiscretization", "MeanAndCovariance"]:
            integrated_states = qz_matrix_1[:, -1]
        else:
            raise NotImplementedError("This discretization method is not supported yet.")

        # Integrator
        self.x_integration_func = cas.Function(
            "F",
            [
                variables_vector.get_collocation_point("q", 1),
                variables_vector.get_chol_cov(1),
            ],
            [integrated_states],
        )

        # Transition defect
        p_previous = self.get_fd(
            ocp_example=ocp_example,
            variables_vector=variables_vector,
            noises_vector=noises_vector,
            nb_total_q=nb_total_q,
            dt=dt,
            z_matrix=qz_matrix_0,
            DqL_func=DqL_func,
            DvL_func=DvL_func,
            i_collocation=self.nb_collocation_points - 1,
            node=0,
        )

        transition_defect = p_previous + self.get_fd(
            ocp_example=ocp_example,
            variables_vector=variables_vector,
            noises_vector=noises_vector,
            nb_total_q=nb_total_q,
            dt=dt,
            z_matrix=qz_matrix_1,
            DqL_func=DqL_func,
            DvL_func=DvL_func,
            i_collocation=0,
            node=1,
        )

        slope_defects = []
        for i_collocation in range(1, self.nb_collocation_points - 1):
            slope_defects += [
                variables_vector.reshape_matrix_to_vector(self.get_fd(
                    ocp_example=ocp_example,
                    variables_vector=variables_vector,
                    noises_vector=noises_vector,
                    nb_total_q=nb_total_q,
                    dt=dt,
                    z_matrix=qz_matrix_1,
                    DqL_func=DqL_func,
                    DvL_func=DvL_func,
                    i_collocation=i_collocation,
                    node=1,
                ))
            ]
        # TODO: add state continuity and slope defects for variables that are not q and qdot

        # Defects
        # First collocation state = x
        if discretization_method.name == "UnscentedTransform":
            first_defect = [
                qz_matrix_1[:, 0] - variables_vector.reshape_matrix_to_vector(variables_vector.get_sigma_states(1, sigma_ww)[:nb_q, :]),
                qz_matrix_1[:, -1] - variables_vector.reshape_matrix_to_vector(variables_vector.get_sigma_states(2, sigma_ww)[:nb_q, :]),
            ]
        elif discretization_method.name in ["MeanAndCovariance", "NoiseDiscretization", "Deterministic"]:
            first_defect = [qz_matrix_1[:, 0] - q_1]
        else:
            raise NotImplementedError(f"discretization method not recognized :{discretization_method.name}")


        # Defect function
        defects = cas.vertcat(*first_defect, *slope_defects)
        self.defect_func = cas.Function(
            "defects",
            [
                variables_vector.get_time(),
                variables_vector.get_state("q", 1),
                variables_vector.get_state("q", 2),
                variables_vector.get_collocation_point("q", 1),
                variables_vector.get_chol_cov(1),
                variables_vector.get_chol_cov(2),
                cas.vertcat(*variables_vector.get_states_list(0)),  # Should not be used
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
                variables_vector.get_chol_cov(0),
                variables_vector.get_chol_cov(1),
                cas.vertcat(*variables_vector.get_states_list(0)),  # Should not be used
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
        if self.discretization_method.name in ["Determinisitc", "MeanAndCovariance", "NoiseDiscretization"]:
            qdot_0 = variables_vector.get_state("qdot", 0)
            p0 = self.discretization_method.get_momentum(
                ocp_example=ocp_example,
                q=variables_vector.get_state_list(name="q", node=0),
                qdot=variables_vector.get_state_list(name="qdot", node=0),
                u=variables_vector.get_controls(node=0),
            )(
                q_0,
                qdot_0,
                variables_vector.get_controls(0),
            )
        elif self.discretization_method.name == "UnscentedTransform":
            states_0 = variables_vector.get_sigma_states(0, sigma_ww)
            p0 = variables_vector.cx.zeros(ocp_example.model.nb_q, variables_vector.nb_sigma_points)
            momentum_func = self.discretization_method.get_momentum(
                ocp_example=ocp_example,
                q=variables_vector.get_state_list(name="q", node=0),
                qdot=variables_vector.get_state_list(name="qdot", node=0),
                u=variables_vector.get_controls(node=0),
            )
            for i_sigma in range(variables_vector.nb_sigma_points):
                p0[:, i_sigma] = momentum_func(
                    states_0[i_sigma * variables_vector.nb_states + ocp_example.model.q_indices.start: i_sigma * variables_vector.nb_states + ocp_example.model.q_indices.stop],
                    states_0[
                        i_sigma * variables_vector.nb_states + ocp_example.model.qdot_indices.start: i_sigma * variables_vector.nb_states + ocp_example.model.qdot_indices.stop],
                    variables_vector.get_controls(0),
                )
        else:
            raise NotImplementedError(f"Discretization method {self.discretization_method.name} not implemented")

        initial_defect = p0 + self.get_fd(
            ocp_example=ocp_example,
            variables_vector=variables_vector,
            noises_vector=noises_vector,
            nb_total_q=nb_total_q,
            dt=dt,
            z_matrix=qz_matrix_0,
            DqL_func=DqL_func,
            DvL_func=DvL_func,
            i_collocation=0,
            node=0,
        )

        self.initial_defect_func = cas.Function(
            "initial_defects",
            [
                variables_vector.get_time(),
                variables_vector.get_state("q", 0),
                variables_vector.get_state("qdot", 0),
                variables_vector.get_collocation_point("q", 0),
                variables_vector.get_chol_cov(0),
                cas.vertcat(*variables_vector.get_states_list(0)),   # Should not be used for now
                variables_vector.get_controls(0),
                variables_vector.get_controls(1),
                noises_vector.get_noise_single(0),
                noises_vector.get_noise_single(1),
            ],
            [initial_defect],
        )

        # Final defect
        if self.discretization_method.name in ["Deterministic", "MeanAndCovariance", "NoiseDiscretization"]:
            q_N = variables_vector.get_state("q", variables_vector.n_shooting)
            qdot_N = variables_vector.get_state("qdot", variables_vector.n_shooting)
            pN = self.discretization_method.get_momentum(
                ocp_example=ocp_example,
                q=variables_vector.get_state_list(name="q", node=0),
                qdot=variables_vector.get_state_list(name="qdot", node=0),
                u=variables_vector.get_controls(node=0),
            )(
                q_N,
                qdot_N,
                variables_vector.get_controls(variables_vector.n_shooting - 1),
            )
        elif self.discretization_method.name == "UnscentedTransform":
            sigma_ww_N = noises_vector.get_noise_matrix(variables_vector.n_shooting)
            states_N = variables_vector.get_sigma_states(variables_vector.n_shooting, sigma_ww_N)
            pN = variables_vector.cx.zeros(ocp_example.model.nb_q, variables_vector.nb_sigma_points)
            for i_sigma in range(variables_vector.nb_sigma_points):
                pN[:, i_sigma] = momentum_func(
                    states_N[
                        i_sigma * variables_vector.nb_states + ocp_example.model.q_indices.start: i_sigma * variables_vector.nb_states + ocp_example.model.q_indices.stop],
                    states_N[
                        i_sigma * variables_vector.nb_states + ocp_example.model.qdot_indices.start: i_sigma * variables_vector.nb_states + ocp_example.model.qdot_indices.stop],
                    variables_vector.get_controls(variables_vector.n_shooting - 1)
                )
        else:
            raise NotImplementedError(f"Discretization method {self.discretization_method.name} not implemented")

        p_penultimate = self.get_fd(
            ocp_example=ocp_example,
            variables_vector=variables_vector,
            noises_vector=noises_vector,
            nb_total_q=nb_total_q,
            dt=dt,
            z_matrix=qz_matrix_penultimate,
            DqL_func=DqL_func,
            DvL_func=DvL_func,
            i_collocation=self.nb_collocation_points - 1,
            node=variables_vector.n_shooting - 1,
        )
        final_defect = p_penultimate - pN

        self.final_defect_func = cas.Function(
            "final_defects",
            [
                variables_vector.get_time(),
                variables_vector.get_state("q", variables_vector.n_shooting),
                variables_vector.get_state("qdot", variables_vector.n_shooting),
                variables_vector.get_collocation_point("q", variables_vector.n_shooting - 1),
                variables_vector.get_chol_cov(variables_vector.n_shooting),
                cas.vertcat(*variables_vector.get_states_list(0)), # Should not be used for now
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

            states_end = qz_matrix_1[:, 0]
            for j_collocation in range(self.nb_collocation_points):
                states_end += (
                    dt
                    * self.lobatto.weights[j_collocation]
                    * self.get_slope(
                        nb_total_q=nb_total_q,
                        dt=dt,
                        z_matrix=qz_matrix_1,
                        j_collocation=j_collocation,
                    )
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
                    cas.vertcat(*variables_vector.get_states_list(0)),  # Should not be used
                    variables_vector.get_controls(0),
                    variables_vector.get_controls(1),
                    variables_vector.get_controls(2),
                    noises_vector.get_noise_single(0),
                    noises_vector.get_noise_single(1),
                    noises_vector.get_noise_single(2),
                ],
                [dGdx, dGdz, dGdw, dFdz],
            )
            cov_matrix = variables_vector.get_cov_matrix(1)[:nb_q, :nb_q]
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
                    cas.vertcat(*variables_vector.get_states_list(0)),  # Should not be used
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
            sigma_ww_first = noises_vector.get_noise_matrix(0)

            states_end_first = qz_matrix_0[:, 0]
            for j_collocation in range(self.nb_collocation_points):
                states_end_first += (
                    dt
                    * self.lobatto.weights[j_collocation]
                    * self.get_slope(
                        nb_total_q=nb_total_q,
                        dt=dt,
                        z_matrix=qz_matrix_0,
                        j_collocation=j_collocation,
                    )
                )

            first_defect_first = [qz_matrix_0[:, 0] - q_0]

            slope_defects_first = []
            for i_collocation in range(1, self.nb_collocation_points - 1):
                slope_defects_first += [
                    self.get_fd(
                        ocp_example=ocp_example,
                        variables_vector=variables_vector,
                        noises_vector=noises_vector,
                        nb_total_q=nb_total_q,
                        dt=dt,
                        z_matrix=qz_matrix_0,
                        DqL_func=DqL_func,
                        DvL_func=DvL_func,
                        i_collocation=i_collocation,
                        node=0,
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
                    cas.vertcat(*variables_vector.get_states_list(0)),  # Should not be used
                    variables_vector.get_controls(0),
                    variables_vector.get_controls(1),
                    noises_vector.get_noise_single(0),
                    noises_vector.get_noise_single(1),
                ],
                [dGdx_first, dGdz_first, dGdw_first, dFdz_first],
            )
            cov_matrix_first = variables_vector.get_cov_matrix(0)[:nb_q, :nb_q]
            cov_integrated_first = (
                m_matrix_first
                @ (dGdx_first @ cov_matrix_first @ dGdx_first.T + dGdw_first @ sigma_ww_first @ dGdw_first.T)
                @ m_matrix_first.T
            )
            cov_integrated_vector_first = variables_vector.reshape_matrix_to_vector(cov_integrated_first)

            self.cov_integration_func_first = cas.Function(
                "F",
                [
                    variables_vector.get_time(),
                    variables_vector.get_state("q", 0),
                    variables_vector.get_state("qdot", 0),
                    variables_vector.get_collocation_point("q", 0),
                    cas.vertcat(*variables_vector.get_states_list(0)),  # Should not be used
                    variables_vector.get_cov(0),
                    variables_vector.get_ms(0),
                    variables_vector.get_controls(0),
                    variables_vector.get_controls(1),
                    noises_vector.get_noise_single(0),
                    noises_vector.get_noise_single(1),
                ],
                [cov_integrated_vector_first],
            )
        elif self.discretization_method.name == "UnscentedTransform":
            diff = variables_vector.reshape_vector_to_matrix(qz_matrix_1[:, -1], (variables_vector.nb_q, variables_vector.nb_sigma_points))[:nb_q, :] - integrated_states
            cov_integrated_matrix = (diff @ diff.T) / (variables_vector.nb_sigma_points - 1)
            self.chol_cov_integration_func = cas.Function(
                "chol_cov_integration",
                [
                    variables_vector.get_collocation_points(1),
                ],
                [variables_vector.reshape_matrix_to_vector(cov_integrated_matrix)],
            )
        elif self.discretization_method.name in ["Deterministic", "NoiseDiscretization"]:
            pass
        else:
            raise NotImplementedError("This discretization method is not supported yet.")

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
            cas.vertcat(*variables_vector.get_states_list(0)),  # Should not be used
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
                cas.vertcat(*variables_vector.get_states_list(0)),  # Should not be used
                variables_vector.get_controls(0),
                variables_vector.get_controls(1),
                variables_vector.get_controls(2),
                variables_vector.get_ms(1),
            ],
            [variables_vector.reshape_matrix_to_vector(dFdz.T - dGdz.T @ m_matrix.T)],
        )

    def set_dynamics_constraints(
        self,
        ocp_example: ExampleAbstract,
        variables_vector: VariablesAbstract,
        noises_vector: NoisesAbstract,
        constraints: Constraints,
        n_threads: int = 8,
    ) -> None:

        nb_q = ocp_example.model.nb_q

        if self.discretization_method.name in ["Deterministic", "MeanAndCovariance"]:
            nb_defects = ocp_example.model.nb_q
            nb_continuity = ocp_example.model.nb_q
            multiplier = 1
        elif self.discretization_method.name in ["NoiseDiscretization"]:
            nb_defects = ocp_example.model.nb_q * variables_vector.nb_random
            nb_continuity = ocp_example.model.nb_q * variables_vector.nb_random
            multiplier = variables_vector.nb_random
        elif self.discretization_method.name == "UnscentedTransform":
            nb_defects = ocp_example.model.nb_q * variables_vector.nb_sigma_points
            nb_continuity = ocp_example.model.nb_q
            multiplier = variables_vector.nb_sigma_points
        else:
            raise NotImplementedError("This discretization method is not supported yet.")

        n_shooting = variables_vector.n_shooting

        # Multi-thread state continuity constraint
        multi_threaded_constraint = self.x_integration_func.map(n_shooting, "thread", n_threads)
        x_integrated = multi_threaded_constraint(
            cas.horzcat(*[variables_vector.get_collocation_point("q", i_node) for i_node in range(0, n_shooting)]),
            cas.horzcat(*[variables_vector.get_chol_cov(i_node) for i_node in range(0, n_shooting)]),
        )
        states_next = cas.horzcat(*[variables_vector.get_state("q", i_node) for i_node in range(1, n_shooting + 1)])

        g_continuity = x_integrated - states_next
        for i_node in range(n_shooting):
            constraints.add(
                g=g_continuity[:, i_node],
                lbg=[0] * nb_continuity,
                ubg=[0] * nb_continuity,
                g_names=[f"dynamics_continuity_node_{i_node+1}"] * nb_continuity,
                node=i_node + 1,
            )

        # Cov continuity constraint
        if self.discretization_method.name == "MeanAndCovariance":
            nb_cov_variables = nb_q * nb_q

            multi_threaded_constraint = self.cov_integration_func.map(n_shooting - 1, "thread", n_threads)
            cov_integrated = multi_threaded_constraint(
                variables_vector.get_time(),
                cas.horzcat(*[variables_vector.get_state("q", i_node) for i_node in range(1, n_shooting)]),
                cas.horzcat(
                    *[variables_vector.get_collocation_point("q", i_node) for i_node in range(0, n_shooting - 1)]
                ),
                cas.horzcat(*[variables_vector.get_collocation_point("q", i_node) for i_node in range(1, n_shooting)]),
                cas.horzcat(*[variables_vector.get_states(0) for i_node in range(1, n_shooting)]),  # Should not be used
                cas.horzcat(*[variables_vector.get_cov(i_node) for i_node in range(1, n_shooting)]),
                cas.horzcat(*[variables_vector.get_ms(i_node) for i_node in range(1, n_shooting)]),
                cas.horzcat(*[variables_vector.get_controls(i_node) for i_node in range(0, n_shooting - 1)]),
                cas.horzcat(*[variables_vector.get_controls(i_node) for i_node in range(1, n_shooting)]),
                cas.horzcat(*[variables_vector.get_controls(i_node) for i_node in range(2, n_shooting + 1)]),
                cas.horzcat(*[noises_vector.get_one_vector_numerical(i_node) for i_node in range(0, n_shooting - 1)]),
                cas.horzcat(*[noises_vector.get_one_vector_numerical(i_node) for i_node in range(1, n_shooting)]),
                cas.horzcat(*[noises_vector.get_one_vector_numerical(i_node) for i_node in range(2, n_shooting + 1)]),
            )
            cov_next = cas.horzcat(*[variables_vector.get_cov(i_node) for i_node in range(2, n_shooting + 1)])

            for i_node in range(n_shooting - 1):
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
                variables_vector.get_states(0),  # Should not be used
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
        elif self.discretization_method.name in ["Deterministic", "NoiseDiscretization"]:
            pass
        elif self.discretization_method.name == "UnscentedTransform":
            nb_cov_variables = nb_q * nb_q

            # multi_threaded_integrator = self.chol_cov_integration_func.map(n_shooting, "thread", n_threads)
            # cov_integrated = multi_threaded_integrator(
            #     cas.horzcat(*[variables_vector.get_collocation_points(i_node) for i_node in range(0, n_shooting)]),
            # )
            #
            # cov_next = cas.horzcat(*[variables_vector.reshape_matrix_to_vector(
            #     variables_vector.get_chol_cov_matrix(i_node)[:nb_q, :nb_q] @
            #     variables_vector.get_chol_cov_matrix(i_node)[:nb_q, :nb_q].T,
            # ) for i_node in range(1, n_shooting + 1)])
            #
            # for i_node in range(n_shooting):
            #     constraints.add(
            #         g=cov_next[:, i_node] - cov_integrated[:, i_node],
            #         lbg=[0] * nb_cov_variables,
            #         ubg=[0] * nb_cov_variables,
            #         g_names=[f"cov_continuity"] * nb_cov_variables,
            #         node=i_node,
            #     )
        else:
            raise NotImplementedError("This discretization method is not supported yet.")

        # Multi-thread defect constraints
        multi_threaded_constraint = self.defect_func.map(n_shooting, "thread", n_threads)
        defects = multi_threaded_constraint(
            variables_vector.get_time(),
            cas.horzcat(*[variables_vector.get_state("q", i_node) for i_node in range(0, n_shooting)]),
            cas.horzcat(*[variables_vector.get_state("q", i_node) for i_node in range(1, n_shooting+1)]),
            cas.horzcat(*[variables_vector.get_collocation_point("q", i_node) for i_node in range(0, n_shooting)]),
            cas.horzcat(*[variables_vector.get_chol_cov(i_node) for i_node in range(0, n_shooting)]),
            cas.horzcat(*[variables_vector.get_chol_cov(i_node) for i_node in range(1, n_shooting+1)]),
            cas.horzcat(*[variables_vector.get_states(0) for i_node in range(0, n_shooting)]),  # Should not be used
            cas.horzcat(*[variables_vector.get_controls(i_node) for i_node in range(0, n_shooting)]),
            cas.horzcat(*[variables_vector.get_controls(i_node) for i_node in range(1, n_shooting + 1)]),
            cas.horzcat(
                *[
                    cas.DM.zeros(ocp_example.model.nb_noises * multiplier)
                    for i_node in range(0, n_shooting)
                ]
            ),
            cas.horzcat(
                *[
                    cas.DM.zeros(ocp_example.model.nb_noises * multiplier)
                    for i_node in range(0, n_shooting)
                ]
            ),
        )

        if self.discretization_method.name == "UnscentedTransform":
            for i_node in range(n_shooting):
                constraints.add(
                    g=defects[:, i_node],
                    lbg=[0] * nb_defects * (self.order + 1),
                    ubg=[0] * nb_defects * (self.order + 1),
                    g_names=[f"collocation_defect"] * nb_defects * (self.order + 1),
                    node=i_node,
                )
        else:
            for i_node in range(n_shooting):
                constraints.add(
                    g=defects[:, i_node],
                    lbg=[0] * nb_defects * self.order,
                    ubg=[0] * nb_defects * self.order,
                    g_names=[f"collocation_defect"] * nb_defects * self.order,
                    node=i_node,
                )

        # Multi-thread M_matrix constraint
        if self.discretization_method.name == "MeanAndCovariance":
            # Constrain M at all collocation points to follow df_integrated/dz.T - dg_integrated/dz @ m.T = 0
            multi_threaded_constraint = self.m_constraint(
                ocp_example=ocp_example,
                variables_vector=variables_vector,
            ).map(n_shooting - 1, "thread", n_threads)
            m_constraint = multi_threaded_constraint(
                variables_vector.get_time(),
                cas.horzcat(*[variables_vector.get_state("q", i_node) for i_node in range(0, n_shooting - 1)]),
                cas.horzcat(
                    *[variables_vector.get_collocation_point("q", i_node) for i_node in range(0, n_shooting - 1)]
                ),
                cas.horzcat(*[variables_vector.get_collocation_point("q", i_node) for i_node in range(1, n_shooting)]),
                cas.horzcat(*[variables_vector.get_states(0) for i_node in range(1, n_shooting)]),  # Should not be used
                cas.horzcat(*[variables_vector.get_controls(i_node) for i_node in range(0, n_shooting - 1)]),
                cas.horzcat(*[variables_vector.get_controls(i_node) for i_node in range(1, n_shooting)]),
                cas.horzcat(*[variables_vector.get_controls(i_node) for i_node in range(2, n_shooting + 1)]),
                cas.horzcat(*[variables_vector.get_ms(i_node) for i_node in range(1, n_shooting)]),
            )

            for i_node in range(n_shooting - 1):
                nb_components = m_constraint[:, i_node].shape[0]
                constraints.add(
                    g=m_constraint[:, i_node],
                    lbg=[0] * nb_components,
                    ubg=[0] * nb_components,
                    g_names=[f"collocation_defect"] * nb_components,
                    node=i_node + 1,
                )

            # First node m constraint
            m_matrix_first = variables_vector.get_m_matrix(0)
            _, dGdz_first, _, dFdz_first = self.jacobian_funcs_first(
                variables_vector.get_time(),
                variables_vector.get_state("q", 0),
                variables_vector.get_state("qdot", 0),
                variables_vector.get_collocation_point("q", 0),
                variables_vector.get_states(0),  # Should not be used
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
        elif self.discretization_method.name in ["Deterministic", "NoiseDiscretization", "UnscentedTransform"]:
            pass
        else:
            raise NotImplementedError("This discretization method is not supported yet.")

        # Ld transition defect
        multi_threaded_constraint = self.transition_defects_func.map(n_shooting - 1, "thread", n_threads)
        ld_transition_defect = multi_threaded_constraint(
            variables_vector.get_time(),
            cas.horzcat(*[variables_vector.get_state("q", i_node) for i_node in range(0, n_shooting - 1)]),
            cas.horzcat(*[variables_vector.get_state("q", i_node) for i_node in range(1, n_shooting)]),
            cas.horzcat(*[variables_vector.get_collocation_point("q", i_node) for i_node in range(0, n_shooting - 1)]),
            cas.horzcat(*[variables_vector.get_collocation_point("q", i_node) for i_node in range(1, n_shooting)]),
            cas.horzcat(*[variables_vector.get_chol_cov(i_node) for i_node in range(0, n_shooting - 1)]),
            cas.horzcat(*[variables_vector.get_chol_cov(i_node) for i_node in range(1, n_shooting)]),
            cas.horzcat(*[variables_vector.get_states(0) for i_node in range(1, n_shooting)]),  # Should not be used
            cas.horzcat(*[variables_vector.get_controls(i_node) for i_node in range(0, n_shooting - 1)]),
            cas.horzcat(*[variables_vector.get_controls(i_node) for i_node in range(1, n_shooting)]),
            cas.horzcat(*[variables_vector.get_controls(i_node) for i_node in range(2, n_shooting + 1)]),
            cas.horzcat(*[noises_vector.get_one_vector_numerical(i_node) for i_node in range(0, n_shooting - 1)]),
            cas.horzcat(*[noises_vector.get_one_vector_numerical(i_node) for i_node in range(1, n_shooting)]),
            cas.horzcat(*[noises_vector.get_one_vector_numerical(i_node) for i_node in range(2, n_shooting + 1)]),
        )

        for i_node in range(n_shooting - 1):
            constraints.add(
                g=ld_transition_defect[:, i_node],
                lbg=[0] * nb_continuity,
                ubg=[0] * nb_continuity,
                g_names=[f"Ld_continuity_node_{i_node+1}"] * nb_continuity,
                node=i_node + 1,
            )

        # First node defect
        initial_defect = self.initial_defect_func(
            variables_vector.get_time(),
            variables_vector.get_state("q", node=0),
            variables_vector.get_state("qdot", node=0),
            variables_vector.get_collocation_point("q", node=0),
            variables_vector.get_chol_cov(node=0),
            variables_vector.get_states(node=0),
            variables_vector.get_controls(node=0),
            variables_vector.get_controls(node=1),
            noises_vector.get_one_vector_numerical(node=0),
            noises_vector.get_one_vector_numerical(node=1),
        )
        constraints.add(
            g=variables_vector.reshape_matrix_to_vector(initial_defect),
            lbg=[0] * nb_defects,
            ubg=[0] * nb_defects,
            g_names=[f"dynamics_initial_defect"] * nb_defects,
            node=0,
        )

        # Last node defect
        final_defect = self.final_defect_func(
            variables_vector.get_time(),
            variables_vector.get_state("q", node=n_shooting),
            variables_vector.get_state("qdot", node=n_shooting),
            variables_vector.get_collocation_point("q", node=n_shooting - 1),
            variables_vector.get_chol_cov(node=variables_vector.n_shooting),
            variables_vector.get_states(node=0),  # Should not be used for now
            variables_vector.get_controls(node=n_shooting - 1),
            variables_vector.get_controls(node=n_shooting),
            noises_vector.get_one_vector_numerical(node=n_shooting - 1),
            noises_vector.get_one_vector_numerical(node=n_shooting),
        )
        constraints.add(
            g=variables_vector.reshape_matrix_to_vector(final_defect),
            lbg=[0] * nb_defects,
            ubg=[0] * nb_defects,
            g_names=[f"dynamics_final_defect"] * nb_defects,
            node=n_shooting,
        )
