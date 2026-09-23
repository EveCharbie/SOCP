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
        qdot_sym: list[cas.MX | cas.SX],
        i_collocation: int,
        node: int,
    ) -> cas.MX | cas.SX:

        controls_0 = variables_vector.get_controls(node)
        controls_1 = variables_vector.get_controls(node + 1)
        ref = variables_vector.get_ref(node)
        noises_0 = noises_vector.get_noise_single(node)

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
                    noises = noises_0[ocp_example.model.nb_noises * i_sigma : ocp_example.model.nb_noises * (i_sigma + 1)]
                else:
                    this_z_matrix = z_matrix
                    noises = noises_0

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
                    qdot=qdot_sym,
                    padded_x=variables_vector.get_states_list(0),
                    u=variables_vector.get_controls(node=0),
                    ref_sym=ref,
                    noise=noises_single,
                )(
                    this_z_matrix[:, j_collocation],
                    DP,
                    cas.DM.zeros(cas.vertcat(*variables_vector.get_states_list(0)).shape[0]),  # TODO: see what to do in this case for not q and qdot states!
                    controls,
                    ref,
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
        q_1 = variables_vector.get_state("q", 1)
        qz_matrix_1 = variables_vector.get_collocation_point("q", 1)
        p_1 = variables_vector.get_state("p", 1)
        pz_matrix_1 = variables_vector.get_collocation_point("p", 1)

        # Declare the noise matrix
        sigma_std = noises_vector.noise_magnitude_matrix
        sigma_ww = noises_vector.get_noise_matrix(1).T @ noises_vector.get_noise_matrix(1)

        # Declare some useful functions
        nb_qdot = len(variables_vector.state_indices["qdot"])
        if self.discretization_method.name in ["Deterministic", "MeanAndCovariance", "UnscentedTransform"]:
            qdot_sym = [variables_vector.cx.sym("qdot", nb_qdot)]
        elif self.discretization_method.name == "NoiseDiscretization":
            qdot_sym = [variables_vector.cx.sym(f"qdot_{i_random}", nb_qdot) for i_random in range(variables_vector.nb_random)]
        else:
            raise RuntimeError(f"Discretization method {self.discretization_method.name} not implemented.")

        lagrangian_func = self.discretization_method.get_lagrangian(
            ocp_example=ocp_example,
            q=variables_vector.get_state_list(name="q", node=0),
            qdot=qdot_sym,
            u=variables_vector.get_controls(node=0),
        )
        DqL_func = cas.Function(
            "DqL_func",
            [
                cas.vertcat(*variables_vector.get_state_list(name="q", node=0)),
                cas.vertcat(*qdot_sym),
                variables_vector.get_controls(node=0),
            ],
            [
                self.discretization_method.get_lagrangian_jacobian_q(
                    ocp_example,
                    lagrangian_func(
                        q=cas.vertcat(*variables_vector.get_state_list(name="q", node=0)),
                        qdot=cas.vertcat(*qdot_sym),
                        u=variables_vector.get_controls(node=0),
                    )["L"],
                    q=variables_vector.get_state_list(name="q", node=0),
                    qdot=qdot_sym,
                )(
                    cas.vertcat(*variables_vector.get_state_list(name="q", node=0)),
                    cas.vertcat(*qdot_sym),
                )
            ],
        )
        DvL_func = cas.Function(
            "DvL_func",
            [
                cas.vertcat(*variables_vector.get_state_list(name="q", node=0)),
                cas.vertcat(*qdot_sym),
                variables_vector.get_controls(node=0),
            ],
            [
                self.discretization_method.get_lagrangian_jacobian_qdot(
                    ocp_example,
                    lagrangian_func(
                        q=cas.vertcat(*variables_vector.get_state_list(name="q", node=0)),
                        qdot=cas.vertcat(*qdot_sym),
                        u=variables_vector.get_controls(node=0),
                    )["L"],
                    q=variables_vector.get_state_list(name="q", node=0),
                    qdot=qdot_sym,
                )(
                    cas.vertcat(*variables_vector.get_state_list(name="q", node=0)),
                    cas.vertcat(*qdot_sym),
                )
            ],
        )

        # Integration - p
        integrated_p = self.get_fd(
            ocp_example=ocp_example,
            variables_vector=variables_vector,
            noises_vector=noises_vector,
            nb_total_q=nb_total_q,
            dt=dt,
            z_matrix=qz_matrix_1,
            DqL_func=DqL_func,
            DvL_func=DvL_func,
            qdot_sym=qdot_sym,
            i_collocation=self.nb_collocation_points - 1,
            node=1,
        )
        integrated_p = variables_vector.reshape_matrix_to_vector(integrated_p)

        # Integration - q
        integrated_q = qz_matrix_1[:, 0]
        for j_collocation in range(self.nb_collocation_points):
            integrated_q += (
                dt
                * self.lobatto.weights[j_collocation]
                * self.get_slope(
                    nb_slopes=nb_total_q,
                    dt=dt,
                    z_matrix=qz_matrix_1,
                    j_collocation=j_collocation,
                )
            )

        # Integrator
        if discretization_method.name == "UnscentedTransform":
            integrated_states_matrix = cas.vertcat(
                variables_vector.reshape_vector_to_matrix(integrated_q, (nb_q, variables_vector.nb_sigma_points)),
                variables_vector.reshape_vector_to_matrix(integrated_p, (nb_q, variables_vector.nb_sigma_points)),
            )
            integrated_states = variables_vector.get_mean_sigma(variables_vector.reshape_matrix_to_vector(integrated_states_matrix))
        else:
            integrated_states = cas.vertcat(integrated_q, integrated_p)
        self.x_integration_func = cas.Function(
            "F",
            [
                variables_vector.get_time(),
                variables_vector.get_collocation_points(node=1),
                variables_vector.get_controls(node=1),
                variables_vector.get_controls(node=2),
                variables_vector.get_ref(node=1),
                noises_vector.get_noise_single(node=1),
            ],
            [integrated_states],
        )

        # Transition defect
        initial_p_defect = pz_matrix_1[:, 0] + variables_vector.reshape_matrix_to_vector(self.get_fd(
            ocp_example=ocp_example,
            variables_vector=variables_vector,
            noises_vector=noises_vector,
            nb_total_q=nb_total_q,
            dt=dt,
            z_matrix=qz_matrix_1,
            DqL_func=DqL_func,
            DvL_func=DvL_func,
            qdot_sym=qdot_sym,
            i_collocation=0,
            node=1,
        ))

        # Slopes are constrained only for the q, since the qdot are constrained by the same polynomials
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
                    qdot_sym=qdot_sym,
                    i_collocation=i_collocation,
                    node=1,
                ))
            ]
        # TODO: add state continuity and slope defects for variables that are not q and qdot

        # Defects
        # First collocation state = x and initial momentum = p
        if discretization_method.name == "UnscentedTransform":
            z_matrix_1 = variables_vector.get_collocation_points(node=1)
            initial_states_defect = z_matrix_1[:, 0] - variables_vector.reshape_matrix_to_vector(variables_vector.get_sigma_states(1, sigma_std)[:variables_vector.nb_states, :])
        elif discretization_method.name in ["MeanAndCovariance", "NoiseDiscretization", "Deterministic"]:
            initial_states_defect = qz_matrix_1[:, 0] - q_1
        else:
            raise NotImplementedError(f"discretization method not recognized :{discretization_method.name}")
        first_defect = [cas.vertcat(initial_states_defect, initial_p_defect)]

        # Defect function
        defects = cas.vertcat(*first_defect, *slope_defects)
        self.defect_func = cas.Function(
            "defects",
            [
                variables_vector.get_time(),
                variables_vector.get_state("q", 1),
                variables_vector.get_state("p", 1),
                variables_vector.get_collocation_points(node=1),
                variables_vector.get_chol_cov(1),
                cas.vertcat(*variables_vector.get_states_list(0)),  # Should not be used
                variables_vector.get_controls(1),
                variables_vector.get_controls(2),
                variables_vector.get_ref(1),
                noises_vector.get_noise_single(1),
            ],
            [defects],
        )

        self.jacobian_funcs = None
        if self.discretization_method.name == "MeanAndCovariance":
            m_matrix = variables_vector.get_m_matrix(1)

            dGdx = cas.jacobian(
                defects,
                cas.vertcat(
                    variables_vector.get_state("q", 1),
                    variables_vector.get_state("p", 1),
                ),
            )
            dGdz = cas.jacobian(defects,variables_vector.get_collocation_point("q", 1))
            dGdw = cas.jacobian(defects, noises_vector.get_noise_single(1))
            dFdz = cas.jacobian(integrated_states, variables_vector.get_collocation_point("q", 1))

            self.jacobian_funcs = cas.Function(
                "jacobian_func",
                [
                    variables_vector.get_time(),
                    variables_vector.get_state("q", 1),
                    variables_vector.get_collocation_point("q", 1),
                    cas.vertcat(*variables_vector.get_states_list(0)),  # Should not be used
                    variables_vector.get_controls(1),
                    variables_vector.get_controls(2),
                    variables_vector.get_ref(1),
                    noises_vector.get_noise_single(1),
                ],
                [dGdx, dGdz, dGdw, dFdz],
            )
            cov_matrix = variables_vector.get_cov_matrix(1)
            cov_integrated = m_matrix @ (dGdx @ cov_matrix @ dGdx.T + dGdw @ sigma_ww @ dGdw.T) @ m_matrix.T

            cov_integrated_vector = variables_vector.reshape_matrix_to_vector(cov_integrated)

            # Cov integrator
            self.cov_integration_func = cas.Function(
                "F",
                [
                    variables_vector.get_time(),
                    variables_vector.get_state("q", 1),
                    variables_vector.get_collocation_point("q", 1),
                    cas.vertcat(*variables_vector.get_states_list(0)),  # Should not be used
                    variables_vector.get_cov(1),
                    variables_vector.get_ms(1),
                    variables_vector.get_controls(1),
                    variables_vector.get_controls(2),
                    variables_vector.get_ref(1),
                    noises_vector.get_noise_single(1),
                ],
                [cov_integrated_vector],
            )

        elif self.discretization_method.name == "UnscentedTransform":
            integrated_q_matrix = variables_vector.reshape_vector_to_matrix(
                integrated_q, (nb_q, variables_vector.nb_sigma_points)
            )
            integrated_p_matrix = variables_vector.reshape_vector_to_matrix(
                integrated_p, (nb_q, variables_vector.nb_sigma_points)
            )
            integrated_states_matrix = cas.vertcat(integrated_q_matrix, integrated_p_matrix)

            diff = integrated_states_matrix - integrated_states
            cov_integrated_matrix = (diff @ diff.T) / (variables_vector.nb_sigma_points - 1)
            self.chol_cov_integration_func = cas.Function(
                "chol_cov_integration",
                [
                    variables_vector.get_time(),
                    variables_vector.get_collocation_points(node=1),
                    variables_vector.get_controls(node=1),
                    variables_vector.get_controls(node=2),
                    variables_vector.get_ref(node=1),
                    noises_vector.get_noise_single(node=1),
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
            variables_vector.get_state("q", 1),
            variables_vector.get_collocation_point("q", 1),
            cas.vertcat(*variables_vector.get_states_list(0)),  # Should not be used
            variables_vector.get_controls(1),
            variables_vector.get_controls(2),
            variables_vector.get_ref(1),
            cas.DM.zeros(ocp_example.model.nb_noises * variables_vector.nb_random),
        )
        return cas.Function(
            "m_constraint",
            [
                variables_vector.get_time(),
                variables_vector.get_state("q", 1),
                variables_vector.get_collocation_point("q", 1),
                cas.vertcat(*variables_vector.get_states_list(0)),  # Should not be used
                variables_vector.get_controls(1),
                variables_vector.get_controls(2),
                variables_vector.get_ref(1),
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
            nb_total_q = ocp_example.model.nb_q
        elif self.discretization_method.name == "NoiseDiscretization":
            nb_total_q = ocp_example.model.nb_q * variables_vector.nb_random
        elif self.discretization_method.name == "UnscentedTransform":
            nb_total_q = ocp_example.model.nb_q * variables_vector.nb_sigma_points
        else:
            raise NotImplementedError(f"Discretization method {self.discretization_method.name} not implemented.")

        dt = variables_vector.get_time() / ocp_example.n_shooting

        if self.discretization_method.name in ["Deterministic", "MeanAndCovariance"]:
            nb_defects = ocp_example.model.nb_q
            nb_continuity = 2*ocp_example.model.nb_q
            multiplier = 1
        elif self.discretization_method.name in ["NoiseDiscretization"]:
            nb_defects = ocp_example.model.nb_q * variables_vector.nb_random
            nb_continuity = 2*ocp_example.model.nb_q * variables_vector.nb_random
            multiplier = variables_vector.nb_random
        elif self.discretization_method.name == "UnscentedTransform":
            nb_defects = ocp_example.model.nb_q * variables_vector.nb_sigma_points
            nb_continuity = 2*ocp_example.model.nb_q
            multiplier = variables_vector.nb_sigma_points
        else:
            raise NotImplementedError("This discretization method is not supported yet.")

        n_shooting = variables_vector.n_shooting

        # Multi-thread state continuity constraint
        multi_threaded_constraint = self.x_integration_func.map(n_shooting, "thread", n_threads)
        x_integrated = multi_threaded_constraint(
            cas.horzcat(*[variables_vector.get_time() for _ in range(0, n_shooting)]),
            cas.horzcat(*[variables_vector.get_collocation_points(i_node) for i_node in range(0, n_shooting)]),
            cas.horzcat(*[variables_vector.get_controls(i_node) for i_node in range(0, n_shooting)]),
            cas.horzcat(*[variables_vector.get_controls(i_node) for i_node in range(1, n_shooting+1)]),
            cas.horzcat(*[variables_vector.get_ref(i_node) for i_node in range(0, n_shooting)]),
            cas.horzcat(*[noises_vector.get_one_vector_numerical(i_node) for i_node in range(0, n_shooting)]),
        )

        states_next = cas.horzcat(*[
            cas.vertcat(
                variables_vector.get_state("q", i_node),
                variables_vector.get_state("p", i_node),
            ) for i_node in range(1, n_shooting + 1)])

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
            nb_cov_variables = (2 * nb_q) * (2 * nb_q)

            multi_threaded_constraint = self.cov_integration_func.map(n_shooting, "thread", n_threads)
            cov_integrated = multi_threaded_constraint(
                variables_vector.get_time(),
                cas.horzcat(*[variables_vector.get_state("q", i_node) for i_node in range(0, n_shooting)]),
                cas.horzcat(
                    *[variables_vector.get_collocation_point("q", i_node) for i_node in range(0, n_shooting)]
                ),
                cas.horzcat(*[variables_vector.get_states(0) for _ in range(1, n_shooting+1)]),  # Should not be used
                cas.horzcat(*[variables_vector.get_cov(i_node) for i_node in range(0, n_shooting)]),
                cas.horzcat(*[variables_vector.get_ms(i_node) for i_node in range(0, n_shooting)]),
                cas.horzcat(*[variables_vector.get_controls(i_node) for i_node in range(0, n_shooting)]),
                cas.horzcat(*[variables_vector.get_controls(i_node) for i_node in range(1, n_shooting+1)]),
                cas.horzcat(*[variables_vector.get_ref(i_node) for i_node in range(0, n_shooting)]),
                cas.horzcat(*[noises_vector.get_one_vector_numerical(i_node) for i_node in range(0, n_shooting)]),
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

        elif self.discretization_method.name in ["Deterministic", "NoiseDiscretization"]:
            pass
        elif self.discretization_method.name == "UnscentedTransform":
            nb_cov_variables = (2 * nb_q) * (2 * nb_q)

            multi_threaded_integrator = self.chol_cov_integration_func.map(n_shooting, "thread", n_threads)
            cov_integrated = multi_threaded_integrator(
                cas.horzcat(*[variables_vector.get_time() for _ in range(0, n_shooting)]),
                cas.horzcat(*[variables_vector.get_collocation_points(node=i_node) for i_node in range(0, n_shooting)]),
                cas.horzcat(*[variables_vector.get_controls(node=i_node) for i_node in range(0, n_shooting)]),
                cas.horzcat(*[variables_vector.get_controls(node=i_node) for i_node in range(1, n_shooting+1)]),
                cas.horzcat(*[variables_vector.get_ref(node=i_node) for i_node in range(0, n_shooting)]),
                cas.horzcat(*[noises_vector.get_one_vector_numerical(node=i_node) for i_node in range(0, n_shooting)]),
            )

            cov_next = cas.horzcat(*[variables_vector.reshape_matrix_to_vector(
                variables_vector.get_chol_cov_matrix(i_node) @
                variables_vector.get_chol_cov_matrix(i_node).T,
            ) for i_node in range(1, n_shooting + 1)])

            for i_node in range(n_shooting):
                constraints.add(
                    g=cov_next[:, i_node] - cov_integrated[:, i_node],
                    lbg=[0] * nb_cov_variables,
                    ubg=[0] * nb_cov_variables,
                    g_names=[f"cov_continuity"] * nb_cov_variables,
                    node=i_node,
                )
        else:
            raise NotImplementedError("This discretization method is not supported yet.")

        # Multi-thread defect constraints
        multi_threaded_constraint = self.defect_func.map(n_shooting, "thread", n_threads)
        defects = multi_threaded_constraint(
            variables_vector.get_time(),
            cas.horzcat(*[variables_vector.get_state("q", i_node) for i_node in range(0, n_shooting)]),
            cas.horzcat(*[variables_vector.get_state("p", i_node) for i_node in range(0, n_shooting)]),
            cas.horzcat(*[variables_vector.get_collocation_points(node=i_node) for i_node in range(0, n_shooting)]),
            cas.horzcat(*[variables_vector.get_chol_cov(i_node) for i_node in range(0, n_shooting)]),
            cas.horzcat(*[variables_vector.get_states(0) for _ in range(1, n_shooting+1)]),  # Should not be used
            cas.horzcat(*[variables_vector.get_controls(i_node) for i_node in range(0, n_shooting)]),
            cas.horzcat(*[variables_vector.get_controls(i_node) for i_node in range(1, n_shooting + 1)]),
            cas.horzcat(*[variables_vector.get_ref(i_node) for i_node in range(0, n_shooting)]),
            cas.horzcat(*[noises_vector.get_one_vector_numerical(i_node) for i_node in range(0, n_shooting)]),
        )

        if self.discretization_method.name == "UnscentedTransform":
            for i_node in range(n_shooting):
                constraints.add(
                    g=defects[:, i_node],
                    lbg=[0] * nb_defects * (self.order + 2),
                    ubg=[0] * nb_defects * (self.order + 2),
                    g_names=[f"collocation_defect"] * nb_defects * (self.order + 2),
                    node=i_node,
                )
        else:
            for i_node in range(n_shooting):
                constraints.add(
                    g=defects[:, i_node],
                    lbg=[0] * nb_defects * (self.order + 1),
                    ubg=[0] * nb_defects * (self.order + 1),
                    g_names=[f"collocation_defect"] * nb_defects * (self.order + 1),
                    node=i_node,
                )

        # Multi-thread M_matrix constraint
        if self.discretization_method.name in ["MeanAndCovariance", "UnscentedTransform"]:
            if self.discretization_method.name == "MeanAndCovariance":
                # Constrain M at all collocation points to follow df_integrated/dz.T - dg_integrated/dz @ m.T = 0
                multi_threaded_constraint = self.m_constraint(
                    ocp_example=ocp_example,
                    variables_vector=variables_vector,
                ).map(n_shooting, "thread", n_threads)
                m_constraint = multi_threaded_constraint(
                    variables_vector.get_time(),
                    cas.horzcat(
                        *[variables_vector.get_state("q", i_node) for i_node in range(0, n_shooting)]
                    ),
                    cas.horzcat(
                        *[variables_vector.get_collocation_point("q", i_node) for i_node in range(0, n_shooting)]
                    ),
                    cas.horzcat(*[variables_vector.get_states(0) for _ in range(1, n_shooting+1)]),  # Should not be used
                    cas.horzcat(*[variables_vector.get_controls(i_node) for i_node in range(0, n_shooting)]),
                    cas.horzcat(*[variables_vector.get_controls(i_node) for i_node in range(1, n_shooting+1)]),
                    cas.horzcat(*[variables_vector.get_ref(i_node) for i_node in range(0, n_shooting)]),
                    cas.horzcat(*[variables_vector.get_ms(i_node) for i_node in range(0, n_shooting)]),
                )

                for i_node in range(n_shooting):
                    nb_components = m_constraint[:, i_node].shape[0]
                    constraints.add(
                        g=m_constraint[:, i_node],
                        lbg=[0] * nb_components,
                        ubg=[0] * nb_components,
                        g_names=[f"m_constraint"] * nb_components,
                        node=i_node + 1,
                    )

        elif self.discretization_method.name in ["Deterministic", "NoiseDiscretization"]:
            pass

        else:
            raise NotImplementedError("This discretization method is not supported yet.")

        # ref_sym = real ref
        nb_qdot = len(ocp_example.model.qdot_indices)
        for i_node in range(n_shooting + 1):
            # Please note that this is false for the last node (n_shooting) because the qdot computed will be zero,
            # But it is useful to fix a value for ref and this value should never be used in any constraint
            ref_sym = variables_vector.get_ref(i_node)
            if isinstance(ref_sym, (cas.MX, cas.SX)):
                z_matrix = variables_vector.reshape_vector_to_matrix(
                    variables_vector.get_collocation_point("q", node=i_node),
                    (nb_total_q, self.nb_collocation_points),
                )
                qdot_from_collocation = self.get_slope(
                    nb_slopes=nb_total_q,
                    dt=dt,
                    z_matrix=z_matrix,
                    j_collocation=0,  # At the node only
                )
                q_list = variables_vector.get_state_list("q", node=i_node)
                qdot_list = [qdot_from_collocation[i_current*nb_qdot: (i_current+1)*nb_qdot] for i_current in range(variables_vector.nb_random)]
                real_ref = self.discretization_method.get_reference(
                    ocp_example,
                    q_list,
                    qdot_list,
                    variables_vector.get_states_list(node=i_node),
                    variables_vector.get_controls(node=i_node),
                )
                nb_components = ref_sym.shape[0]
                constraints.add(
                    g=ref_sym - real_ref,
                    lbg=[0] * nb_components,
                    ubg=[0] * nb_components,
                    g_names=[f"ref"] * nb_components,
                    node=i_node,
                )
            elif ocp_example.model.nb_references > 0 and self.discretization_method.name != "Deterministic":
                raise RuntimeError(
                    f"The get_ref method was not implemented for discretization method {self.discretization_method.name}.")
