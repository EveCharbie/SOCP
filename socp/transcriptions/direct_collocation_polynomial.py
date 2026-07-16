"""
Legendre was used as in Gillis et al. 2013, but Radau polynomials might be preferred because of their better robustness
for collocation methods. However, it might be less accurate for the same number of collocation points.
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


class DirectCollocationPolynomial(TranscriptionAbstract):

    def __init__(self, order: int = 5) -> None:

        super().__init__()  # Does nothing
        self.order = order
        self.lagrange_polynomial = LagrangePolynomial(order)

    @property
    def name(self) -> str:
        return "DirectCollocationPolynomial"

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
        """
        Formulate discrete time dynamics integration using a Radau collocation scheme.
        """

        # Note: The first x and u used to declare the casadi functions, but all nodes will be used during the evaluation of the functions
        self.discretization_method = discretization_method

        if discretization_method.name == "UnscentedTransform":
            nb_total_z = variables_vector.nb_states * variables_vector.nb_sigma_points
        elif discretization_method.name == "NoiseDiscretization":
            nb_total_z = variables_vector.nb_states * variables_vector.nb_random
        elif discretization_method.name in ["Deterministic", "MeanAndCovariance"]:
            nb_total_z = variables_vector.nb_states
        else:
            raise NotImplementedError(f"discretization method not recognized :{discretization_method.name}")

        z_matrix = variables_vector.reshape_vector_to_matrix(
            variables_vector.get_collocation_points(0),
            (nb_total_z, self.nb_collocation_points),
        )
        states_end = self.lagrange_polynomial.get_states_end(z_matrix)
        dt = variables_vector.get_time() / ocp_example.n_shooting

        # State dynamics
        ref, ref_sym = discretization_method.get_reference(
            ocp_example,
            variables_vector.get_state("q", node=0),
            variables_vector.get_state("qdot", node=0),
            variables_vector.get_states(node=0),
            variables_vector.get_controls(node=0),
        )
        xdot = self.discretization_method.state_dynamics(
            ocp_example,
            variables_vector.get_states(0),
            variables_vector.get_controls(0),
            ref_sym,
            noises_vector.get_noise_single(0),
            with_q_qdot=True,
        )
        self.dynamics_func = cas.Function(
            f"dynamics",
            [
                variables_vector.get_states(0),
                variables_vector.get_controls(0),
                ref_sym,
                noises_vector.get_noise_single(0),
                ],
            [xdot],
            ["x", "u", "ref", "noise"],
            ["xdot"],
        )

        # Declare the noise matrix
        sigma_ww = noises_vector.get_noise_matrix(0)

        # Defects
        # First collocation state = x
        if discretization_method.name == "UnscentedTransform":
            sigma_ww_magnitude = noises_vector.noise_magnitude_matrix
            first_defect = [variables_vector.reshape_matrix_to_vector(variables_vector.get_sigma_states(0, sigma_ww_magnitude)[:variables_vector.nb_states, :]) - z_matrix[:, 0]]
        elif discretization_method.name in ["MeanAndCovariance", "NoiseDiscretization", "Deterministic"]:
            first_defect = [variables_vector.get_states(0) - z_matrix[:, 0]]
        else:
            raise NotImplementedError(f"discretization method not recognized :{discretization_method.name}")

        # Collocation slopes
        slope_defects = []
        for j_collocation in range(1, self.nb_collocation_points):
            vertical_variation = self.lagrange_polynomial.interpolate_first_derivative(
                z_matrix, self.lagrange_polynomial.time_grid[j_collocation]
            )

            # To follow Gillis et al., it should be :
            # slope = vertical_variation
            # xdot = self.discretization_method.state_dynamics(
            #     ocp_example,
            #     z_matrix[:, j_collocation],
            #     variables_vector.get_controls(0),
            #     ref_sym,
            #     noises_vector.get_noise_single(0),
            #     with_q_qdot = True,
            # ) * dt
            # but it has an impact on convergence, so I will leave it as is for now.

            this_control = self.discretization_method.interpolate_between_nodes(
                var_pre=variables_vector.get_controls(0),
                var_post=variables_vector.get_controls(1),
                time_ratio=j_collocation / (self.nb_collocation_points - 1),
            )
            slope = vertical_variation / dt
            if self.discretization_method.name in ["Deterministic", "MeanAndCovariance", "NoiseDiscretization"]:
                xdot = self.discretization_method.state_dynamics(
                    ocp_example,
                    z_matrix[:, j_collocation],
                    this_control,
                    ref_sym,
                    noises_vector.get_noise_single(0),
                    with_q_qdot=True,
                )
                slope_defects += [slope - xdot]
            elif self.discretization_method.name == "UnscentedTransform":
                for i_sigma in range(variables_vector.nb_sigma_points):
                    these_indices = range(variables_vector.nb_states * i_sigma, variables_vector.nb_states * (i_sigma + 1))
                    xdot = self.discretization_method.state_dynamics(
                        ocp_example,
                        z_matrix[these_indices, j_collocation],
                        this_control,
                        ref_sym,
                        noises_vector.get_noise_single(0),
                        with_q_qdot=True,
                    )
                    slope_defects += [slope[these_indices] - xdot]
            else:
                raise NotImplementedError(f"Discretization method {self.discretization_method.name} not recognized")

        # Integration
        if self.discretization_method.name == "UnscentedTransform":
            integrated_states = variables_vector.get_mean_sigma(states_end)
        elif self.discretization_method.name in ["Deterministic", "NoiseDiscretization", "MeanAndCovariance"]:
            integrated_states = states_end
        else:
            raise NotImplementedError("This discretization method is not supported yet.")

        # Integrator
        self.x_integration_func = cas.Function(
            "F",
            [
                variables_vector.get_collocation_points(0),
            ],
            [integrated_states],
        )

        # Defects
        defects = cas.vertcat(*first_defect, *slope_defects)
        if self.discretization_method.name == "MeanAndCovariance":
            m_matrix = variables_vector.get_m_matrix(0)

            dGdx = cas.jacobian(defects, variables_vector.get_states(0))
            dGdz = cas.jacobian(defects, variables_vector.get_collocation_points(0))
            dGdw = cas.jacobian(defects, noises_vector.get_noise_single(0))
            dFdz = cas.jacobian(states_end, variables_vector.get_collocation_points(0))

            self.jacobian_funcs = cas.Function(
                "jacobian_func",
                [
                    variables_vector.get_time(),
                    variables_vector.get_states(0),
                    variables_vector.get_collocation_points(0),
                    variables_vector.get_controls(0),
                    variables_vector.get_controls(1),
                    noises_vector.get_noise_single(0),
                ],
                [dGdx, dGdz, dGdw, dFdz],
            )
            cov_matrix = variables_vector.get_cov_matrix(0)
            cov_integrated = m_matrix @ (dGdx @ cov_matrix @ dGdx.T + dGdw @ sigma_ww @ dGdw.T) @ m_matrix.T

            cov_integrated_vector = variables_vector.reshape_matrix_to_vector(cov_integrated)

            self.cov_integration_func = cas.Function(
                "F",
                [
                    variables_vector.get_time(),
                    variables_vector.get_states(0),
                    variables_vector.get_collocation_points(0),
                    variables_vector.get_cov(0),
                    variables_vector.get_ms(0),
                    variables_vector.get_controls(0),
                    variables_vector.get_controls(1),
                    noises_vector.get_noise_single(0),
                ],
                [cov_integrated_vector],
            )
        elif self.discretization_method.name == "UnscentedTransform":
            diff = variables_vector.reshape_vector_to_matrix(states_end, (variables_vector.nb_states, variables_vector.nb_sigma_points)) - integrated_states
            cov_integrated_matrix = (diff @ diff.T) / (variables_vector.nb_sigma_points - 1)
            self.chol_cov_integration_func = cas.Function(
                "chol_cov_integration",
                [
                    variables_vector.get_time(),
                    variables_vector.get_collocation_points(0),
                    variables_vector.get_chol_cov(0),
                    variables_vector.get_controls(0),
                    variables_vector.get_controls(1),
                    noises_vector.get_noise_single(0),
                ],
                [variables_vector.reshape_matrix_to_vector(cov_integrated_matrix)],
            )
        elif self.discretization_method.name in ["Deterministic", "NoiseDiscretization"]:
            pass
        else:
            raise NotImplementedError("This discretization method is not supported yet.")

        # Defect function
        self.defect_func = cas.Function(
            "defects",
            [
                variables_vector.get_time(),
                variables_vector.get_states(0),
                variables_vector.get_collocation_points(0),
                variables_vector.get_chol_cov(0),
                variables_vector.get_controls(0),
                variables_vector.get_controls(1),
                noises_vector.get_noise_single(0),
            ],
            [defects],
        )
        return

    def m_constraint(
        self,
        ocp_example: ExampleAbstract,
        variables_vector: VariablesAbstract,
    ) -> cas.Function:

        m_matrix = variables_vector.get_m_matrix(0)

        _, dGdz, _, dFdz = self.jacobian_funcs(
            variables_vector.get_time(),
            variables_vector.get_states(0),
            variables_vector.get_collocation_points(0),
            variables_vector.get_controls(0),
            variables_vector.get_controls(1),
            cas.DM.zeros(ocp_example.model.nb_noises * variables_vector.nb_random),
        )

        return cas.Function(
            "m_constraint",
            [
                variables_vector.get_time(),
                variables_vector.get_states(0),
                variables_vector.get_collocation_points(0),
                variables_vector.get_controls(0),
                variables_vector.get_controls(1),
                variables_vector.get_ms(0),
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

        nb_states = ocp_example.model.nb_states
        nb_variables = ocp_example.model.nb_states * variables_vector.nb_random
        n_shooting = variables_vector.n_shooting

        # Multi-thread continuity constraint
        multi_threaded_integrator = self.x_integration_func.map(n_shooting, "thread", n_threads)
        x_integrated = multi_threaded_integrator(
            cas.horzcat(*[variables_vector.get_collocation_points(i_node) for i_node in range(0, n_shooting)]),
        )
        x_next = cas.horzcat(*[variables_vector.get_states(i_node) for i_node in range(1, n_shooting + 1)])

        g_continuity = x_integrated - x_next
        for i_node in range(n_shooting):
            constraints.add(
                g=g_continuity[:, i_node],
                lbg=[0] * nb_variables,
                ubg=[0] * nb_variables,
                g_names=[f"dynamics_continuity"] * nb_variables,
                node=i_node,
            )

        if self.discretization_method.name == "MeanAndCovariance":
            nb_cov_variables = nb_states * nb_states

            multi_threaded_integrator = self.cov_integration_func.map(n_shooting, "thread", n_threads)
            cov_integrated = multi_threaded_integrator(
                variables_vector.get_time(),
                cas.horzcat(*[variables_vector.get_states(i_node) for i_node in range(0, n_shooting)]),
                cas.horzcat(*[variables_vector.get_collocation_points(i_node) for i_node in range(0, n_shooting)]),
                cas.horzcat(*[variables_vector.get_cov(i_node) for i_node in range(0, n_shooting)]),
                cas.horzcat(*[variables_vector.get_ms(i_node) for i_node in range(0, n_shooting)]),
                cas.horzcat(*[variables_vector.get_controls(i_node) for i_node in range(0, n_shooting)]),
                cas.horzcat(*[variables_vector.get_controls(i_node) for i_node in range(1, n_shooting + 1)]),
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
            nb_cov_variables = nb_states * nb_states

            multi_threaded_integrator = self.chol_cov_integration_func.map(n_shooting, "thread", n_threads)
            cov_integrated = multi_threaded_integrator(
                variables_vector.get_time(),
                cas.horzcat(*[variables_vector.get_collocation_points(i_node) for i_node in range(0, n_shooting)]),
                cas.horzcat(*[variables_vector.get_chol_cov(i_node) for i_node in range(0, n_shooting)]),
                cas.horzcat(*[variables_vector.get_controls(i_node) for i_node in range(0, n_shooting)]),
                cas.horzcat(*[variables_vector.get_controls(i_node) for i_node in range(1, n_shooting + 1)]),
                cas.horzcat(*[noises_vector.get_one_vector_numerical(i_node) for i_node in range(0, n_shooting)]),
            )

            cov_next = cas.horzcat(*[variables_vector.reshape_matrix_to_vector(
                variables_vector.get_chol_cov_matrix(i_node)[:nb_states, :nb_states] @
                variables_vector.get_chol_cov_matrix(i_node)[:nb_states, :nb_states].T,
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
            cas.horzcat(*[variables_vector.get_states(i_node) for i_node in range(0, n_shooting)]),
            cas.horzcat(*[variables_vector.get_collocation_points(i_node) for i_node in range(0, n_shooting)]),
            cas.horzcat(*[variables_vector.get_chol_cov(i_node) for i_node in range(0, n_shooting)]),
            cas.horzcat(*[variables_vector.get_controls(i_node) for i_node in range(0, n_shooting)]),
            cas.horzcat(*[variables_vector.get_controls(i_node) for i_node in range(1, n_shooting + 1)]),
            cas.horzcat(
                *[
                    cas.DM.zeros(ocp_example.model.nb_noises * variables_vector.nb_random)
                    for i_node in range(0, n_shooting)
                ]
            ),
        )

        if self.discretization_method.name in ["Deterministic", "NoiseDiscretization", "MeanAndCovariance"]:
            nb_defects = ocp_example.model.nb_states * variables_vector.nb_random * (self.order + 1)
        elif self.discretization_method.name == "UnscentedTransform":
            nb_defects = ocp_example.model.nb_states * variables_vector.nb_sigma_points * (self.order + 1)
        else:
            raise NotImplementedError("This discretization method is not supported yet.")

        for i_node in range(n_shooting):
            constraints.add(
                g=defects[:, i_node],
                lbg=[0] * nb_defects,
                ubg=[0] * nb_defects,
                g_names=[f"collocation_defect"] * nb_defects,
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
                cas.horzcat(*[variables_vector.get_states(i_node) for i_node in range(0, n_shooting)]),
                cas.horzcat(*[variables_vector.get_collocation_points(i_node) for i_node in range(0, n_shooting)]),
                cas.horzcat(*[variables_vector.get_controls(i_node) for i_node in range(0, n_shooting)]),
                cas.horzcat(*[variables_vector.get_controls(i_node) for i_node in range(1, n_shooting + 1)]),
                cas.horzcat(*[variables_vector.get_ms(i_node) for i_node in range(0, n_shooting)]),
            )

            for i_node in range(n_shooting):
                nb_components = m_constraint[:, i_node].shape[0]
                constraints.add(
                    g=m_constraint[:, i_node],
                    lbg=[0] * nb_components,
                    ubg=[0] * nb_components,
                    g_names=[f"collocation_defect"] * nb_components,
                    node=i_node + 1,
                )
        elif self.discretization_method.name in ["Deterministic", "NoiseDiscretization", "UnscentedTransform"]:
            pass
        else:
            raise NotImplementedError("This discretization method is not supported yet.")