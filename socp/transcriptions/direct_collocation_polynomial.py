"""
Legendre was used as in Gillis et al. 2013, but Radau polynomials might be preferred because of their better robustness
for collocation methods. However, it might be less accurate for the same number of collocation points.
"""

import casadi as cas
import numpy as np

from .discretization_abstract import DiscretizationAbstract
from .noises_abstract import NoisesAbstract
from .transcription_abstract import TranscriptionAbstract
from .variables_abstract import VariablesAbstract
from ..examples.example_abstract import ExampleAbstract
from ..constraints import Constraints


class DirectCollocationPolynomial(TranscriptionAbstract):

    def __init__(self, order: int = 5) -> None:

        super().__init__()  # Does nothing
        self.order = order
        self.time_grid = [0] + cas.collocation_points(self.order, "legendre")

    @property
    def nb_collocation_points(self):
        return self.order + 1

    def initialize_dynamics_integrator(
        self,
        ocp_example: ExampleAbstract,
        discretization_method: DiscretizationAbstract,
        variables_vector: VariablesAbstract,
        noises_vector: NoisesAbstract,
    ) -> None:

        # Note: The first x and u used to declare the casadi functions, but all nodes will be used during the evaluation of the functions
        self.discretization_method = discretization_method
        dynamics_func, integration_func, defect_func, jacobian_funcs = self.declare_dynamics_integrator(
            ocp_example,
            discretization_method,
            variables_vector,
            noises_vector,
        )

        self.dynamics_func = dynamics_func
        self.integration_func = integration_func
        self.defect_func = defect_func
        self.jacobian_funcs = jacobian_funcs

    @property
    def name(self) -> str:
        return "DirectCollocationPolynomial"

    def partial_lagrange_polynomial(
        self, j_collocation: int, time_control_interval: cas.SX, i_collocation: int
    ) -> cas.SX:
        _l = 1
        for r_collocation in range(self.nb_collocation_points):
            if r_collocation != j_collocation and r_collocation != i_collocation:
                _l *= (time_control_interval - self.time_grid[r_collocation]) / (
                    self.time_grid[j_collocation] - self.time_grid[r_collocation]
                )
        return _l

    def lagrange_polynomial(self, j_collocation: int, time_control_interval: cas.SX) -> cas.SX:
        _l = 1
        for r_collocation in range(self.nb_collocation_points):
            if r_collocation != j_collocation:
                _l *= (time_control_interval - self.time_grid[r_collocation]) / (
                    self.time_grid[j_collocation] - self.time_grid[r_collocation]
                )
        return _l

    def lagrange_polynomial_derivative(self, j_collocation: int, time_control_interval: cas.SX) -> cas.SX:

        sum_term = 0
        for k_collocation in range(self.nb_collocation_points):
            if k_collocation == j_collocation:
                continue

            partial_Ljk = self.partial_lagrange_polynomial(j_collocation, time_control_interval, k_collocation)
            sum_term += 1.0 / (self.time_grid[j_collocation] - self.time_grid[k_collocation]) * partial_Ljk

        return sum_term

    def get_states_end(self, z_matrix: cas.SX) -> cas.SX:

        states_end = 0
        for j_collocation in range(self.nb_collocation_points):
            sum_term = self.lagrange_polynomial(
                j_collocation=j_collocation,
                time_control_interval=1.0,
            )
            states_end += z_matrix[:, j_collocation] * sum_term
        return states_end

    def interpolate_first_derivative(self, z_matrix: cas.SX, time_control_interval: cas.SX) -> cas.SX:
        interpolated_value = 0
        for j_collocation in range(self.nb_collocation_points):
            interpolated_value += z_matrix[:, j_collocation] * self.lagrange_polynomial_derivative(
                j_collocation, time_control_interval
            )
        return interpolated_value

    def declare_dynamics_integrator(
        self,
        ocp_example: ExampleAbstract,
        discretization_method: DiscretizationAbstract,
        variables_vector: VariablesAbstract,
        noises_vector: NoisesAbstract,
    ) -> tuple[cas.Function, cas.Function, cas.Function, cas.Function]:
        """
        Formulate discrete time dynamics integration using a Radau collocation scheme.
        """
        nb_total_states = ocp_example.model.nb_states * variables_vector.nb_random

        # Create z without the first points (as it is z_sym_first)
        z_matrix = variables_vector.reshape_vector_to_matrix(
            variables_vector.get_collocation_points(0),
            (nb_total_states, self.nb_collocation_points),
        )
        states_end = self.get_states_end(z_matrix)
        dt = variables_vector.get_time() / ocp_example.n_shooting

        # State dynamics
        xdot = discretization_method.state_dynamics(
            ocp_example,
            variables_vector.get_states(0),
            variables_vector.get_controls(0),
            noises_vector.get_noise_single(0),
        )
        dynamics_func = cas.Function(
            f"dynamics",
            [variables_vector.get_states(0), variables_vector.get_controls(0), noises_vector.get_noise_single(0)],
            [xdot],
            ["x", "u", "noise"],
            ["xdot"],
        )
        # dynamics_func = dynamics_func.expand()

        # Defects
        # First collocation state = x
        first_defect = [variables_vector.get_states(0) - z_matrix[:, 0]]

        # Collocation slopes
        slope_defects = []
        for j_collocation in range(1, self.nb_collocation_points):
            # vertical_variation = self.interpolate_first_derivative(z_matrix, self.time_grid[j_collocation])
            vertical_variation = self.interpolate_first_derivative(z_matrix, self.time_grid[j_collocation])
            slope = vertical_variation / dt
            xdot = discretization_method.state_dynamics(
                ocp_example,
                z_matrix[:, j_collocation],
                variables_vector.get_controls(0),
                noises_vector.get_noise_single(0),
            )
            slope_defects += [slope - xdot]

        defects = cas.vertcat(*first_defect, *slope_defects)

        cov_integrated_vector = cas.SX()
        jacobian_funcs = None
        if discretization_method.name == "MeanAndCovariance":
            if discretization_method.with_helper_matrix:
                m_matrix = variables_vector.get_m_matrix(0)

                sigma_ww = cas.diag(noises_vector.get_noise_single(0))

                states_end = self.get_states_end(z_matrix)

                dGdx = cas.jacobian(defects, variables_vector.get_states(0))
                dGdz = cas.jacobian(defects, z_matrix)
                dGdw = cas.jacobian(defects, noises_vector.get_noise_single(0))
                dFdz = cas.jacobian(states_end, z_matrix)

                jacobian_funcs = cas.Function(
                    "jacobian_func",
                    [
                        variables_vector.get_time(),
                        variables_vector.get_states(0),
                        variables_vector.get_collocation_points(0),
                        variables_vector.get_controls(0),
                        noises_vector.get_noise_single(0),
                    ],
                    [dGdx, dGdz, dGdw, dFdz],
                )
                cov_matrix = variables_vector.get_cov_matrix(0)
                cov_integrated = m_matrix @ (dGdx @ cov_matrix @ dGdx.T + dGdw @ sigma_ww @ dGdw.T) @ m_matrix.T

                cov_integrated_vector = variables_vector.reshape_matrix_to_vector(cov_integrated)

            else:
                raise NotImplementedError(
                    "Covariance dynamics with helper matrix is the only supported method for now."
                )
                cov_dot = discretization_method.covariance_dynamics(ocp_example, x_single, u_single, noises_single)
                # cov_integrated = (cov_pre + (cov_dot_pre + cov_dot_post) / 2 * dt)
                # cov_integrated_vector = ocp_example.model.reshape_matrix_to_vector(cov_integrated)

        # Integrator
        x_next = cas.vertcat(states_end, cov_integrated_vector)
        integration_func = cas.Function(
            "F",
            [
                variables_vector.get_time(),
                variables_vector.get_states(0),
                variables_vector.get_collocation_points(0),
                variables_vector.get_cov(0),
                variables_vector.get_ms(0),
                variables_vector.get_controls(0),
                noises_vector.get_noise_single(0),
            ],
            [x_next],
        )
        # integration_func = integration_func.expand()

        # Defect function
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

        return dynamics_func, integration_func, defect_func, jacobian_funcs

    def add_other_internal_constraints(
        self,
        ocp_example: ExampleAbstract,
        discretization_method: DiscretizationAbstract,
        variables_vector: VariablesAbstract,
        noises_vector: NoisesAbstract,
        i_node: int,
        constraints: Constraints,
    ) -> None:

        nb_variables = ocp_example.model.nb_states * variables_vector.nb_random
        defects = self.defect_func(
            variables_vector.get_time(),
            variables_vector.get_states(i_node),
            variables_vector.get_collocation_points(i_node),
            variables_vector.get_controls(i_node),
            cas.DM.zeros(ocp_example.model.nb_noises * variables_vector.nb_random),
        )

        # First collocation state = x and slopes defects
        constraints.add(
            g=defects,
            lbg=[0] * (nb_variables * (self.order + 1)),
            ubg=[0] * (nb_variables * (self.order + 1)),
            g_names=[f"collocation_defect"] * nb_variables * (self.order + 1),
            node=i_node,
        )

        if discretization_method.with_helper_matrix:
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

        nb_states = ocp_example.model.nb_states
        nb_variables = ocp_example.model.nb_states * variables_vector.nb_random
        n_shooting = variables_vector.n_shooting

        # Multi-thread continuity constraint
        multi_threaded_integrator = self.integration_func.map(n_shooting, "thread", n_threads)
        x_integrated = multi_threaded_integrator(
            variables_vector.get_time(),
            cas.horzcat(*[variables_vector.get_states(i_node) for i_node in range(0, n_shooting)]),
            cas.horzcat(*[variables_vector.get_collocation_points(i_node) for i_node in range(0, n_shooting)]),
            cas.horzcat(*[variables_vector.get_cov(i_node) for i_node in range(0, n_shooting)]),
            cas.horzcat(*[variables_vector.get_ms(i_node) for i_node in range(0, n_shooting)]),
            cas.horzcat(*[variables_vector.get_controls(i_node) for i_node in range(0, n_shooting)]),
            cas.horzcat(*[noises_vector.get_one_vector_numerical(i_node) for i_node in range(0, n_shooting)]),
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
