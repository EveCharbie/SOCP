"""
Legendre was used as in Gillis et al. 2013, but Radau polynomials might be preferred because of their better robustness
for collocation methods. However, it might be less accurate for the same number of collocation points.
"""

import casadi as cas
import numpy as np

from .discretization_abstract import DiscretizationAbstract
from .transcription_abstract import TranscriptionAbstract
from ..models.model_abstract import ModelAbstract
from ..examples.example_abstract import ExampleAbstract
from .variables_abstract import VariablesAbstract


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
        noises_single: cas.SX.sym,
    ) -> None:

        # Note: The first x and u used to declare the casadi functions, but all nodes will be used during the evaluation of the functions
        self.discretization_method = discretization_method
        dynamics_func, integration_func, defect_func, jacobian_funcs = self.declare_dynamics_integrator(
            ocp_example,
            discretization_method,
            variables_vector,
            noises_single=noises_single,
        )

        self.dynamics_func = dynamics_func
        self.integration_func = integration_func
        self.defect_func = defect_func
        self.jacobian_funcs = jacobian_funcs

    def name(self) -> str:
        return "DirectCollocationPolynomial"

    def partial_lagrange_polynomial(self, j_collocation: int, time_control_interval: cas.SX, i_collocation: int) -> cas.SX:
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

    def interpolate_first_derivative(self,  z_matrix: cas.SX, time_control_interval: cas.SX) -> cas.SX:
        interpolated_value = cas.SX.zeros(z_matrix.shape[0])
        for j_collocation in range(self.nb_collocation_points):
            interpolated_value += z_matrix[:, j_collocation] * self.lagrange_polynomial_derivative(j_collocation, time_control_interval)
        return interpolated_value

    def get_m_matrix(
        self,
        ocp_example: ExampleAbstract,
        m_sym: cas.SX,
    ) -> cas.SX:

        nb_states = ocp_example.model.nb_states
        m_matrix = None
        offset = 0
        for i_collocation in range(self.nb_collocation_points):
            m_vector = m_sym[offset : offset + nb_states * nb_states]
            m_matrix_i = ocp_example.model.reshape_vector_to_matrix(
                m_vector,
                (nb_states, nb_states),
            )
            if m_matrix is None:
                m_matrix = m_matrix_i
            else:
                m_matrix = cas.horzcat(m_matrix, m_matrix_i)
            offset += nb_states * nb_states
        return m_matrix

    def declare_dynamics_integrator(
        self,
        ocp_example,
        discretization_method,
        variables_vector: VariablesAbstract,
        noises_single: cas.SX.sym,
    ) -> tuple[cas.Function, cas.Function, cas.Function, cas.Function]:
        """
        Formulate discrete time dynamics integration using a Radau collocation scheme.
        """
        # TODO: remove all the chenanigans with tempo functions now that variables_vector is fixed !!!!!!!!!!!!!!!!!!!!!

        nb_states = ocp_example.model.nb_states
        if discretization_method.with_cholesky:
            nb_cov_variables = ocp_example.nb_cholesky_components(nb_states)
        else:
            nb_cov_variables = nb_states * nb_states

        x_sym = cas.SX.sym("x_sym", nb_states)
        z_sym = cas.SX.sym("z_sym", nb_states * self.nb_collocation_points)
        cov_sym = cas.SX.sym("cov_sym", nb_cov_variables)
        m_sym = cas.SX.sym("m_sym", nb_states * nb_states * self.nb_collocation_points)

        # Create z without the first points (as it is z_sym_first)
        z_matrix_middle = ocp_example.model.reshape_vector_to_matrix(
            z_sym,
            (nb_states, self.nb_collocation_points),
        )
        states_end = self.get_states_end(z_matrix_middle)
        dt = variables_vector.get_time() / ocp_example.n_shooting

        # State dynamics
        xdot = discretization_method.state_dynamics(ocp_example, x_sym, variables_vector.get_controls(0), noises_single)
        dynamics_tempo_func = cas.Function(
            f"dynamics", [x_sym, variables_vector.get_controls(0), noises_single], [xdot], ["x_sym", "u", "noise"], ["xdot"]
        )
        dynamics_tempo_func_eval = dynamics_tempo_func(
            variables_vector.get_states(0),
            variables_vector.get_controls(0),
            noises_single,
        )
        dynamics_func = cas.Function(
            f"dynamics", [
                variables_vector.get_states(0),
                variables_vector.get_controls(0),
                noises_single
            ],
            [dynamics_tempo_func_eval],
            ["x", "u", "noise"],
            ["xdot"]
        )
        # dynamics_func = dynamics_func.expand()

        # Defects
        # First collocation state = x
        first_defect = [x_sym - z_matrix_middle[:, 0]]

        # Collocation slopes
        slope_defects = []
        for j_collocation in range(1, self.nb_collocation_points):
            vertical_variation = self.interpolate_first_derivative(
                z_matrix_middle, self.time_grid[j_collocation]
            )
            slope = vertical_variation / dt
            xdot = discretization_method.state_dynamics(
                ocp_example, z_matrix_middle[:, j_collocation], variables_vector.get_controls(0), noises_single
            )
            slope_defects += [slope - xdot]

        defects = cas.vertcat(*first_defect, *slope_defects)

        cov_integrated_vector = cas.SX()
        if hasattr(discretization_method, "covariance_dynamics"):
            # Covariance dynamics
            if discretization_method.with_cholesky:
                triangular = ocp_example.model.reshape_vector_to_cholesky_matrix(cov_sym, (nb_states, nb_states))
                cov_matrix = triangular @ triangular.T
            else:
                cov_matrix = ocp_example.model.reshape_vector_to_matrix(
                    cov_sym,
                    (nb_states, nb_states),
                )

            if discretization_method.with_helper_matrix:
                m_matrix = self.get_m_matrix(ocp_example, m_sym)

                sigma_ww = cas.diag(noises_single)

                states_end = self.get_states_end(z_matrix_middle)

                dGdx = cas.jacobian(defects, x_sym)
                dGdz = cas.jacobian(defects, z_sym)
                dGdw = cas.jacobian(defects, noises_single)
                dFdz = cas.jacobian(states_end, z_sym)

                jacobian_tempo_funcs = cas.Function(
                    "jacobian_func",
                    [variables_vector.get_time(), x_sym, z_sym, variables_vector.get_controls(0), noises_single],
                    [dGdx, dGdz, dGdw, dFdz],
                )
                jacobian_tempo_funcs_evalueated = jacobian_tempo_funcs(
                    variables_vector.get_time(),
                    variables_vector.get_states(0),
                    variables_vector.get_collocation_points(0),
                    variables_vector.get_controls(0),
                    cas.DM.zeros(ocp_example.model.nb_noises),
                )
                dGdx_evaluated = jacobian_tempo_funcs_evalueated[0]
                dGdz_evaluated = jacobian_tempo_funcs_evalueated[1]
                dGdw_evaluated = jacobian_tempo_funcs_evalueated[2]
                dFdz_evaluated = jacobian_tempo_funcs_evalueated[3]
                jacobian_funcs = cas.Function(
                    "jacobian_func",
                    [
                        variables_vector.get_time(),
                        variables_vector.get_states(0),
                        variables_vector.get_collocation_points(0),
                        variables_vector.get_controls(0),
                        noises_single
                    ],
                    [dGdx_evaluated, dGdz_evaluated, dGdw_evaluated, dFdz_evaluated],
                )
                cov_integrated = m_matrix @ (dGdx @ cov_matrix @ dGdx.T + dGdw @ sigma_ww @ dGdw.T) @ m_matrix.T

                cov_integrated_vector = ocp_example.model.reshape_matrix_to_vector(cov_integrated)

            else:
                raise NotImplementedError(
                    "Covariance dynamics with helper matrix is the only supported method for now."
                )
                cov_dot = discretization_method.covariance_dynamics(ocp_example, x_single, u_single, noises_single)
                # cov_integrated = (cov_pre + (cov_dot_pre + cov_dot_post) / 2 * dt)
                # cov_integrated_vector = ocp_example.model.reshape_matrix_to_vector(cov_integrated)

        # Integrator
        x_next = cas.vertcat(states_end, cov_integrated_vector)
        integration_tempo_func = cas.Function(
            "F",
            [variables_vector.get_time(), x_sym, z_sym, cov_sym, m_sym, variables_vector.get_controls(0), noises_single],
            [x_next],
            ["T", "x_sym", "z_sym", "cov_sym", "m_sym", "u_single", "noise"],
            ["x_next"],
        )
        integration_tempo_func_eval = integration_tempo_func(
            variables_vector.get_time(),
            variables_vector.get_states(0),
            variables_vector.get_collocation_points(0),
            variables_vector.get_cov(0),
            variables_vector.get_ms(0),
            variables_vector.get_controls(0),
            noises_single,
        )
        integration_func = cas.Function(
            "F",
            [
                variables_vector.get_time(),
                variables_vector.get_states(0),
                variables_vector.get_collocation_points(0),
                variables_vector.get_cov(0),
                variables_vector.get_ms(0),
                variables_vector.get_controls(0),
                noises_single,
            ],
            [integration_tempo_func_eval],
        )
        # integration_func = integration_func.expand()

        # Defect function
        defect_tempo_func = cas.Function(
            "defects",
            [
                variables_vector.get_time(),
                x_sym,
                z_sym,
                variables_vector.get_controls(0),
                noises_single,
            ],
            [defects],
        )
        defects_tempo_func_eval = defect_tempo_func(
            variables_vector.get_time(),
            variables_vector.get_states(0),
            variables_vector.get_collocation_points(0),
            variables_vector.get_controls(0),
            noises_single,
        )
        defect_func = cas.Function(
            "defects",
            [
                variables_vector.get_time(),
                variables_vector.get_states(0),
                variables_vector.get_collocation_points(0),
                variables_vector.get_controls(0),
                noises_single,
            ],
            [defects_tempo_func_eval],
        )
        # defect_func = defect_func.expand()

        return dynamics_func, integration_func, defect_func, jacobian_funcs

    def other_internal_constraints(
        self,
        ocp_example: ExampleAbstract,
        discretization_method: DiscretizationAbstract,
        variables_vector: VariablesAbstract,
        noises_single: cas.SX.sym,
    ):
        g = []
        lbg = []
        ubg = []
        g_names = []

        nb_states = ocp_example.model.nb_states
        defects = self.defect_func(
            variables_vector.get_time(),
            variables_vector.get_states(0),
            variables_vector.get_collocation_points(0),
            variables_vector.get_controls(0),
            cas.DM.zeros(ocp_example.model.nb_noises),
        )

        # First collocation state = x and slopes defects
        g += [defects]
        lbg += [0] * (nb_states * (self.order + 1))
        ubg += [0] * (nb_states * (self.order + 1))
        g_names += [f"collocation_defect"] * nb_states * (self.order + 1)

        if discretization_method.with_helper_matrix:
            # Constrain M at all collocation points to follow df_integrated/dz.T - dg_integrated/dz @ m.T = 0
            m_matrix = self.get_m_matrix(ocp_example, variables_vector.get_ms(0))
            _, dGdz, _, dFdz = self.jacobian_funcs(
                variables_vector.get_time(),
                variables_vector.get_states(0),
                variables_vector.get_collocation_points(0),
                variables_vector.get_controls(0),
                cas.DM.zeros(ocp_example.model.nb_noises),
            )

            constraint = dFdz.T - dGdz.T @ m_matrix.T
            g += [ocp_example.model.reshape_matrix_to_vector(constraint)]
            lbg += [0] * (dFdz.shape[1] * dFdz.shape[0])
            ubg += [0] * (dFdz.shape[1] * dFdz.shape[0])
            g_names += [f"helper_matrix_defect"] * (dFdz.shape[1] * dFdz.shape[0])


        # # Semi-definite constraint on the covariance matrix (Sylvester's criterion)
        # if not discretization_method.with_cholesky:
        #     cov_matrix = ocp_example.model.reshape_vector_to_matrix(
        #         x_single[nb_states : nb_states + nb_states * nb_states],
        #         (nb_states, nb_states),
        #     )
        #     epsilon = 1e-6
        #     for k in range(1, nb_states + 1):
        #         minor = cas.det(cov_matrix[:k, :k])
        #         g += [minor]
        #         lbg += [epsilon]
        #         ubg += [cas.inf]
        #         g_names += ["covariance_positive_definite_minor_" + str(k)]


        return g, lbg, ubg, g_names

    def get_dynamics_constraints(
        self,
        ocp_example: ExampleAbstract,
        discretization_method: DiscretizationAbstract,
        n_shooting: int,
        variables_vector: VariablesAbstract,
        noises_single: cas.SX.sym,
        noises_numerical: np.ndarray,
        n_threads: int = 8,
    ) -> tuple[list[cas.SX], list[float], list[float], list[str]]:

        nb_states = ocp_example.model.nb_states

        # Multi-thread continuity constraint
        multi_threaded_integrator = self.integration_func.map(n_shooting, "thread", n_threads)
        x_integrated = multi_threaded_integrator(
            variables_vector.get_time(),
            cas.horzcat(*[variables_vector.get_states(i_node) for i_node in range(0, variables_vector.n_shooting)]),
            cas.horzcat(*[variables_vector.get_collocation_points(i_node) for i_node in range(0, variables_vector.n_shooting)]),
            cas.horzcat(*[variables_vector.get_cov(i_node) for i_node in range(0, variables_vector.n_shooting)]),
            cas.horzcat(*[variables_vector.get_ms(i_node) for i_node in range(0, variables_vector.n_shooting)]),
            cas.horzcat(*[variables_vector.get_controls(i_node) for i_node in range(0, variables_vector.n_shooting)]),
            cas.horzcat(*noises_numerical),
        )

        if discretization_method.with_cholesky:
            nb_cov_variables = ocp_example.model.nb_cholesky_components(nb_states)
            x_next = None
            for i_node in range(n_shooting):
                states_next_vector = variables_vector.get_states(i_node+1)
                cov_vector = variables_vector.get_cov(i_node+1)
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
            states_next = cas.horzcat(*[variables_vector.get_states(i_node) for i_node in range(1, variables_vector.n_shooting + 1)])
            cov_next = cas.horzcat(*[variables_vector.get_cov(i_node) for i_node in range(1, variables_vector.n_shooting + 1)])
            x_next = cas.vertcat(states_next, cov_next)

        if discretization_method.with_helper_matrix:
            g_continuity = cas.reshape(
                x_integrated[: nb_states + nb_cov_variables, :] - x_next[: nb_states + nb_cov_variables, :],
                (-1, 1),
            )
        else:
            g_continuity = cas.reshape(x_integrated - x_next, -1, 1)

        g = [g_continuity]
        lbg = [0] * g_continuity.shape[0]
        ubg = [0] * g_continuity.shape[0]
        g_names = [f"dynamics_continuity"] * g_continuity.shape[0]

        # Add other constraints if any
        for i_node in range(n_shooting):
            g_other, lbg_other, ubg_other, g_names_other = self.other_internal_constraints(
                ocp_example,
                discretization_method,
                variables_vector,
                noises_single,
            )

            g += g_other
            lbg += lbg_other
            ubg += ubg_other
            g_names += g_names_other

        return g, lbg, ubg, g_names
