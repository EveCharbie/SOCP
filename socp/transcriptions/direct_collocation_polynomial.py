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


class DirectCollocationPolynomial(TranscriptionAbstract):

    def __init__(self, order: int = 5) -> None:

        super().__init__()  # Does nothing
        self.order = order
        self.time_grid = cas.collocation_points(self.order, "legendre")

    @property
    def nb_collocation_points(self):
        return self.order + 2

    def initialize_dynamics_integrator(
        self,
        ocp_example: ExampleAbstract,
        discretization_method: DiscretizationAbstract,
        x: list[cas.SX.sym],
        u: list[cas.SX.sym],
        noises_single: cas.SX.sym,
    ) -> None:

        # Note: The first x and u used to declare the casadi functions, but all nodes will be used during the evaluation of the functions
        self.discretization_method = discretization_method
        dynamics_func, integration_func, defect_func, jacobian_func = self.declare_dynamics_integrator(
            ocp_example,
            discretization_method,
            x_single=x[0],
            u_single=u[0],
            noises_single=noises_single
        )
        self.dynamics_func = dynamics_func
        self.integration_func = integration_func
        self.defect_func = defect_func
        self.jacobian_func = jacobian_func

    def name(self) -> str:
        return "DirectCollocationPolynomial"

    def lagrange_polynomial_derivative(
            self,
            j_collocation: int,
            time_control_interval: cas.SX
    ) -> cas.SX:
        sum_term = 0
        for k_collocation in range(self.order):
            if k_collocation == j_collocation:
                continue

            _l = 1
            for r_collocation in range(self.order):
                if r_collocation != j_collocation and r_collocation != k_collocation:
                    _l *= (time_control_interval - self.time_grid[r_collocation]) / (
                                self.time_grid[j_collocation] - self.time_grid[r_collocation])

            partial_Ljk = _l
            sum_term += 1.0 / (self.time_grid[j_collocation] - self.time_grid[k_collocation]) * partial_Ljk

        return sum_term

    def get_states_end(self, z_matrix: cas.SX) -> cas.SX:
        states_end = 0
        for j_collocation in range(self.order):
            sum_term = self.lagrange_polynomial_derivative(
                j_collocation=j_collocation,
                time_control_interval=1.0,
            )
            states_end += z_matrix[:, j_collocation] * sum_term
        return states_end

    def get_m_matrix(
            self,
            ocp_example: ExampleAbstract,
            x_single: cas.SX,
    ) -> cas.SX:
        nb_states = ocp_example.model.nb_states
        nb_cov_variables = nb_states * nb_states
        m_matrix = None
        offset = nb_states + nb_cov_variables
        for i_collocation in range(self.nb_collocation_points):
            m_vector = x_single[offset: nb_states + offset + nb_states * nb_states]
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
        x_single: cas.SX.sym,
        z_single: cas.SX.sym,
        u_single: cas.SX.sym,
        noises_single: cas.SX.sym,
    ) -> tuple[cas.Function, cas.Function, cas.Function, cas.Function]:
        """
        Formulate discrete time dynamics integration using a Radau collocation scheme.
        """
        nb_states = ocp_example.model.nb_states
        offset = 0
        z_matrix = ocp_example.model.reshape_vector_to_matrix(
            z_single,
            (nb_states, self.nb_collocation_points),
        )
        states_end = self.get_states_end(z_matrix)
        dt = ocp_example.dt

        # State dynamics
        xdot = discretization_method.state_dynamics(ocp_example, x_single, u_single, noises_single)
        dynamics_func = cas.Function(
            f"dynamics", [x_single, u_single, noises_single], [xdot], ["x", "u", "noise"], ["xdot"]
        )
        # dynamics_func = dynamics_func.expand()
        offset += nb_states

        nb_collocation_variables = nb_states * nb_states * self.nb_collocation_points
        offset += nb_collocation_variables

        # Defects
        # First collocation state = x
        x_sym = cas.SX.sym("x_sym", nb_states)  # Need to define a new symbol for jacobian computation (x without cov)
        first_defect = [z_matrix[:, 0] - x_sym]

        # Collocation slopes
        slope_defects = []
        for j_collocation in range(1, self.order + 1):
            time_control_interval = self.time_grid[j_collocation - 1]
            vertical_variation = 0
            for k_collocation in range(self.order):
                if k_collocation == 0:
                    state_at_collocation = x_sym
                else:
                    state_at_collocation = z_matrix[:, k_collocation]
                vertical_variation += state_at_collocation * self.lagrange_polynomial_derivative(k_collocation, time_control_interval)
            slope = vertical_variation / dt
            xdot = discretization_method.state_dynamics(ocp_example, z_matrix[:, j_collocation], u_single, noises_single)
            slope_defects += [slope - xdot]

        defects = cas.vertcat(*first_defect, *slope_defects)

        cov_integrated_eval = cas.SX()
        if hasattr(discretization_method, "covariance_dynamics"):
            # Covariance dynamics
            if discretization_method.with_cholesky:
                nb_cov_variables = ocp_example.nb_cholesky_components(nb_states)
                triangular = ocp_example.model.reshape_vector_to_cholesky_matrix(x_single[offset : offset + nb_cov_variables])
                cov_matrix = triangular @ triangular.T
            else:
                nb_cov_variables = nb_states * nb_states
                cov_matrix = ocp_example.model.reshape_vector_to_matrix(
                    x_single[nb_states : nb_states + nb_cov_variables],
                    (nb_states, nb_states),
                )

            if discretization_method.with_helper_matrix:
                # If this code is validated, mode the jacobian somewhere else to avoid duplicating computation
                m_matrix = self.get_m_matrix(ocp_example, x_single)

                sigma_ww = cas.diag(noises_single)

                states_end = self.get_states_end(z_matrix)

                dGdx = cas.jacobian(defects, x_sym)
                dGdz = cas.jacobian(defects, z_single)
                dGdw = cas.jacobian(defects, noises_single)
                dFdz = cas.jacobian(states_end, z_single)

                jacobian_funcs = cas.Function(
                    "jacobian_func",
                    [x_sym, z_single, u_single, noises_single],
                    [dGdx, dGdz, dGdw, dFdz],
                )
                cov_integrated = m_matrix @ (dGdx @ cov_matrix @ dGdx.T + dGdw @ sigma_ww @ dGdw.T) @ m_matrix.T

                cov_integrated_vector = ocp_example.model.reshape_matrix_to_vector(cov_integrated)

                cov_integrated_func = cas.Function(
                    "cov_integrated",
                    [x_sym, z_single, u_single, noises_single],
                    [cov_integrated_vector],
                )
                cov_integrated_eval = cov_integrated_func(
                    x_single[:nb_states],
                    z_single,
                    u_single,
                    noises_single,
                )
            else:
                raise NotImplementedError("Covariance dynamics with helper matrix is the only supported method for now.")
                cov_dot = discretization_method.covariance_dynamics(ocp_example, x_single, u_single, noises_single)
                # cov_integrated = (cov_pre + (cov_dot_pre + cov_dot_post) / 2 * dt)
                # cov_integrated_vector = ocp_example.model.reshape_matrix_to_vector(cov_integrated)

        # Integrator
        x_next = cas.vertcat(states_end, cov_integrated_eval)
        integration_func = cas.Function(
            "F", [x_single, z_single, u_single, noises_single], [x_next], ["x_single", "u_single", "noise"], ["x_next"]
        )
        # integration_func = integration_func.expand()

        # Defect function
        defect_func = cas.Function(
            "defects",
            [x_sym, z_single, u_single, noises_single],
            [defects],
        )

        return dynamics_func, integration_func, defect_func, jacobian_funcs

    def other_internal_constraints(
            self,
            ocp_example: ExampleAbstract,
            discretization_method: DiscretizationAbstract,
            x_single: cas.SX.sym,
            z_single: cas.SX.sym,
            u_single: cas.SX.sym,
            noises_single: cas.SX.sym,
    ):
        g = []
        lbg = []
        ubg = []
        g_names = []

        nb_states = ocp_example.model.nb_states
        z_matrix = ocp_example.model.reshape_vector_to_matrix(
            z_single,
            (nb_states, self.nb_collocation_points),
        )

        defects = self.defect_func(
            x_single[:nb_states],
            z_single,
            u_single,
            noises_single,
        )

        # First collocation state = x
        g += [defects]
        lbg += [0] * nb_states * (self.order + 1)
        ubg += [0] * nb_states * (self.order + 1)
        g_names += [f"collocation_defect_{i}" for i in range(nb_states * (self.order + 1))]

        if not discretization_method.with_helper_matrix:
            # Constrain M at all collocation points to follow df_integrated/dz.T - dg_integrated/dz @ m.T = 0
            m_matrix = self.get_m_matrix(ocp_example, x_single)
            _, dGdz, _, dFdz = self.jacobian_func(
                x_single[:nb_states],
                z_single,
                u_single,
                noises_single,
            )

            g += [dFdz.T - dGdz.T @ m_matrix.T]
            lbg += [0] * dFdz.shape[1] * dFdz.shape[0]
            ubg += [0] * dFdz.shape[1] * dFdz.shape[0]
            g_names += [f"helper_matrix_defect_{i}" for i in range(dFdz.shape[1] * dFdz.shape[0])]

        return g, lbg, ubg, g_names

    def get_dynamics_constraints(
        self,
        ocp_example: ExampleAbstract,
        discretization_method: DiscretizationAbstract,
        n_shooting: int,
        x: list[cas.SX.sym],
        z: list[cas.SX.sym],
        u: list[cas.SX.sym],
        noises_single: cas.SX.sym,
        noises_numerical: np.ndarray,
        dt: float,
        n_threads: int = 8,
    ) -> tuple[list[cas.SX], list[float], list[float], list[str]]:

        nb_states = ocp_example.model.nb_states

        # Multi-thread continuity constraint
        multi_threaded_integrator = self.integration_func.map(n_shooting, "thread", n_threads)
        x_integrated = multi_threaded_integrator(
            cas.horzcat(*x[:-1]),
            cas.horzcat(*x[1:]),
            cas.horzcat(*u),
            cas.horzcat(*(u[1:] + cas.MX.zeros(ocp_example.model.nb_controls, 1))),
            cas.horzcat(*noises_numerical),
        )
        if discretization_method.with_cholesky:
            x_next = None
            for i_node in range(n_shooting):
                states_vector = x[i_node][: nb_states]
                nb_cov_variables = ocp_example.model.nb_cholesky_components(nb_states)
                triangular_matrix = ocp_example.model.reshape_vector_to_cholesky_matrix(
                    states_vector[nb_states : nb_states + nb_cov_variables],
                    (nb_states, nb_states),
                )
                cov_matrix = triangular_matrix @ triangular_matrix.T
                cov_vector = ocp_example.model.reshape_matrix_to_vector(cov_matrix)
                if x_next is None:
                    x_next = cas.vertcat(states_vector, cov_vector)
                else:
                    x_next = cas.horzcat(x_next, cas.vertcat(states_vector, cov_vector))
        else:
            x_next = cas.horzcat(*x[1:])
        g_continuity = cas.reshape(x_integrated - x_next, -1, 1)

        g = [g_continuity]
        lbg = [0] * x[0].shape[0] * n_shooting
        ubg = [0] * x[0].shape[0] * n_shooting
        g_names = [f"dynamics_continuity"] * x[0].shape[0] * n_shooting

        # Add other constraints if any
        for i_node in range(n_shooting):
            g_other, lbg_other, ubg_other, g_names_other = self.other_internal_constraints(
                ocp_example,
                discretization_method,
                x[i_node],
                z[i_node],
                u[i_node],
                noises_single,
            )

            g += g_other
            lbg += lbg_other
            ubg += ubg_other
            g_names += g_names_other

        return g, lbg, ubg, g_names
