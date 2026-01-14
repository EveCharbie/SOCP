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
        self.time_grid = [0] + cas.collocation_points(self.order, "legendre")

    @property
    def nb_collocation_points(self):
        return self.order + 2

    def initialize_dynamics_integrator(
        self,
        ocp_example: ExampleAbstract,
        discretization_method: DiscretizationAbstract,
        T: cas.SX.sym,
        x_all: list[cas.SX.sym],
        z_all: list[cas.SX.sym],
        u_all: list[cas.SX.sym],
        noises_single: cas.SX.sym,
    ) -> None:

        # Note: The first x and u used to declare the casadi functions, but all nodes will be used during the evaluation of the functions
        self.discretization_method = discretization_method
        dynamics_func, integration_func, defect_func, jacobian_funcs = self.declare_dynamics_integrator(
            ocp_example,
            discretization_method,
            T=T,
            x_single=x_all[0],
            z_single=z_all[0],
            u_single=u_all[0],
            noises_single=noises_single,
        )

        self.dynamics_func = dynamics_func
        self.integration_func = integration_func
        self.defect_func = defect_func
        self.jacobian_funcs = jacobian_funcs

    def name(self) -> str:
        return "DirectCollocationPolynomial"

    def lagrange_polynomial_derivative(self, j_collocation: int, time_control_interval: cas.SX) -> cas.SX:

        sum_term = 0
        for k_collocation in range(self.order + 1):
            if k_collocation == j_collocation:
                continue

            _l = 1
            for r_collocation in range(self.order + 1):
                if r_collocation != j_collocation and r_collocation != k_collocation:
                    _l *= (time_control_interval - self.time_grid[r_collocation]) / (
                        self.time_grid[j_collocation] - self.time_grid[r_collocation]
                    )

            partial_Ljk = _l
            sum_term += 1.0 / (self.time_grid[j_collocation] - self.time_grid[k_collocation]) * partial_Ljk

        return sum_term

    def get_states_end(self, z_matrix: cas.SX) -> cas.SX:

        states_end = 0
        for j_collocation in range(self.order + 1):
            sum_term = self.lagrange_polynomial_derivative(
                j_collocation=j_collocation,
                time_control_interval=1.0,
            )
            states_end += z_matrix[:, j_collocation] * sum_term
        return states_end

    def get_m_matrix(
        self,
        ocp_example: ExampleAbstract,
        m_sym: cas.SX,
    ) -> cas.SX:

        nb_states = ocp_example.model.nb_states
        m_matrix = None
        offset = 0
        for i_collocation in range(self.nb_collocation_points - 1):
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
        T: cas.SX.sym,
        x_single: cas.SX.sym,
        z_single: cas.SX.sym,
        u_single: cas.SX.sym,
        noises_single: cas.SX.sym,
    ) -> tuple[cas.Function, cas.Function, cas.Function, cas.Function]:
        """
        Formulate discrete time dynamics integration using a Radau collocation scheme.
        """
        nb_states = ocp_example.model.nb_states
        if discretization_method.with_cholesky:
            nb_cov_variables = ocp_example.nb_cholesky_components(nb_states)
        else:
            nb_cov_variables = nb_states * nb_states
        offset = 0
        # Need to define a new symbol for jacobian computation (x without cov)
        x_sym = cas.SX.sym("x_sym", nb_states)
        z_sym_first = cas.SX.sym("z_sym_first", nb_states)
        z_sym_middle = cas.SX.sym(
            "z_sym_middle", nb_states * (self.nb_collocation_points - 1)
        )  # The furst points is excluded
        cov_sym = cas.SX.sym("cov_sym", nb_cov_variables)
        m_sym = cas.SX.sym(
            "m_sym", nb_states * nb_states * (self.nb_collocation_points - 1)
        )  # The last point is excluded

        # Create z without the first points (as it is z_sym_first)
        z_matrix_middle = ocp_example.model.reshape_vector_to_matrix(
            z_sym_middle,
            (nb_states, (self.nb_collocation_points - 1)),
        )
        states_end = self.get_states_end(cas.horzcat(z_sym_first, z_matrix_middle))
        dt = T / ocp_example.n_shooting

        # State dynamics
        xdot = discretization_method.state_dynamics(ocp_example, x_sym, u_single, noises_single)
        dynamics_tempo_func = cas.Function(
            f"dynamics", [x_sym, u_single, noises_single], [xdot], ["x_sym", "u", "noise"], ["xdot"]
        )
        dynamics_tempo_func_eval = dynamics_tempo_func(
            x_single[:nb_states],
            u_single,
            noises_single,
        )
        dynamics_func = cas.Function(
            f"dynamics", [x_single, u_single, noises_single], [dynamics_tempo_func_eval], ["x", "u", "noise"], ["xdot"]
        )
        # dynamics_func = dynamics_func.expand()
        offset += nb_states

        nb_collocation_variables = nb_states * nb_states * self.nb_collocation_points
        offset += nb_collocation_variables

        # Defects
        # First collocation state = x
        first_defect = [z_sym_first - x_sym]

        # Collocation slopes
        slope_defects = []
        for j_collocation in range(1, self.order + 1):
            time_control_interval = self.time_grid[j_collocation]
            vertical_variation = 0
            for k_collocation in range(self.order + 1):
                if k_collocation == 0:
                    state_at_collocation = x_sym
                else:
                    state_at_collocation = z_matrix_middle[:, k_collocation-1]
                vertical_variation += state_at_collocation * self.lagrange_polynomial_derivative(
                    k_collocation, time_control_interval
                )
            slope = vertical_variation / dt
            if j_collocation == 0:
                xdot = discretization_method.state_dynamics(ocp_example, z_sym_first, u_single, noises_single)
            else:
                xdot = discretization_method.state_dynamics(
                    ocp_example, z_matrix_middle[:, j_collocation - 1], u_single, noises_single
                )
            slope_defects += [slope - xdot]

        defects = cas.vertcat(*first_defect, *slope_defects)

        cov_integrated_vector = cas.SX()
        if hasattr(discretization_method, "covariance_dynamics"):
            # Covariance dynamics
            if discretization_method.with_cholesky:
                triangular = ocp_example.model.reshape_vector_to_cholesky_matrix(cov_sym, (nb_states, nb_states))
                cov_matrix = triangular @ triangular.T
                # x_single[offset: offset + nb_cov_variables]
            else:
                cov_matrix = ocp_example.model.reshape_vector_to_matrix(
                    cov_sym,
                    (nb_states, nb_states),
                )

            if discretization_method.with_helper_matrix:
                # If this code is validated, mode the jacobian somewhere else to avoid duplicating computation
                m_matrix = self.get_m_matrix(ocp_example, m_sym)

                sigma_ww = cas.diag(noises_single)

                states_end = self.get_states_end(cas.horzcat(z_sym_first, z_matrix_middle))

                dGdx = cas.jacobian(defects, x_sym)
                dGdz = cas.jacobian(defects, z_sym_middle)
                dGdw = cas.jacobian(defects, noises_single)
                dFdz = cas.jacobian(states_end, z_sym_middle)

                jacobian_tempo_funcs = cas.Function(
                    "jacobian_func",
                    [T, x_sym, z_sym_first, z_sym_middle, u_single, noises_single],
                    [dGdx, dGdz, dGdw, dFdz],
                )
                jacobian_tempo_funcs_evalueated = jacobian_tempo_funcs(
                    T,  # T
                    x_single[:nb_states],  # x_sym
                    z_single[:nb_states],  # z_sym_first
                    z_single[nb_states : nb_states + nb_states * (self.nb_collocation_points - 1)],  # z_sym_middle
                    u_single,
                    noises_single,
                )
                dGdx_evaluated = jacobian_tempo_funcs_evalueated[0]
                dGdz_evaluated = jacobian_tempo_funcs_evalueated[1]
                dGdw_evaluated = jacobian_tempo_funcs_evalueated[2]
                dFdz_evaluated = jacobian_tempo_funcs_evalueated[3]
                jacobian_funcs = cas.Function(
                    "jacobian_func",
                    [T, x_single, z_single, u_single, noises_single],
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
            [T, x_sym, z_sym_first, z_sym_middle, cov_sym, m_sym, u_single, noises_single],
            [x_next],
            ["T", "x_sym", "z_sym_first", "z_sym_middle", "cov_sym", "m_sym", "u_single", "noise"],
            ["x_next"],
        )
        integration_tempo_func_eval = integration_tempo_func(
            T,  # T
            x_single[:nb_states],  # x_sym
            z_single[:nb_states],  # z_sym_first
            z_single[nb_states : nb_states + nb_states * (self.nb_collocation_points - 1)],  # z_sym_middle
            x_single[nb_states : nb_states + nb_cov_variables],  # cov_sym
            x_single[
                nb_states
                + nb_cov_variables : nb_states
                + nb_cov_variables
                + nb_states * nb_states * (self.nb_collocation_points - 1)
            ],  # m_sym
            u_single,
            noises_single,
        )
        integration_func = cas.Function(
            "F",
            [T, x_single, z_single, u_single, noises_single],
            [integration_tempo_func_eval],
            ["T", "x_single", "z_single", "u_single", "noise"],
            ["x_next"],
        )
        # integration_func = integration_func.expand()

        # Defect function
        defect_tempo_func = cas.Function(
            "defects",
            [T, x_sym, z_sym_first, z_sym_middle, u_single, noises_single],
            [defects],
        )
        defects_tempo_func_eval = defect_tempo_func(
            T,
            x_single[:nb_states],
            z_single[:nb_states],
            z_single[nb_states : nb_states + nb_states * (self.nb_collocation_points - 1)],
            u_single,
            noises_single,
        )
        defect_func = cas.Function(
            "defects",
            [T, x_single, z_single, u_single, noises_single],
            [defects_tempo_func_eval],
        )
        # defect_func = defect_func.expand()

        return dynamics_func, integration_func, defect_func, jacobian_funcs

    def other_internal_constraints(
        self,
        ocp_example: ExampleAbstract,
        discretization_method: DiscretizationAbstract,
        T: cas.SX.sym,
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
        defects = self.defect_func(
            T,
            x_single,
            z_single,
            u_single,
            noises_single,
        )

        # First collocation state = x and slopes defects
        g += [defects]
        lbg += [0] * (nb_states * (self.order + 1))
        ubg += [0] * (nb_states * (self.order + 1))
        g_names += [f"collocation_defect_{i}" for i in range(nb_states * (self.order + 1))]

        if not discretization_method.with_helper_matrix:
            # Constrain M at all collocation points to follow df_integrated/dz.T - dg_integrated/dz @ m.T = 0
            m_matrix = self.get_m_matrix(ocp_example, x_single)
            _, dGdz, _, dFdz = self.jacobian_funcs(
                T,
                x_single,
                z_single,
                u_single,
                noises_single,
            )

            g += [dFdz.T - dGdz.T @ m_matrix.T]  # Bioptim version
            # g += [dFdz.T - m_matrix.T @ dGdz.T]  # Paper version
            lbg += [0] * (dFdz.shape[1] * dFdz.shape[0])
            ubg += [0] * (dFdz.shape[1] * dFdz.shape[0])
            g_names += [f"helper_matrix_defect_{i}" for i in range(dFdz.shape[1] * dFdz.shape[0])]

        return g, lbg, ubg, g_names

    def get_dynamics_constraints(
        self,
        ocp_example: ExampleAbstract,
        discretization_method: DiscretizationAbstract,
        n_shooting: int,
        T: cas.SX.sym,
        x: list[cas.SX.sym],
        z: list[cas.SX.sym],
        u: list[cas.SX.sym],
        noises_single: cas.SX.sym,
        noises_numerical: np.ndarray,
        n_threads: int = 8,
    ) -> tuple[list[cas.SX], list[float], list[float], list[str]]:

        nb_states = ocp_example.model.nb_states

        # Multi-thread continuity constraint
        multi_threaded_integrator = self.integration_func.map(n_shooting, "thread", n_threads)
        x_integrated = multi_threaded_integrator(
            T,
            cas.horzcat(*x[:-1]),
            cas.horzcat(*z[:-1]),
            cas.horzcat(*u[:-1]),
            cas.horzcat(*noises_numerical),
        )
        if discretization_method.with_cholesky:
            nb_cov_variables = ocp_example.model.nb_cholesky_components(nb_states)
            x_next = None
            for i_node in range(n_shooting):
                states_vector = x[i_node + 1][:nb_states]
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
            nb_cov_variables = nb_states * nb_states
            x_next = cas.horzcat(*x[1:])

        if discretization_method.with_helper_matrix:
            g_continuity = cas.reshape(
                x_integrated[: nb_states + nb_cov_variables, :] - x_next[: nb_states + nb_cov_variables, :], -1, 1
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
                T,
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
