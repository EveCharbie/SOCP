import casadi as cas
import numpy as np

from .discretization_abstract import DiscretizationAbstract
from .transcription_abstract import TranscriptionAbstract
from ..models.model_abstract import ModelAbstract
from ..examples.example_abstract import ExampleAbstract
from ..constraints import Constraints


class DirectCollocationTrapezoidal(TranscriptionAbstract):

    def __init__(self) -> None:

        super().__init__()  # Does nothing

    def initialize_dynamics_integrator(
        self,
        ocp_example: ExampleAbstract,
        discretization_method: DiscretizationAbstract,
        T: cas.SX,
        x_all: list[cas.SX.sym],
        z_all: list[cas.SX.sym],
        u_all: list[cas.SX.sym],
        noises_single: cas.SX.sym,
    ) -> None:

        # Note: The first and second x and u used to declare the casadi functions, but all nodes will be used during the evaluation of the functions
        self.discretization_method = discretization_method
        dynamics_func, integration_func = self.declare_dynamics_integrator(
            ocp_example,
            discretization_method,
            x_pre=x_all[0],
            x_post=x_all[1],
            u_pre=u_all[0],
            u_post=u_all[1],
            noises_single=noises_single,
        )
        self.dynamics_func = dynamics_func
        self.integration_func = integration_func

    @property
    def name(self) -> str:
        return "DirectCollocationTrapezoidal"

    @property
    def nb_collocation_points(self):
        return 1

    def declare_dynamics_integrator(
        self,
        ocp_example,
        discretization_method,
        T: cas.SX.sym,
        x_pre: cas.SX.sym,
        x_post: cas.SX.sym,
        u_pre: cas.SX.sym,
        u_post: cas.SX.sym,
        noises_single: cas.SX.sym,
    ) -> tuple[cas.Function, cas.Function]:
        """
        Formulate discrete time dynamics integration using a trapezoidal collocation scheme.
        """
        nb_states = ocp_example.model.nb_states
        dt = T / ocp_example.n_shooting

        # State dynamics
        xdot_pre = discretization_method.state_dynamics(ocp_example, x_pre, u_pre, noises_single)
        xdot_post = discretization_method.state_dynamics(ocp_example, x_post, u_post, noises_single)
        dynamics_func = cas.Function(
            f"dynamics", [x_pre, u_pre, noises_single], [xdot_pre], ["x", "u", "noise"], ["xdot"]
        )
        # dynamics_func = dynamics_func.expand()

        cov_integrated_vector = cas.SX()
        if hasattr(discretization_method, "covariance_dynamics"):
            # Covariance dynamics
            if discretization_method.with_cholesky:
                nb_cov_variables = ocp_example.nb_cholesky_components(nb_states)
                triangular_pre = ocp_example.model.reshape_vector_to_cholesky_matrix(
                    x_pre[nb_states : nb_states + nb_cov_variables]
                )
                cov_pre = triangular_pre @ triangular_pre.T
            else:
                nb_cov_variables = nb_states * nb_states
                cov_pre = ocp_example.model.reshape_vector_to_matrix(
                    x_pre[nb_states : nb_states + nb_cov_variables],
                    (nb_states, nb_states),
                )

            if self.discretization_method.with_helper_matrix:
                m_matrix, df_dx, df_dw, sigma_w = discretization_method.covariance_dynamics(
                    ocp_example, x_pre, u_pre, noises_single
                )

                # Trapezoidal integration of the covariance dynamics with helper matrix
                dg_dx = -(df_dx * dt / 2 + cas.SX.eye(df_dx.shape))
                dg_dw = -(df_dw * dt)

                cov_integrated = m_matrix @ (dg_dx @ cov_pre @ dg_dx.T + dg_dw @ sigma_w @ dg_dw.T) * m_matrix.T
                cov_integrated_vector = ocp_example.model.reshape_matrix_to_vector(cov_integrated)
            else:
                cov_dot_pre = discretization_method.covariance_dynamics(ocp_example, x_pre, u_pre, noises_single)
                cov_dot_post = discretization_method.covariance_dynamics(ocp_example, x_post, u_post, noises_single)
                cov_integrated = cov_pre + (cov_dot_pre + cov_dot_post) / 2 * dt
                cov_integrated_vector = ocp_example.model.reshape_matrix_to_vector(cov_integrated)

        # Integrator
        states_integrated = x_pre + (xdot_pre + xdot_post) / 2 * dt
        x_next = cas.vertcat(states_integrated, cov_integrated_vector)
        integration_func = cas.Function(
            "F",
            [x_pre, x_post, u_pre, u_post, noises_single],
            [x_next],
            ["x_pre", "x_post", "u_pre", "u_post", "noise"],
            ["x_next"],
        )
        # integration_func = integration_func.expand()
        return dynamics_func, integration_func

    def set_dynamics_constraints(
        self,
        ocp_example: ExampleAbstract,
        discretization_method: DiscretizationAbstract,
        n_shooting: int,
        T: cas.SX.sym,
        x_all: list[cas.SX.sym],
        z_all: list[cas.SX.sym],
        u_all: list[cas.SX.sym],
        noises_single: cas.SX.sym,
        noises_numerical: np.ndarray,
        constraints: Constraints,
        n_threads: int = 8,
    ) -> tuple[list[cas.SX], list[float], list[float], list[str]]:

        nb_states = ocp_example.nb_states

        # Multi-thread continuity constraint
        multi_threaded_integrator = self.integration_func.map(n_shooting, "thread", n_threads)
        x_integrated = multi_threaded_integrator(
            cas.horzcat(*x_all[:-1]),
            cas.horzcat(*x_all[1:]),
            cas.horzcat(*u_all[:-1]),
            cas.horzcat(*(u_all[1:])),
            cas.horzcat(*noises_numerical),
        )
        if discretization_method.with_cholesky:
            x_next = None
            for i_node in range(n_shooting):
                states_vector = x_all[i_node][:nb_states]
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
            x_next = cas.horzcat(*x_all[1:])
        g_continuity = cas.reshape(x_integrated - x_next, -1, 1)

        g = [g_continuity]
        lbg = [0] * x_all[0].shape[0] * n_shooting
        ubg = [0] * x_all[0].shape[0] * n_shooting
        g_names = [f"dynamics_continuity"] * x_all[0].shape[0] * n_shooting

        # Add other constraints if any
        for i_node in range(n_shooting):
            self.add_other_internal_constraints(
                ocp_example,
                discretization_method,
                variables_vector,
                noises_single,
                i_node,
                constraints,
            )

        return g, lbg, ubg, g_names
