import casadi as cas
import numpy as np

from .transcription_abstract import TranscriptionAbstract
from ..models.model_abstract import ModelAbstract


class DirectMultipleShooting(TranscriptionAbstract):

    def name(self) -> str:
        return "DirectMultipleShooting"

    @staticmethod
    def declare_dynamics_integrator(
        model,
        x_single: cas.MX.sym,
        u_single: cas.MX.sym,
        noises_single: cas.MX.sym,
        dt: float,
    ):
        """
        Formulate discrete time dynamics integration using a fixed step Runge-Kutta 4 integrator.
        """

        n_steps = 5  # RK4 steps per interval
        h = dt / n_steps

        # Dynamics
        xdot = model.dynamics(x_single, u_single, noises_single)
        dynamics_func = cas.Function(
            f"dynamics", [x_single, u_single, noises_single], [xdot], ["x", "u", "noise"], ["xdot"]
        )
        # dynamics_func = dynamics_func.expand()

        # Integrator
        x_next = x_single[:]
        for j in range(n_steps):
            k1 = dynamics_func(x_next, u_single, noises_single)
            k2 = dynamics_func(x_next + h / 2 * k1, u_single, noises_single)
            k3 = dynamics_func(x_next + h / 2 * k2, u_single, noises_single)
            k4 = dynamics_func(x_next + h * k3, u_single, noises_single)
            x_next += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        integration_func = cas.Function(
            "F", [x_single, u_single, noises_single], [x_next], ["x", "u", "noise"], ["x_next"]
        )
        # integration_func = integration_func.expand()
        return dynamics_func, integration_func

    def get_dynamics_constraints(
        self,
        model: ModelAbstract,
        n_shooting: int,
        x: list[cas.MX.sym],
        u: list[cas.MX.sym],
        noises_single: cas.MX.sym,
        noises_numerical: np.ndarray,
        dt: float,
        n_threads: int = 8,
    ) -> tuple[list[cas.MX], list[float], list[float], list[str]]:

        n_random = model.n_random

        # Note: The first x and u used to declare the casadi functions, but all nodes will be used during the evaluation of the functions
        dynamics_func, integration_func = self.declare_dynamics_integrator(
            model, x_single=x[0], u_single=u[0], noises_single=noises_single, dt=dt
        )

        # Multi-thread continuity constraint
        multi_threaded_integrator = integration_func.map(n_shooting, "thread", n_threads)
        x_integrated = multi_threaded_integrator(cas.horzcat(*x[:-1]), cas.horzcat(*u), cas.horzcat(*noises_numerical))
        g_continuity = cas.reshape(x_integrated - cas.horzcat(*x[1:]), -1, 1)

        g = [g_continuity]
        lbg = [0] * ((model.nb_states * n_random) * n_shooting)
        ubg = [0] * ((model.nb_states * n_random) * n_shooting)
        g_names = [f"dynamics_continuity"] * ((model.nb_states * n_random) * n_shooting)

        return g, lbg, ubg, g_names
