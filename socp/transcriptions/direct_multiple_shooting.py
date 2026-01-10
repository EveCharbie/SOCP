import casadi as cas
import numpy as np

from .discretization_abstract import DiscretizationAbstract
from .transcription_abstract import TranscriptionAbstract
from ..models.model_abstract import ModelAbstract
from ..examples.example_abstract import ExampleAbstract


class DirectMultipleShooting(TranscriptionAbstract):

    def __init__(self) -> None:

        super().__init__() # Does nothing

    def initialize_dynamics_integrator(
        self,
        ocp_example: ExampleAbstract,
        discretization_method: DiscretizationAbstract,
        x: list[cas.MX.sym],
        u: list[cas.MX.sym],
        noises_single: cas.MX.sym,
    ) -> None:

        # Note: The first x and u used to declare the casadi functions, but all nodes will be used during the evaluation of the functions
        dynamics_func, integration_func = self.declare_dynamics_integrator(
            ocp_example, discretization_method, x_single=x[0], u_single=u[0], noises_single=noises_single
        )
        self.dynamics_func = dynamics_func
        self.integration_func = integration_func

    def name(self) -> str:
        return "DirectMultipleShooting"

    @staticmethod
    def declare_dynamics_integrator(
        ocp_example,
        discretization,
        x_single: cas.MX.sym,
        u_single: cas.MX.sym,
        noises_single: cas.MX.sym,
    ) -> tuple[cas.Function, cas.Function]:
        """
        Formulate discrete time dynamics integration using a fixed step Runge-Kutta 4 integrator.
        """

        n_steps = 5  # RK4 steps per interval
        h = ocp_example.dt / n_steps

        # Dynamics
        xdot = discretization.state_dynamics(ocp_example, x_single, u_single, noises_single)
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
        discretization_method: DiscretizationAbstract,
        n_shooting: int,
        x: list[cas.MX.sym],
        u: list[cas.MX.sym],
        noises_single: cas.MX.sym,
        noises_numerical: np.ndarray,
        dt: float,
        n_threads: int = 8,
    ) -> tuple[list[cas.MX], list[float], list[float], list[str]]:

        # Multi-thread continuity constraint
        multi_threaded_integrator = self.integration_func.map(n_shooting, "thread", n_threads)
        x_integrated = multi_threaded_integrator(cas.horzcat(*x[:-1]), cas.horzcat(*u), cas.horzcat(*noises_numerical))
        g_continuity = cas.reshape(x_integrated - cas.horzcat(*x[1:]), -1, 1)

        g = [g_continuity]
        lbg = [0] * x[0].shape[0] * n_shooting
        ubg = [0] * x[0].shape[0] * n_shooting
        g_names = [f"dynamics_continuity"] * x[0].shape[0] * n_shooting

        return g, lbg, ubg, g_names
