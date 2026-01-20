import casadi as cas
import numpy as np

from .discretization_abstract import DiscretizationAbstract
from .transcription_abstract import TranscriptionAbstract
from .variables_abstract import VariablesAbstract
from ..models.model_abstract import ModelAbstract
from ..examples.example_abstract import ExampleAbstract
from ..constraints import Constraints


class DirectMultipleShooting(TranscriptionAbstract):

    def __init__(self) -> None:

        super().__init__()  # Does nothing

    def initialize_dynamics_integrator(
        self,
        ocp_example: ExampleAbstract,
        discretization_method: DiscretizationAbstract,
        variables_vector: VariablesAbstract,
        noises_single: cas.SX.sym,
    ) -> None:

        # Note: The first x and u used to declare the casadi functions, but all nodes will be used during the evaluation of the functions
        dynamics_func, integration_func = self.declare_dynamics_integrator(
            ocp_example,
            discretization_method,
            variables_vector,
            noises_single=noises_single,
        )
        self.dynamics_func = dynamics_func
        self.integration_func = integration_func

    def name(self) -> str:
        return "DirectMultipleShooting"

    def declare_dynamics_integrator(
        self,
        ocp_example,
        discretization_method,
        variables_vector: VariablesAbstract,
        noises_single: cas.SX.sym,
    ) -> tuple[cas.Function, cas.Function]:
        """
        Formulate discrete time dynamics integration using a fixed step Runge-Kutta 4 integrator.
        """

        n_steps = 5  # RK4 steps per interval
        dt = variables_vector.get_time() / ocp_example.n_shooting
        h = dt / n_steps

        # Dynamics
        xdot = discretization_method.state_dynamics(
            ocp_example,
            variables_vector.get_states(0),
            variables_vector.get_controls(0),
            noises_single,
        )
        dynamics_func = cas.Function(
            f"dynamics", [
                variables_vector.get_states(0),
                variables_vector.get_controls(0),
                noises_single,
            ], [xdot], ["x", "u", "noise"], ["xdot"]
        )
        # dynamics_func = dynamics_func.expand()

        # Integrator
        x_next = variables_vector.get_states(0)
        for j in range(n_steps):
            u_single = variables_vector.get_controls(0)
            k1 = dynamics_func(x_next, u_single, noises_single)
            k2 = dynamics_func(x_next + h / 2 * k1, u_single, noises_single)
            k3 = dynamics_func(x_next + h / 2 * k2, u_single, noises_single)
            k4 = dynamics_func(x_next + h * k3, u_single, noises_single)
            x_next += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        integration_func = cas.Function(
            "F",
            [variables_vector.get_states(0), variables_vector.get_controls(0), noises_single],
            [x_next],
            ["x", "u", "noise"],
            ["x_next"],
        )
        # integration_func = integration_func.expand()
        return dynamics_func, integration_func

    def set_dynamics_constraints(
        self,
        ocp_example: ExampleAbstract,
        discretization_method: DiscretizationAbstract,
        variables_vector: VariablesAbstract,
        noises_single: cas.SX.sym,
        noises_numerical: np.ndarray,
        constraints: Constraints,
        n_threads: int = 8,
    ) -> None:

        n_shooting = variables_vector.n_shooting
        nb_states = variables_vector.get_states(0).shape[0]

        # Multi-thread continuity constraint
        multi_threaded_integrator = self.integration_func.map(n_shooting, "thread", n_threads)
        x_integrated = multi_threaded_integrator(
            variables_vector.get_time(),
            cas.horzcat(*[variables_vector.get_states(i_node) for i_node in range(0, n_shooting)]),
            cas.horzcat(*[variables_vector.get_controls(i_node) for i_node in range(0, n_shooting)]),
            cas.horzcat(*noises_numerical),
        )
        x_next = cas.horzcat(*[variables_vector.get_states(i_node) for i_node in range(1, n_shooting + 1)])
        g_continuity = cas.reshape(x_integrated - x_next, (-1, 1))

        for i_node in range(n_shooting):
            constraints.add(
                g=g_continuity[i_node * nb_states : (i_node + 1) * nb_states],
                lbg=[0] * nb_states,
                ubg=[0] * nb_states,
                g_names=["dynamics_continuity"] * nb_states,
                node=i_node,
            )

        # Add other constraints if any
        for i_node in range(n_shooting):
            g_other, lbg_other, ubg_other, g_names_other = self.other_internal_constraints(
                ocp_example,
                discretization_method,
                variables_vector,
                noises_single,
            )
            constraints.add(
                g=g_other,
                lbg=ubg_other,
                ubg=ubg_other,
                g_names=g_names_other,
                node=i_node,
            )

        return
