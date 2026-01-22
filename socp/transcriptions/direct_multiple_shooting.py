import casadi as cas
import numpy as np

from .discretization_abstract import DiscretizationAbstract
from .noises_abstract import NoisesAbstract
from .transcription_abstract import TranscriptionAbstract
from .variables_abstract import VariablesAbstract
from ..constraints import Constraints
from ..examples.example_abstract import ExampleAbstract


class DirectMultipleShooting(TranscriptionAbstract):

    def __init__(self) -> None:

        super().__init__()  # Does nothing

    def initialize_dynamics_integrator(
        self,
        ocp_example: ExampleAbstract,
        discretization_method: DiscretizationAbstract,
        variables_vector: VariablesAbstract,
        noises_vector: NoisesAbstract,
    ) -> None:

        dynamics_func, integration_func, _, jacobian_funcs = self.declare_dynamics_integrator(
            ocp_example,
            discretization_method,
            variables_vector,
            noises_vector,
        )
        self.dynamics_func = dynamics_func
        self.integration_func = integration_func
        self.defect_func = None
        self.jacobian_funcs = jacobian_funcs

    @property
    def name(self) -> str:
        return "DirectMultipleShooting"

    def declare_dynamics_integrator(
        self,
        ocp_example,
        discretization_method,
        variables_vector: VariablesAbstract,
        noises_vector: NoisesAbstract,
    ) -> tuple[cas.Function, cas.Function, cas.Function, cas.Function]:
        """
        Formulate discrete time dynamics integration using a fixed step Runge-Kutta 4 integrator.
        Note: The first x and u used to declare the casadi functions, but all nodes will be used during the evaluation
        of the functions
        """

        n_steps = 5  # RK4 steps per interval
        dt = variables_vector.get_time() / ocp_example.n_shooting
        h = dt / n_steps

        # Dynamics
        xdot = discretization_method.state_dynamics(
            ocp_example,
            variables_vector.get_states(0),
            variables_vector.get_controls(0),
            noises_vector.get_noise_single(),
        )
        dynamics_func = cas.Function(
            f"dynamics",
            [
                variables_vector.get_states(0),
                variables_vector.get_controls(0),
                noises_vector.get_noise_single(),
            ],
            [xdot],
            ["x", "u", "noise"],
            ["xdot"],
        )
        # dynamics_func = dynamics_func.expand()


        # Integrator
        states_integrated = variables_vector.get_states(0)
        noises_single = noises_vector.get_noise_single()
        for j in range(n_steps):
            u_single = variables_vector.get_controls(0)
            k1 = dynamics_func(states_integrated, u_single, noises_single)
            k2 = dynamics_func(states_integrated + h / 2 * k1, u_single, noises_single)
            k3 = dynamics_func(states_integrated + h / 2 * k2, u_single, noises_single)
            k4 = dynamics_func(states_integrated + h * k3, u_single, noises_single)
            states_integrated += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        states_integration_func = cas.Function(
            "F",
            [
                variables_vector.get_time(),
                variables_vector.get_states(0),
                variables_vector.get_controls(0),
                noises_vector.get_noise_single(),
            ],
            [states_integrated],
            ["T", "x", "u", "noise"],
            ["x_next"],
        )

        # Covariance
        cov_integrated_vector = cas.SX()
        jacobian_funcs = None
        if discretization_method.name == "MeanAndCovariance":

            sigma_ww = cas.diag(noises_vector.get_noise_single())

            dFdx = cas.jacobian(states_integrated, variables_vector.get_states(0))
            dFdw = cas.jacobian(states_integrated, noises_vector.get_noise_single())

            jacobian_funcs = cas.Function(
                "jacobian_func",
                [
                    variables_vector.get_time(),
                    variables_vector.get_states(0),
                    variables_vector.get_controls(0),
                    noises_vector.get_noise_single(),
                ],
                [dFdx, dFdw],
            )

            cov_matrix = variables_vector.get_cov_matrix(0)
            cov_integrated = dFdx @ cov_matrix @ dFdx.T + dFdw @ sigma_ww @ dFdw.T

            cov_integrated_vector = variables_vector.reshape_matrix_to_vector(cov_integrated)

        # This function evaluation shields the mean state dynamics from the noises, where as the P dynamics needs a
        # numerical noise value.
        states_next = states_integration_func(
            variables_vector.get_time(),
            variables_vector.get_states(0),
            variables_vector.get_controls(0),
            cas.DM.zeros(ocp_example.model.nb_noises * variables_vector.nb_random),
        )
        x_next = cas.vertcat(states_next, cov_integrated_vector)
        integration_func = cas.Function(
            "F",
            [
                variables_vector.get_time(),
                variables_vector.get_states(0),
                variables_vector.get_cov(0),
                variables_vector.get_controls(0),
                noises_vector.get_noise_single(),
            ],
            [x_next],
            ["T", "x", "cov", "u", "noise"],
            ["x_next"],
        )
        # integration_func = integration_func.expand()
        return dynamics_func, integration_func, None, jacobian_funcs

    def set_dynamics_constraints(
        self,
        ocp_example: ExampleAbstract,
        discretization_method: DiscretizationAbstract,
        variables_vector: VariablesAbstract,
        noises_vector: NoisesAbstract,
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
            cas.horzcat(*[variables_vector.get_cov(i_node) for i_node in range(0, n_shooting)]),
            cas.horzcat(*[variables_vector.get_controls(i_node) for i_node in range(0, n_shooting)]),
            cas.horzcat(*[noises_vector.get_one_vector_numerical(i_node) for i_node in range(0, n_shooting)]),
        )

        if discretization_method.name == "MeanAndCovariance":
            states_next = cas.horzcat(*[variables_vector.get_states(i_node) for i_node in range(1, n_shooting + 1)])
            cov_next = cas.horzcat(*[variables_vector.get_cov(i_node) for i_node in range(1, n_shooting + 1)])
            x_next = cas.vertcat(states_next, cov_next)
            nb_variables = nb_states + nb_states * nb_states
        else:
            x_next = cas.horzcat(*[variables_vector.get_states(i_node) for i_node in range(1, n_shooting + 1)])
            nb_variables = nb_states

        g_continuity = cas.reshape(x_integrated - x_next, (-1, 1))

        for i_node in range(n_shooting):
            constraints.add(
                g=g_continuity[i_node * nb_variables : (i_node + 1) * nb_variables],
                lbg=[0] * nb_variables,
                ubg=[0] * nb_variables,
                g_names=["dynamics_continuity"] * nb_variables,
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

        return
