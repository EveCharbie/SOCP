import casadi as cas
import numpy as np

from .discretization_abstract import DiscretizationAbstract
from ..models.model_abstract import ModelAbstract


class NoiseDiscretization(DiscretizationAbstract):

    def name(self) -> str:
        return "NoiseDiscretization"

    def declare_variables(
        self,
        model: ModelAbstract,
        n_shooting: int,
        states_lower_bounds: dict[str, np.ndarray],
        states_upper_bounds: dict[str, np.ndarray],
        states_initial_guesses: dict[str, np.ndarray],
        controls_lower_bounds: dict[str, np.ndarray],
        controls_upper_bounds: dict[str, np.ndarray],
        controls_initial_guesses: dict[str, np.ndarray],
    ) -> tuple[list[cas.MX], list[cas.MX], list[cas.MX], list[float], list[float], list[float]]:
        """
        Declare all symbolic variables for the states and controls with their bounds and initial guesses
        """
        nb_random = model.nb_random

        x = []
        u = []
        w = []
        w_lower_bound = []
        w_upper_bound = []
        w_initial_guess = []
        for i_node in range(n_shooting + 1):

            # States
            this_x = []
            for state_name in states_lower_bounds.keys():
                # Create the symbolic variables
                n_components = states_lower_bounds[state_name].shape[0]
                this_x += [cas.MX.sym(f"{state_name}_{i_node}", n_components * nb_random)]
                # Add bounds and initial guess
                w_lower_bound += states_lower_bounds[state_name][:, i_node].tolist() * nb_random
                w_upper_bound += states_upper_bounds[state_name][:, i_node].tolist() * nb_random
                w_initial_guess += states_initial_guesses[state_name][:, i_node].tolist() * nb_random

            # Add the variables to a larger vector for easy access later
            x += [cas.vertcat(*this_x)]
            w += [cas.vertcat(*this_x)]

            # Controls
            if i_node < n_shooting:
                this_u = []
                for control_name in controls_lower_bounds.keys():
                    # Create the symbolic variables
                    n_components = controls_lower_bounds[control_name].shape[0]
                    this_u += [cas.MX.sym(f"{control_name}_{i_node}", n_components)]
                    # Add bounds and initial guess
                    w_lower_bound += controls_lower_bounds[control_name][:, i_node].tolist()
                    w_upper_bound += controls_upper_bounds[control_name][:, i_node].tolist()
                    w_initial_guess += controls_initial_guesses[control_name][:, i_node].tolist()
                # Add the variables to a larger vector for easy access later
                u += [cas.vertcat(*this_u)]
                w += [cas.vertcat(*this_u)]

        return x, u, w, w_lower_bound, w_upper_bound, w_initial_guess

    def get_variables_from_vector(
        self,
        model: ModelAbstract,
        states_lower_bounds: dict[str, np.ndarray],
        controls_lower_bounds: dict[str, np.ndarray],
        vector: cas.DM,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], cas.DM, cas.DM]:
        """
        Extract the states and controls from the optimization vector.
        """
        nb_random = model.nb_random
        n_shooting = states_lower_bounds[list(states_lower_bounds.keys())[0]].shape[1] - 1

        offset = 0
        states = {
            key: np.zeros((states_lower_bounds[key].shape[0], n_shooting + 1, nb_random))
            for key in states_lower_bounds.keys()
        }
        controls = {key: np.zeros_like(controls_lower_bounds[key]) for key in controls_lower_bounds.keys()}
        x = []
        u = []
        for i_node in range(n_shooting + 1):

            # States
            for state_name in states_lower_bounds.keys():
                for i_random in range(nb_random):
                    n_components = states_lower_bounds[state_name].shape[0]
                    states[state_name][:, i_node, i_random] = np.array(vector[offset : offset + n_components]).flatten()
                    x += [vector[offset : offset + n_components]]
                    offset += n_components

            # Controls
            if i_node < n_shooting:
                for control_name in controls_lower_bounds.keys():
                    n_components = controls_lower_bounds[control_name].shape[0]
                    controls[control_name][:, i_node] = np.array(vector[offset : offset + n_components]).flatten()
                    u += [vector[offset : offset + n_components]]
                    offset += n_components

        return states, controls, cas.vertcat(*x), cas.vertcat(*u)

    def declare_noises(
        self,
        model: ModelAbstract,
        n_shooting: int,
        nb_random: int,
        motor_noise_magnitude: np.ndarray,
        sensory_noise_magnitude: np.ndarray,
    ) -> tuple[np.ndarray, cas.MX]:
        """
        Sample the noise values and declare the symbolic variables for the noises.
        """
        n_motor_noises = motor_noise_magnitude.shape[0]
        nb_references = sensory_noise_magnitude.shape[0]

        noises_numerical = []
        for i_node in range(n_shooting):
            this_noises_numerical = []
            if i_node == 0:
                this_noises_single = []
            for i_random in range(nb_random):
                this_motor_noise_vector = np.random.normal(
                    loc=np.zeros((model.nb_q,)),
                    scale=np.reshape(np.array(motor_noise_magnitude), (n_motor_noises,)),
                    size=n_motor_noises,
                )
                if i_node == 0:
                    this_noises_single += [cas.MX.sym(f"motor_noise_{i_random}_{i_node}", n_motor_noises)]
                this_noises_numerical += [this_motor_noise_vector]

            for i_random in range(nb_random):  # to remove
                this_sensory_noise_vector = np.random.normal(
                    loc=np.zeros((nb_references,)),
                    scale=np.reshape(np.array(sensory_noise_magnitude), (nb_references,)),
                    size=nb_references,
                )
                if i_node == 0:
                    this_noises_single += [cas.MX.sym(f"sensory_noise_{i_random}_{i_node}", nb_references)]
                this_noises_numerical += [this_sensory_noise_vector]

            noises_numerical += [cas.vertcat(*this_noises_numerical)]

        return noises_numerical, cas.vertcat(*this_noises_single)

    def get_mean_states(
        self,
        model: ModelAbstract,
        x,
        squared: bool = False,
    ):
        exponent = 2 if squared else 1
        states = type(x).zeros(model.nb_states, model.nb_random)

        offset = 0
        for state_indices in model.state_indices:
            n_components = state_indices.stop - state_indices.start
            for i_random in range(model.nb_random):
                states[state_indices, i_random] = (
                    x[offset + i_random * n_components : offset + (i_random + 1) * n_components] ** exponent
                )
            offset += n_components * model.nb_random

        states_mean = cas.sum2(states) / model.nb_random
        return states_mean

    def get_states_variance(
        self,
        model: ModelAbstract,
        x,
        squared: bool = False,
    ):
        exponent = 2 if squared else 1
        states = type(x).zeros(model.nb_states, model.nb_random)

        offset = 0
        for state_indices in model.state_indices:
            n_components = state_indices.stop - state_indices.start
            for i_random in range(model.nb_random):
                states[state_indices, i_random] = (
                    x[offset + i_random * n_components : offset + (i_random + 1) * n_components] ** exponent
                )
            offset += n_components * model.nb_random
        states_mean = cas.sum2(states) / model.nb_random

        variations = cas.sum2((states - states_mean) ** 2) / model.nb_random
        return variations

    def get_reference(
        self,
        model: ModelAbstract,
        x: cas.MX,
        u: cas.MX,
    ):
        """
        Compute the mean sensory feedback to get the reference over all random simulations.

        Parameters
        ----------
        model : ModelAbstract
            The model used for the computation.
        x : cas.MX
            The state vector for all randoms (e.g., [q_1, qdot_1, q_2, qdot_2, ...]) at a specific time node.
        """

        ref = type(x).zeros(model.nb_references, 1)
        n_components = model.q_indices.stop - model.q_indices.start
        offset = n_components * model.nb_random
        for i_random in range(model.nb_random):
            q_this_time = x[i_random * n_components : (i_random + 1) * n_components]
            qdot_this_time = x[offset + i_random * n_components : offset + (i_random + 1) * n_components]
            ref += model.sensory_output(q_this_time, qdot_this_time, cas.DM.zeros(model.nb_references))

        ref /= model.nb_random
        return ref

    def state_dynamics(
        self,
        model: ModelAbstract,
        x,
        u,
        noise,
    ) -> cas.MX:

        ref = self.get_reference(
            model,
            x,
            u,
        )

        dxdt = cas.MX.zeros(x.shape)
        for i_random in range(model.nb_random):

            # Code looks messier, but easier to extract the casadi variables from the printed casadi expressions
            offset = 0
            x_this_time = None
            for state_indices in model.state_indices:
                n_components = state_indices.stop - state_indices.start
                if x_this_time is None:
                    x_this_time = x[offset + i_random * n_components : offset + (i_random + 1) * n_components]
                else:
                    x_this_time = cas.vertcat(
                        x_this_time, x[offset + i_random * n_components : offset + (i_random + 1) * n_components]
                    )
                offset += n_components * model.nb_random

            offset = 0
            noise_this_time = None
            for noise_indices in model.noise_indices:
                n_components = noise_indices.stop - noise_indices.start
                if noise_this_time is None:
                    noise_this_time = noise[offset + i_random * n_components : offset + (i_random + 1) * n_components]
                else:
                    noise_this_time = cas.vertcat(
                        noise_this_time,
                        noise[offset + i_random * n_components : offset + (i_random + 1) * n_components],
                    )
                offset += n_components * model.nb_random

            dxdt_this_time = model.dynamics(
                x_this_time,
                u,
                ref,
                noise_this_time,
            )

            offset = 0
            for state_indices in model.state_indices:
                n_components = state_indices.stop - state_indices.start
                dxdt[offset + i_random * n_components : offset + (i_random + 1) * n_components] = dxdt_this_time[
                    state_indices
                ]
                offset += n_components * model.nb_random

        return dxdt
