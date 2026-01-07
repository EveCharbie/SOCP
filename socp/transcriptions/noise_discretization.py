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
        n_random = model.n_random

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
                this_x += [cas.MX.sym(f"{state_name}_{i_node}", n_components * n_random)]
                # Add bounds and initial guess
                w_lower_bound += states_lower_bounds[state_name][:, i_node].tolist() * n_random
                w_upper_bound += states_upper_bounds[state_name][:, i_node].tolist() * n_random
                w_initial_guess += states_initial_guesses[state_name][:, i_node].tolist() * n_random
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

    def declare_noises(
        self,
        model: ModelAbstract,
        n_shooting: int,
        n_random: int,
        motor_noise_magnitude: np.ndarray,
        sensory_noise_magnitude: np.ndarray,
    ) -> tuple[np.ndarray, cas.MX]:
        """
        Sample the noise values and declare the symbolic variables for the noises.
        """
        n_motor_noises = motor_noise_magnitude.shape[0]
        n_references = sensory_noise_magnitude.shape[0]

        noises_numerical = []
        for i_shooting in range(n_shooting):
            this_motor_noise_vector = np.zeros((n_motor_noises * n_random,))
            this_sensory_noise_vector = np.zeros((n_references * n_random,))
            for i_random in range(n_random):
                this_motor_noise_vector[model.motor_noise_indices_this_random(i_random)] = np.random.normal(
                    loc=np.zeros((model.nb_q,)),
                    scale=np.reshape(np.array(motor_noise_magnitude), (n_motor_noises,)),
                    size=n_motor_noises,
                )
                this_sensory_noise_vector[model.sensory_noise_indices_this_random(i_random)] = np.random.normal(
                    loc=np.zeros((n_references,)),
                    scale=np.reshape(np.array(sensory_noise_magnitude), (n_references,)),
                    size=n_references,
                )
            noises_numerical += [cas.vertcat(this_motor_noise_vector, this_sensory_noise_vector)]
        noises_single = cas.MX.sym("noises_single", (n_motor_noises + n_references) * n_random)
        return noises_numerical, noises_single
