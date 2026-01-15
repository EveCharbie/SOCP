import casadi as cas
import numpy as np

from .discretization_abstract import DiscretizationAbstract
from ..examples.example_abstract import ExampleAbstract
from ..models.model_abstract import ModelAbstract
from .direct_collocation_polynomial import DirectCollocationPolynomial


class NoiseDiscretization(DiscretizationAbstract):

    def __init__(
        self,
        dynamics_transcription: DiscretizationAbstract,
        with_cholesky: bool = False,
        with_helper_matrix: bool = False,
    ) -> None:

        # Checks
        if with_cholesky:
            raise ValueError("The NoiseDiscretization method does not support/need the Cholesky decomposition.")
        if with_helper_matrix:
            raise ValueError("The NoiseDiscretization method does not support/need the helper matrix.")

        super().__init__()  # Does nothing

        self.discretization_transcription = dynamics_transcription

    def name(self) -> str:
        return "NoiseDiscretization"

    def declare_variables(
        self,
        ocp_example: ExampleAbstract,
        states_lower_bounds: dict[str, np.ndarray],
        states_upper_bounds: dict[str, np.ndarray],
        states_initial_guesses: dict[str, np.ndarray],
        controls_lower_bounds: dict[str, np.ndarray],
        controls_upper_bounds: dict[str, np.ndarray],
        controls_initial_guesses: dict[str, np.ndarray],
        collocation_points_initial_guesses: dict[str, np.ndarray]
    ) -> tuple[list[cas.SX], list[cas.SX], list[cas.SX], list[cas.SX], list[float], list[float], list[float]]:
        """
        Declare all symbolic variables for the states and controls with their bounds and initial guesses
        """
        # TODO : !!!!!!!!!!!!!!!!!!!
        nb_random = ocp_example.nb_random
        state_names = list(ocp_example.model.state_indices.keys())

        x = []
        z = []
        u = []
        w = []
        w_lower_bound = []
        w_upper_bound = []
        w_initial_guess = []

        T = cas.SX.sym("final_time", 1)
        w += [T]
        w_lower_bound += [ocp_example.min_time]
        w_upper_bound += [ocp_example.max_time]

        for i_node in range(ocp_example.n_shooting + 1):

            # States
            this_x = []
            this_z = []
            for state_name in state_names:

                # Create the symbolic variables
                n_components = states_lower_bounds[state_name].shape[0]
                this_x += [cas.SX.sym(f"{state_name}_{i_node}", n_components * nb_random)]

                # Add bounds and initial guess
                if i_node == 0:
                    # At the first node, the random initial state is imposed
                    this_init = states_initial_guesses[state_name][:, i_node].tolist()
                    initial_configuration = np.random.normal(
                        loc=this_init * nb_random,
                        scale=np.repeat(
                            ocp_example.initial_state_variability[ocp_example.model.state_indices[state_name]],
                            nb_random,
                        ),
                        size=len(this_init) * nb_random,
                    )
                    w_initial_guess += initial_configuration.tolist()
                    w_lower_bound += initial_configuration.tolist()
                    w_upper_bound += initial_configuration.tolist()
                else:
                    w_lower_bound += states_lower_bounds[state_name][:, i_node].tolist() * nb_random
                    w_upper_bound += states_upper_bounds[state_name][:, i_node].tolist() * nb_random
                    w_initial_guess += states_initial_guesses[state_name][:, i_node].tolist() * nb_random

                if isinstance(self.dynamics_transcription, DirectCollocationPolynomial):
                    # Create the symbolic variables for the mean states collocation points
                    collocation_order = self.dynamics_transcription.order
                    this_z += [
                        cas.SX.sym(f"{state_name}_{i_node}_z", n_components * nb_random * (collocation_order + 2))
                    ]
                    for i_collocation in range(collocation_order + 2):
                        # Add bounds and initial guess as linear interpolation between the two nodes
                        w_lower_bound += (
                            self.interpolate_between_nodes(
                                var_pre=states_lower_bounds[state_name][:, i_node],
                                var_post=states_lower_bounds[state_name][:, i_node + 1],
                                nb_points=collocation_order + 2,
                                current_point=i_collocation,
                            ).tolist()
                            * nb_random
                        )
                        w_upper_bound += (
                            self.interpolate_between_nodes(
                                var_pre=states_upper_bounds[state_name][:, i_node],
                                var_post=states_upper_bounds[state_name][:, i_node + 1],
                                nb_points=collocation_order + 2,
                                current_point=i_collocation,
                            ).tolist()
                            * nb_random
                        )
                        w_initial_guess += (
                            self.interpolate_between_nodes(
                                var_pre=states_initial_guesses[state_name][:, i_node],
                                var_post=states_initial_guesses[state_name][:, i_node + 1],
                                nb_points=collocation_order + 2,
                                current_point=i_collocation,
                            ).tolist()
                            * nb_random
                        )

            # Add the variables to a larger vector for easy access later
            x += [cas.vertcat(*this_x)]
            z += [cas.vertcat(*this_z)]
            w += [cas.vertcat(cas.vertcat(*this_x), cas.vertcat(*this_z))]

            # Controls
            if i_node < ocp_example.n_shooting:
                this_u = []
                for control_name in controls_lower_bounds.keys():
                    # Create the symbolic variables
                    n_components = controls_lower_bounds[control_name].shape[0]
                    this_u += [cas.SX.sym(f"{control_name}_{i_node}", n_components)]
                    # Add bounds and initial guess
                    w_lower_bound += controls_lower_bounds[control_name][:, i_node].tolist()
                    w_upper_bound += controls_upper_bounds[control_name][:, i_node].tolist()
                    w_initial_guess += controls_initial_guesses[control_name][:, i_node].tolist()
                # Add the variables to a larger vector for easy access later
                u += [cas.vertcat(*this_u)]
                w += [cas.vertcat(*this_u)]

        return x, z, u, w, w_lower_bound, w_upper_bound, w_initial_guess

    def get_variables_from_vector(
        self,
        model: ModelAbstract,
        states_lower_bounds: dict[str, np.ndarray],
        controls_lower_bounds: dict[str, np.ndarray],
        vector: cas.DM,
    ) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], cas.DM, cas.DM, cas.DM]:
        """
        Extract the states and controls from the optimization vector.
        """
        nb_random = model.nb_random
        n_shooting = states_lower_bounds[list(states_lower_bounds.keys())[0]].shape[1] - 1
        state_names = list(model.state_indices.keys())

        offset = 0
        T = vector[offset]
        offset += 1

        states = {
            key: np.zeros((states_lower_bounds[key].shape[0], n_shooting + 1, nb_random))
            for key in states_lower_bounds.keys()
        }
        collocation_points = {}
        if isinstance(self.dynamics_transcription, DirectCollocationPolynomial):
            nb_collocation_points = self.dynamics_transcription.order + 2
            collocation_points = {
                key: np.zeros((states_lower_bounds[key].shape[0], nb_collocation_points, n_shooting + 1, nb_random))
                for key in states_lower_bounds.keys()
            }
        controls = {key: np.zeros_like(controls_lower_bounds[key]) for key in controls_lower_bounds.keys()}
        x = []
        z = []
        u = []
        for i_node in range(n_shooting + 1):

            # States
            for state_name in state_names:
                n_components = states_lower_bounds[state_name].shape[0]
                for i_random in range(nb_random):
                    states[state_name][:, i_node, i_random] = np.array(vector[offset : offset + n_components]).flatten()
                    x += [vector[offset : offset + n_components]]
                    offset += n_components
                for i_random in range(nb_random):
                    for i_collocation in range(nb_collocation_points):
                        collocation_points[state_name][:, i_collocation, i_node, i_random] = np.array(
                            vector[offset : offset + n_components]
                        ).flatten()
                        z += [vector[offset : offset + n_components]]
                        offset += n_components

            # Controls
            if i_node < n_shooting:
                for control_name in controls_lower_bounds.keys():
                    n_components = controls_lower_bounds[control_name].shape[0]
                    controls[control_name][:, i_node] = np.array(vector[offset : offset + n_components]).flatten()
                    u += [vector[offset : offset + n_components]]
                    offset += n_components

        return T, states, collocation_points, controls, cas.vertcat(*x), cas.vertcat(*z), cas.vertcat(*u)

    def declare_noises(
        self,
        model: ModelAbstract,
        n_shooting: int,
        nb_random: int,
        motor_noise_magnitude: np.ndarray,
        sensory_noise_magnitude: np.ndarray,
        seed: int = 0,
    ) -> tuple[np.ndarray, cas.SX]:
        """
        Sample the noise values and declare the symbolic variables for the noises.
        """
        np.random.seed(seed)

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
                    this_noises_single += [cas.SX.sym(f"motor_noise_{i_random}", n_motor_noises)]
                this_noises_numerical += [this_motor_noise_vector]

            for i_random in range(nb_random):  # to remove
                this_sensory_noise_vector = np.random.normal(
                    loc=np.zeros((nb_references,)),
                    scale=np.reshape(np.array(sensory_noise_magnitude), (nb_references,)),
                    size=nb_references,
                )
                if i_node == 0:
                    this_noises_single += [cas.SX.sym(f"sensory_noise_{i_random}", nb_references)]
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
        for state_name, state_indices in model.state_indices.values():
            n_components = state_indices.stop - state_indices.start
            for i_random in range(model.nb_random):
                states[state_indices, i_random] = (
                    x[offset + i_random * n_components : offset + (i_random + 1) * n_components] ** exponent
                )
            offset += n_components * model.nb_random

        states_mean = cas.sum2(states) / model.nb_random
        return states_mean

    def get_covariance(
        self,
        model: ModelAbstract,
        x,
    ):
        states = type(x).zeros(model.nb_states, model.nb_random)
        states_mean = self.get_mean_states(model, x, squared=False)
        offset = 0
        for state_name, state_indices in model.state_indices.values():
            n_components = state_indices.stop - state_indices.start
            for i_random in range(model.nb_random):
                states[state_indices, i_random] = (
                    x[offset + i_random * n_components : offset + (i_random + 1) * n_components]
                )
            offset += n_components * model.nb_random

        diff = (states - states_mean)
        covariance = (diff @ diff.T) / (model.nb_random - 1)

        return covariance

    # def get_states_variance(
    #     self,
    #     model: ModelAbstract,
    #     x,
    #     squared: bool = False,
    # ):
    #     exponent = 2 if squared else 1
    #     states = type(x).zeros(model.nb_states, model.nb_random)
    #
    #     offset = 0
    #     for state_indices in model.state_indices:
    #         n_components = state_indices.stop - state_indices.start
    #         for i_random in range(model.nb_random):
    #             states[state_indices, i_random] = (
    #                 x[offset + i_random * n_components : offset + (i_random + 1) * n_components] ** exponent
    #             )
    #         offset += n_components * model.nb_random
    #     states_mean = cas.sum2(states) / model.nb_random
    #
    #     variations = cas.sum2((states - states_mean) ** 2) / model.nb_random
    #     return variations

    def get_reference(
        self,
        model: ModelAbstract,
        x: cas.SX,
        u: cas.SX,
    ):
        """
        Compute the mean sensory feedback to get the reference over all random simulations.

        Parameters
        ----------
        model : ModelAbstract
            The model used for the computation.
        x : cas.SX
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

    def get_ee_variance(
        self,
        model: ModelAbstract,
        x: cas.SX,
        u: cas.SX,
        HAND_FINAL_TARGET: np.ndarray,
    ):
        """

        Parameters
        ----------
        model : ModelAbstract
            The model used for the computation.
        x : cas.SX
            The state vector for all randoms (e.g., [q_1, qdot_1, q_2, qdot_2, ...]) at a specific time node.
        """

        sensory = type(x).zeros(model.nb_references, model.nb_random)
        n_components = model.q_indices.stop - model.q_indices.start
        offset = n_components * model.nb_random
        for i_random in range(model.nb_random):
            q_this_time = x[i_random * n_components : (i_random + 1) * n_components]
            qdot_this_time = x[offset + i_random * n_components : offset + (i_random + 1) * n_components]
            sensory[:, i_random] = model.sensory_output(q_this_time, qdot_this_time, cas.DM.zeros(model.nb_references))

        ee_pos_variability_x = cas.sum2((sensory[0, :] - HAND_FINAL_TARGET[0]) ** 2) / model.nb_random
        ee_pos_variability_y = cas.sum2((sensory[1, :][1, :] - HAND_FINAL_TARGET[1]) ** 2) / model.nb_random

        return ee_pos_variability_x, ee_pos_variability_y

    def get_mus_variance(
        self,
        model: ModelAbstract,
        x,
    ):
        states = type(x).zeros(model.nb_states, model.nb_random)

        offset = 0
        for state_name, state_indices in model.state_indices.values():
            n_components = state_indices.stop - state_indices.start
            for i_random in range(model.nb_random):
                states[state_indices, i_random] = x[
                    offset + i_random * n_components : offset + (i_random + 1) * n_components
                ]
            offset += n_components * model.nb_random
        states_mean = cas.sum2(states) / model.nb_random

        activations_variations = cas.sum2((states - states_mean) ** 2) / model.nb_random
        mus_variations = activations_variations[4 : 4 + model.nb_muscles]
        sum_variations = cas.sum1(mus_variations)

        return sum_variations

    def state_dynamics(
        self,
        ocp_example: ExampleAbstract,
        x,
        u,
        noise,
    ) -> cas.SX:

        nb_random = ocp_example.model.nb_random

        ref = self.get_reference(
            ocp_example.model,
            x,
            u,
        )

        dxdt = cas.SX.zeros(x.shape)
        for i_random in range(nb_random):

            # Code looks messier, but easier to extract the casadi variables from the printed casadi expressions
            offset = 0
            x_this_time = None
            for state_name, state_indices in ocp_example.model.state_indices.values():
                n_components = state_indices.stop - state_indices.start
                if x_this_time is None:
                    x_this_time = x[offset + i_random * n_components : offset + (i_random + 1) * n_components]
                else:
                    x_this_time = cas.vertcat(
                        x_this_time, x[offset + i_random * n_components : offset + (i_random + 1) * n_components]
                    )
                offset += n_components * nb_random

            offset = 0
            noise_this_time = None
            for noise_indices in ocp_example.model.noise_indices:
                n_components = noise_indices.stop - noise_indices.start
                if noise_this_time is None:
                    noise_this_time = noise[offset + i_random * n_components : offset + (i_random + 1) * n_components]
                else:
                    noise_this_time = cas.vertcat(
                        noise_this_time,
                        noise[offset + i_random * n_components : offset + (i_random + 1) * n_components],
                    )
                offset += n_components * nb_random

            dxdt_this_time = ocp_example.model.dynamics(
                x_this_time,
                u,
                ref,
                noise_this_time,
            )

            offset = 0
            for state_name, state_indices in ocp_example.model.state_indices.values():
                n_components = state_indices.stop - state_indices.start
                dxdt[offset + i_random * n_components : offset + (i_random + 1) * n_components] = dxdt_this_time[
                    state_indices
                ]
                offset += n_components * ocp_example.model.nb_random

        return dxdt

    def create_state_plots(
        self,
        ocp_example: ExampleAbstract,
        colors,
        axs,
        i_row,
        i_col,
        time_vector,
    ):
        # TODO: Add collocation points
        states_plots = []
        for i_random in range(ocp_example.nb_random):
            # Placeholder to plot the variables
            color = colors(i_random / ocp_example.nb_random)
            states_plots += axs[i_row, i_col].plot(time_vector, np.zeros_like(time_vector), marker=".", color=color)
        return states_plots

    def update_state_plots(
        self,
        ocp_example: ExampleAbstract,
        states_plots,
        i_state,
        states_opt,
        key,
        i_col,
        time_vector: np.ndarray,
    ) -> int:
        # TODO: Add collocation points
        for i_random in range(ocp_example.nb_random):
            states_plots[i_state].set_ydata(states_opt[key][i_col, :, i_random])
            i_state += 1
        return i_state
