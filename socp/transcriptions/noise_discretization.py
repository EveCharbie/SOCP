import casadi as cas
import numpy as np

from .direct_collocation_polynomial import DirectCollocationPolynomial
from .discretization_abstract import DiscretizationAbstract
from .variables_abstract import VariablesAbstract
from ..examples.example_abstract import ExampleAbstract
from ..models.model_abstract import ModelAbstract


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
        self.with_cholesky = False
        self.with_helper_matrix = False

    class Variables(VariablesAbstract):
        def __init__(
            self,
            n_shooting: int,
            nb_collocation_points: int,
            nb_random: int,
            state_indices: dict[str, range],
            control_indices: dict[str, range],
        ):
            self.n_shooting = n_shooting
            self.nb_collocation_points = nb_collocation_points
            self.nb_random = nb_random
            self.state_indices = state_indices
            self.control_indices = control_indices
            self.state_names = list(state_indices.keys())
            self.control_names = list(control_indices.keys())

            self.t = None
            self.x_list = [
                {state_name: [None for _ in range(nb_random)] for state_name in state_names}
                for _ in range(n_shooting + 1)
            ]
            self.z_list = [
                {
                    state_name: [[None for _ in range(nb_random)] for _ in range(nb_collocation_points)]
                    for state_name in state_names
                }
                for _ in range(n_shooting + 1)
            ]
            self.u_list = [{control_name: None for control_name in control_names} for _ in range(n_shooting + 1)]

        @staticmethod
        def transform_to_dm(value: cas.SX | cas.DM | np.ndarray | list) -> cas.DM:
            if isinstance(value, np.ndarray):
                return cas.DM(value.flatten())
            elif isinstance(value, list):
                return cas.DM(np.array(value).flatten())
            else:
                return value

        # --- Add --- #
        def add_time(self, value: cas.SX | cas.DM):
            self.t = self.transform_to_dm(value)

        def add_state(self, name: str, node: int, random: int, value: cas.SX | cas.DM):
            self.x_list[node][name][random] = self.transform_to_dm(value)

        def add_collocation_point(self, name: str, node: int, random: int, point: int, value: cas.SX | cas.DM):
            self.z_list[node][name][random][point] = self.transform_to_dm(value)

        def add_control(self, name: str, node: int, value: cas.SX | cas.DM):
            self.u_list[node][name] = self.transform_to_dm(value)

        # --- Nb --- #
        @property
        def nb_states(self):
            nb_states = 0
            for state_name in self.state_names:
                nb_states += self.x_list[0][state_name][0].shape[0]
            return nb_states

        @property
        def nb_controls(self):
            nb_controls = 0
            for control_name in self.control_names:
                nb_controls += self.u_list[0][control_name].shape[0]
            return nb_controls

        # --- Get --- #
        def get_time(self):
            return self.t

        def get_state(self, name: str, node: int, random: int):
            return self.x_list[node][name][random]

        def get_states_matrix(self, node: int):
            states_matrix = None
            for i_random in range(self.nb_random):
                states_vector = None
                for state_name in self.state_names:
                    if states_vector is None:
                        states_vector = self.x_list[node][state_name][i_random]
                    else:
                        states_vector = cas.vertcat(states_vector, self.x_list[node][state_name][i_random])
                if states_matrix is None:
                    states_matrix = states_vector
                else:
                    states_matrix = cas.horzcat(states_matrix, states_vector)
            return states_matrix

        def get_specific_collocation_point(self, name: str, node: int, random: int, point: int):
            return self.z_list[node][name][random][point]

        def get_collocation_point(self, node: int, point: int):
            collocation_points_matrix = None
            for i_random in range(self.nb_random):
                collocation_points_vector = None
                for state_name in self.state_names:
                    if collocation_points_vector is None:
                        collocation_points_vector = self.z_list[node][state_name][i_random][point]
                    else:
                        collocation_points_vector = cas.vertcat(
                            collocation_points_vector, self.z_list[node][state_name][i_random][point]
                        )
                if collocation_points_matrix is None:
                    collocation_points_matrix = collocation_points_vector
                else:
                    collocation_points_matrix = cas.horzcat(
                        collocation_points_matrix,
                        collocation_points_vector,
                    )
            return collocation_points_matrix

        def get_collocation_points_matrix(self, node: int):
            collocation_points_matrix = None
            for i_random in range(self.nb_random):
                collocation_points_vector = None
                for i_collocation in range(self.nb_collocation_points):
                    for state_name in self.state_names:
                        if collocation_points_vector is None:
                            collocation_points_vector = self.z_list[node][state_name][i_random][i_collocation]
                        else:
                            collocation_points_vector = cas.vertcat(
                                collocation_points_vector, self.z_list[node][state_name][i_random][i_collocation]
                            )
                if collocation_points_matrix is None:
                    collocation_points_matrix = collocation_points_vector
                else:
                    collocation_points_matrix = cas.horzcat(
                        collocation_points_matrix,
                        collocation_points_vector,
                    )
            return collocation_points_matrix

        def get_control(self, name: str, node: int):
            return self.u_list[node][name]

        def get_controls(self, node: int):
            controls = None
            for control_name in self.control_names:
                if controls is None:
                    controls = self.u_list[node][control_name]
                else:
                    controls = cas.vertcat(controls, self.u_list[node][control_name])
            return controls

        # --- Get vectors --- #
        def get_one_vector(self, node: int, keep_only_symbolic: bool = False):
            nb_random = self.nb_random

            vector = []
            # X
            for i_random in range(nb_random):
                for state_name in self.state_names:
                    vector += [self.x_list[node][state_name][i_random]]
            # Z
            for i_random in range(nb_random):
                for i_collocation in range(self.nb_collocation_points):
                    for state_name in self.state_names:
                        if node < self.n_shooting:
                            vector += [self.z_list[node][state_name][i_random][i_collocation]]
                        else:
                            if not keep_only_symbolic:
                                vector += [self.z_list[node][state_name][i_random][i_collocation]]
            # U
            for control_name in self.control_names:
                if node < self.n_shooting:
                    vector += [self.u_list[node][control_name]]
                else:
                    if not keep_only_symbolic:
                        vector += [self.u_list[node][control_name]]

            return cas.vertcat(*vector)

        def get_full_vector(self, keep_only_symbolic: bool = False):
            vector = []
            vector += [self.t]
            for i_node in range(self.n_shooting + 1):
                vector += [self.get_one_vector(i_node, keep_only_symbolic)]
            return cas.vertcat(*vector)

        # --- Set vectors --- #
        def set_from_vector(self, vector: cas.DM, only_has_symbolics: bool = False):
            nb_random = self.nb_random

            offset = 0
            self.t = vector[offset]
            offset += 1

            for i_node in range(self.n_shooting + 1):
                # X
                for i_random in range(nb_random):
                    for state_name in self.state_names:
                        n_components = self.state_indices[state_name].stop - self.state_indices[state_name].start
                        self.x_list[i_node][state_name][i_random] = vector[offset : offset + n_components]
                        offset += n_components
                # Z
                for i_random in range(nb_random):
                    for i_collocation in range(self.nb_collocation_points):
                        for state_name in self.state_names:
                            if not only_has_symbolics or i_node < self.n_shooting:
                                n_components = (
                                    self.state_indices[state_name].stop - self.state_indices[state_name].start
                                )
                                self.z_list[i_node][state_name][i_random][i_collocation] = vector[
                                    offset : offset + n_components
                                ]
                                offset += n_components

                # U
                if not only_has_symbolics or i_node < self.n_shooting:
                    for control_name in self.control_names:
                        n_components = (
                            self.control_indices[control_name].stop - self.control_indices[control_name].start
                        )
                        self.u_list[i_node][control_name] = vector[offset : offset + n_components]
                        offset += n_components

        # --- Get array --- #
        def get_states_array(self) -> np.ndarray:
            states_var_array = np.zeros((self.nb_states, self.n_shooting + 1, self.nb_random))
            for i_random in range(self.nb_random):
                for i_node in range(self.n_shooting + 1):
                    states = None
                    for state_name in self.state_names:
                        if states is None:
                            states = np.array(self.x_list[i_node][state_name][i_random])
                        else:
                            states = np.vstack((states, self.x_list[i_node][state_name][i_random]))
                    states_var_array[:, i_node, i_random] = states
            return states_var_array

        def get_collocation_points_array(self) -> np.ndarray:
            collocation_points_var_array = np.zeros(
                (self.nb_states * self.nb_collocation_points, self.n_shooting + 1, self.nb_random)
            )
            for i_random in range(self.nb_random):
                for i_node in range(self.n_shooting + 1):
                    coll = None
                    for i_collocation in range(self.nb_collocation_points):
                        for state_name in self.state_names:
                            if coll is None:
                                coll = np.array(self.z_list[i_node][state_name][i_random][i_collocation])
                            else:
                                coll = np.vstack((coll, self.z_list[i_node][state_name][i_random][i_collocation]))
                    collocation_points_var_array[:, i_node, i_random] = coll
            return collocation_points_var_array

        def get_controls_array(self) -> np.ndarray:
            controls_var_array = np.zeros((self.nb_controls, self.n_shooting + 1))
            for i_node in range(self.n_shooting + 1):
                control = None
                for control_name in self.control_names:
                    if control is None:
                        control = np.array(self.u_list[i_node][control_name])
                    else:
                        control = np.vstack((control, self.u_list[i_node][control_name]))
                controls_var_array[:, i_node] = control
            return controls_var_array

        def validate_vector(self):
            # TODO
            pass

    def name(self) -> str:
        return "NoiseDiscretization"

    def declare_variables(
        self,
        ocp_example: ExampleAbstract,
        states_lower_bounds: dict[str, np.ndarray],
        controls_lower_bounds: dict[str, np.ndarray],
    ) -> Variables:
        """
        Declare all symbolic variables for the states and controls with their bounds and initial guesses
        """
        nb_random = ocp_example.nb_random
        n_shooting = ocp_example.n_shooting
        nb_collocation_points = self.dynamics_transcription.nb_collocation_points
        state_names = list(ocp_example.model.state_indices.keys())
        control_names = list(ocp_example.model.control_indices.keys())

        variables = self.Variables(
            n_shooting=n_shooting,
            nb_collocation_points=nb_collocation_points,
            nb_random=nb_random,
            state_names=state_names,
            control_names=control_names,
        )

        T = cas.SX.sym("final_time", 1)
        variables.add_time(T)

        for i_node in range(n_shooting + 1):
            for state_name in state_names:
                n_components = states_lower_bounds[state_name].shape[0]
                for i_random in range(nb_random):
                    x_sym = cas.SX.sym(f"{state_name}_{i_node}_{i_random}", n_components)
                    variables.add_state(state_name, i_node, i_random, x_sym)

                if isinstance(self.dynamics_transcription, DirectCollocationPolynomial):
                    # Create the symbolic variables for the states collocation points
                    for i_random in range(nb_random):
                        for i_collocation in range(nb_collocation_points):
                            if i_node < n_shooting:
                                z_sym = cas.SX.sym(f"{state_name}_{i_node}_{i_random}_{i_collocation}_z", n_components)
                            else:
                                z_sym = cas.SX.zeros(n_components)
                            variables.add_collocation_point(state_name, i_node, i_random, i_collocation, z_sym)

            # Controls
            for control_name in controls_lower_bounds.keys():
                n_components = controls_lower_bounds[control_name].shape[0]
                if i_node < ocp_example.n_shooting:
                    u = cas.SX.sym(f"{control_name}_{i_node}", n_components)
                else:
                    u = cas.SX.zeros(n_components)
                variables.add_control(control_name, i_node, u)

        return variables

    def declare_bounds_and_init(
        self,
        ocp_example: ExampleAbstract,
        states_lower_bounds: dict[str, np.ndarray],
        states_upper_bounds: dict[str, np.ndarray],
        states_initial_guesses: dict[str, np.ndarray],
        controls_lower_bounds: dict[str, np.ndarray],
        controls_upper_bounds: dict[str, np.ndarray],
        controls_initial_guesses: dict[str, np.ndarray],
        collocation_points_initial_guesses: dict[str, np.ndarray],
    ) -> tuple[Variables, Variables, Variables]:
        """
        Declare all symbolic variables for the states and controls with their bounds and initial guesses
        """
        nb_random = ocp_example.nb_random
        n_shooting = ocp_example.n_shooting
        nb_collocation_points = self.dynamics_transcription.nb_collocation_points
        state_names = list(ocp_example.model.state_indices.keys())
        control_names = list(ocp_example.model.control_indices.keys())

        w_lower_bound = self.Variables(
            n_shooting=n_shooting,
            nb_random=nb_random,
            nb_collocation_points=nb_collocation_points,
            state_names=state_names,
            control_names=control_names,
        )
        w_upper_bound = self.Variables(
            n_shooting=n_shooting,
            nb_random=nb_random,
            nb_collocation_points=nb_collocation_points,
            state_names=state_names,
            control_names=control_names,
        )
        w_initial_guess = self.Variables(
            n_shooting=n_shooting,
            nb_random=nb_random,
            nb_collocation_points=nb_collocation_points,
            state_names=state_names,
            control_names=control_names,
        )

        w_initial_guess.add_time(ocp_example.final_time)
        w_lower_bound.add_time(ocp_example.min_time)
        w_upper_bound.add_time(ocp_example.max_time)

        for i_node in range(n_shooting + 1):

            # X - states
            for state_name in state_names:
                if i_node == 0:
                    # At the first node, the random initial state is imposed
                    this_init = states_initial_guesses[state_name][:, i_node].tolist()
                    initial_configuration = np.random.normal(
                        loc=this_init * nb_random,
                        scale=np.repeat(
                            ocp_example.initial_state_variability[ocp_example.model.state_indices[state_name]],
                            nb_random,
                        ),
                        size=(len(this_init), nb_random),
                    )
                    for i_random in range(nb_random):
                        w_lower_bound.add_state(state_name, i_node, i_random, initial_configuration[:, i_random])
                        w_upper_bound.add_state(state_name, i_node, i_random, initial_configuration[:, i_random])
                        w_initial_guess.add_state(state_name, i_node, i_random, initial_configuration[:, i_random])
                else:
                    for i_random in range(nb_random):
                        w_lower_bound.add_state(
                            state_name, i_node, i_random, states_lower_bounds[state_name][:, i_node]
                        )
                        w_upper_bound.add_state(
                            state_name, i_node, i_random, states_upper_bounds[state_name][:, i_node]
                        )
                        w_initial_guess.add_state(
                            state_name, i_node, i_random, states_initial_guesses[state_name][:, i_node]
                        )

                # Z - collocation points
                if isinstance(self.dynamics_transcription, DirectCollocationPolynomial):
                    for state_name in state_names:
                        for i_random in range(nb_random):
                            # The last interval does not have collocation points
                            for i_collocation in range(nb_collocation_points):
                                if i_node < n_shooting:
                                    # Add bounds and initial guess as linear interpolation between the two nodes
                                    w_lower_bound.add_collocation_point(
                                        state_name,
                                        i_node,
                                        i_random,
                                        i_collocation,
                                        self.interpolate_between_nodes(
                                            var_pre=states_lower_bounds[state_name][:, i_node],
                                            var_post=states_lower_bounds[state_name][:, i_node + 1],
                                            nb_points=nb_collocation_points,
                                            current_point=i_collocation,
                                        ).tolist(),
                                    )
                                    w_upper_bound.add_collocation_point(
                                        state_name,
                                        i_node,
                                        i_random,
                                        i_collocation,
                                        self.interpolate_between_nodes(
                                            var_pre=states_upper_bounds[state_name][:, i_node],
                                            var_post=states_upper_bounds[state_name][:, i_node + 1],
                                            nb_points=nb_collocation_points,
                                            current_point=i_collocation,
                                        ).tolist(),
                                    )
                                    if collocation_points_initial_guesses is None:
                                        w_initial_guess.add_collocation_point(
                                            state_name,
                                            i_node,
                                            i_random,
                                            i_collocation,
                                            self.interpolate_between_nodes(
                                                var_pre=states_initial_guesses[state_name][:, i_node],
                                                var_post=states_initial_guesses[state_name][:, i_node + 1],
                                                nb_points=nb_collocation_points,
                                                current_point=i_collocation,
                                            ).tolist(),
                                        )
                                    else:
                                        w_initial_guess.add_collocation_point(
                                            state_name,
                                            i_node,
                                            i_random,
                                            i_collocation,
                                            collocation_points_initial_guesses[state_name][
                                                :, i_collocation, i_node
                                            ].tolist(),
                                        )
                                elif i_collocation == 0:
                                    # Add bounds and initial guess as linear interpolation between the two nodes
                                    w_lower_bound.add_collocation_point(
                                        state_name,
                                        i_node,
                                        i_random,
                                        i_collocation,
                                        states_lower_bounds[state_name][:, i_node].tolist(),
                                    )
                                    w_upper_bound.add_collocation_point(
                                        state_name,
                                        i_node,
                                        i_random,
                                        i_collocation,
                                        states_upper_bounds[state_name][:, i_node].tolist(),
                                    )
                                    if collocation_points_initial_guesses is None:
                                        w_initial_guess.add_collocation_point(
                                            state_name,
                                            i_node,
                                            i_random,
                                            i_collocation,
                                            states_initial_guesses[state_name][:, i_node].tolist(),
                                        )
                                    else:
                                        w_initial_guess.add_collocation_point(
                                            state_name,
                                            i_node,
                                            i_random,
                                            i_collocation,
                                            collocation_points_initial_guesses[state_name][
                                                :, i_collocation, i_node
                                            ].tolist(),
                                        )
                                else:
                                    nb_components = states_lower_bounds[state_name].shape[0]
                                    w_lower_bound.add_collocation_point(
                                        state_name, i_node, i_random, i_collocation, [0] * nb_components
                                    )
                                    w_upper_bound.add_collocation_point(
                                        state_name, i_node, i_random, i_collocation, [0] * nb_components
                                    )
                                    w_initial_guess.add_collocation_point(
                                        state_name, i_node, i_random, i_collocation, [0] * nb_components
                                    )

            # U - controls
            for control_name in controls_lower_bounds.keys():
                if i_node < ocp_example.n_shooting:
                    w_lower_bound.add_control(
                        control_name, i_node, controls_lower_bounds[control_name][:, i_node].tolist()
                    )
                    w_upper_bound.add_control(
                        control_name, i_node, controls_upper_bounds[control_name][:, i_node].tolist()
                    )
                    w_initial_guess.add_control(
                        control_name, i_node, controls_initial_guesses[control_name][:, i_node].tolist()
                    )
                else:
                    n_components = controls_lower_bounds[control_name].shape[0]
                    w_lower_bound.add_control(control_name, i_node, [0] * n_components)
                    w_upper_bound.add_control(control_name, i_node, [0] * n_components)
                    w_initial_guess.add_control(control_name, i_node, [0] * n_components)

        return w_lower_bound, w_upper_bound, w_initial_guess

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
                states[state_indices, i_random] = x[
                    offset + i_random * n_components : offset + (i_random + 1) * n_components
                ]
            offset += n_components * model.nb_random

        diff = states - states_mean
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
