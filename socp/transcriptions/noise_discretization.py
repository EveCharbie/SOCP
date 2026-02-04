import casadi as cas
import numpy as np

from .direct_collocation_polynomial import DirectCollocationPolynomial
from .discretization_abstract import DiscretizationAbstract
from .noises_abstract import NoisesAbstract
from .variables_abstract import VariablesAbstract
from .variational_polynomial import VariationalPolynomial
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

        self.dynamics_transcription = dynamics_transcription
        self.with_cholesky = False
        self.with_helper_matrix = False

    class Variables(VariablesAbstract):
        def __init__(
            self,
            n_shooting: int,
            nb_collocation_points: int,
            state_indices: dict[str, range],
            control_indices: dict[str, range],
            nb_random: int,
            with_cholesky: bool = False,
            with_helper_matrix: bool = False,
        ):
            if with_cholesky or with_helper_matrix:
                raise ValueError(
                    "The NoiseDiscretization method does not support/need the cholesky decomposition not the helper matrix."
                )

            self.n_shooting = n_shooting
            self.nb_collocation_points = nb_collocation_points
            self.nb_random = nb_random
            self.state_indices = state_indices
            self.control_indices = control_indices
            self.state_names = list(state_indices.keys())
            self.control_names = list(control_indices.keys())

            self.t = None
            self.x_list = [
                {state_name: [None for _ in range(nb_random)] for state_name in self.state_names}
                for _ in range(n_shooting + 1)
            ]
            self.z_list = [
                {
                    state_name: [[None for _ in range(nb_collocation_points)] for _ in range(nb_random)]
                    for state_name in self.state_names
                }
                for _ in range(n_shooting + 1)
            ]
            self.u_list = [{control_name: None for control_name in self.control_names} for _ in range(n_shooting + 1)]

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
                this_x = self.x_list[0][state_name][0]
                if this_x is not None:
                    nb_states += self.x_list[0][state_name][0].shape[0]
            return nb_states

        @property
        def nb_total_states(self):
            return self.nb_states * self.nb_random

        @property
        def nb_controls(self):
            nb_controls = 0
            for control_name in self.control_names:
                nb_controls += self.u_list[0][control_name].shape[0]
            return nb_controls

        # --- Get --- #
        def get_time(self):
            return self.t

        def get_state(self, name: str, node: int):
            states = None
            for i_random in range(self.nb_random):
                if states is None:
                    states = self.x_list[node][name][i_random]
                else:
                    states = cas.vertcat(states, self.x_list[node][name][i_random])
            return states

        def get_states(self, node: int):
            states = None
            for i_random in range(self.nb_random):
                for state_name in self.state_names:
                    if states is None:
                        states = self.x_list[node][state_name][i_random]
                    else:
                        states = cas.vertcat(states, self.x_list[node][state_name][i_random])
            return states

        def get_states_matrix(self, node: int):
            states_matrix = None
            for i_random in range(self.nb_random):
                states_vector = None
                for state_name in self.state_names:
                    this_state = self.x_list[node][state_name][i_random]
                    if this_state is None:
                        this_state = cas.DM.ones(self.x_list[node]["q"][i_random].shape[0]) * np.nan
                    if states_vector is None:
                        states_vector = this_state
                    else:
                        states_vector = cas.vertcat(states_vector, this_state)
                if states_matrix is None:
                    states_matrix = states_vector
                else:
                    states_matrix = cas.horzcat(states_matrix, states_vector)
            return states_matrix

        def get_state_matrix(self, name: str, node: int):
            states_matrix = None
            for i_random in range(self.nb_random):
                this_state = self.x_list[node][name][i_random]
                if this_state is None:
                    this_state = cas.DM.ones(self.x_list[node]["q"][i_random].shape[0]) * np.nan
                if states_matrix is None:
                    states_matrix = this_state
                else:
                    states_matrix = cas.horzcat(states_matrix, this_state)
            return states_matrix

        def get_specific_collocation_point(self, name: str, node: int, random: int, point: int):
            return self.z_list[node][name][random][point]

        def get_collocation_point(self, name: str, node: int):
            collocation_points_matrix = None
            for i_collocation in range(self.nb_collocation_points):
                collocation_points_vector = None
                for i_random in range(self.nb_random):
                    if collocation_points_vector is None:
                        collocation_points_vector = self.z_list[node][name][i_random][i_collocation]
                    else:
                        collocation_points_vector = cas.vertcat(
                            collocation_points_vector, self.z_list[node][name][i_random][i_collocation]
                        )
                if collocation_points_matrix is None:
                    collocation_points_matrix = collocation_points_vector
                else:
                    collocation_points_matrix = cas.horzcat(
                        collocation_points_matrix,
                        collocation_points_vector,
                    )
            return collocation_points_matrix

        def get_collocation_points(self, node: int):
            collocation_points_matrix = None
            for i_collocation in range(self.nb_collocation_points):
                collocation_points_vector = None
                for i_random in range(self.nb_random):
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
        def get_one_vector(self, node: int, keep_only_symbolic: bool = False, skip_qdot_variables: bool = False):
            nb_random = self.nb_random

            vector = []
            # X
            for i_random in range(nb_random):
                for state_name in self.state_names:
                    if node == 0 or node == self.n_shooting or not (state_name == "qdot" and skip_qdot_variables):
                        vector += [self.x_list[node][state_name][i_random]]
            # Z
            for i_random in range(nb_random):
                for i_collocation in range(self.nb_collocation_points):
                    for state_name in self.state_names:
                        if node == 0 or node == self.n_shooting or not (state_name == "qdot" and skip_qdot_variables):
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

        def get_full_vector(self, keep_only_symbolic: bool = False, skip_qdot_variables: bool = False):
            vector = []
            vector += [self.t]
            for i_node in range(self.n_shooting + 1):
                vector += [self.get_one_vector(i_node, keep_only_symbolic, skip_qdot_variables)]
            return cas.vertcat(*vector)

        def get_states_time_series_vector(self, name: str):
            n_components = self.x_list[0][name][0].shape[0]
            vector = np.zeros((n_components, self.n_shooting + 1, self.nb_random))
            for i_node in range(self.n_shooting + 1):
                for i_random in range(self.nb_random):
                    vector[:, i_node, i_random] = np.array(self.x_list[i_node][name][i_random]).flatten()
            return vector

        def get_controls_time_series_vector(self, name: str):
            n_components = self.u_list[0][name].shape[0]
            vector = np.zeros((n_components, self.n_shooting))
            for i_node in range(self.n_shooting):
                vector[:, i_node] = np.array(self.u_list[i_node][name]).flatten()
            return vector

        # --- Set vectors --- #
        def set_from_vector(self, vector: cas.DM, only_has_symbolics: bool, qdot_variables_skipped: bool):
            nb_random = self.nb_random

            offset = 0
            self.t = vector[offset]
            offset += 1

            for i_node in range(self.n_shooting + 1):
                # X
                for i_random in range(nb_random):
                    for state_name in self.state_names:
                        if (
                            i_node == 0
                            or i_node == self.n_shooting
                            or not (state_name == "qdot" and qdot_variables_skipped)
                        ):
                            n_components = self.state_indices[state_name].stop - self.state_indices[state_name].start
                            self.x_list[i_node][state_name][i_random] = vector[offset : offset + n_components]
                            offset += n_components
                # Z
                for i_random in range(nb_random):
                    for i_collocation in range(self.nb_collocation_points):
                        for state_name in self.state_names:
                            if (
                                i_node == 0
                                or i_node == self.n_shooting
                                or not (state_name == "qdot" and qdot_variables_skipped)
                            ):
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
                        this_state = np.array(self.x_list[i_node][state_name][i_random])
                        if np.all(this_state == None):
                            this_state = np.ones(self.x_list[i_node]["q"][i_random].shape) * np.nan
                        if states is None:
                            states = this_state
                        else:
                            states = np.vstack((states, this_state))
                    states_var_array[:, i_node, i_random] = states.reshape(
                        -1,
                    )
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
                    collocation_points_var_array[:, i_node, i_random] = coll.reshape(
                        -1,
                    )
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
                controls_var_array[:, i_node] = control.reshape(
                    -1,
                )
            return controls_var_array

        def validate_vector(self):
            # TODO
            pass

    class Noises(NoisesAbstract):
        def __init__(
            self,
            n_shooting: int,
            nb_random: int = 0,
        ):
            self.n_shooting = n_shooting
            self.nb_random = nb_random

            self.motor_noise = [[None for _ in range(n_shooting + 1)] for _ in range(n_shooting)]
            self.sensory_noise = [[None for _ in range(n_shooting + 1)] for _ in range(n_shooting)]
            self.motor_noises_numerical = [[None for _ in range(nb_random)] for _ in range(n_shooting + 1)]
            self.sensory_noises_numerical = [[None for _ in range(nb_random)] for _ in range(n_shooting + 1)]

        @staticmethod
        def transform_to_dm(value: cas.SX | cas.DM | np.ndarray | list) -> cas.DM:
            if isinstance(value, np.ndarray):
                return cas.DM(value.flatten())
            elif isinstance(value, list):
                return cas.DM(np.array(value).flatten())
            else:
                return value

        # --- Add --- #
        def add_motor_noise(self, index: int, random: int, value: cas.SX | cas.DM):
            self.motor_noise[index][random] = self.transform_to_dm(value)

        def add_sensory_noise(self, index: int, random: int, value: cas.SX | cas.DM):
            self.sensory_noise[index][random] = self.transform_to_dm(value)

        def add_motor_noise_numerical(self, node: int, random: int, value: cas.SX | cas.DM):
            self.motor_noises_numerical[node][random] = self.transform_to_dm(value)

        def add_sensory_noise_numerical(self, node: int, random: int, value: cas.SX | cas.DM):
            self.sensory_noises_numerical[node][random] = self.transform_to_dm(value)

        # --- Get vectors --- #
        def get_noise_single(self, index: int) -> cas.SX:
            noise_single = None
            for i_random in range(self.nb_random):
                if noise_single is None:
                    noise_single = cas.vertcat(self.motor_noise[index][i_random], self.sensory_noise[index][i_random])
                else:
                    noise_single = cas.vertcat(
                        noise_single,
                        cas.vertcat(self.motor_noise[index][i_random], self.sensory_noise[index][i_random]),
                    )
            return noise_single

        def get_one_noise_numerical(self, node: int, random: int) -> cas.DM:
            if self.motor_noises_numerical[node][random] is None:
                return self.sensory_noises_numerical[node][random]
            elif self.sensory_noises_numerical[node][random] is None:
                return self.motor_noises_numerical[node][random]
            else:
                return cas.vertcat(
                    self.motor_noises_numerical[node][random],
                    self.sensory_noises_numerical[node][random],
                )

        def get_one_vector_numerical(self, node: int) -> cas.DM:
            vector = None
            for i_random in range(self.nb_random):
                if vector is None:
                    vector = self.get_one_noise_numerical(node, i_random)
                else:
                    vector = cas.vertcat(vector, self.get_one_noise_numerical(node, i_random))
            return vector

        def get_full_matrix_numerical(self):
            vector = []
            for i_node in range(self.n_shooting + 1):
                vector += [self.get_one_vector_numerical(i_node)]
            return cas.horzcat(*vector)

    @property
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

        variables = self.Variables(
            n_shooting=n_shooting,
            nb_collocation_points=nb_collocation_points,
            state_indices=ocp_example.model.state_indices,
            control_indices=ocp_example.model.control_indices,
            nb_random=nb_random,
        )

        T = cas.SX.sym("final_time", 1)
        variables.add_time(T)

        for i_node in range(n_shooting + 1):
            for state_name in state_names:
                n_components = states_lower_bounds[state_name].shape[0]
                for i_random in range(nb_random):
                    x_sym = cas.SX.sym(f"{state_name}_{i_node}_{i_random}", n_components)
                    variables.add_state(state_name, i_node, i_random, x_sym)

                if isinstance(self.dynamics_transcription, (DirectCollocationPolynomial, VariationalPolynomial)):
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
                if i_node < n_shooting:
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

        w_lower_bound = self.Variables(
            n_shooting=n_shooting,
            nb_collocation_points=nb_collocation_points,
            state_indices=ocp_example.model.state_indices,
            control_indices=ocp_example.model.control_indices,
            nb_random=nb_random,
        )
        w_upper_bound = self.Variables(
            n_shooting=n_shooting,
            nb_collocation_points=nb_collocation_points,
            state_indices=ocp_example.model.state_indices,
            control_indices=ocp_example.model.control_indices,
            nb_random=nb_random,
        )
        w_initial_guess = self.Variables(
            n_shooting=n_shooting,
            nb_collocation_points=nb_collocation_points,
            state_indices=ocp_example.model.state_indices,
            control_indices=ocp_example.model.control_indices,
            nb_random=nb_random,
        )

        w_initial_guess.add_time(ocp_example.final_time)
        w_lower_bound.add_time(ocp_example.min_time)
        w_upper_bound.add_time(ocp_example.max_time)

        for i_node in range(n_shooting + 1):

            # X - states
            for state_name in state_names:
                for i_random in range(nb_random):
                    # Some randomness is given on the state initial guess
                    this_init = states_initial_guesses[state_name][:, i_node].tolist()
                    initial_configuration = np.array(
                        np.random.normal(
                            loc=this_init * nb_random,
                            scale=np.repeat(
                                ocp_example.initial_state_variability[ocp_example.model.state_indices[state_name]],
                                nb_random,
                            ),
                        )
                    ).reshape(len(this_init), nb_random, order="F")

                    w_lower_bound.add_state(state_name, i_node, i_random, states_lower_bounds[state_name][:, i_node])
                    w_upper_bound.add_state(state_name, i_node, i_random, states_upper_bounds[state_name][:, i_node])
                    for i_random in range(nb_random):
                        w_initial_guess.add_state(state_name, i_node, i_random, initial_configuration[:, i_random])

                # Z - collocation points
                if isinstance(self.dynamics_transcription, (DirectCollocationPolynomial, VariationalPolynomial)):
                    for state_name in state_names:
                        for i_random in range(nb_random):
                            # The last interval does not have collocation points
                            for i_collocation in range(nb_collocation_points):
                                if i_node < n_shooting:
                                    if isinstance(self.dynamics_transcription, VariationalPolynomial):
                                        z_basis = states_initial_guesses[state_name][:, i_node]
                                    else:
                                        z_basis = 0
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
                                                var_pre=states_initial_guesses[state_name][:, i_node] - z_basis,
                                                var_post=states_initial_guesses[state_name][:, i_node + 1] - z_basis,
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
                                            (collocation_points_initial_guesses[state_name][
                                                :, i_collocation, i_node
                                            ] - z_basis).tolist(),
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
                                            (states_initial_guesses[state_name][:, i_node] - z_basis).tolist(),
                                        )
                                    else:
                                        w_initial_guess.add_collocation_point(
                                            state_name,
                                            i_node,
                                            i_random,
                                            i_collocation,
                                            (collocation_points_initial_guesses[state_name][
                                                :, i_collocation, i_node
                                            ] - z_basis).tolist(),
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

    def declare_noises(
        self,
        model: ModelAbstract,
        n_shooting: int,
        nb_random: int,
        motor_noise_magnitude: np.ndarray,
        sensory_noise_magnitude: np.ndarray,
        seed: int = 0,
    ) -> NoisesAbstract:
        """
        Sample the noise values and declare the symbolic variables for the noises.
        """
        np.random.seed(seed)

        noises_vector = self.Noises(n_shooting, nb_random)

        n_motor_noises = motor_noise_magnitude.shape[0] if motor_noise_magnitude is not None else 0
        nb_references = sensory_noise_magnitude.shape[0] if sensory_noise_magnitude is not None else 0

        for i_random in range(nb_random):
            for i_index in range(n_shooting):
                noises_vector.add_motor_noise(
                    i_index, i_random, cas.SX.sym(f"motor_noise_{i_random}_{i_index}", n_motor_noises)
                )
                noises_vector.add_sensory_noise(
                    i_index, i_random, cas.SX.sym(f"sensory_noise_{i_random}_{i_index}", nb_references)
                )

            for i_node in range(n_shooting + 1):
                if n_motor_noises > 0:
                    this_motor_noise_vector = np.random.normal(
                        loc=np.zeros((model.nb_q,)),
                        scale=np.reshape(np.array(motor_noise_magnitude), (n_motor_noises,)),
                        size=n_motor_noises,
                    )
                    noises_vector.add_motor_noise_numerical(i_node, i_random, this_motor_noise_vector)

                if nb_references > 0:
                    this_sensory_noise_vector = np.random.normal(
                        loc=np.zeros((nb_references,)),
                        scale=np.reshape(np.array(sensory_noise_magnitude), (nb_references,)),
                        size=nb_references,
                    )
                    noises_vector.add_sensory_noise_numerical(i_node, i_random, this_sensory_noise_vector)

        return noises_vector

    def get_mean_states(
        self,
        variables_vector: VariablesAbstract,
        node: int,
        squared: bool = False,
    ):
        states = variables_vector.get_states_matrix(node)

        exponent = 2 if squared else 1
        states_sq = states**exponent

        states_mean = cas.sum2(states_sq) / variables_vector.nb_random
        return states_mean

    def get_covariance(
        self,
        variables_vector: VariablesAbstract,
        node: int,
        is_matrix: bool = False,
    ):
        states = variables_vector.get_states_matrix(node)
        states_mean = self.get_mean_states(variables_vector, node, squared=False)

        diff = states - states_mean
        covariance = (diff @ diff.T) / (variables_vector.nb_random - 1)

        if is_matrix:
            return covariance
        else:
            return variables_vector.reshape_matrix_to_vector(covariance)

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
        if model.nb_references > 0:
            ref = type(x).zeros(model.nb_references, 1)
            n_components = model.q_indices.stop - model.q_indices.start
            offset = n_components * model.nb_random
            for i_random in range(model.nb_random):
                q_this_time = x[i_random * n_components : (i_random + 1) * n_components]
                qdot_this_time = x[offset + i_random * n_components : offset + (i_random + 1) * n_components]
                ref += model.sensory_output(q_this_time, qdot_this_time, cas.DM.zeros(model.nb_references))

            ref /= model.nb_random
        else:
            ref = cas.DM.zeros(0, 1)
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
        for state_name, state_indices in model.state_indices.items():
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
        x: cas.SX,
        u: cas.SX,
        noise: cas.SX,
    ) -> cas.SX:

        nb_random = ocp_example.model.nb_random

        ref = self.get_reference(
            ocp_example.model,
            x,
            u,
        )

        dxdt = cas.SX.zeros(x.shape)
        states_offset = 0
        noise_offset = 0
        dxdt_offset = 0
        for i_random in range(nb_random):
            # Code looks messier, but easier to extract the casadi variables from the printed casadi expressions
            x_this_time = None
            for state_name, state_indices in ocp_example.model.state_indices.items():
                n_components = state_indices.stop - state_indices.start
                if x_this_time is None:
                    x_this_time = x[states_offset : states_offset + n_components]
                else:
                    x_this_time = cas.vertcat(x_this_time, x[states_offset : states_offset + n_components])
                states_offset += n_components

            noise_this_time = None
            for noise_indices in ocp_example.model.noise_indices:
                n_components = noise_indices.stop - noise_indices.start
                if noise_this_time is None:
                    noise_this_time = noise[noise_offset : noise_offset + n_components]
                else:
                    noise_this_time = cas.vertcat(
                        noise_this_time,
                        noise[noise_offset : noise_offset + n_components],
                    )
                noise_offset += n_components

            dxdt_this_time = ocp_example.model.dynamics(
                x_this_time,
                u,
                ref,
                noise_this_time,
            )

            for state_name, state_indices in ocp_example.model.state_indices.items():
                n_components = state_indices.stop - state_indices.start
                dxdt[dxdt_offset : dxdt_offset + n_components] = dxdt_this_time[state_indices]
                dxdt_offset += n_components

        return dxdt

    def get_non_conservative_forces(
        self,
        ocp_example: ExampleAbstract,
        q: cas.SX,
        qdot: cas.SX,
        u: cas.SX,
        noise: cas.SX,
    ) -> cas.SX:

        nb_random = ocp_example.model.nb_random
        nb_q = ocp_example.model.nb_q
        nb_noises = ocp_example.model.nb_noises

        f = cas.SX.zeros(u.shape[0] * nb_random)
        q_offset = 0
        noise_offset = 0
        f_offset = 0
        for i_random in range(nb_random):
            q_this_time = q[q_offset : q_offset + nb_q]
            qdot_this_time = qdot[q_offset : q_offset + nb_q]
            q_offset += nb_q

            noise_this_time = noise[noise_offset : noise_offset + nb_noises]
            noise_offset += nb_noises

            f_this_time = ocp_example.model.non_conservative_forces(
                q_this_time,
                qdot_this_time,
                u,
                noise_this_time,
            )

            n_components = f_this_time.shape[0]
            f[f_offset : f_offset + n_components] = f_this_time
            f_offset += n_components

        return f

    def get_lagrangian(
        self,
        ocp_example: ExampleAbstract,
        q: cas.SX,
        qdot: cas.SX,
        u: cas.SX,
    ) -> cas.SX:

        nb_random = ocp_example.model.nb_random
        nb_q = ocp_example.model.nb_q

        l = cas.SX.zeros(nb_random)
        q_offset = 0
        l_offset = 0
        for i_random in range(nb_random):
            q_this_time = q[q_offset : q_offset + nb_q]
            qdot_this_time = qdot[q_offset : q_offset + nb_q]
            q_offset += nb_q

            l_this_time = ocp_example.model.lagrangian(
                q_this_time,
                qdot_this_time,
                u,
            )

            l[l_offset : l_offset + 1] = l_this_time
            l_offset += 1

        return l

    def get_lagrangian_jacobian(self, ocp_example: ExampleAbstract, discrete_lagrangian: cas.SX, q: cas.SX):
        nb_q = ocp_example.model.nb_q
        nb_random = ocp_example.nb_random

        p = cas.SX.zeros(nb_q * nb_random)
        for i_random in range(nb_random):
            p[i_random * nb_q : (i_random + 1) * nb_q] = cas.transpose(
                cas.jacobian(
                    discrete_lagrangian[i_random],
                    q[i_random * nb_q : (i_random + 1) * nb_q],
                )
            )

        return p

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
        variable_opt,
        key,
        i_col,
        time_vector: np.ndarray,
    ) -> int:
        # TODO: Add collocation points
        states_data = variable_opt.get_states_time_series_vector(key)

        for i_random in range(ocp_example.nb_random):
            states_plots[i_state].set_ydata(states_data[i_col, :, i_random])
            i_state += 1

        return i_state
