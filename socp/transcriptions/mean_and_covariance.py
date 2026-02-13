import casadi as cas
import numpy as np

from .discretization_abstract import DiscretizationAbstract
from .noises_abstract import NoisesAbstract
from .variables_abstract import VariablesAbstract
from ..examples.example_abstract import ExampleAbstract
from ..models.model_abstract import ModelAbstract
from ..transcriptions.transcription_abstract import TranscriptionAbstract
from ..transcriptions.direct_collocation_polynomial import DirectCollocationPolynomial
from ..transcriptions.variational import Variational
from ..transcriptions.variational_polynomial import VariationalPolynomial


class MeanAndCovariance(DiscretizationAbstract):

    def __init__(
        self,
        dynamics_transcription: TranscriptionAbstract,
    ) -> None:

        super().__init__()  # Does nothing

        self.dynamics_transcription = dynamics_transcription

    class Variables(VariablesAbstract):
        def __init__(
            self,
            n_shooting: int,
            nb_collocation_points: int,
            nb_m_points: int,
            state_indices: dict[str, range],
            control_indices: dict[str, range],
            nb_random: int = 1,
        ):
            self.n_shooting = n_shooting
            self.nb_random = nb_random
            self.nb_collocation_points = nb_collocation_points
            self.nb_m_points = nb_m_points
            self.state_indices = state_indices
            self.control_indices = control_indices
            self.state_names = list(state_indices.keys())
            self.control_names = list(control_indices.keys())

            self.t = None
            self.x_list = [{state_name: None for state_name in self.state_names} for _ in range(n_shooting + 1)]
            self.cov_list = [{"cov": None} for _ in range(n_shooting + 1)]
            self.m_list = None
            self.m_list = [{"m": [None for _ in range(self.nb_m_points)]} for _ in range(n_shooting + 1)]
            self.z_list = [
                {state_name: [None for _ in range(nb_collocation_points)] for state_name in self.state_names}
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

        def add_state(self, name: str, node: int, value: cas.SX | cas.DM):
            self.x_list[node][name] = self.transform_to_dm(value)

        def add_collocation_point(self, name: str, node: int, point: int, value: cas.SX | cas.DM):
            self.z_list[node][name][point] = self.transform_to_dm(value)

        def add_cov(self, node: int, value: cas.SX | cas.DM):
            self.cov_list[node]["cov"] = self.transform_to_dm(value)

        def add_m(self, node: int, point: int, value: cas.SX | cas.DM):
            self.m_list[node]["m"][point] = self.transform_to_dm(value)

        def add_control(self, name: str, node: int, value: cas.SX | cas.DM):
            self.u_list[node][name] = self.transform_to_dm(value)

        # --- Nb --- #
        @property
        def nb_states(self):
            nb_states = 0
            for state_name in self.state_indices.keys():
                nb_states += self.state_indices[state_name].stop - self.state_indices[state_name].start
            return nb_states

        @property
        def nb_total_states(self):
            return self.nb_states

        @property
        def nb_controls(self):
            nb_controls = 0
            for control_name in self.control_names:
                nb_controls += self.u_list[0][control_name].shape[0]
            return nb_controls

        @property
        def nb_cov(self):
            nb_states = int(np.sqrt(self.cov_list[0]["cov"].shape[0]))
            return nb_states * nb_states

        # --- Get --- #
        def get_time(self):
            return self.t

        def get_state(self, name: str, node: int):
            return self.x_list[node][name]

        def get_states(self, node: int):
            states = None
            for state_name in self.state_names:
                this_state = self.x_list[node][state_name]
                if this_state is not None:
                    if states is None:
                        states = this_state
                    else:
                        states = cas.vertcat(states, this_state)
            return states

        def get_specific_collocation_point(self, name: str, node: int, point: int):
            return self.z_list[node][name][point]

        def get_collocation_point(self, name: int, node: int):
            collocation_points = None
            for i_collocation in range(self.nb_collocation_points):
                if collocation_points is None:
                    collocation_points = self.z_list[node][name][i_collocation]
                else:
                    collocation_points = cas.vertcat(collocation_points, self.z_list[node][name][i_collocation])
            return collocation_points

        def get_collocation_points(self, node: int):
            collocation_points = None
            for i_collocation in range(self.nb_collocation_points):
                for state_name in self.state_names:
                    if collocation_points is None:
                        collocation_points = self.z_list[node][state_name][i_collocation]
                    else:
                        collocation_points = cas.vertcat(
                            collocation_points, self.z_list[node][state_name][i_collocation]
                        )
            return collocation_points

        def get_cov(self, node: int):
            return self.cov_list[node]["cov"]

        def get_m(self, node: int, point: int):
            return self.m_list[node]["m"][point]

        def get_ms(self, node: int):
            m = None
            for i_collocation in range(self.nb_m_points):
                if m is None:
                    m = self.m_list[node]["m"][i_collocation]
                else:
                    m = cas.vertcat(m, self.m_list[node]["m"][i_collocation])
            return m

        def get_m_matrix(self, node: int):
            m_matrix = None
            offset = 0
            for i_collocation in range(self.nb_m_points):
                nb_states = int(np.sqrt(self.m_list[node]["m"][i_collocation].shape[0]))
                m_vector = self.m_list[node]["m"][i_collocation]
                m_matrix_i = self.reshape_vector_to_matrix(
                    m_vector,
                    (nb_states, nb_states),
                )
                if m_matrix is None:
                    m_matrix = m_matrix_i
                else:
                    m_matrix = cas.horzcat(m_matrix, m_matrix_i)
                offset += nb_states * nb_states
            return m_matrix

        def get_cov_matrix(self, node: int):
            nb_states = int(np.sqrt(self.cov_list[node]["cov"].shape[0]))
            return self.reshape_vector_to_matrix(
                self.cov_list[node]["cov"],
                (nb_states, nb_states),
            )

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
            vector = []
            # X
            for state_name in self.state_names:
                if node == 0 or node == self.n_shooting or not (state_name == "qdot" and skip_qdot_variables):
                    vector += [self.x_list[node][state_name]]
            # COV
            vector += [self.cov_list[node]["cov"]]
            # M
            for i_collocation in range(self.nb_m_points):
                if node < self.n_shooting:
                    vector += [self.m_list[node]["m"][i_collocation]]
                else:
                    if not keep_only_symbolic:
                        nb_states = int(np.sqrt(self.cov_list[node]["cov"].shape[0]))
                        vector += [cas.DM.zeros(nb_states * nb_states)]
            # Z
            for i_collocation in range(self.nb_collocation_points):
                for state_name in self.state_names:
                    if node == 0 or node == self.n_shooting or not (state_name == "qdot" and skip_qdot_variables):
                        if node < self.n_shooting:
                            vector += [self.z_list[node][state_name][i_collocation]]
                        else:
                            if not keep_only_symbolic:
                                vector += [self.z_list[node][state_name][i_collocation]]
            # U
            for control_name in self.control_names:
                vector += [self.u_list[node][control_name]]

            return cas.vertcat(*vector)

        def get_full_vector(self, keep_only_symbolic: bool = False, skip_qdot_variables: bool = False):
            vector = []
            vector += [self.t]
            for i_node in range(self.n_shooting + 1):
                vector += [self.get_one_vector(i_node, keep_only_symbolic, skip_qdot_variables)]
            return cas.vertcat(*vector)

        def get_states_time_series_vector(self, name: str):
            n_components = self.x_list[0][name].shape[0]
            vector = np.zeros((n_components, self.n_shooting + 1))
            for i_node in range(self.n_shooting + 1):
                vector[:, i_node] = np.array(self.x_list[i_node][name]).flatten()
            return vector

        def get_cov_time_series_vector(self):
            nb_states = int(np.sqrt(self.cov_list[0]["cov"].shape[0]))
            matrix = np.zeros((nb_states, nb_states, self.n_shooting + 1))
            for i_node in range(self.n_shooting + 1):
                matrix[:, :, i_node] = self.get_cov_matrix(i_node)
            return matrix

        def get_controls_time_series_vector(self, name: str):
            n_components = self.u_list[0][name].shape[0]
            vector = np.zeros((n_components, self.n_shooting + 1))
            for i_node in range(self.n_shooting + 1):
                vector[:, i_node] = np.array(self.u_list[i_node][name]).flatten()
            return vector

        # --- Set vectors --- #
        def set_from_vector(self, vector: cas.DM, only_has_symbolics: bool, qdot_variables_skipped: bool):
            offset = 0
            self.t = vector[offset]
            offset += 1

            if qdot_variables_skipped:
                nb_states = self.state_indices["q"].stop - self.state_indices["q"].start
            else:
                nb_states = self.nb_states

            for i_node in range(self.n_shooting + 1):
                # X
                for state_name in self.state_names:
                    if (
                        i_node == 0
                        or i_node == self.n_shooting
                        or not (state_name == "qdot" and qdot_variables_skipped)
                    ):
                        n_components = self.state_indices[state_name].stop - self.state_indices[state_name].start
                        self.x_list[i_node][state_name] = vector[offset : offset + n_components]
                        offset += n_components

                # COV
                nb_cov_variables = nb_states * nb_states
                self.cov_list[i_node]["cov"] = vector[offset : offset + nb_cov_variables]
                offset += nb_cov_variables

                # M
                if not only_has_symbolics or i_node < self.n_shooting:
                    for i_collocation in range(self.nb_m_points):
                        nb_m_variables = nb_states * nb_states
                        self.m_list[i_node]["m"][i_collocation] = vector[offset : offset + nb_m_variables]
                        offset += nb_m_variables

                # Z
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
                                self.z_list[i_node][state_name][i_collocation] = vector[offset : offset + n_components]
                                offset += n_components

                # U
                for control_name in self.control_names:
                    n_components = (
                        self.control_indices[control_name].stop - self.control_indices[control_name].start
                    )
                    self.u_list[i_node][control_name] = vector[offset : offset + n_components]
                    offset += n_components

        # --- Get array --- #
        def get_states_array(self) -> np.ndarray:
            states_var_array = np.zeros((self.nb_states, self.n_shooting + 1))
            for i_node in range(self.n_shooting + 1):
                for state_name in self.state_names:
                    states = np.array(self.x_list[i_node][state_name])
                    states_var_array[self.state_indices[state_name], i_node] = states.reshape(
                        -1,
                    )
            return states_var_array

        def get_cov_array(self) -> np.ndarray:
            nb_states = int(np.sqrt(self.cov_list[0]["cov"].shape[0]))
            cov_var_array = np.zeros((nb_states * nb_states, self.n_shooting + 1))
            for i_node in range(self.n_shooting + 1):
                cov_var_array[:, i_node] = self.cov_list[i_node]["cov"].reshape(
                    -1,
                )
            return cov_var_array

        def get_m_array(self) -> np.ndarray:
            nb_states = int(np.sqrt(self.cov_list[0]["cov"].shape[0]))
            m_var_array = np.zeros((nb_states * nb_states * self.nb_m_points, self.n_shooting + 1))
            for i_node in range(self.n_shooting + 1):
                m = None
                for i_collocation in range(self.nb_m_points):
                    if m is None:
                        m = np.array(self.m_list[i_node]["m"][i_collocation])
                    else:
                        m = np.vstack((m, self.m_list[i_node]["m"][i_collocation]))
                m_var_array[:, i_node] = m
            return m_var_array

        def get_collocation_points_array(self) -> np.ndarray:
            nb_states = int(np.sqrt(self.cov_list[0]["cov"].shape[0]))
            collocation_points_var_array = np.zeros((nb_states * self.nb_collocation_points, self.n_shooting + 1))
            for i_node in range(self.n_shooting + 1):
                coll = None
                for i_collocation in range(self.nb_collocation_points):
                    for state_name in self.state_names:
                        if coll is None:
                            coll = np.array(self.z_list[i_node][state_name][i_collocation])
                        else:
                            coll = np.hstack((coll, self.z_list[i_node][state_name][i_collocation]))
                        print(coll)
                collocation_points_var_array[:, i_node] = coll
            return collocation_points_var_array

        def get_controls_array(self) -> np.ndarray:
            controls_var_array = np.zeros((self.nb_controls, self.n_shooting + 1))
            for i_node in range(self.n_shooting + 1):
                for control_name in self.control_names:
                    control = np.array(self.u_list[i_node][control_name])
                    controls_var_array[self.control_indices[control_name], i_node] = control.reshape(
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
            nb_random: int = 1,
        ):
            self.n_shooting = n_shooting

            self.motor_noise = [None for _ in range(n_shooting + 1)]
            self.sensory_noise = [None for _ in range(n_shooting + 1)]
            self.motor_noises_numerical = [None for _ in range(n_shooting + 1)]
            self.sensory_noises_numerical = [None for _ in range(n_shooting + 1)]

        @staticmethod
        def transform_to_dm(value: cas.SX | cas.DM | np.ndarray | list) -> cas.DM:
            if isinstance(value, np.ndarray):
                return cas.DM(value.flatten())
            elif isinstance(value, list):
                return cas.DM(np.array(value).flatten())
            else:
                return value

        # --- Add --- #
        def add_motor_noise(self, node: int, value: cas.SX | cas.DM):
            self.motor_noise[node] = self.transform_to_dm(value)

        def add_sensory_noise(self, node: int, value: cas.SX | cas.DM):
            self.sensory_noise[node] = self.transform_to_dm(value)

        def add_motor_noise_numerical(self, node: int, value: cas.SX | cas.DM):
            self.motor_noises_numerical[node] = self.transform_to_dm(value)

        def add_sensory_noise_numerical(self, node: int, value: cas.SX | cas.DM):
            self.sensory_noises_numerical[node] = self.transform_to_dm(value)

        # --- Get vectors --- #
        def get_noise_single(self, node: int) -> cas.SX:
            return cas.vertcat(self.motor_noise[node], self.sensory_noise[node])

        def get_one_vector_numerical(self, node: int):
            if self.motor_noises_numerical[node] is None:
                return self.sensory_noises_numerical[node]
            elif self.sensory_noises_numerical[node] is None:
                return self.motor_noises_numerical[node]
            else:
                return cas.vertcat(self.motor_noises_numerical[node], self.sensory_noises_numerical[node])

        def get_full_matrix_numerical(self):
            vector = []
            for i_node in range(self.n_shooting + 1):
                vector += [self.get_one_vector_numerical(i_node)]
            return cas.horzcat(*vector)

        def get_noises_array(self) -> np.ndarray:
            nb_noises = 0
            if self.motor_noises_numerical[0] is not None:
                nb_noises += self.motor_noises_numerical[0].shape[0]
            if self.sensory_noises_numerical[0]is not None:
                nb_noises += self.sensory_noises_numerical[0].shape[0]

            noises_array = np.zeros((nb_noises, self.n_shooting + 1))
            for i_node in range(self.n_shooting + 1):
                if self.motor_noises_numerical[0] is not None and self.sensory_noises_numerical[0] is not None:
                    noises_array[:, i_node] = np.hstack((
                        np.array(self.motor_noises_numerical[i_node]).reshape(-1, ),
                        np.array(self.sensory_noises_numerical[i_node]).reshape(-1, ),
                    ))
                elif self.motor_noises_numerical[0] is not None:
                    noises_array[:, i_node] = np.array(self.motor_noises_numerical[i_node]).reshape(-1, )
                elif self.sensory_noises_numerical[0] is not None:
                    noises_array[:, i_node] = np.array(self.sensory_noises_numerical[i_node]).reshape(-1, )
                else:
                    raise RuntimeError("At least motor or sensory noise should be included to the problem if you want to solve a SOCP.")

            return noises_array

    @property
    def name(self) -> str:
        return "MeanAndCovariance"

    def declare_variables(
        self,
        ocp_example: ExampleAbstract,
        states_lower_bounds: dict[str, np.ndarray],
        controls_lower_bounds: dict[str, np.ndarray],
    ) -> Variables:
        """
        Declare all symbolic variables for the states and controls with their bounds and initial guesses
        """
        nb_states = ocp_example.model.nb_states
        n_shooting = ocp_example.n_shooting
        nb_collocation_points = self.dynamics_transcription.nb_collocation_points
        nb_m_points = self.dynamics_transcription.nb_m_points
        state_names = list(ocp_example.model.state_indices.keys())
        control_names = list(ocp_example.model.control_indices.keys())

        variables = self.Variables(
            n_shooting=n_shooting,
            nb_collocation_points=nb_collocation_points,
            nb_m_points=nb_m_points,
            state_indices=ocp_example.model.state_indices,
            control_indices=ocp_example.model.control_indices,
        )
        nb_m_points = variables.nb_m_points

        T = cas.SX.sym("final_time", 1)
        variables.add_time(T)

        for i_node in range(n_shooting + 1):
            for state_name in state_names:
                # X
                n_components = states_lower_bounds[state_name].shape[0]
                mean_x = cas.SX.sym(f"{state_name}_{i_node}", n_components)
                variables.add_state(state_name, i_node, mean_x)

                # Z
                if isinstance(
                    self.dynamics_transcription, (DirectCollocationPolynomial, Variational, VariationalPolynomial)
                ):
                    # Create the symbolic variables for the mean states collocation points
                    for i_collocation in range(nb_collocation_points):
                        if i_node < n_shooting:
                            mean_z = cas.SX.sym(f"{state_name}_{i_node}_{i_collocation}_z", n_components)
                        else:
                            mean_z = cas.SX.zeros(n_components)
                        variables.add_collocation_point(state_name, i_node, i_collocation, mean_z)

            # Create the symbolic variables for the state covariance
            if isinstance(self.dynamics_transcription, (Variational, VariationalPolynomial)):
                nb_cov_variables = ocp_example.model.nb_q * ocp_example.model.nb_q
            else:
                nb_cov_variables = nb_states * nb_states
            cov = cas.SX.sym(f"cov_{i_node}", nb_cov_variables)
            variables.add_cov(i_node, cov)

            # Create the symbolic variables for the helper matrix
            for i_collocation in range(nb_m_points):
                if i_node < n_shooting:
                    m = cas.SX.sym(f"m_{i_node}_{i_collocation}", nb_cov_variables)
                else:
                    m = cas.SX.zeros(nb_cov_variables)
                variables.add_m(i_node, i_collocation, m)

            # Controls
            for control_name in controls_lower_bounds.keys():
                n_components = controls_lower_bounds[control_name].shape[0]
                u = cas.SX.sym(f"{control_name}_{i_node}", n_components)
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
        if isinstance(self.dynamics_transcription, (Variational, VariationalPolynomial)):
            nb_states = ocp_example.model.nb_q
        else:
            nb_states = ocp_example.model.nb_states

        n_shooting = ocp_example.n_shooting
        nb_collocation_points = self.dynamics_transcription.nb_collocation_points
        nb_m_points = self.dynamics_transcription.nb_m_points
        state_names = list(ocp_example.model.state_indices.keys())
        control_names = list(ocp_example.model.control_indices.keys())

        w_lower_bound = self.Variables(
            n_shooting=n_shooting,
            nb_collocation_points=nb_collocation_points,
            nb_m_points=nb_m_points,
            state_indices=ocp_example.model.state_indices,
            control_indices=ocp_example.model.control_indices,
        )
        w_upper_bound = self.Variables(
            n_shooting=n_shooting,
            nb_collocation_points=nb_collocation_points,
            nb_m_points=nb_m_points,
            state_indices=ocp_example.model.state_indices,
            control_indices=ocp_example.model.control_indices,
        )
        w_initial_guess = self.Variables(
            n_shooting=n_shooting,
            nb_collocation_points=nb_collocation_points,
            nb_m_points=nb_m_points,
            state_indices=ocp_example.model.state_indices,
            control_indices=ocp_example.model.control_indices,
        )

        w_initial_guess.add_time(ocp_example.final_time)
        w_lower_bound.add_time(ocp_example.min_time)
        w_upper_bound.add_time(ocp_example.max_time)

        for i_node in range(n_shooting + 1):

            # X - states
            for state_name in state_names:
                w_lower_bound.add_state(state_name, i_node, states_lower_bounds[state_name][:, i_node])
                w_upper_bound.add_state(state_name, i_node, states_upper_bounds[state_name][:, i_node])
                w_initial_guess.add_state(state_name, i_node, states_initial_guesses[state_name][:, i_node])

            # COV - covariance
            cov_init = np.diag(ocp_example.initial_state_variability.tolist())
            # Declare cov variables
            nb_cov_variables = nb_states * nb_states
            p_init = (
                np.array(w_initial_guess.reshape_matrix_to_vector(cov_init[:nb_states, :nb_states])).flatten().tolist()
            )

            w_initial_guess.add_cov(i_node, p_init)
            w_lower_bound.add_cov(i_node, [-cas.inf] * nb_cov_variables)
            w_upper_bound.add_cov(i_node, [cas.inf] * nb_cov_variables)

            # M - Helper matrix
            n_components = nb_states * nb_states
            for i_collocation in range(nb_m_points):
                if i_node < n_shooting:
                    if "m" in states_initial_guesses.keys():
                        w_initial_guess.add_m(
                            i_node,
                            i_collocation,
                            w_initial_guess.reshape_matrix_to_vector(
                                states_initial_guesses["m"][:, :, i_collocation, i_node],
                            ),
                        )
                    else:
                        w_initial_guess.add_m(i_node, i_collocation, [0.01] * n_components)
                    w_lower_bound.add_m(i_node, i_collocation, [-cas.inf] * n_components)
                    w_upper_bound.add_m(i_node, i_collocation, [cas.inf] * n_components)
                else:
                    w_initial_guess.add_m(i_node, i_collocation, [0.0] * n_components)
                    w_lower_bound.add_m(i_node, i_collocation, [0.0] * n_components)
                    w_upper_bound.add_m(i_node, i_collocation, [0.0] * n_components)

            # Z - collocation points
            if isinstance(
                self.dynamics_transcription, (DirectCollocationPolynomial, Variational, VariationalPolynomial)
            ):
                for state_name in state_names:
                    # The last interval does not have collocation points
                    for i_collocation in range(nb_collocation_points):
                        if i_node < n_shooting:
                            # Add bounds and initial guess as linear interpolation between the two nodes
                            w_lower_bound.add_collocation_point(
                                state_name,
                                i_node,
                                i_collocation,
                                self.interpolate_between_nodes(
                                    var_pre=states_lower_bounds[state_name][:, i_node],
                                    var_post=states_lower_bounds[state_name][:, i_node + 1],
                                    time_ratio=i_collocation / (nb_collocation_points - 1),
                                ).tolist(),
                            )
                            w_upper_bound.add_collocation_point(
                                state_name,
                                i_node,
                                i_collocation,
                                self.interpolate_between_nodes(
                                    var_pre=states_upper_bounds[state_name][:, i_node],
                                    var_post=states_upper_bounds[state_name][:, i_node + 1],
                                    time_ratio=i_collocation / (nb_collocation_points - 1)
                                ).tolist(),
                            )
                            if collocation_points_initial_guesses is None:
                                w_initial_guess.add_collocation_point(
                                    state_name,
                                    i_node,
                                    i_collocation,
                                    self.interpolate_between_nodes(
                                        var_pre=states_initial_guesses[state_name][:, i_node],
                                        var_post=states_initial_guesses[state_name][:, i_node + 1],
                                        time_ratio=i_collocation / (nb_collocation_points - 1)
                                    ).tolist(),
                                )
                            else:
                                w_initial_guess.add_collocation_point(
                                    state_name,
                                    i_node,
                                    i_collocation,
                                    collocation_points_initial_guesses[state_name][:, i_collocation, i_node].tolist(),
                                )
                        elif i_collocation == 0:
                            # Add bounds and initial guess as linear interpolation between the two nodes
                            w_lower_bound.add_collocation_point(
                                state_name,
                                i_node,
                                i_collocation,
                                states_lower_bounds[state_name][:, i_node].tolist(),
                            )
                            w_upper_bound.add_collocation_point(
                                state_name,
                                i_node,
                                i_collocation,
                                states_upper_bounds[state_name][:, i_node].tolist(),
                            )
                            if collocation_points_initial_guesses is None:
                                w_initial_guess.add_collocation_point(
                                    state_name,
                                    i_node,
                                    i_collocation,
                                    states_initial_guesses[state_name][:, i_node].tolist(),
                                )
                            else:
                                w_initial_guess.add_collocation_point(
                                    state_name,
                                    i_node,
                                    i_collocation,
                                    collocation_points_initial_guesses[state_name][:, i_collocation, i_node].tolist(),
                                )
                        else:
                            nb_components = states_lower_bounds[state_name].shape[0]
                            w_lower_bound.add_collocation_point(state_name, i_node, i_collocation, [0] * nb_components)
                            w_upper_bound.add_collocation_point(state_name, i_node, i_collocation, [0] * nb_components)
                            w_initial_guess.add_collocation_point(
                                state_name, i_node, i_collocation, [0] * nb_components
                            )

            # U - controls
            for control_name in controls_lower_bounds.keys():
                w_lower_bound.add_control(
                    control_name, i_node, controls_lower_bounds[control_name][:, i_node].tolist()
                )
                w_upper_bound.add_control(
                    control_name, i_node, controls_upper_bounds[control_name][:, i_node].tolist()
                )
                w_initial_guess.add_control(
                    control_name, i_node, controls_initial_guesses[control_name][:, i_node].tolist()
                )

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

        noises_vector = self.Noises(n_shooting)
        n_motor_noises = motor_noise_magnitude.shape[0] if motor_noise_magnitude is not None else 0
        nb_references = sensory_noise_magnitude.shape[0] if sensory_noise_magnitude is not None else 0

        for i_node in range(n_shooting + 1):
            if motor_noise_magnitude is not None:
                noises_vector.add_motor_noise_numerical(i_node, motor_noise_magnitude.tolist())
            if sensory_noise_magnitude is not None:
                noises_vector.add_sensory_noise_numerical(i_node, sensory_noise_magnitude.tolist())

        for i_node in range(n_shooting + 1):
            noises_vector.add_motor_noise(i_node, cas.SX.sym(f"motor_noise_{i_node}", n_motor_noises))
            noises_vector.add_sensory_noise(i_node, cas.SX.sym(f"sensory_noise_{i_node}", nb_references))

        return noises_vector

    def initialize_m(
        self,
        ocp_example: ExampleAbstract,
        vector_initial_guess: VariablesAbstract,
        jacobian_funcs: cas.Function,
    ) -> None:

        if isinstance(self.dynamics_transcription, (Variational, VariationalPolynomial)):
            nb_states = ocp_example.model.nb_q
        else:
            nb_states = ocp_example.model.nb_states

        n_shooting = ocp_example.n_shooting
        nb_m_points = self.dynamics_transcription.nb_m_points

        for i_node in range(n_shooting):
            _, dg_dz, _, df_dz = jacobian_funcs(
                vector_initial_guess.get_time(),
                vector_initial_guess.get_states(i_node),
                vector_initial_guess.get_collocation_points(i_node),
                vector_initial_guess.get_controls(i_node),
                vector_initial_guess.get_controls(i_node+1),
                np.zeros((ocp_example.model.nb_noises,)),
            )

            m_this_time = df_dz @ np.linalg.inv(dg_dz)

            for i_collocation in range(nb_m_points):
                m_vector = vector_initial_guess.reshape_matrix_to_vector(
                    m_this_time[:, i_collocation * nb_states : (i_collocation + 1) * nb_states]
                )
                vector_initial_guess.add_m(i_node, i_collocation, m_vector)

        # Wrong, but necessary since we do not have the collocation points at the last node
        for i_collocation in range(nb_m_points):
            vector_initial_guess.add_m(
                n_shooting,
                i_collocation,
                m_vector,
            )
        return

    def modify_init(
        self,
        ocp_example: ExampleAbstract,
        vector_initial_guess: VariablesAbstract,
    ):
        """
        Modify bounds and initial guesses if needed.
        This is needed when the bounds and init from one variable depend on the dynamics of the system.
        """
        if isinstance(
            self.dynamics_transcription,
            DirectCollocationPolynomial,
        ):
            self.initialize_m(
                ocp_example=ocp_example,
                vector_initial_guess=vector_initial_guess,
                jacobian_funcs=self.dynamics_transcription.jacobian_funcs,
            )
        return

    def get_mean_states(
        self,
        variables_vector: VariablesAbstract,
        node: int,
        squared: bool = False,
    ):
        exponent = 2 if squared else 1
        states_mean = variables_vector.get_states(node) ** exponent

        return states_mean

    def get_covariance(
        self,
        variables_vector: VariablesAbstract,
        node: int,
        is_matrix: bool = False,
    ):
        if is_matrix:
            cov = variables_vector.get_cov_matrix(node)
        else:
            cov = variables_vector.get_cov(node)
        return cov

    # def get_states_variance(
    #     self,
    #     model: ModelAbstract,
    #     x,
    #     squared: bool = False,
    # ):
    #     exponent = 2 if squared else 1
    #
    #     offset = model.state_indices[-1].stop
    #     nb_components = model.nb_states
    #     cov = x[offset: offset + nb_components * nb_components] * exponent
    #
    #     return cov

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
        n_components = model.q_indices.stop - model.q_indices.start
        q = x[:n_components]
        qdot = x[n_components : 2 * n_components]
        ref = model.sensory_output(q, qdot, cas.DM.zeros(model.nb_references))
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
        # Create temporary symbolic variables and functions
        if isinstance(self.dynamics_transcription, (Variational, VariationalPolynomial)):
            nb_states = model.nb_q
        else:
            nb_states = model.nb_states

        q = cas.SX.sym("q", model.nb_q)
        qdot = cas.SX.sym("qdot", model.nb_q)

        # No noise for mean
        dee_dq = cas.jacobian(
            model.sensory_output(q, qdot, cas.DM.zeros(model.nb_references)),
            q,
        )
        nb_cov_variables = nb_states * nb_states
        covariance = cas.SX.sym("cov", nb_cov_variables)
        cov = model.reshape_vector_to_matrix(
            covariance,
            (nb_states, nb_states),
        )

        end_effector_covariance = dee_dq @ cov[model.q_indices, model.q_indices] @ cas.transpose(dee_dq)
        end_effector_covariance_func = cas.Function(
            "end_effector_covariance_func",
            [q, qdot, covariance],
            [end_effector_covariance[0, 0], end_effector_covariance[1, 1]],
            ["q", "qdot", "covariance"],
            ["end_effector_covariance_x", "end_effector_covariance_y"],
        )
        end_effector_covariance_eval_x, end_effector_covariance_eval_y = end_effector_covariance_func(
            x[model.q_indices],  # Q
            x[model.qdot_indices],  # Qdot
            x[nb_states : nb_states + nb_cov_variables],  # Cov
        )

        return end_effector_covariance_eval_x, end_effector_covariance_eval_y

    def get_mus_variance(
        self,
        model: ModelAbstract,
        x,
    ):
        if isinstance(self.dynamics_transcription, (Variational, VariationalPolynomial)):
            nb_states = model.nb_q
        else:
            nb_states = model.nb_states

        state_names = list(model.state_indices.keys())
        offset = model.state_indices[state_names[-1]].stop
        nb_components = nb_states * nb_states
        cov = x[offset : offset + nb_components]
        cov_matrix = model.reshape_vector_to_matrix(
            cov,
            (nb_states, nb_states),
        )
        sum_variations = cas.trace(cov_matrix[model.muscle_activation_indices, model.muscle_activation_indices])
        return sum_variations

    def state_dynamics(
        self,
        example_ocp: ExampleAbstract,
        x,
        u,
        noise,
    ) -> cas.SX:

        if isinstance(self.dynamics_transcription, (Variational, VariationalPolynomial)):
            nb_states = example_ocp.model.nb_q
        else:
            nb_states = example_ocp.model.nb_states

        # Mean state
        ref_mean = self.get_reference(
            example_ocp.model,
            x,
            u,
        )
        dxdt_mean = example_ocp.model.dynamics(
            x[:nb_states],
            u,
            ref_mean,
            noise,
        )

        return dxdt_mean

    def get_non_conservative_forces(
        self,
        ocp_example: ExampleAbstract,
        q: cas.SX,
        qdot: cas.SX,
        u: cas.SX,
        noise: cas.SX,
    ) -> cas.SX:
        f = ocp_example.model.non_conservative_forces(
            q,
            qdot,
            u,
            noise,
        )
        return f

    def get_lagrangian(
        self,
        ocp_example: ExampleAbstract,
        q: cas.SX,
        qdot: cas.SX,
        u: cas.SX,
    ) -> cas.SX:
        l = ocp_example.model.lagrangian(
            q,
            qdot,
            u,
        )
        return l

    def get_lagrangian_func(
        self,
        ocp_example: ExampleAbstract,
        q_shape: int,
        qdot_shape: int,
        u_shape: int,
    ) -> tuple[cas.Function, dict[str, cas.SX.sym]]:

        q = cas.SX.sym("q", q_shape)
        qdot = cas.SX.sym("qdot", qdot_shape)
        u = cas.SX.sym("u", u_shape)
        variables = {
            "q": q,
            "qdot": qdot,
            "u": u,
        }

        l = ocp_example.model.lagrangian(
            q,
            qdot,
            u,
        )
        l_func = cas.Function("Lagrangian", [q, qdot, u], [l])

        return l_func, variables

    def get_lagrangian_jacobian(self, ocp_example: ExampleAbstract, discrete_lagrangian: cas.SX, q: cas.SX):
        p = cas.transpose(
            cas.jacobian(
                discrete_lagrangian,
                q,
            )
        )
        return p

    def get_momentum(
        self,
        ocp_example: ExampleAbstract,
        q: cas.SX,
        qdot: cas.SX,
        u: cas.SX,
    ) -> cas.SX:
        p = ocp_example.model.mometum(
            q,
            qdot,
            u,
        )
        return p

    def create_state_plots(
        self,
        ocp_example: ExampleAbstract,
        colors,
        axs,
        i_row,
        i_col,
        time_vector: np.ndarray,
    ):
        states_plots = []
        # Placeholder to plot the variables
        color = "tab:blue"
        states_plots += axs[i_row, i_col].plot(time_vector, np.zeros_like(time_vector), marker=".", color=color)
        # states_plots += [
        #     axs[i_row, i_col].fill_between(
        #         time_vector,
        #         np.zeros_like(time_vector),
        #         np.zeros_like(time_vector),
        #         color=color,
        #         alpha=0.3,
        #     )
        # ]
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

        states_data = variable_opt.get_states_time_series_vector(key)[i_col, :]

        # Update mean state plot
        states_plots[i_state].set_ydata(
            states_data,
        )
        i_state += 1

        # # Update covariance fill
        # cov = variable_opt.get_cov_time_series_vector()[i_col, i_col, :]
        # verts = np.vstack(
        #     [
        #         np.column_stack([time_vector, states_data - np.sqrt(np.abs(cov))]),
        #         np.column_stack([time_vector[::-1], (states_data + np.sqrt(np.abs(cov)))[::-1]]),
        #     ]
        # )
        # states_plots[i_state].get_paths()[0].vertices[i_col][:] = verts
        # i_state += 1

        return i_state
