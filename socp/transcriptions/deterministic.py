import casadi as cas
import numpy as np

from .direct_collocation_polynomial import DirectCollocationPolynomial
from .discretization_abstract import DiscretizationAbstract
from .noises_abstract import NoisesAbstract
from .variables_abstract import VariablesAbstract
from .variational import Variational
from .variational_polynomial import VariationalPolynomial
from ..examples.example_abstract import ExampleAbstract
from ..models.biorbd_model import cache_function


class Deterministic(DiscretizationAbstract):

    def __init__(
        self,
        dynamics_transcription: DiscretizationAbstract,
    ) -> None:

        super().__init__()
        self.dynamics_transcription = dynamics_transcription

    class Variables(VariablesAbstract):
        def __init__(
            self,
            n_shooting: int,
            nb_collocation_points: int,
            state_indices: dict[str, range],
            control_indices: dict[str, range],
            nb_m_points: int = 0,
            nb_random: int = 1,
        ):

            if nb_m_points != 0 or nb_random != 1:
                raise RuntimeError(f"Something went wrong, nb_m_points ({nb_m_points}) != 0 or nb_random ({nb_random}) != 1")

            self.nb_random = 1
            self.n_shooting = n_shooting
            self.nb_collocation_points = nb_collocation_points
            self.state_indices = state_indices
            self.control_indices = control_indices
            self.state_names = list(state_indices.keys())
            self.control_names = list(control_indices.keys())

            self.t = None
            self.x_list = [{state_name: None for state_name in self.state_names} for _ in range(n_shooting + 1)]
            self.z_list = [
                {state_name: [None for _ in range(nb_collocation_points)] for state_name in self.state_names}
                for _ in range(n_shooting + 1)
            ]
            self.u_list = [{control_name: None for control_name in self.control_names} for _ in range(n_shooting + 1)]

        # --- Add --- #
        def add_time(self, value: cas.MX | cas.SX | cas.DM):
            self.t = self.transform_to_dm(value)

        def add_state(self, name: str, node: int, value: cas.MX | cas.SX | cas.DM):
            self.x_list[node][name] = self.transform_to_dm(value)

        def add_collocation_point(self, name: str, node: int, point: int, value: cas.MX | cas.SX | cas.DM):
            self.z_list[node][name][point] = self.transform_to_dm(value)

        def add_control(self, name: str, node: int, value: cas.MX | cas.SX | cas.DM):
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
                    n_components = self.control_indices[control_name].stop - self.control_indices[control_name].start
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

        def get_collocation_points_array(self) -> np.ndarray:
            collocation_points_var_array = np.zeros(
                (self.nb_states * self.nb_collocation_points, self.n_shooting + 1)
            )
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
        ):
            self.n_shooting = n_shooting

            self.motor_noise = [None for _ in range(n_shooting + 1)]
            self.sensory_noise = [None for _ in range(n_shooting + 1)]
            self.motor_noises_numerical = [cas.DM() for _ in range(n_shooting + 1)]
            self.sensory_noises_numerical = [cas.DM() for _ in range(n_shooting + 1)]

        # --- Add --- #
        def add_motor_noise(self, node: int, value: cas.MX | cas.SX | cas.DM):
            self.motor_noise[node] = self.transform_to_dm(value)

        def add_sensory_noise(self, node: int, value: cas.MX | cas.SX | cas.DM):
            self.sensory_noise[node] = self.transform_to_dm(value)

        # --- Get vectors --- #
        def get_noise_single(self, node: int) -> cas.MX | cas.SX:
            return cas.vertcat(self.motor_noise[node], self.sensory_noise[node])

        def get_sensory_noise(self, node: int) -> cas.MX | cas.SX:
            return self.sensory_noise[node]

        def get_motor_noise(self, node: int) -> cas.MX | cas.SX:
            return self.motor_noise[node]

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
            if self.sensory_noises_numerical[0] is not None:
                nb_noises += self.sensory_noises_numerical[0].shape[0]

            noises_array = np.zeros((nb_noises, self.n_shooting + 1))
            for i_node in range(self.n_shooting + 1):
                if self.motor_noises_numerical[0] is not None and self.sensory_noises_numerical[0] is not None:
                    noises_array[:, i_node] = np.hstack(
                        (
                            np.array(self.motor_noises_numerical[i_node]).reshape(
                                -1,
                            ),
                            np.array(self.sensory_noises_numerical[i_node]).reshape(
                                -1,
                            ),
                        )
                    )
                elif self.motor_noises_numerical[0] is not None:
                    noises_array[:, i_node] = np.array(self.motor_noises_numerical[i_node]).reshape(
                        -1,
                    )
                elif self.sensory_noises_numerical[0] is not None:
                    noises_array[:, i_node] = np.array(self.sensory_noises_numerical[i_node]).reshape(
                        -1,
                    )
                else:
                    raise RuntimeError(
                        "At least motor or sensory noise should be included to the problem if you want to solve a SOCP."
                    )

            return noises_array

    @property
    def name(self) -> str:
        return "Deterministic"

    def declare_variables(
        self,
        ocp_example: ExampleAbstract,
        states_lower_bounds: dict[str, np.ndarray],
        controls_lower_bounds: dict[str, np.ndarray],
    ) -> Variables:
        """
        Declare all symbolic variables for the states and controls with their bounds and initial guesses
        """
        n_shooting = ocp_example.n_shooting
        nb_collocation_points = self.dynamics_transcription.nb_collocation_points
        state_names = list(ocp_example.model.state_indices.keys())

        variables = self.Variables(
            n_shooting=n_shooting,
            nb_collocation_points=nb_collocation_points,
            state_indices=ocp_example.model.state_indices,
            control_indices=ocp_example.model.control_indices,
        )

        use_sx = ocp_example.model.use_sx
        T = cas.SX.sym("final_time", 1) if use_sx else cas.MX.sym("final_time", 1)
        variables.add_time(T)

        if isinstance(self.dynamics_transcription, (Variational, VariationalPolynomial)):
            skip_qdot_variables = True
        else:
            skip_qdot_variables = False

        for i_node in range(n_shooting + 1):
            for state_name in state_names:
                if i_node == 0 or i_node == n_shooting or not (state_name == "qdot" and skip_qdot_variables):
                    n_components = states_lower_bounds[state_name].shape[0]
                    if use_sx:
                        x_sym = cas.SX.sym(f"{state_name}_{i_node}", n_components)
                    else:
                        x_sym = cas.MX.sym(f"{state_name}_{i_node}", n_components)
                    variables.add_state(state_name, i_node, x_sym)

                if isinstance(self.dynamics_transcription, (DirectCollocationPolynomial, VariationalPolynomial)):
                    # Create the symbolic variables for the states collocation points
                    if not (state_name == "qdot" and skip_qdot_variables):
                        for i_collocation in range(nb_collocation_points):
                            if i_node < n_shooting:
                                if use_sx:
                                    z_sym = cas.SX.sym(
                                        f"{state_name}_{i_node}_{i_collocation}_z", n_components
                                    )
                                else:
                                    z_sym = cas.MX.sym(
                                        f"{state_name}_{i_node}_{i_collocation}_z", n_components
                                    )
                            else:
                                if use_sx:
                                    z_sym = cas.SX.zeros(n_components)
                                else:
                                    z_sym = cas.MX.zeros(n_components)
                            variables.add_collocation_point(state_name, i_node, i_collocation, z_sym)

            # Controls
            for control_name in controls_lower_bounds.keys():
                n_components = controls_lower_bounds[control_name].shape[0]
                if use_sx:
                    u = cas.SX.sym(f"{control_name}_{i_node}", n_components)
                else:
                    u = cas.MX.sym(f"{control_name}_{i_node}", n_components)
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
        nb_m_points = self.dynamics_transcription.nb_m_points
        state_names = list(ocp_example.model.state_indices.keys())

        w_lower_bound = self.Variables(
            n_shooting=n_shooting,
            nb_collocation_points=nb_collocation_points,
            state_indices=ocp_example.model.state_indices,
            control_indices=ocp_example.model.control_indices,
        )
        w_upper_bound = self.Variables(
            n_shooting=n_shooting,
            nb_collocation_points=nb_collocation_points,
            state_indices=ocp_example.model.state_indices,
            control_indices=ocp_example.model.control_indices,
        )
        w_initial_guess = self.Variables(
            n_shooting=n_shooting,
            nb_collocation_points=nb_collocation_points,
            state_indices=ocp_example.model.state_indices,
            control_indices=ocp_example.model.control_indices,
        )

        w_initial_guess.add_time(ocp_example.final_time)
        w_lower_bound.add_time(ocp_example.min_time)
        w_upper_bound.add_time(ocp_example.max_time)

        for i_node in range(n_shooting + 1):

            # X - states
            for state_name in state_names:

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

                w_lower_bound.add_state(state_name, i_node, states_lower_bounds[state_name][:, i_node])
                w_upper_bound.add_state(state_name, i_node, states_upper_bounds[state_name][:, i_node])
                w_initial_guess.add_state(state_name, i_node, this_init)

                # Z - collocation points
                if isinstance(self.dynamics_transcription, (DirectCollocationPolynomial, VariationalPolynomial)):
                    for state_name in state_names:
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
                                        time_ratio=i_collocation / (nb_collocation_points - 1),
                                    ).tolist(),
                                )
                                if collocation_points_initial_guesses is None:
                                    w_initial_guess.add_collocation_point(
                                        state_name,
                                        i_node,
                                        i_collocation,
                                        self.interpolate_between_nodes(
                                            var_pre=states_initial_guesses[state_name][:, i_node] - z_basis,
                                            var_post=states_initial_guesses[state_name][:, i_node + 1] - z_basis,
                                            time_ratio=i_collocation / (nb_collocation_points - 1),
                                        ).tolist(),
                                    )
                                else:
                                    w_initial_guess.add_collocation_point(
                                        state_name,
                                        i_node,
                                        i_collocation,
                                        (
                                            collocation_points_initial_guesses[state_name][:, i_collocation, i_node]
                                            - z_basis
                                        ).tolist(),
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
                                        (states_initial_guesses[state_name][:, i_node] - z_basis).tolist(),
                                    )
                                else:
                                    w_initial_guess.add_collocation_point(
                                        state_name,
                                        i_node,
                                        i_collocation,
                                        (
                                            collocation_points_initial_guesses[state_name][:, i_collocation, i_node]
                                            - z_basis
                                        ).tolist(),
                                    )
                            else:
                                nb_components = states_lower_bounds[state_name].shape[0]
                                w_lower_bound.add_collocation_point(
                                    state_name, i_node, i_collocation, [0] * nb_components
                                )
                                w_upper_bound.add_collocation_point(
                                    state_name, i_node, i_collocation, [0] * nb_components
                                )
                                w_initial_guess.add_collocation_point(
                                    state_name, i_node, i_collocation, [0] * nb_components
                                )

            # U - controls
            for control_name in controls_lower_bounds.keys():
                w_lower_bound.add_control(control_name, i_node, controls_lower_bounds[control_name][:, i_node].tolist())
                w_upper_bound.add_control(control_name, i_node, controls_upper_bounds[control_name][:, i_node].tolist())
                w_initial_guess.add_control(
                    control_name, i_node, controls_initial_guesses[control_name][:, i_node].tolist()
                )

        return w_lower_bound, w_upper_bound, w_initial_guess

    def declare_noises(
        self,
        ocp_example: ExampleAbstract,
        n_shooting: int,
        nb_random: int,
        motor_noise_magnitude: np.ndarray,
        sensory_noise_magnitude: np.ndarray,
        seed: int = 0,
    ) -> NoisesAbstract:
        """
        Sample the noise values and declare the symbolic variables for the noises.
        """

        motor_noise_magnitude = cas.DM()
        sensory_noise_magnitude = cas.DM()

        noises_vector = self.Noises(n_shooting)

        for i_node in range(n_shooting + 1):
            if ocp_example.model.use_sx:
                noises_vector.add_motor_noise(i_node, cas.SX())
                noises_vector.add_sensory_noise(i_node, cas.SX())
            else:
                noises_vector.add_motor_noise(i_node, cas.MX())
                noises_vector.add_sensory_noise(i_node, cas.MX())

        return noises_vector

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
        pass

    def get_reference(
        self,
        ocp_example: ExampleAbstract,
        x: cas.MX | cas.SX | np.ndarray,
        u: cas.MX | cas.SX | np.ndarray,
    ) -> cas.MX | cas.SX | np.ndarray:

        n_components = ocp_example.model.q_indices.stop - ocp_example.model.q_indices.start
        q = x[:n_components]
        qdot = x[n_components : 2 * n_components]
        ref = ocp_example.model.sensory_output(q, qdot, cas.DM.zeros(ocp_example.model.nb_references))
        return ref

    def state_dynamics(
        self,
        ocp_example: ExampleAbstract,
        x: cas.MX | cas.SX,
        u: cas.MX | cas.SX,
        noise: cas.MX | cas.SX,
    ) -> cas.MX | cas.SX:
        if isinstance(self.dynamics_transcription, (Variational, VariationalPolynomial)):
            nb_states = ocp_example.model.nb_q
        else:
            nb_states = ocp_example.model.nb_states

        # Mean state
        ref_mean = self.get_reference(
            ocp_example=ocp_example,
            x=x,
            u=u,
        )
        dxdt_mean = ocp_example.model.dynamics(
            x[:nb_states],
            u,
            ref_mean,
            noise,
        )

        return dxdt_mean

    @cache_function
    def get_non_conservative_forces(
        self,
        ocp_example: ExampleAbstract,
        q: list[cas.MX | cas.SX],
        qdot: list[cas.MX | cas.SX],
        u: cas.MX | cas.SX,
        noise: cas.MX | cas.SX,
    ) -> cas.Function:
        f = ocp_example.model.non_conservative_forces(
            q[0],
            qdot[0],
            u,
            noise,
        )
        return cas.Function(
            "NonConservativeForces",
            [
                cas.vertcat(*q),
                cas.vertcat(*qdot),
                u,
                noise,
            ],
            [f],
        )

    @cache_function
    def get_lagrangian(
        self,
        ocp_example: ExampleAbstract,
        q: list[cas.MX | cas.SX],
        qdot: list[cas.MX | cas.SX],
        u: cas.MX | cas.SX,
    ) -> cas.MX | cas.SX:
        l = ocp_example.model.lagrangian(
            q[0],
            qdot[0],
            u,
        )
        return cas.Function(
            "Lagrangian",
            [
                cas.vertcat(*q),
                cas.vertcat(*qdot),
                u,
            ],
            [l],
            ["q", "qdot", "u"],
            ["L"],
        )

    def get_temporary_variables(
        self,
        ocp_example: ExampleAbstract,
        nb_q: int,
        nb_u: int,
    ) -> dict[str, list[cas.MX | cas.SX] | cas.MX | cas.SX]:

        if ocp_example.model.use_sx:
            q = [cas.SX.sym("q", nb_q)]
            qdot = [cas.SX.sym("qdot", nb_q)]
            u = cas.SX.sym("u", nb_u)
        else:
            q = [cas.MX.sym("q", nb_q)]
            qdot = [cas.MX.sym("qdot", nb_q)]
            u = cas.MX.sym("u", nb_u)

        variables = {
            "q": q,
            "qdot": qdot,
            "u": u,
        }
        return variables

    @cache_function
    def get_lagrangian_jacobian_q(
        self,
        ocp_example: ExampleAbstract,
        discrete_lagrangian: cas.MX | cas.SX,
        q: list[cas.MX | cas.SX],
        qdot: list[cas.MX | cas.SX],
    ) -> cas.Function:
        p = cas.transpose(
            cas.jacobian(
                discrete_lagrangian,
                q[0],
            )
        )
        return cas.Function(
            "LagrangianJacobian",
            [
                cas.vertcat(*q),
                cas.vertcat(*qdot),
            ],
            [p],
        )

    @cache_function
    def get_lagrangian_jacobian_qdot(
        self,
        ocp_example: ExampleAbstract,
        discrete_lagrangian: cas.MX | cas.SX,
        q: list[cas.MX | cas.SX],
        qdot: list[cas.MX | cas.SX],
    ) -> cas.Function:
        p = cas.transpose(
            cas.jacobian(
                discrete_lagrangian,
                qdot[0],
            )
        )
        return cas.Function(
            "LagrangianJacobian",
            [
                cas.vertcat(*q),
                cas.vertcat(*qdot),
            ],
            [p],
        )

    @cache_function
    def get_momentum(
        self,
        ocp_example: ExampleAbstract,
        q: list[cas.MX | cas.SX],
        qdot: list[cas.MX | cas.SX],
        u: cas.MX | cas.SX,
    ) -> cas.Function:
        p = ocp_example.model.momentum(
            q[0],
            qdot[0],
            u,
        )
        return cas.Function(
            "Momentum",
            [
                cas.vertcat(*q),
                cas.vertcat(*qdot),
                u,
            ],
            [p],
        )

    def get_lagrangian_jacobian(
        self,
        ocp_example: ExampleAbstract,
        discrete_lagrangian: cas.MX | cas.SX,
        variable_to_derivate_for: cas.MX | cas.SX,
    ) -> cas.MX | cas.SX:
        """
        Watch out that this version is only ised when get_lagrangian_jacobian is only called to create a casadi function
        afterward. Otherwise, please use a cache_function version.
        """

        p = cas.transpose(
            cas.jacobian(
                discrete_lagrangian,
                variable_to_derivate_for,
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
        states_plots = []
        # Placeholder to plot the variables
        color = "tab:blue"
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

        states_data = variable_opt.get_states_time_series_vector(key)[i_col, :]

        # Update mean state plot
        states_plots[i_state].set_ydata(
            states_data,
        )
        i_state += 1

        return i_state
