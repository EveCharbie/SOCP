import casadi as cas
import numpy as np

from .discretization_abstract import DiscretizationAbstract
from ..transcriptions.transcription_abstract import TranscriptionAbstract
from ..transcriptions.direct_collocation_polynomial import DirectCollocationPolynomial
from ..transcriptions.direct_collocation_trapezoidal import DirectCollocationTrapezoidal
from ..examples.example_abstract import ExampleAbstract
from ..models.model_abstract import ModelAbstract


class MeanAndCovariance(DiscretizationAbstract):

    def __init__(
        self,
        dynamics_transcription: TranscriptionAbstract,
        with_cholesky: bool = False,
        with_helper_matrix: bool = False,
    ) -> None:

        super().__init__()  # Does nothing

        self.dynamics_transcription = dynamics_transcription
        self.with_cholesky = with_cholesky
        self.with_helper_matrix = with_helper_matrix

    def name(self) -> str:
        return "MeanAndCovariance"

    def declare_variables(
        self,
        ocp_example: ExampleAbstract,
        states_lower_bounds: dict[str, np.ndarray],
        controls_lower_bounds: dict[str, np.ndarray],
    ) -> tuple[cas.SX, list[cas.SX], list[cas.SX], list[cas.SX], list[cas.SX]]:
        """
        Declare all symbolic variables for the states and controls with their bounds and initial guesses
        """
        nb_states = ocp_example.model.nb_states
        n_shooting = ocp_example.n_shooting
        nb_collocation_points = self.dynamics_transcription.nb_collocation_points
        state_names = list(ocp_example.model.state_indices.keys())

        x = []
        z = []
        u = []
        w = []

        T = cas.SX.sym("final_time", 1)
        w += [T]

        for i_node in range(n_shooting + 1):

            mean_x = []
            mean_z = []
            for state_name in state_names:
                n_components = states_lower_bounds[state_name].shape[0]
                mean_x += [cas.SX.sym(f"{state_name}_{i_node}", n_components)]

                if isinstance(self.dynamics_transcription, DirectCollocationPolynomial):
                    # Create the symbolic variables for the mean states collocation points
                    if i_node < n_shooting:
                        mean_z += [cas.SX.sym(f"{state_name}_{i_node}_z", n_components * nb_collocation_points)]
                    else:
                        mean_z += [cas.SX.zeros(n_components * (nb_collocation_points))]

            # Create the symbolic variables for the state covariance
            if self.with_cholesky:
                nb_cov_variables = ocp_example.model.nb_cholesky_components(nb_states)
                cov = [cas.SX.sym(f"cov_{i_node}", nb_cov_variables)]
            else:
                nb_cov_variables = nb_states * nb_states
                cov = [cas.SX.sym(f"cov_{i_node}", nb_cov_variables)]

            m = []
            if self.with_helper_matrix:
                # Create the symbolic variables for the helper matrix
                nb_m_variables = nb_states * nb_states
                if i_node < n_shooting:
                    for i_collocation in range(nb_collocation_points):
                        m += [cas.SX.sym(f"m_{i_node}_{i_collocation}", nb_m_variables)]
                else:
                    for i_collocation in range(nb_collocation_points):
                        m += [cas.SX.zeros(nb_m_variables)]

            # Add the variables to a larger vector for easy access later
            x += [cas.vertcat(cas.vertcat(*mean_x), cas.vertcat(*cov), cas.vertcat(*m))]
            if i_node < n_shooting:
                w += [cas.vertcat(cas.vertcat(*mean_x), cas.vertcat(*cov), cas.vertcat(*m))]
            else:
                w += [cas.vertcat(cas.vertcat(*mean_x), cas.vertcat(*cov))]

            if isinstance(self.dynamics_transcription, DirectCollocationPolynomial):
                # Add the collocation points variables
                z += [cas.vertcat(*mean_z)]
                if i_node < n_shooting:
                    w += [cas.vertcat(*mean_z)]

            # Controls
            if i_node < ocp_example.n_shooting:
                this_u = []
                for control_name in controls_lower_bounds.keys():
                    n_components = controls_lower_bounds[control_name].shape[0]
                    this_u += [cas.SX.sym(f"{control_name}_{i_node}", n_components)]
            else:
                this_u = [
                    cas.SX.zeros(controls_lower_bounds[control_name].shape[0])
                    for control_name in controls_lower_bounds.keys()
                ]
            # Add the variables to a larger vector for easy access later
            u += [cas.vertcat(*this_u)]
            if i_node < ocp_example.n_shooting:
                w += [cas.vertcat(*this_u)]

        return T, x, z, u, w

    @staticmethod
    def get_state_from_single_vector(
            ocp_example: ExampleAbstract,
            state_single: cas.SX,
            state_name: str,
    ) -> cas.SX:
        """
        Extract the state variable from the single vector at a given node.
        """
        state_indices = ocp_example.model.state_indices[state_name]
        return state_single[state_indices.start: state_indices.stop]

    def declare_bounds_and_init(
        self,
        ocp_example: ExampleAbstract,
        states_lower_bounds: dict[str, np.ndarray],
        states_upper_bounds: dict[str, np.ndarray],
        states_initial_guesses: dict[str, np.ndarray],
        controls_lower_bounds: dict[str, np.ndarray],
        controls_upper_bounds: dict[str, np.ndarray],
        controls_initial_guesses: dict[str, np.ndarray],
        collocation_points_initial_guesses: dict[str, np.ndarray]
    ) -> tuple[list[float], list[float], list[float]]:
        """
        Declare all symbolic variables for the states and controls with their bounds and initial guesses
        """
        nb_states = ocp_example.model.nb_states
        n_shooting = ocp_example.n_shooting
        nb_collocation_points = self.dynamics_transcription.nb_collocation_points
        state_names = list(ocp_example.model.state_indices.keys())

        w_lower_bound = []
        w_upper_bound = []
        w_initial_guess = []

        w_initial_guess += [ocp_example.final_time]
        w_lower_bound += [ocp_example.min_time]
        w_upper_bound += [ocp_example.max_time]

        for i_node in range(n_shooting + 1):

            # X - states
            for state_name in state_names:
                w_lower_bound += states_lower_bounds[state_name][:, i_node].tolist()
                w_upper_bound += states_upper_bounds[state_name][:, i_node].tolist()
                w_initial_guess += states_initial_guesses[state_name][:, i_node].tolist()

            # COV - covariance
            cov_init = states_initial_guesses["covariance"][:, :, i_node]
            if self.with_cholesky:
                nb_cov_variables = ocp_example.model.nb_cholesky_components(nb_states)
                p_init = np.array(ocp_example.model.reshape_cholesky_matrix_to_vector(cov_init)).flatten().tolist()
            else:
                # Declare cov variables
                nb_cov_variables = nb_states * nb_states
                p_init = np.array(ocp_example.model.reshape_matrix_to_vector(cov_init)).flatten().tolist()

            w_initial_guess += p_init
            if i_node == 0:
                w_lower_bound += p_init
                w_upper_bound += p_init
            else:
                w_lower_bound += [-10] * nb_cov_variables
                w_upper_bound += [10] * nb_cov_variables

            # M - Helper matrix
            if self.with_helper_matrix:
                n_components = nb_states * nb_states
                if i_node < n_shooting:
                    for i_collocation in range(nb_collocation_points):
                        if "m" in states_initial_guesses.keys():
                            w_initial_guess += ocp_example.model.reshape_matrix_to_vector(
                                states_initial_guesses["m"][:, :, i_collocation, i_node],
                            ).flatten().tolist()
                        else:
                            w_initial_guess += [0.0] * n_components
                        w_lower_bound += [-10] * n_components
                        w_upper_bound += [10] * n_components

            # Z - collocation points
            if isinstance(self.dynamics_transcription, DirectCollocationPolynomial):
                for state_name in state_names:
                    if i_node < n_shooting:
                        # The last interval does not have collocation points
                        for i_collocation in range(nb_collocation_points):
                            # Add bounds and initial guess as linear interpolation between the two nodes
                            w_lower_bound += self.interpolate_between_nodes(
                                var_pre=states_lower_bounds[state_name][:, i_node],
                                var_post=states_lower_bounds[state_name][:, i_node + 1],
                                nb_points=nb_collocation_points,
                                current_point=i_collocation,
                            ).tolist()
                            w_upper_bound += self.interpolate_between_nodes(
                                var_pre=states_upper_bounds[state_name][:, i_node],
                                var_post=states_upper_bounds[state_name][:, i_node + 1],
                                nb_points=nb_collocation_points,
                                current_point=i_collocation,
                            ).tolist()
                            if collocation_points_initial_guesses is None:
                                w_initial_guess += self.interpolate_between_nodes(
                                    var_pre=states_initial_guesses[state_name][:, i_node],
                                    var_post=states_initial_guesses[state_name][:, i_node + 1],
                                    nb_points=nb_collocation_points,
                                    current_point=i_collocation,
                                ).tolist()
                            else:
                                w_initial_guess += collocation_points_initial_guesses[state_name][
                                    :, i_collocation, i_node
                                ].tolist()

            # Controls
            if i_node < ocp_example.n_shooting:
                for control_name in controls_lower_bounds.keys():
                    w_lower_bound += controls_lower_bounds[control_name][:, i_node].tolist()
                    w_upper_bound += controls_upper_bounds[control_name][:, i_node].tolist()
                    w_initial_guess += controls_initial_guesses[control_name][:, i_node].tolist()

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
        n_shooting = states_lower_bounds[list(states_lower_bounds.keys())[0]].shape[1] - 1
        nb_collocation_points = self.dynamics_transcription.nb_collocation_points
        state_names = list(model.state_indices.keys())

        offset = 0
        T = vector[offset]
        offset += 1

        states = {
            name: np.zeros((states_lower_bounds[name].shape[0], n_shooting + 1)) for name in state_names
        }
        states["covariance"] = np.zeros((model.nb_states, model.nb_states, n_shooting + 1))
        if self.with_helper_matrix:
            states["m"] = np.zeros((model.nb_states, model.nb_states, nb_collocation_points, n_shooting + 1))

        collocation_points = {
            key: np.zeros((states_lower_bounds[key].shape[0], nb_collocation_points, n_shooting + 1))
            for key in states_lower_bounds.keys()
        }

        controls = {key: np.zeros_like(controls_lower_bounds[key]) for key in controls_lower_bounds.keys()}
        x = []
        z = []
        u = []
        for i_node in range(n_shooting + 1):

            # X - states
            for state_name in state_names:
                n_components = states_lower_bounds[state_name].shape[0]
                this_state = vector[offset : offset + n_components]
                states[state_name][:, i_node] = np.array(this_state).flatten()
                x += [this_state]
                offset += n_components

            # COV - covariance
            nb_states = model.nb_states
            if self.with_cholesky:
                nb_cov_variables = model.nb_cholesky_components(nb_states)
                this_cov = vector[offset : offset + nb_cov_variables]
                triangular_matrix = np.array(model.reshape_vector_to_cholesky_matrix(
                    np.array(this_cov),
                    (nb_states, nb_states),
                ))
                states["covariance"][:, :, i_node] = triangular_matrix @ cas.transpose(triangular_matrix)
            else:
                nb_cov_variables = nb_states * nb_states
                this_cov = vector[offset: offset + nb_cov_variables]
                states["covariance"][:, :, i_node] = model.reshape_vector_to_matrix(
                    np.array(this_cov),
                    (nb_states, nb_states),
                )
            x += [this_cov]
            offset += nb_cov_variables

            # M - Helper matrix
            if self.with_helper_matrix:
                nb_m_variables = nb_states * nb_states
                if i_node < n_shooting:
                    for i_collocation in range(nb_collocation_points):
                        this_m = vector[offset : offset + nb_m_variables]
                        states["m"][:, :, i_collocation, i_node] = model.reshape_vector_to_matrix(
                            np.array(this_m),
                            (nb_states, nb_states),
                        )
                        x += [this_m]
                        offset += nb_m_variables
                else:
                    for i_collocation in range(nb_collocation_points):
                        states["m"][:, :, i_collocation, i_node] = np.zeros((nb_states, nb_states))
                        x += [np.zeros(nb_m_variables)]

            # Z - collocation points
            for state_name in state_names:
                if i_node < n_shooting:
                    for i_collocation in range(nb_collocation_points):
                        collocation_points[state_name][:, i_collocation, i_node] = np.array(
                            vector[offset : offset + n_components]
                        ).flatten()
                        z += [vector[offset : offset + n_components]]
                        offset += n_components
                else:
                    collocation_points[state_name][:, :, i_node] = np.zeros((n_components, nb_collocation_points))
                    z += np.zeros((n_components * nb_collocation_points, )).tolist()

            # U - Controls
            if i_node < n_shooting:
                for control_name in controls_lower_bounds.keys():
                    n_components = controls_lower_bounds[control_name].shape[0]
                    controls[control_name][:, i_node] = np.array(vector[offset : offset + n_components]).flatten()
                    u += [vector[offset : offset + n_components]]
                    offset += n_components

        return T, states, collocation_points, controls, cas.vertcat(*x), cas.vertcat(*z), cas.vertcat(*u)

    def get_var_arrays(
            self,
            ocp_example: ExampleAbstract,
            discretization_method: DiscretizationAbstract,
            states_var: dict[str, cas.DM | np.ndarray],
            collocation_vars: dict[str, cas.DM | np.ndarray],
            controls_var: dict[str, cas.DM | np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert the states and controls from dict into array format to use the same functions as during the optimization with x and u.
        """
        nb_states = ocp_example.model.nb_states
        nb_controls = ocp_example.model.nb_controls
        n_shooting = ocp_example.n_shooting
        nb_collocation_points = self.dynamics_transcription.nb_collocation_points

        # States
        states_var_array = np.zeros((nb_states, n_shooting + 1))
        state_names = list(ocp_example.model.state_indices.keys())
        for state_name in state_names:
            indices = ocp_example.model.state_indices[state_name]
            states_var_array[indices, :] = states_var[state_name]

        if "covariance" in states_var.keys():
            cov_vector = None
            for i_node in range(n_shooting + 1):
                if discretization_method.with_cholesky:
                    vect = ocp_example.model.reshape_cholesky_matrix_to_vector(
                        states_var["covariance"][:, :, i_node]
                    )
                else:
                    vect = ocp_example.model.reshape_matrix_to_vector(
                        states_var["covariance"][:, :, i_node]
                    )

                if cov_vector is None:
                    cov_vector = vect
                else:
                    cov_vector = np.hstack((cov_vector, vect))

            states_var_array = np.vstack((states_var_array, cov_vector))

        if "m" in states_var.keys():
            # The last interval does not have collocation points
            m_vector = None
            for i_node in range(n_shooting):
                tempo = None
                for i_collocation in range(nb_collocation_points):
                    vect = ocp_example.model.reshape_matrix_to_vector(states_var["m"][:, :, i_collocation, i_node])
                    tempo = vect if tempo is None else np.vstack((tempo, vect))

                if m_vector is None:
                    m_vector = tempo
                else:
                    m_vector = np.hstack((m_vector, tempo))

            m_vector = np.hstack((m_vector, np.zeros_like(tempo)))  # Add the final zeros
            states_var_array = np.vstack((states_var_array, m_vector))

        # Collocation points
        collocation_points_var_array = np.zeros((nb_states * nb_collocation_points, n_shooting + 1))
        collocation_offset = 0
        for state_name in state_names:
            indices = ocp_example.model.state_indices[state_name]
            n_components = indices.stop - indices.start
            for i_collocation in range(nb_collocation_points):
                collocation_points_var_array[collocation_offset: collocation_offset + n_components, :] = collocation_vars[state_name][:, i_collocation, :]
                collocation_offset += n_components

        # Controls
        controls_var_array = np.zeros((nb_controls, n_shooting))
        control_names = [name for name in controls_var.keys()]
        for control_name in control_names:
            indices = ocp_example.model.control_indices[control_name]
            controls_var_array[indices, :] = controls_var[control_name]

        return states_var_array, collocation_points_var_array, controls_var_array

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

        n_motor_noises = motor_noise_magnitude.shape[0] if motor_noise_magnitude is not None else 0
        nb_references = sensory_noise_magnitude.shape[0] if sensory_noise_magnitude is not None else 0

        noises_numerical = []  # No numerical values needed as only the covariance is used
        this_noises_single = []
        this_noises_single += [cas.SX.sym(f"motor_noise", n_motor_noises)]
        this_noises_single += [cas.SX.sym(f"sensory_noise", nb_references)]

        return noises_numerical, cas.vertcat(*this_noises_single)

    def initialize_m(
        self,
        ocp_example: ExampleAbstract,
        final_time_init: float,
        states_initial_guesses: dict[str, np.ndarray],
        controls_initial_guesses: dict[str, np.ndarray],
        collocation_points_initial_guesses: dict[str, np.ndarray],
        jacobian_funcs: cas.Function,
    ) -> np.ndarray:

        nb_states = ocp_example.model.nb_states
        n_shooting = ocp_example.n_shooting
        nb_collocation_points = self.dynamics_transcription.nb_collocation_points

        states_init_array, collocation_points_init_array, controls_init_array = self.get_var_arrays(
            ocp_example,
            self,
            states_initial_guesses,
            collocation_points_initial_guesses,
            controls_initial_guesses,
        )
        m_init = np.zeros((nb_states, nb_states, nb_collocation_points, n_shooting + 1))
        for i_node in range(n_shooting):
            _, dg_dz, _, df_dz = jacobian_funcs(
                final_time_init,
                states_init_array[:, i_node],
                collocation_points_init_array[:, i_node],
                controls_init_array[:, i_node],
                np.zeros((ocp_example.model.nb_noises, )),
            )
            m_this_time = df_dz @ np.linalg.inv(dg_dz)

            for i_collocation in range(nb_collocation_points):
                m_init[:, :, i_collocation, i_node] = m_this_time[:, i_collocation * nb_states : (i_collocation + 1) * nb_states]

        # Wrong, but necessary since we do not have the collocation points at the last node
        m_init[:, :, 0, -1] = m_init[:, :, 0, -2]
        return m_init

    def modify_init(
            self,
            ocp_example: ExampleAbstract,
            states_initial_guesses: dict[str, np.ndarray],
            collocation_points_initial_guesses: dict[str, np.ndarray],
            controls_initial_guesses: dict[str, np.ndarray],
    ):
        """
        Modify bounds and initial guesses if needed.
        This is needed when the bounds and init from one variable depend on the dynamics of the system.
        """
        if self.with_helper_matrix and isinstance(
                self.dynamics_transcription, DirectCollocationPolynomial,
        ):
            m_init = self.initialize_m(
                ocp_example=ocp_example,
                final_time_init=ocp_example.final_time,
                states_initial_guesses=states_initial_guesses,
                controls_initial_guesses=controls_initial_guesses,
                collocation_points_initial_guesses=collocation_points_initial_guesses,
                jacobian_funcs=self.dynamics_transcription.jacobian_funcs
            )
        states_initial_guesses["m"] = m_init
        return states_initial_guesses, collocation_points_initial_guesses, controls_initial_guesses

    def get_mean_states(
        self,
        model: ModelAbstract,
        x,
        squared: bool = False,
    ):
        exponent = 2 if squared else 1

        state_names = list(model.state_indices.keys())
        start = model.state_indices[state_names[0]].start
        stop = model.state_indices[state_names[-1]].stop
        states_mean = x[start:stop] ** exponent

        return states_mean

    def get_covariance(
        self,
        model: ModelAbstract,
        x,
    ):
        state_names = list(model.state_indices.keys())
        offset = model.state_indices[state_names[-1]].stop
        nb_states = model.nb_states

        if self.with_cholesky:
            nb_cov_variables = model.nb_cholesky_components(nb_states)
            covariance = x[offset : offset + nb_cov_variables]
            triangular_matrix = model.reshape_vector_to_cholesky_matrix(
                covariance,
                (nb_states, nb_states),
            )
            cov = triangular_matrix @ cas.transpose(triangular_matrix)
        else:
            nb_cov_variables = nb_states * nb_states
            covariance = x[offset: offset + nb_cov_variables]
            cov = model.reshape_vector_to_matrix(
                covariance,
                (nb_states, nb_states),
            )

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
        nb_states = model.nb_states
        q = cas.SX.sym("q", model.nb_q)
        qdot = cas.SX.sym("qdot", model.nb_q)

        # No noise for mean
        dee_dq = cas.jacobian(
            model.sensory_output(q, qdot, cas.DM.zeros(model.nb_references)),
            q,
        )
        if self.with_cholesky:
            nb_cov_variables = model.nb_cholesky_components(nb_states)
            covariance = cas.SX.sym("cov", nb_cov_variables)
            triangular_matrix = model.reshape_vector_to_cholesky_matrix(
                covariance,
                (nb_states, nb_states),
            )
            cov = triangular_matrix @ cas.transpose(triangular_matrix)
        else:
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
            x[model.q_indices],
            x[model.qdot_indices],
            x[nb_states : nb_states + nb_cov_variables],
        )

        return end_effector_covariance_eval_x, end_effector_covariance_eval_y

    def get_mus_variance(
        self,
        model: ModelAbstract,
        x,
    ):
        state_names = list(model.state_indices.keys())
        offset = model.state_indices[state_names[-1]].stop
        nb_states = model.nb_states
        if self.with_cholesky:
            nb_components = model.nb_cholesky_components(nb_states)
        else:
            nb_components = nb_states * nb_states
        cov = x[offset : offset + nb_components]
        if self.with_cholesky:
            triangular_matrix = model.reshape_vector_to_cholesky_matrix(
                cov,
                (nb_states, nb_states),
            )
            cov_matrix = triangular_matrix @ cas.transpose(triangular_matrix)
        else:
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

        nb_noises = example_ocp.model.nb_noises
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
            cas.DM.zeros(nb_noises),
        )

        return dxdt_mean

    def covariance_dynamics(
        self,
        example_ocp: ExampleAbstract,
        x,
        u,
        noise,
    ) -> cas.SX:
        """
        This should work at optimality, but in the meantime we do not have any guarantee that the Cholesky decomposition exists...
        Therefore, it gives NaNs during the optimization, and we have to put the continuity constraint on all cov terms.
            triangular_matrix = cas.transpose(cas.chol(dxdt_cov))
        """
        nb_noises = example_ocp.model.nb_noises
        nb_states = example_ocp.model.nb_states

        ref_mean = self.get_reference(
            example_ocp.model,
            x,
            u,
        )

        # State covariance
        # Temporary symbolic variables and functions
        states = cas.SX.sym("x", nb_states)
        if self.with_cholesky:
            nb_cov_variables = example_ocp.model.nb_cholesky_components(nb_states)
        else:
            nb_cov_variables = nb_states * nb_states
        covariance = cas.SX.sym("cov", nb_cov_variables)

        # TODO: move in trapezoidal
        dxdt = example_ocp.model.dynamics(
            states,
            u,
            ref_mean,
            noise,
        )
        df_dx = cas.jacobian(dxdt, states)
        df_dw = cas.jacobian(dxdt, noise)

        if self.with_cholesky:
            triangular_matrix = example_ocp.model.reshape_vector_to_cholesky_matrix(
                covariance,
                (nb_states, nb_states),
            )
            current_cov = triangular_matrix @ cas.transpose(triangular_matrix)
        else:
            current_cov = example_ocp.model.reshape_vector_to_matrix(
                covariance,
                (nb_states, nb_states),
            )

        sigma_w = noise * cas.SX_eye(nb_noises)
        if self.with_helper_matrix:
            """
            When helper matrix is used, the output is a list of elements needed to compute the covariance at the next time step
            """
            m_vector = x[nb_states + nb_cov_variables : nb_states + nb_cov_variables + nb_states * nb_states]
            m_matrix = example_ocp.model.reshape_vector_to_matrix(
                m_vector,
                (nb_states, nb_states),
            )
            cov_output = [m_matrix, df_dx, df_dw, sigma_w]
        else:
            """
            When helper matrix is not used, the output is the derivative of the covariance
            """
            dxdt_cov = df_dx @ current_cov + current_cov @ cas.transpose(df_dx) + df_dw @ sigma_w @ cas.transpose(df_dw)

            if self.with_cholesky:
                triangular_matrix = cas.transpose(cas.chol(dxdt_cov))
                cov_output = [example_ocp.model.reshape_cholesky_matrix_to_vector(triangular_matrix)]
            else:
                cov_output = [example_ocp.model.reshape_matrix_to_vector(dxdt_cov)]

        dxdt_cov_func = cas.Function(
            "dxdt_cov_func",
            [states, covariance, u, noise],
            cov_output,
        )
        motor_noise_magnitude, sensory_noise_magnitude = example_ocp.get_noises_magnitude()
        numerical_noise = cas.vertcat(motor_noise_magnitude, sensory_noise_magnitude)
        output = dxdt_cov_func(
            x[:nb_states],
            x[nb_states : nb_states + nb_cov_variables],
            u,
            numerical_noise,
        )

        return output

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
        states_plots += [
            axs[i_row, i_col].fill_between(
                time_vector,
                np.zeros_like(time_vector),
                np.zeros_like(time_vector),
                color=color,
                alpha=0.3,
            )
        ]
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

        # Update mean state plot
        states_plots[i_state].set_ydata(states_opt[key][i_col, :])
        i_state += 1

        # Update covariance fill
        cov = states_opt["covariance"][i_col, i_col, :]
        verts = np.vstack(
            [
                np.column_stack([time_vector, states_opt[key][i_col, :] - np.sqrt(cov)]),
                np.column_stack([time_vector[::-1], (states_opt[key][i_col, :] + np.sqrt(cov))[::-1]]),
            ]
        )
        states_plots[i_state].get_paths()[0].vertices = verts
        i_state += 1

        return i_state
