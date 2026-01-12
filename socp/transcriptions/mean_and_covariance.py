import casadi as cas
import numpy as np

from .discretization_abstract import DiscretizationAbstract
from ..transcriptions.transcription_abstract import TranscriptionAbstract
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
        states_upper_bounds: dict[str, np.ndarray],
        states_initial_guesses: dict[str, np.ndarray],
        controls_lower_bounds: dict[str, np.ndarray],
        controls_upper_bounds: dict[str, np.ndarray],
        controls_initial_guesses: dict[str, np.ndarray],
    ) -> tuple[list[cas.SX], list[cas.SX], list[cas.SX], list[float], list[float], list[float]]:
        """
        Declare all symbolic variables for the states and controls with their bounds and initial guesses
        """

        x = []
        u = []
        w = []
        w_lower_bound = []
        w_upper_bound = []
        w_initial_guess = []
        for i_node in range(ocp_example.n_shooting + 1):

            # Create the symbolic variables for the mean states
            mean_x = []
            for state_name in states_lower_bounds.keys():
                n_components = states_lower_bounds[state_name].shape[0]
                mean_x += [cas.SX.sym(f"{state_name}_{i_node}", n_components)]
                # Add bounds and initial guess
                w_lower_bound += states_lower_bounds[state_name][:, i_node].tolist()
                w_upper_bound += states_upper_bounds[state_name][:, i_node].tolist()
                w_initial_guess += states_initial_guesses[state_name][:, i_node].tolist()

            # Create the symbolic variables for the state covariance
            n_components = ocp_example.model.nb_states
            cov_init = cas.DM.eye(n_components) * ocp_example.initial_state_variability
            if self.with_cholesky:
                # Declare cov variables
                nb_cov_variables = ocp_example.model.nb_cholesky_components(n_components)
                cov = [cas.SX.sym(f"cov_{i_node}", nb_cov_variables)]
                # Add cov bounds and initial guess
                p_init = ocp_example.model.reshape_cholesky_matrix_to_vector(cov_init).full().flatten().tolist()
            else:
                # Declare cov variables
                nb_cov_variables = n_components * n_components
                cov = [cas.SX.sym(f"cov_{i_node}", nb_cov_variables)]
                # Add cov bounds and initial guess
                p_init = ocp_example.model.reshape_matrix_to_vector(cov_init).full().flatten().tolist()

            w_initial_guess += p_init
            if i_node == 0:
                w_lower_bound += p_init
                w_upper_bound += p_init
            else:
                w_lower_bound += [-10] * nb_cov_variables
                w_upper_bound += [10] * nb_cov_variables

            # Create the symbolic variables for the helper matrix
            if self.with_helper_matrix:
                nb_states = ocp_example.model.nb_states
                nb_m_variables = nb_states * nb_states
                # Declare m variables
                m = [cas.SX.sym(f"m_{i_node}", nb_m_variables)]
                # Add m bounds and initial guess
                w_initial_guess += [0.01] * nb_m_variables
                w_lower_bound += [-10] * nb_m_variables
                w_upper_bound += [10] * nb_m_variables

                # Add the variables to a larger vector for easy access later
                x += [cas.vertcat(cas.vertcat(*mean_x), cas.vertcat(*cov), cas.vertcat(*m))]
                w += [cas.vertcat(cas.vertcat(*mean_x), cas.vertcat(*cov), cas.vertcat(*m))]
            else:
                # Add the variables to a larger vector for easy access later
                x += [cas.vertcat(cas.vertcat(*mean_x), cas.vertcat(*cov))]
                w += [cas.vertcat(cas.vertcat(*mean_x), cas.vertcat(*cov))]

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
        n_shooting = states_lower_bounds[list(states_lower_bounds.keys())[0]].shape[1] - 1
        n_collocation_points = self.dynamics_transcription.nb_collocation_points

        offset = 0
        states = {
            key: np.zeros((states_lower_bounds[key].shape[0], n_shooting + 1)) for key in states_lower_bounds.keys()
        }
        states["covariance"] = np.zeros((model.nb_states, model.nb_states, n_shooting + 1))
        controls = {key: np.zeros_like(controls_lower_bounds[key]) for key in controls_lower_bounds.keys()}
        x = []
        u = []
        for i_node in range(n_shooting + 1):

            # States mean
            for state_name in states_lower_bounds.keys():
                n_components = states_lower_bounds[state_name].shape[0]
                states[state_name][:, i_node] = np.array(vector[offset : offset + n_components]).flatten()
                x += [vector[offset : offset + n_components]]
                offset += n_components

            # States covariance
            nb_states = model.nb_states
            if self.with_cholesky:
                nb_cov_variables = model.nb_cholesky_components(nb_states)
                triangular_matrix = model.reshape_vector_to_cholesky_matrix(
                    vector[offset : offset + nb_cov_variables],
                    (nb_states, nb_states),
                ).full()
                states["covariance"][:, :, i_node] = triangular_matrix @ cas.transpose(triangular_matrix)
            else:
                nb_cov_variables = nb_states * nb_states
                states["covariance"][:, :, i_node] = model.reshape_vector_to_matrix(
                    vector[offset : offset + nb_states * nb_states],
                    (nb_states, nb_states),
                ).full()
            x += [vector[offset : offset + nb_cov_variables]]
            offset += nb_cov_variables

            # Helper matrix
            if self.with_helper_matrix:
                nb_m_variables = nb_states * nb_states
                states["m"][:, :, i_node] = model.reshape_vector_to_matrix(
                    vector[offset: offset + nb_m_variables],
                    (nb_states, nb_states),
                ).full()
                offset += nb_m_variables

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
    ) -> tuple[np.ndarray, cas.SX]:
        """
        Sample the noise values and declare the symbolic variables for the noises.
        """
        n_motor_noises = motor_noise_magnitude.shape[0]
        nb_references = sensory_noise_magnitude.shape[0]

        noises_numerical = []  # No numerical values needed as only the covariance is used
        this_noises_single = []
        this_noises_single += [cas.SX.sym(f"motor_noise", n_motor_noises)]
        this_noises_single += [cas.SX.sym(f"sensory_noise", nb_references)]

        return noises_numerical, cas.vertcat(*this_noises_single)

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

    @staticmethod
    def other_internal_constraints(
        model: ModelAbstract,
        x: cas.SX,
        u: cas.SX,
    ) -> tuple[list[cas.SX], list[float], list[float], list[str]]:
        """
        Other internal constraints specific to this discretization method.
        """
        g = []
        lbg = []
        ubg = []
        g_names = []

        # TODO: ref - mean_ref = 0

        # Semi-definite constraint on the covariance matrix (Sylvester's criterion)
        if not self.with_cholesky:
            cov_matrix = model.reshape_vector_to_matrix(
                x[model.nb_states : model.nb_states + model.nb_states * model.nb_states],
                (model.nb_states, model.nb_states),
            )
            epsilon = 1e-6
            for k in range(1, model.nb_states + 1):
                minor = cas.det(cov_matrix[:k, :k])
                g += [minor]
                lbg += [epsilon]
                ubg += [cas.inf]
                g_names += ["covariance_positive_definite_minor_" + str(k)]

        # Helper matrix constraint
        # TODO

        return g, lbg, ubg, g_names

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
