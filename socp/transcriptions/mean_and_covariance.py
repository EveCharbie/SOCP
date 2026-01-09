import casadi as cas
import numpy as np

from .discretization_abstract import DiscretizationAbstract
from ..examples.example_abstract import ExampleAbstract
from ..models.model_abstract import ModelAbstract


class MeanAndCovariance(DiscretizationAbstract):

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
    ) -> tuple[list[cas.MX], list[cas.MX], list[cas.MX], list[float], list[float], list[float]]:
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
                mean_x += [cas.MX.sym(f"{state_name}_{i_node}", n_components)]
                # Add bounds and initial guess
                w_lower_bound += states_lower_bounds[state_name][:, i_node].tolist()
                w_upper_bound += states_upper_bounds[state_name][:, i_node].tolist()
                w_initial_guess += states_initial_guesses[state_name][:, i_node].tolist()

            # Add the variables to a larger vector for easy access later
            x += [cas.vertcat(*mean_x)]
            w += [cas.vertcat(*mean_x)]

            # Create the symbolic variables for the state covariance
            n_components = ocp_example.model.nb_states
            cov = [cas.MX.sym(f"cov_{i_node}", n_components * n_components)]
            # Add bounds and initial guess
            p_init = ocp_example.model.reshape_matrix_to_vector(cas.DM.eye(n_components) * ocp_example.initial_state_covariance).full().flatten().tolist()
            w_initial_guess += p_init
            if i_node == 0:
                w_lower_bound += p_init
                w_upper_bound += p_init
            else:
                w_lower_bound += [-10] * (n_components * n_components)
                w_upper_bound += [10] * (n_components * n_components)

            # Add the variables to a larger vector for easy access later
            x += [cas.vertcat(*cov)]
            w += [cas.vertcat(*cov)]

            # Controls
            if i_node < ocp_example.n_shooting:
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

            # States mean
            for state_name in states_lower_bounds.keys():
                n_components = states_lower_bounds[state_name].shape[0]
                states[state_name][:, i_node] = np.array(vector[offset : offset + n_components]).flatten()
                x += [vector[offset : offset + n_components]]
                offset += n_components

            # States covariance
            n_components = model.nb_states
            states["cov"][:, :, i_node] = model.reshape_vector_to_matrix(
                vector[offset : offset + n_components * n_components]
            ).full()
            x += [vector[offset : offset + n_components * n_components]]
            offset += n_components * n_components

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

        noises_numerical = []  # No numerical values needed as only the covariance is used
        this_noises_single = []
        this_noises_single += [cas.MX.sym(f"motor_noise", n_motor_noises)]
        this_noises_single += [cas.MX.sym(f"sensory_noise", nb_references)]

        return noises_numerical, cas.vertcat(*this_noises_single)

    def get_mean_states(
        self,
        model: ModelAbstract,
        x,
        squared: bool = False,
    ):
        exponent = 2 if squared else 1

        start = model.state_indices[0].start
        stop = model.state_indices[-1].stop
        states_mean = x[start: stop] ** exponent

        return states_mean

    def get_states_variance(
        self,
        model: ModelAbstract,
        x,
        squared: bool = False,
    ):
        exponent = 2 if squared else 1

        offset = model.state_indices[-1].stop
        nb_components = model.nb_states
        cov = x[offset: offset + nb_components * nb_components]

        return cov  # ??

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
        n_components = model.q_indices.stop - model.q_indices.start
        q = x[: n_components]
        qdot = x[n_components: 2 * n_components]
        ref = model.sensory_output(q, qdot, cas.DM.zeros(model.nb_references))
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
        dxdt_mean = model.dynamics(
            x[: model.nb_states],
            u,
            ref,
            cas.DM.zeros(model.nb_noises),
        )

        dxdt_cov = 2 ### WAS HERE

        dxdt = cas.vertcat(
            dxdt_mean,
            dxdt_cov,
        )
        return dxdt

    def other_internal_constraints(
        self,
        model: ModelAbstract,
        x: cas.MX,
        u: cas.MX,
        noises_single: cas.MX,
        noises_numerical: cas.MX,
    ) -> tuple[list[cas.MX], list[float], list[float], list[str]]:
        """
        Other internal constraints specific to this discretization method.
        """
        # TODO: ref - mean_ref = 0
        pass
