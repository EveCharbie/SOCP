import casadi as cas
import numpy as np

from .discretization_abstract import DiscretizationAbstract
from .noises_abstract import NoisesAbstract
from .variables_abstract import VariablesAbstract
from ..examples.example_abstract import ExampleAbstract
from ..models.model_abstract import ModelAbstract
from ..models.biorbd_model import cache_function
from ..transcriptions.transcription_abstract import TranscriptionAbstract
from ..transcriptions.direct_collocation_polynomial import DirectCollocationPolynomial
from ..transcriptions.variational import Variational
from ..transcriptions.variational_polynomial import VariationalPolynomial


class UnscentedTransform(DiscretizationAbstract):

    def __init__(
        self,
        dynamics_transcription: TranscriptionAbstract,
    ) -> None:

        super().__init__()
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
            nb_sigma_points: int = 1,
        ):
            super().__init__(
                n_shooting=n_shooting,
                nb_random=nb_random,
                nb_sigma_points=nb_sigma_points,
                nb_collocation_points=nb_collocation_points,
                nb_m_points=nb_m_points,
                state_indices=state_indices,
                control_indices=control_indices
            )

            self.t = None
            # self.x_list = [{state_name: None for state_name in self.state_names} for _ in range(n_shooting + 1)]
            # self.padded_x_list = [{state_name: None for state_name in self.state_names} for _ in range(n_shooting + 1)]
            # self.chol_cov_list = [{"chol_cov": None} for _ in range(n_shooting + 1)]
            # self.z_list = [
            #     {
            #         state_name: [[None for _ in range(nb_collocation_points)] for _ in range(nb_sigma_points)]
            #         for state_name in self.state_names
            #     }
            #     for _ in range(n_shooting + 1)
            # ]

            # TODO: remove !
            self.x_list = [
                {state_name: [None for _ in range(nb_sigma_points)] for state_name in self.state_names}
                for _ in range(n_shooting + 1)
            ]
            self.padded_x_list = [
                {state_name: [None for _ in range(nb_sigma_points)] for state_name in self.state_names}
                for _ in range(n_shooting + 1)
            ]
            self.z_list = [
                {
                    state_name: [[None for _ in range(nb_collocation_points)] for _ in range(nb_sigma_points)]
                    for state_name in self.state_names
                }
                for _ in range(n_shooting + 1)
            ]

            self.u_list = [{control_name: None for control_name in self.control_names} for _ in range(n_shooting + 1)]

        # --- Add --- #
        def add_time(self, value: cas.MX | cas.SX | cas.DM):
            self.t = self.transform_to_dm(value)

        # def add_state(self, name: str, node: int, value: cas.MX | cas.SX | cas.DM):
        #     self.x_list[node][name] = self.transform_to_dm(value)
        #
        # def add_padded_state(self, name: str, node: int):
        #     if name in ["q", "qdot"]:
        #         state_that_should_be = self.x_list[0][name]
        #         if isinstance(state_that_should_be, (cas.MX, cas.SX)):
        #             self.padded_x_list[node][name] = self.cx.sym(f"{name}_DO_NOT_USE_{node}", state_that_should_be.shape)



        def add_state(self, name: str, node: int, sigma_point: int, value: cas.MX | cas.SX | cas.DM):
            self.x_list[node][name][sigma_point] = self.transform_to_dm(value)

        def add_padded_state(self, name: str, node: int, random: int):
            if name in ["q", "qdot"]:
                state_that_should_be = self.x_list[0][name][0]
                if isinstance(state_that_should_be, (cas.MX, cas.SX)):
                    cx = type(state_that_should_be)
                    self.padded_x_list[node][name][random] = cx.sym(f"{name}_DO_NOT_USE_{node}_{random}", state_that_should_be.shape)



        def add_collocation_point(self, name: str, node: int, sigma_point: int, point: int, value: cas.MX | cas.SX | cas.DM):
            self.z_list[node][name][sigma_point][point] = self.transform_to_dm(value)

        # def add_chol_cov(self, node: int, value: cas.MX | cas.SX | cas.DM):
        #     self.chol_cov_list[node]["chol_cov"] = self.transform_to_dm(value)

        def add_control(self, name: str, node: int, value: cas.MX | cas.SX | cas.DM):
            self.u_list[node][name] = self.transform_to_dm(value)

        # --- Nb --- #
        # @property
        # def nb_states(self):
        #     nb_states = 0
        #     for state_name in self.state_indices.keys():
        #         nb_states += self.state_indices[state_name].stop - self.state_indices[state_name].start
        #     return nb_states

        @property
        def nb_states(self):
            nb_states = 0
            for state_name in self.state_names:
                this_x = self.x_list[0][state_name][0]
                if this_x is not None:
                    nb_states += self.x_list[0][state_name][0].shape[0]
            return nb_states

        @property
        def nb_q(self) -> int:
            nb_q = self.state_indices["q"].stop - self.state_indices["q"].start
            return nb_q

        # @property
        # def nb_total_states(self):
        #     return self.nb_states


        @property
        def nb_total_states(self):
            return self.nb_states * self.nb_sigma_points


        @property
        def nb_controls(self):
            nb_controls = 0
            for control_name in self.control_names:
                nb_controls += self.u_list[0][control_name].shape[0]
            return nb_controls

        # --- Get --- #
        def get_time(self):
            return self.t

        # def get_state(self, name: str, node: int):
        #     return self.x_list[node][name]


        def get_specific_state(self, name: str, node: int, random: int):
            return self.x_list[node][name][random]

        def get_state(self, name: str, node: int):
            states = None
            for i_random in range(self.nb_sigma_points):
                if states is None:
                    states = self.x_list[node][name][i_random]
                else:
                    states = cas.vertcat(states, self.x_list[node][name][i_random])
            return states


        def get_state_list(self, name: str, node: int):
            """
            Get a list of symbolic variables for a specific state at the first node.
            """
            if name not in self.state_names:
                raise RuntimeError(f"There is no state named {name} in the model, cannot get its list.")
            return [self.x_list[node][name][0]]

        # def get_states(self, node: int) -> cas.MX | cas.SX | cas.DM:
        #     states = None
        #     for state_name in self.state_names:
        #         this_state = self.x_list[node][state_name]
        #         if this_state is not None:
        #             if states is None:
        #                 states = this_state
        #             else:
        #                 states = cas.vertcat(states, this_state)
        #     return states


        def get_states(self, node: int) -> cas.MX | cas.SX | cas.DM:
            states = None
            for i_random in range(self.nb_sigma_points):
                for state_name in self.state_names:
                    this_state = self.x_list[node][state_name][i_random]
                    if this_state is not None:
                        if states is None:
                            states = this_state
                        else:
                            states = cas.vertcat(states, this_state)
            return states


        def get_sigma_states(self, node: int, noise_matrix: cas.DM):

            # Detect if variational is used
            if self.x_list[1]["qdot"] is None:
                skipped_qdot = True
            else:
                skipped_qdot = False

            # # Get the mean states
            # x_mean = None
            # for state_name in self.state_names:
            #     this_state = self.x_list[node][state_name]
            #     if this_state is not None and (not skipped_qdot or state_name != "qdot"):
            #         if x_mean is None:
            #             x_mean = this_state
            #         else:
            #             x_mean = cas.vertcat(x_mean, this_state)
            # x_mean = cas.vertcat(x_mean, cas.DM.zeros(noise_matrix.shape[0]))
            #
            # # Get the +- one STD sigma points (Eq. 4 from D'Hondt et al. 2026 preprint)
            # l_matrix = self.get_chol_cov_matrix(node=node)
            # augmented_l_matrix = cas.vertcat(
            #     cas.horzcat(l_matrix, self.cx.zeros(l_matrix.shape[0], noise_matrix.shape[0])),
            #     cas.horzcat(self.cx.zeros(noise_matrix.shape[0], l_matrix.shape[0]), noise_matrix),
            # )
            # sigma_minus = self.cx.zeros(augmented_l_matrix.shape[0], augmented_l_matrix.shape[1])
            # sigma_plus = self.cx.zeros(augmented_l_matrix.shape[0], augmented_l_matrix.shape[1])
            # for i_col in range(augmented_l_matrix.shape[1]):
            #     sigma_minus[:, i_col] = x_mean - augmented_l_matrix[:, i_col]
            #     sigma_plus[:, i_col] = x_mean + augmented_l_matrix[:, i_col]
            #
            # sigma_states = cas.horzcat(
            # x_mean,
            #     sigma_plus,
            #     sigma_minus,
            # )
            #
            # return sigma_states

            # TODO: remove
            collocation_points_matrix = None
            for state_name in self.state_names:
                collocation_points_vector = None
                for i_sigma in range(self.nb_sigma_points):
                    if collocation_points_vector is None:
                        collocation_points_vector = self.z_list[node][state_name][i_sigma][0]
                    else:
                        collocation_points_vector = cas.horzcat(
                            collocation_points_vector, self.z_list[node][state_name][i_sigma][0]
                        )
                if collocation_points_vector is not None and (not skipped_qdot or state_name != "qdot"):
                    if collocation_points_matrix is None:
                        collocation_points_matrix = collocation_points_vector
                    else:
                        collocation_points_matrix = cas.vertcat(
                            collocation_points_matrix,
                            collocation_points_vector,
                        )
            return collocation_points_matrix

        def get_mean_sigma(self, sigma_points_vector: cas.MX | cas.SX | cas.DM):
            if sigma_points_vector.shape[0] == self.nb_states * self.nb_sigma_points:
                nb_states = self.nb_states
            elif sigma_points_vector.shape[0] == self.nb_q * self.nb_sigma_points:
                nb_states = self.nb_q
            else:
                raise RuntimeError(f"The shape of sigma_points_vector {sigma_points_vector.shape[0]} must be nb_states {self.nb_states} * nb_sigma_points {self.nb_sigma_points}.")

            # NOT GOOD
            # mean_sigma = self.cx.zeros(nb_states)
            # for i_state in range(nb_states):
            #     mean_sigma[i_state] = cas.sum1(sigma_points_vector[self.nb_sigma_points * i_state :self.nb_sigma_points * (i_state + 1)]) / self.nb_sigma_points

            mean_sigma = self.cx.zeros(nb_states)
            for i_sigma in range(self.nb_sigma_points):
                mean_sigma += sigma_points_vector[nb_states * i_sigma :nb_states * (i_sigma + 1)]
            mean_sigma /= self.nb_sigma_points

            return mean_sigma

        # def get_states_list(self, node: int) -> list[cas.MX | cas.SX | cas.DM]:
        #     """
        #     Get a list of symbolic variables for all states at the first node.
        #     """
        #     states = None
        #     for state_name in self.state_names:
        #         if state_name in ["q", "qdot"]:
        #             # We remove them from x because otherwise q and qdot are not independent and we cannot declare a casadi function
        #             this_state = self.padded_x_list[node][state_name]
        #         else:
        #             this_state = self.x_list[node][state_name]
        #         if this_state is not None:
        #             if states is None:
        #                 states = this_state
        #             else:
        #                 states = cas.vertcat(states, this_state)
        #     return [states]


        def get_states_list(self, node: int) -> list[cas.MX | cas.SX | cas.DM]:
            states_list = []
            for i_sigma in range(self.nb_sigma_points):
                states = None
                for state_name in self.state_names:
                    if state_name in ["q", "qdot"]:
                        # We remove them from x because otherwise q and qdot are not independent and we cannot declare a casadi function
                        this_state = self.padded_x_list[node][state_name][i_sigma]
                    else:
                        this_state = self.x_list[node][state_name][i_sigma]
                    if this_state is not None:
                        if states is None:
                            states = this_state
                        else:
                            states = cas.vertcat(states, this_state)
                states_list += [states]
            return states_list


        # def get_padded_states(self, node: int):
        #     """
        #     Get a state vector padded so that the shape if the node^th state has the same shape as the 0^th state.
        #     This is wack, but useful for variational.
        #     """
        #     states = None
        #     for state_name in self.state_names:
        #         this_state = self.x_list[node][state_name]
        #
        #         # Padding
        #         if this_state is None:
        #             # We use inf so that if it is accessed by accident the optimization will crash
        #             this_state = cas.DM.ones(self.x_list[0][state_name].shape[0]) * cas.inf
        #
        #         # Append output states
        #         if states is None:
        #             states = this_state
        #         else:
        #             states = cas.vertcat(states, this_state)
        #     return states


        def get_padded_states(self, node: int):
            """
            Get a state vector padded so that the shape if the node^th state has the same shape as the 0^th state.
            This is wack, but useful for variational.
            """
            states = None
            for i_random in range(self.nb_sigma_points):
                for state_name in self.state_names:
                    this_state = self.x_list[node][state_name][i_random]

                    # Padding
                    if this_state is None:
                        # We use inf so that if it is accessed by accident the optimization will crash
                        this_state = cas.DM.ones(self.x_list[0][state_name][i_random].shape[0]) * cas.inf

                    # Append output states
                    if states is None:
                        states = this_state
                    else:
                        states = cas.vertcat(states, this_state)
            return states

        def get_states_matrix(self, node: int):
            states_matrix = None
            for i_random in range(self.nb_sigma_points):
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
            for i_random in range(self.nb_sigma_points):
                this_state = self.x_list[node][name][i_random]
                if this_state is None:
                    this_state = cas.DM.ones(self.x_list[node]["q"][i_random].shape[0]) * np.nan
                if states_matrix is None:
                    states_matrix = this_state
                else:
                    states_matrix = cas.horzcat(states_matrix, this_state)
            return states_matrix


        def get_specific_collocation_point(self, name: str, node: int, sigma_point: int,  point: int):
            return self.z_list[node][name][sigma_point][point]

        def get_collocation_point(self, name: int, node: int):
            collocation_points_matrix = None
            for i_collocation in range(self.nb_collocation_points):
                collocation_points_vector = None
                for i_sigma in range(self.nb_sigma_points):
                    if collocation_points_vector is None:
                        collocation_points_vector = self.z_list[node][name][i_sigma][i_collocation]
                    else:
                        collocation_points_vector = cas.vertcat(
                            collocation_points_vector, self.z_list[node][name][i_sigma][i_collocation]
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
                for i_sigma in range(self.nb_sigma_points):
                    for state_name in self.state_names:
                        this_collocation = self.z_list[node][state_name][i_sigma][i_collocation]
                        if this_collocation is not None:
                            if collocation_points_vector is None:
                                collocation_points_vector = this_collocation
                            else:
                                collocation_points_vector = cas.vertcat(collocation_points_vector, this_collocation)
                if collocation_points_matrix is None:
                    collocation_points_matrix = collocation_points_vector
                else:
                    collocation_points_matrix = cas.horzcat(
                        collocation_points_matrix,
                        collocation_points_vector,
                    )
            return collocation_points_matrix

        # def get_chol_cov(self, node: int):
        #     return self.chol_cov_list[node]["chol_cov"]
        #
        # def get_chol_cov_matrix(self, node: int):
        #     nb_col_cov_variables = self.chol_cov_list[node]["chol_cov"].shape[0]
        #     i = 0
        #     nb_chol_col = 0
        #     while i < nb_col_cov_variables:
        #         nb_chol_col += 1
        #         i += nb_chol_col
        #     return self.reshape_vector_to_cholesky_matrix(
        #         self.chol_cov_list[node]["chol_cov"],
        #         (nb_chol_col, nb_chol_col),
        #     )

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
            # # X
            # for state_name in self.state_names:
            #     if node == 0 or node == self.n_shooting or not (state_name == "qdot" and skip_qdot_variables):
            #         vector += [self.x_list[node][state_name]]
            # # CHOLESKY COV
            # vector += [self.chol_cov_list[node]["chol_cov"]]

            for i_random in range(self.nb_sigma_points):
                for state_name in self.state_names:
                    if node == 0 or node == self.n_shooting or not (state_name == "qdot" and skip_qdot_variables):
                        vector += [self.x_list[node][state_name][i_random]]

            # Z
            for i_sigma in range(self.nb_sigma_points):
                for i_collocation in range(self.nb_collocation_points):
                    for state_name in self.state_names:
                        if not (state_name == "qdot" and skip_qdot_variables):
                            if node < self.n_shooting:
                                vector += [self.z_list[node][state_name][i_sigma][i_collocation]]
                            else:
                                if not keep_only_symbolic:
                                    vector += [self.z_list[node][state_name][i_sigma][i_collocation]]
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

        # def get_states_time_series_vector(self, name: str):
        #     n_components = self.x_list[0][name].shape[0]
        #     vector = np.zeros((n_components, self.n_shooting + 1))
        #     for i_node in range(self.n_shooting + 1):
        #         vector[:, i_node] = np.array(self.x_list[i_node][name]).flatten()
        #     return vector

        def get_states_time_series_vector(self, name: str):
            n_components = self.x_list[0][name][0].shape[0]
            vector = np.zeros((n_components, self.n_shooting + 1, self.nb_sigma_points))
            for i_node in range(self.n_shooting + 1):
                for i_random in range(self.nb_sigma_points):
                    vector[:, i_node, i_random] = np.array(self.x_list[i_node][name][i_random]).flatten()
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
                # # X
                # for state_name in self.state_names:
                #     if (
                #         i_node == 0
                #         or i_node == self.n_shooting
                #         or not (state_name == "qdot" and qdot_variables_skipped)
                #     ):
                #         n_components = self.state_indices[state_name].stop - self.state_indices[state_name].start
                #         self.x_list[i_node][state_name] = vector[offset : offset + n_components]
                #         offset += n_components
                #
                # # CHOLESKY COV
                # nb_col_cov_variables = sum([i+1 for i in range(nb_states)])
                # self.chol_cov_list[i_node]["chol_cov"] = vector[offset : offset + nb_col_cov_variables]
                # offset += nb_col_cov_variables

                for i_random in range(self.nb_sigma_points):
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
                for i_sigma in range(self.nb_sigma_points):
                    for i_collocation in range(self.nb_collocation_points):
                        for state_name in self.state_names:
                            if not (state_name == "qdot" and qdot_variables_skipped):
                                if not only_has_symbolics or i_node < self.n_shooting:
                                    n_components = (
                                        self.state_indices[state_name].stop - self.state_indices[state_name].start
                                    )
                                    self.z_list[i_node][state_name][i_sigma][i_collocation] = vector[
                                        offset : offset + n_components
                                    ]
                                    offset += n_components

                # U
                for control_name in self.control_names:
                    n_components = self.control_indices[control_name].stop - self.control_indices[control_name].start
                    self.u_list[i_node][control_name] = vector[offset : offset + n_components]
                    offset += n_components

        # --- Get array --- #
        # def get_states_array(self) -> np.ndarray:
        #     states_var_array = np.zeros((self.nb_states, self.n_shooting + 1))
        #     for i_node in range(self.n_shooting + 1):
        #         for state_name in self.state_names:
        #             states = np.array(self.x_list[i_node][state_name])
        #             states_var_array[self.state_indices[state_name], i_node] = states.reshape(
        #                 -1,
        #             )
        #     return states_var_array


        def get_states_array(self) -> np.ndarray:
            states_var_array = np.zeros((self.nb_states, self.n_shooting + 1, self.nb_sigma_points))
            for i_random in range(self.nb_sigma_points):
                for i_node in range(self.n_shooting + 1):
                    states = None
                    for state_name in self.state_names:
                        this_state = np.array(self.x_list[i_node][state_name][i_random]).reshape(-1, )
                        if np.all(this_state == None):
                            this_state = np.ones((np.array(self.x_list[i_node]["q"][i_random]).size)) * np.nan
                        if states is None:
                            states = this_state
                        else:
                            states = np.hstack((states, this_state))
                    states_var_array[:, i_node, i_random] = states.reshape(
                        -1,
                    )
            return states_var_array


        # def get_chol_cov_array(self) -> np.ndarray:
        #     nb_chol_cov = self.chol_cov_list[0]["chol_cov"].shape[0]
        #     chol_cov_var_array = np.zeros((nb_chol_cov, self.n_shooting + 1))
        #     for i_node in range(self.n_shooting + 1):
        #         chol_cov_var_array[:, i_node] = self.chol_cov_list[i_node]["chol_cov"].reshape(
        #             -1,
        #         )
        #     return chol_cov_var_array

        def get_collocation_points_array(self) -> np.ndarray:
            collocation_points_var_array = np.zeros(
                (self.nb_states * self.nb_collocation_points, self.n_shooting + 1, self.nb_sigma_points)
            )
            for i_sigma in range(self.nb_sigma_points):
                for i_node in range(self.n_shooting + 1):
                    coll = None
                    for i_collocation in range(self.nb_collocation_points):
                        for state_name in self.state_names:
                            this_collocation = np.array(self.z_list[i_node][state_name][i_sigma][i_collocation]).reshape(-1, )
                            if coll is None:
                                coll = this_collocation
                            else:
                                coll = np.hstack((coll, this_collocation))
                    collocation_points_var_array[:, i_node, i_sigma] = coll.reshape(
                        -1,
                    )
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
            nb_sigma_points: int = 1,
            use_sx: bool = False,
        ):

            super().__init__(use_sx=use_sx)

            self.n_shooting = n_shooting
            self.nb_sigma_points = nb_sigma_points

            self.motor_noise = [[None for _ in range(nb_sigma_points)] for _ in range(n_shooting + 1)]
            self.sensory_noise = [[None for _ in range(nb_sigma_points)] for _ in range(n_shooting + 1)]
            self.motor_noises_numerical = [[None for _ in range(nb_sigma_points)] for _ in range(n_shooting + 1)]
            self.sensory_noises_numerical = [[None for _ in range(nb_sigma_points)] for _ in range(n_shooting + 1)]

        # --- Add --- #
        def add_motor_noise(self, node: int, sigma_point: int, value: cas.MX | cas.SX | cas.DM):
            self.motor_noise[node][sigma_point] = self.transform_to_dm(value)

        def add_sensory_noise(self, node: int, sigma_point: int, value: cas.MX | cas.SX | cas.DM):
            self.sensory_noise[node][sigma_point] = self.transform_to_dm(value)

        def add_motor_noise_numerical(self, node: int, sigma_point: int, value: cas.MX | cas.SX | cas.DM):
            self.motor_noises_numerical[node][sigma_point] = self.transform_to_dm(value)

        def add_sensory_noise_numerical(self, node: int, sigma_point: int, value: cas.MX | cas.SX | cas.DM):
            self.sensory_noises_numerical[node][sigma_point] = self.transform_to_dm(value)

        # --- Get vectors --- #
        def get_noise_single(self, node: int) -> cas.MX | cas.SX:
            noise_single = None
            for i_sigma in range(self.nb_sigma_points):

                this_noise = self.cx()
                if self.motor_noise[node][i_sigma] is not None:
                    this_noise = cas.vertcat(this_noise, self.motor_noise[node][i_sigma])
                if self.sensory_noise[node][i_sigma] is not None:
                    this_noise = cas.vertcat(this_noise, self.sensory_noise[node][i_sigma])

                if noise_single is None:
                    noise_single = this_noise
                else:
                    noise_single = cas.vertcat(
                        noise_single,
                        this_noise,
                    )
            return noise_single

        def get_noise_matrix(self, node: int) -> cas.MX | cas.SX:
            return cas.diag(self.get_one_noise(node, sigma_point=0))

        def get_sensory_noise(self, node: int) -> cas.MX | cas.SX:
            noise_single = None
            for i_sigma in range(self.nb_sigma_points):
                if noise_single is None:
                    noise_single = self.sensory_noise[node][i_sigma]
                else:
                    noise_single = cas.vertcat(
                        noise_single,
                        self.sensory_noise[node][i_sigma],
                    )
            return noise_single

        def get_motor_noise(self, node: int) -> cas.MX | cas.SX:
            noise_single = None
            for i_sigma in range(self.nb_sigma_points):
                if noise_single is None:
                    noise_single = self.motor_noise[node][i_sigma]
                else:
                    noise_single = cas.vertcat(
                        noise_single,
                        self.motor_noise[node][i_sigma],
                    )
            return noise_single

        def get_one_sensory_noise(self, node: int, sigma_point: int) -> cas.MX | cas.SX:
            return self.sensory_noise[node][sigma_point]

        def get_one_motor_noise(self, node: int, sigma_point: int) -> cas.MX | cas.SX:
            return self.motor_noise[node][sigma_point]

        def get_one_noise(self, node: int, sigma_point: int) -> cas.DM:
            if self.motor_noise[node][sigma_point] is None:
                return self.sensory_noise[node][sigma_point]
            elif self.sensory_noise[node][sigma_point] is None:
                return self.motor_noise[node][sigma_point]
            else:
                return cas.vertcat(
                    self.motor_noise[node][sigma_point],
                    self.sensory_noise[node][sigma_point],
                )

        def get_one_noise_numerical(self, node: int, sigma_point: int) -> cas.DM:
            if self.motor_noises_numerical[node][sigma_point] is None:
                return self.sensory_noises_numerical[node][sigma_point]
            elif self.sensory_noises_numerical[node][sigma_point] is None:
                return self.motor_noises_numerical[node][sigma_point]
            else:
                return cas.vertcat(
                    self.motor_noises_numerical[node][sigma_point],
                    self.sensory_noises_numerical[node][sigma_point],
                )

        def get_one_sensory_noise_numerical(self, node: int, sigma_point: int) -> cas.DM:
            return self.sensory_noises_numerical[node][sigma_point]

        def get_one_motor_noise_numerical(self, node: int, sigma_point: int) -> cas.DM:
            return self.motor_noises_numerical[node][sigma_point]

        def get_one_vector_numerical(self, node: int):
            vector = None
            for i_sigma in range(self.nb_sigma_points):
                if vector is None:
                    vector = self.get_one_noise_numerical(node, i_sigma)
                else:
                    vector = cas.vertcat(vector, self.get_one_noise_numerical(node, i_sigma))
            return vector

        def get_full_matrix_numerical(self):
            vector = []
            for i_node in range(self.n_shooting + 1):
                vector += [self.get_one_vector_numerical(i_node)]
            return cas.horzcat(*vector)

    @property
    def name(self) -> str:
        return "UnscentedTransform"

    def declare_variables(
        self,
        ocp_example: ExampleAbstract,
        dynamics_transcription: TranscriptionAbstract,
        states_lower_bounds: dict[str, np.ndarray],
        controls_lower_bounds: dict[str, np.ndarray],
    ) -> Variables:
        """
        Declare all symbolic variables for the states and controls with their bounds and initial guesses
        """
        if dynamics_transcription.name in ["Variational", "VariationalPolynomial"]:
            q_only = True
        elif dynamics_transcription.name in ["DirectCollocationPolynomial", "DirectCollocationTrapezoidal", "DirectMultipleShooting"]:
            q_only = False
        else:
            raise NotImplementedError(f"Transcription {dynamics_transcription.name} not implemented for UnscentedTransform discretization.")

        nb_sigma_points = ocp_example.model.nb_sigma_points(q_only=q_only)
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
            nb_sigma_points=nb_sigma_points,
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
                # # X
                # if i_node == 0 or i_node == n_shooting or not (state_name == "qdot" and skip_qdot_variables):
                #     n_components = states_lower_bounds[state_name].shape[0]
                #     if use_sx:
                #         mean_x = cas.SX.sym(f"{state_name}_{i_node}", n_components)
                #     else:
                #         mean_x = cas.MX.sym(f"{state_name}_{i_node}", n_components)
                #     variables.add_state(state_name, i_node, mean_x)
                # variables.add_padded_state(state_name, i_node)


                n_components = states_lower_bounds[state_name].shape[0]
                for i_random in range(nb_sigma_points):
                    if i_node == 0 or i_node == n_shooting or not (state_name == "qdot" and skip_qdot_variables):
                        if use_sx:
                            x_sym = cas.SX.sym(f"{state_name}_{i_node}_{i_random}", n_components)
                        else:
                            x_sym = cas.MX.sym(f"{state_name}_{i_node}_{i_random}", n_components)
                        variables.add_state(state_name, i_node, i_random, x_sym)
                    variables.add_padded_state(state_name, i_node, i_random)


                # Z
                if isinstance(self.dynamics_transcription, (DirectCollocationPolynomial, VariationalPolynomial)):
                    # Create the symbolic variables for the states collocation points
                    if not (state_name == "qdot" and skip_qdot_variables):
                        for i_sigma in range(nb_sigma_points):
                            for i_collocation in range(nb_collocation_points):
                                if i_node < n_shooting:
                                    if use_sx:
                                        z_sym = cas.SX.sym(
                                            f"{state_name}_{i_node}_{i_sigma}_{i_collocation}_z", n_components
                                        )
                                    else:
                                        z_sym = cas.MX.sym(
                                            f"{state_name}_{i_node}_{i_sigma}_{i_collocation}_z", n_components
                                        )
                                else:
                                    if use_sx:
                                        z_sym = cas.SX.zeros(n_components)
                                    else:
                                        z_sym = cas.MX.zeros(n_components)
                                variables.add_collocation_point(state_name, i_node, i_sigma, i_collocation, z_sym)

            # # Create the symbolic variables for the state covariance
            # if isinstance(self.dynamics_transcription, (Variational, VariationalPolynomial)):
            #     nb_chol_cov_variables = sum([i+1 for i in range(ocp_example.model.nb_q)])
            # else:
            #     nb_chol_cov_variables = sum([i+1 for i in range(ocp_example.model.nb_states)])
            #
            # if use_sx:
            #     chol_cov = cas.SX.sym(f"chol_cov_{i_node}", nb_chol_cov_variables)
            # else:
            #     chol_cov = cas.MX.sym(f"chol_cov_{i_node}", nb_chol_cov_variables)
            # variables.add_chol_cov(i_node, chol_cov)

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
        if isinstance(self.dynamics_transcription, (Variational, VariationalPolynomial)):
            nb_states = ocp_example.model.nb_q
            q_only = True
        else:
            nb_states = ocp_example.model.nb_states
            q_only = False

        nb_sigma_points = ocp_example.model.nb_sigma_points(q_only=q_only)
        n_shooting = ocp_example.n_shooting
        nb_collocation_points = self.dynamics_transcription.nb_collocation_points
        nb_m_points = self.dynamics_transcription.nb_m_points
        state_names = list(ocp_example.model.state_indices.keys())
        control_names = list(ocp_example.model.control_indices.keys())

        w_lower_bound = self.Variables(
            n_shooting=n_shooting,
            nb_collocation_points=nb_collocation_points,
            nb_m_points=nb_m_points,
            nb_sigma_points=nb_sigma_points,
            state_indices=ocp_example.model.state_indices,
            control_indices=ocp_example.model.control_indices,
        )
        w_upper_bound = self.Variables(
            n_shooting=n_shooting,
            nb_collocation_points=nb_collocation_points,
            nb_m_points=nb_m_points,
            nb_sigma_points=nb_sigma_points,
            state_indices=ocp_example.model.state_indices,
            control_indices=ocp_example.model.control_indices,
        )
        w_initial_guess = self.Variables(
            n_shooting=n_shooting,
            nb_collocation_points=nb_collocation_points,
            nb_m_points=nb_m_points,
            nb_sigma_points=nb_sigma_points,
            state_indices=ocp_example.model.state_indices,
            control_indices=ocp_example.model.control_indices,
        )

        w_initial_guess.add_time(ocp_example.final_time)
        w_lower_bound.add_time(ocp_example.min_time)
        w_upper_bound.add_time(ocp_example.max_time)

        for i_node in range(n_shooting + 1):

            # X - states
            for state_name in state_names:
                # if i_node == 0 and (state_name in ocp_example.initial_states_to_impose):
                #     # Initial states are imposed
                #     this_init = states_initial_guesses[state_name][:, i_node].tolist()
                #     w_lower_bound.add_state(state_name, i_node, this_init)
                #     w_upper_bound.add_state(state_name, i_node, this_init)
                #     w_initial_guess.add_state(state_name, i_node, this_init)
                # else:
                #     w_lower_bound.add_state(state_name, i_node, states_lower_bounds[state_name][:, i_node])
                #     w_upper_bound.add_state(state_name, i_node, states_upper_bounds[state_name][:, i_node])
                #     w_initial_guess.add_state(state_name, i_node, states_initial_guesses[state_name][:, i_node])


                # Some randomness is given on the state initial guess
                this_init = states_initial_guesses[state_name][:, i_node].tolist()
                initial_configuration = np.array(
                    np.random.normal(
                        loc=this_init * nb_sigma_points,
                        scale=np.repeat(
                            ocp_example.initial_state_variability[ocp_example.model.state_indices[state_name]],
                            nb_sigma_points,
                        ),
                    )
                ).reshape(len(this_init), nb_sigma_points, order="F")

                for i_random in range(nb_sigma_points):
                    if i_node == 0 and (state_name in ocp_example.initial_states_to_impose):
                        # Impose initial state covariance
                        w_lower_bound.add_state(state_name, i_node, i_random, initial_configuration[:, i_random])
                        w_upper_bound.add_state(state_name, i_node, i_random, initial_configuration[:, i_random])
                        w_initial_guess.add_state(state_name, i_node, i_random, initial_configuration[:, i_random])
                    else:
                        w_lower_bound.add_state(state_name, i_node, i_random, states_lower_bounds[state_name][:, i_node])
                        w_upper_bound.add_state(state_name, i_node, i_random, states_upper_bounds[state_name][:, i_node])
                        w_initial_guess.add_state(state_name, i_node, i_random, initial_configuration[:, i_random])


            # # CHOLESKY COV - covariance
            # cov_init = np.diag(ocp_example.initial_state_variability.tolist()) ** 2
            # # Declare cov variables
            # if isinstance(self.dynamics_transcription, (Variational, VariationalPolynomial)):
            #     nb_chol_cov_variables = sum([i+1 for i in range(ocp_example.model.nb_q)])
            # else:
            #     nb_chol_cov_variables = sum([i+1 for i in range(ocp_example.model.nb_states)])
            # l_init = (
            #     np.array(w_initial_guess.reshape_cholesky_matrix_to_vector(cov_init[:nb_states, :nb_states])).flatten().tolist()
            # )
            #
            # if i_node == 0:
            #     # Initial covariance is imposed
            #     w_initial_guess.add_chol_cov(i_node, l_init)
            #     w_lower_bound.add_chol_cov(i_node, l_init)
            #     w_upper_bound.add_chol_cov(i_node, l_init)
            # else:
            #     w_initial_guess.add_chol_cov(i_node, l_init)
            #     w_lower_bound.add_chol_cov(i_node, [-cas.inf] * nb_chol_cov_variables)
            #     w_upper_bound.add_chol_cov(i_node, [cas.inf] * nb_chol_cov_variables)

            # Z - collocation points
            if isinstance(self.dynamics_transcription, (DirectCollocationPolynomial, VariationalPolynomial)):
                for state_name in state_names:
                    for i_sigma in range(nb_sigma_points):
                        # The last interval does not have collocation points
                        for i_collocation in range(nb_collocation_points):
                            if i_node < n_shooting:
                                # Add bounds and initial guess as linear interpolation between the two nodes
                                w_lower_bound.add_collocation_point(
                                    state_name,
                                    i_node,
                                    i_sigma,
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
                                    i_sigma,
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
                                        i_sigma,
                                        i_collocation,
                                        self.interpolate_between_nodes(
                                            var_pre=states_initial_guesses[state_name][:, i_node],
                                            var_post=states_initial_guesses[state_name][:, i_node + 1],
                                            time_ratio=i_collocation / (nb_collocation_points - 1),
                                        ).tolist(),
                                    )
                                else:
                                    w_initial_guess.add_collocation_point(
                                        state_name,
                                        i_node,
                                        i_sigma,
                                        i_collocation,
                                        (
                                                collocation_points_initial_guesses[state_name][:, i_collocation, i_node]
                                        ).tolist(),
                                    )
                            elif i_collocation == 0:
                                # Add bounds and initial guess as linear interpolation between the two nodes
                                w_lower_bound.add_collocation_point(
                                    state_name,
                                    i_node,
                                    i_sigma,
                                    i_collocation,
                                    states_lower_bounds[state_name][:, i_node].tolist(),
                                )
                                w_upper_bound.add_collocation_point(
                                    state_name,
                                    i_node,
                                    i_sigma,
                                    i_collocation,
                                    states_upper_bounds[state_name][:, i_node].tolist(),
                                )
                                if collocation_points_initial_guesses is None:
                                    w_initial_guess.add_collocation_point(
                                        state_name,
                                        i_node,
                                        i_sigma,
                                        i_collocation,
                                        (states_initial_guesses[state_name][:, i_node]).tolist(),
                                    )
                                else:
                                    w_initial_guess.add_collocation_point(
                                        state_name,
                                        i_node,
                                        i_sigma,
                                        i_collocation,
                                        (
                                                collocation_points_initial_guesses[state_name][:, i_collocation, i_node]
                                        ).tolist(),
                                    )
                            else:
                                nb_components = states_lower_bounds[state_name].shape[0]
                                w_lower_bound.add_collocation_point(
                                    state_name, i_node, i_sigma, i_collocation, [0] * nb_components
                                )
                                w_upper_bound.add_collocation_point(
                                    state_name, i_node, i_sigma, i_collocation, [0] * nb_components
                                )
                                w_initial_guess.add_collocation_point(
                                    state_name, i_node, i_sigma, i_collocation, [0] * nb_components
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
        dynamics_transcription: TranscriptionAbstract,
        n_shooting: int,
        nb_random: int,
        motor_noise_magnitude: np.ndarray,
        sensory_noise_magnitude: np.ndarray,
        seed: int = 0,
    ) -> NoisesAbstract:
        """
        Sample the noise values and declare the symbolic variables for the noises.
        """
        if dynamics_transcription.name in ["Variational", "VariationalPolynomial"]:
            q_only = True
            nb_states = ocp_example.model.nb_q
        elif dynamics_transcription.name in ["DirectCollocationPolynomial", "DirectCollocationTrapezoidal", "DirectMultipleShooting"]:
            q_only = False
            nb_states = ocp_example.model.nb_states
        else:
            raise NotImplementedError(f"Transcription {dynamics_transcription.name} not implemented for UnscentedTransform discretization.")

        nb_sigma_points = ocp_example.model.nb_sigma_points(q_only=q_only)
        noises_vector = self.Noises(n_shooting, nb_sigma_points=nb_sigma_points)
        n_motor_noises = motor_noise_magnitude.shape[0] if motor_noise_magnitude is not None else 0
        nb_references = sensory_noise_magnitude.shape[0] if sensory_noise_magnitude is not None else 0

        nb_noises = ocp_example.model.nb_noises
        noises_magnitude = cas.DM()
        if motor_noise_magnitude is not None:
            noises_magnitude = cas.vertcat(noises_magnitude, motor_noise_magnitude)
        if sensory_noise_magnitude is not None:
            noises_magnitude = cas.vertcat(noises_magnitude, sensory_noise_magnitude)
        noise_matrix = cas.diag(noises_magnitude)

        augmented_l_matrix = cas.vertcat(
            cas.horzcat(cas.DM.zeros(nb_states, nb_states), cas.DM.zeros(nb_states, nb_noises)),
            cas.horzcat(cas.DM.zeros(nb_noises, nb_states), noise_matrix),
        )

        for i_node in range(n_shooting + 1):
            for i_sigma in range(nb_sigma_points):
                if ocp_example.model.use_sx:
                    noises_vector.add_motor_noise(node=i_node, sigma_point=i_sigma, value=cas.SX.sym(f"motor_noise_{i_sigma}_{i_node}", n_motor_noises))
                    noises_vector.add_sensory_noise(node=i_node, sigma_point=i_sigma, value=cas.SX.sym(f"sensory_noise_{i_sigma}_{i_node}", nb_references))
                else:
                    noises_vector.add_motor_noise(node=i_node, sigma_point=i_sigma, value=cas.MX.sym(f"motor_noise_{i_sigma}_{i_node}", n_motor_noises))
                    noises_vector.add_sensory_noise(node=i_node, sigma_point=i_sigma, value=cas.MX.sym(f"sensory_noise_{i_sigma}_{i_node}", nb_references))

            if motor_noise_magnitude is not None:
                motor_noise_indices = range(nb_states + ocp_example.model.motor_noise_indices.start,
                                            nb_states + ocp_example.model.motor_noise_indices.stop)
                noises_vector.add_motor_noise_numerical(node=i_node, sigma_point=0, value=cas.DM.zeros(nb_states))
                index = int((ocp_example.model.nb_sigma_points(q_only=q_only) - 1) / 2)
                for i_sigma in range(index):
                    noises_vector.add_motor_noise_numerical(node=i_node, sigma_point=1+i_sigma, value=augmented_l_matrix[motor_noise_indices, i_sigma])
                    noises_vector.add_motor_noise_numerical(node=i_node, sigma_point=1+index+i_sigma, value=-augmented_l_matrix[motor_noise_indices, i_sigma])
            if sensory_noise_magnitude is not None:
                sensory_noise_indices = range(nb_states + ocp_example.model.sensory_noise_indices.start,
                                              nb_states + ocp_example.model.sensory_noise_indices.stop)
                noises_vector.add_sensory_noise_numerical(node=i_node, sigma_point=0, value=cas.DM.zeros(nb_states))
                index = int((ocp_example.model.nb_sigma_points(q_only=q_only) - 1) / 2)
                for i_sigma in range(index):
                    noises_vector.add_sensory_noise_numerical(node=i_node, sigma_point=1+i_sigma,
                                                            value=augmented_l_matrix[sensory_noise_indices, i_sigma])
                    noises_vector.add_sensory_noise_numerical(node=i_node, sigma_point=1+index + i_sigma,
                                                            value=-augmented_l_matrix[sensory_noise_indices, i_sigma])

        return noises_vector

    # def get_mean_states(
    #     self,
    #     variables_vector: VariablesAbstract,
    #     node: int,
    #     squared: bool = False,
    # ):
    #     exponent = 2 if squared else 1
    #     states_mean = variables_vector.get_states(node) ** exponent
    #
    #     return states_mean


    def get_mean_states(
        self,
        variables_vector: VariablesAbstract,
        node: int,
        squared: bool = False,
    ):
        states = variables_vector.get_states_matrix(node)

        exponent = 2 if squared else 1
        states_sq = states**exponent

        states_mean = cas.sum2(states_sq) / variables_vector.nb_sigma_points
        return states_mean


    # def get_covariance(
    #     self,
    #     variables_vector: VariablesAbstract,
    #     node: int,
    #     is_matrix: bool = False,
    # ):
    #     if is_matrix:
    #         l = variables_vector.get_chol_cov_matrix(node)
    #         cov = l @ l.T
    #     else:
    #         raise NotImplementedError("For cholesky decomposed matrix, this is ambiguous.")
    #         cov = variables_vector.get_chol_cov(node)
    #     return cov


    def get_covariance(
        self,
        variables_vector: VariablesAbstract,
        node: int,
        is_matrix: bool = False,
    ):
        states = variables_vector.get_states_matrix(node)
        states_mean = self.get_mean_states(variables_vector, node, squared=False)
        # np.mean(np.array(states).reshape(4, variables_vector.nb_random), axis=1) OK

        diff = states - states_mean
        covariance = (diff @ diff.T) / (variables_vector.nb_sigma_points - 1)

        if is_matrix:
            return covariance
        else:
            return variables_vector.reshape_matrix_to_vector(covariance)


    def get_tau(
            self,
            ocp_example: ExampleAbstract,
            x: list[cas.MX | cas.SX | np.ndarray],
            u: list[cas.MX | cas.SX | np.ndarray],
    ):
        if "tau" in ocp_example.model.state_indices.keys():
            tau = x[0][ocp_example.model.tau_indices]
        elif "tau" in ocp_example.model.control_indices.keys():
            tau = u[ocp_example.model.control_indices["tau"]]
        else:
            tau = None
        return tau

    # def get_reference(
    #     self,
    #     ocp_example: ExampleAbstract,
    #     q: list[cas.MX | cas.SX | np.ndarray],
    #     qdot: list[cas.MX | cas.SX | np.ndarray],
    #     x: list[cas.MX | cas.SX | np.ndarray],
    #     u: cas.MX | cas.SX | np.ndarray,
    # ) -> cas.MX | cas.SX | np.ndarray:
    #
    #     tau = self.get_tau(ocp_example, x, u)
    #     ref = ocp_example.model.sensory_output(q[0], qdot[0], tau, cas.DM.zeros(ocp_example.model.nb_references))
    #     return ref


    def get_reference(
        self,
        ocp_example: ExampleAbstract,
        q: list[cas.MX | cas.SX | np.ndarray],
        qdot: list[cas.MX | cas.SX | np.ndarray],
        x: list[cas.MX | cas.SX | np.ndarray],
        u: cas.MX | cas.SX | np.ndarray,
    ):

        tau = None
        if "tau" in ocp_example.model.control_indices.keys():
            tau = u[ocp_example.model.control_indices["tau"]]

        if ocp_example.model.nb_references > 0:

            ref = None
            for i_random in range(ocp_example.model.nb_sigma_points):
                q_this_time = q[i_random]
                qdot_this_time = qdot[i_random]
                if "tau" in ocp_example.model.state_indices.keys():
                    tau = x[i_random][ocp_example.model.tau_indices]

                if ref is None:
                    ref = ocp_example.model.sensory_output(
                    q=q_this_time,
                    qdot=qdot_this_time,
                    tau=tau,
                    sensory_noise=np.zeros((ocp_example.model.nb_references, )),
                )
                else:
                    ref += ocp_example.model.sensory_output(
                    q=q_this_time,
                    qdot=qdot_this_time,
                    tau=tau,
                    sensory_noise=np.zeros((ocp_example.model.nb_references, )),
                )
            ref /= ocp_example.model.nb_sigma_points
            if isinstance(x[0], np.ndarray):
                ref = np.array(ref).reshape(-1, )

        else:
            if isinstance(x[0], np.ndarray):
                ref = np.zeros((0, 1))
            else:
                ref = cas.DM.zeros(0, 1)

        return ref

    def get_mean_marker(
        self,
        ocp_example: ExampleAbstract,
        x: cas.MX | cas.SX | np.ndarray,
        u: cas.MX | cas.SX | np.ndarray,
    ) -> cas.MX | cas.SX | np.ndarray:

        n_components = ocp_example.model.q_indices.stop - ocp_example.model.q_indices.start
        q = x[:n_components]
        ref = ocp_example.model.marker_position(q)
        return ref

    def get_sensory_variance(
        self,
        ocp_example: ExampleAbstract,
        q: list[cas.MX | cas.SX | np.ndarray],
        qdot: list[cas.MX | cas.SX | np.ndarray],
        x: list[cas.MX | cas.SX | np.ndarray],
        u: cas.MX | cas.SX | np.ndarray,
        cov_matrix: cas.MX | cas.SX | np.ndarray,
    ):
        # No noise for mean
        tau = self.get_tau(ocp_example, x, u)
        dsensory_dq = cas.jacobian(
            ocp_example.model.sensory_output(q[0], qdot[0], tau, cas.DM.zeros(ocp_example.model.nb_references)),
            q[0],
        )
        sensory_variance = dsensory_dq @ cov_matrix[ocp_example.model.q_indices, ocp_example.model.q_indices] @ cas.transpose(dsensory_dq)

        return cas.diag(sensory_variance)

    def state_dynamics(
            self,
            ocp_example: ExampleAbstract,
            x: cas.MX | cas.SX | np.ndarray,
            u: cas.MX | cas.SX | np.ndarray,
            noise: cas.MX | cas.SX | np.ndarray,
            with_q_qdot: bool = True,
    ) -> cas.MX | cas.SX | np.ndarray:

        if isinstance(self.dynamics_transcription, (Variational, VariationalPolynomial)):
            nb_states = ocp_example.model.nb_q
        else:
            nb_states = ocp_example.model.nb_states

        # Get q and qdot from the states, since state_dynamics should not be used by Variational and VariationalPolynomial
        q = [x[np.array(ocp_example.model.q_indices)]]
        qdot = [x[np.array(ocp_example.model.qdot_indices)]]
        # Mean state
        ref_mean = self.get_reference(
            ocp_example=ocp_example,
            q=q,
            qdot=qdot,
            x=x,
            u=u,
        )
        dxdt_mean = ocp_example.model.dynamics(
            x[:nb_states],
            u,
            ref_mean,
            noise,
            with_q_qdot,
        )

        return dxdt_mean

    @cache_function
    def get_non_conservative_forces(
        self,
        ocp_example: ExampleAbstract,
        q: list[cas.MX | cas.SX],
        qdot: list[cas.MX | cas.SX],
        padded_x: list[cas.MX | cas.SX],
        u: cas.MX | cas.SX,
        noise: cas.MX | cas.SX,
    ) -> cas.Function:

        ref = self.get_reference(ocp_example, q[0], qdot[0], padded_x[0], u)

        f = ocp_example.model.non_conservative_forces(
            q[0],
            qdot[0],
            padded_x[0],
            u,
            noise,
            ref,
        )
        return cas.Function(
            "NonConservativeForces",
            [
                cas.vertcat(*q),
                cas.vertcat(*qdot),
                cas.vertcat(*padded_x),
                u,
                noise,
            ],
            [f],
        )

    @cache_function
    def get_lagrangian(
        self,
        ocp_example: ExampleAbstract,
        q: [cas.MX | cas.SX],
        qdot: [cas.MX | cas.SX],
        u: cas.MX | cas.SX,
    ) -> cas.Function:
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

    @cache_function
    def get_lagrangian_jacobian(
        self,
        ocp_example: ExampleAbstract,
        discrete_lagrangian: cas.MX | cas.SX,
        variable_to_derivate_for: cas.MX | cas.SX,
    ) -> cas.Function:
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
        time_vector: np.ndarray,
    ):
        states_plots = []

        # # Placeholder to plot the variables
        # color = "tab:blue"
        # states_plots += axs[i_row, i_col].plot(time_vector, np.zeros_like(time_vector), marker=".", color=color)

        # Add states
        for i_random in range(ocp_example.nb_sigma_points):
            # Placeholder to plot the variables
            color = colors(i_random / ocp_example.nb_sigma_points)
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

        # states_data = variable_opt.get_states_time_series_vector(key)[i_col, :]
        #
        # # Update mean state plot
        # states_plots[i_state].set_ydata(
        #     states_data,
        # )
        # i_state += 1

        states_data = variable_opt.get_states_time_series_vector(key)
        for i_random in range(ocp_example.nb_sigma_points):
            states_plots[i_state].set_ydata(states_data[i_col, :, i_random])
            i_state += 1

        return i_state
