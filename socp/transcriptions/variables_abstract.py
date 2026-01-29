from abc import ABC
import casadi as cas
import numpy as np


class VariablesAbstract(ABC):

    @staticmethod
    def transform_to_dm(value: cas.SX | cas.DM | np.ndarray | list) -> cas.DM:
        if isinstance(value, np.ndarray):
            return cas.DM(value.flatten())
        elif isinstance(value, list):
            return cas.DM(np.array(value).flatten())
        else:
            return value

    @staticmethod
    def nb_cholesky_components(nb_components: int):
        nb_cholesky = 0
        for i in range(nb_components):
            nb_cholesky += nb_components - i
        return nb_cholesky

    @staticmethod
    def reshape_matrix_to_vector(matrix: cas.SX | cas.DM | np.ndarray) -> cas.SX | cas.DM | np.ndarray:
        if isinstance(matrix, np.ndarray):
            vector = np.array([])
        else:
            vector = type(matrix)()

        matrix_shape = matrix.shape
        for i_shape in range(matrix_shape[1]):
            for j_shape in range(matrix_shape[0]):
                if isinstance(matrix, np.ndarray):
                    vector = np.hstack((vector, matrix[j_shape, i_shape]))
                else:
                    vector = cas.vertcat(vector, matrix[j_shape, i_shape])

        if isinstance(matrix, np.ndarray):
            vector = np.reshape(vector, (-1, 1))
        return vector

    @staticmethod
    def reshape_vector_to_matrix(
        vector: cas.SX | cas.DM | np.ndarray, matrix_shape: tuple[int, ...]
    ) -> cas.SX | cas.DM | np.ndarray:
        if isinstance(vector, np.ndarray):
            matrix = np.zeros(matrix_shape)
        else:
            matrix = type(vector).zeros(matrix_shape)

        idx = 0
        for i_shape in range(matrix_shape[1]):
            for j_shape in range(matrix_shape[0]):
                matrix[j_shape, i_shape] = vector[idx]
                idx += 1
        return matrix

    @staticmethod
    def reshape_cholesky_matrix_to_vector(matrix: cas.SX | cas.DM | np.ndarray) -> cas.SX | cas.DM | np.ndarray:
        if isinstance(matrix, np.ndarray):
            vector = np.array([])
        else:
            vector = type(matrix)()

        matrix_shape = matrix.shape
        for i_shape in range(matrix_shape[0]):
            for j_shape in range(i_shape + 1):
                if isinstance(matrix, np.ndarray):
                    vector = np.hstack((vector, matrix[i_shape, j_shape]))
                else:
                    vector = cas.vertcat(vector, matrix[i_shape, j_shape])

        if isinstance(matrix, np.ndarray):
            vector = np.reshape(vector, (-1, 1))
        return vector

    @staticmethod
    def reshape_vector_to_cholesky_matrix(
        vector: cas.SX | cas.DM | np.ndarray, matrix_shape: tuple[int, ...]
    ) -> cas.SX | cas.DM | np.ndarray:
        if isinstance(vector, np.ndarray):
            matrix = np.zeros(matrix_shape)
        else:
            matrix = type(vector).zeros(matrix_shape)

        idx = 0
        for i_shape in range(matrix_shape[1]):
            for j_shape in range(i_shape + 1):
                matrix[i_shape, j_shape] = vector[idx]
                idx += 1
        return matrix

    # --- Add --- #
    def add_time(self, value: cas.SX | cas.DM):
        pass

    def add_state(self, name: str, node: int, value: cas.SX | cas.DM):
        pass

    def add_collocation_point(self, name: str, node: int, point: int, value: cas.SX | cas.DM):
        pass

    def add_cov(self, node: int, value: cas.SX | cas.DM):
        pass

    def add_m(self, node: int, point: int, value: cas.SX | cas.DM):
        pass

    def add_control(self, name: str, node: int, value: cas.SX | cas.DM):
        pass

    # --- Nb --- #
    @property
    def nb_states(self):
        pass

    @property
    def nb_controls(self):
        pass

    @property
    def nb_cov(self):
        pass

    # --- Get --- #
    def get_time(self):
        return []

    def get_state(self, name: str, node: int):
        return []

    def get_states(self, node: int):
        return []

    def get_specific_collocation_point(self, name: str, node: int, point: int):
        return []

    def get_collocation_point(self, node: int, point: int):
        return []

    def get_collocation_points(self, node: int):
        return []

    def get_cov(self, node: int):
        return []

    def get_m(self, node: int, point: int):
        return []

    def get_ms(self, node: int):
        return []

    def get_m_matrix(self, node: int):
        return []

    def get_cov_matrix(self, node: int):
        return []

    def get_control(self, name: str, node: int):
        return []

    def get_controls(self, node: int):
        return []

    # --- Get vectors --- #
    def get_one_vector(self, node: int, keep_only_symbolic: bool = False):
        pass

    def get_full_vector(self, keep_only_symbolic: bool = False):
        pass

    # --- Set vectors --- #
    def set_from_vector(self, vector: cas.DM, only_has_symbolics: bool, qdot_variables_skipped: bool):
        pass

    # --- Get array --- #
    def get_states_array(self) -> np.ndarray:
        pass

    def get_cov_array(self) -> np.ndarray:
        pass

    def get_m_array(self) -> np.ndarray:
        pass

    def get_collocation_points_array(self) -> np.ndarray:
        pass

    def get_controls_array(self) -> np.ndarray:
        pass

    def validate_vector(self):
        pass
