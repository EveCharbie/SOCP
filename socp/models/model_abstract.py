from abc import ABC, abstractmethod
import casadi as cas


class ModelAbstract(ABC):
    """Abstract base class for biomechanics models compatible with the transcriptions suggested."""

    # TODO

    @staticmethod
    def reshape_matrix_to_vector(matrix: cas.MX | cas.DM) -> cas.MX | cas.DM:
        matrix_shape = matrix.shape
        vector = type(matrix)()
        for i_shape in range(matrix_shape[0]):
            for j_shape in range(matrix_shape[1]):
                vector = cas.vertcat(vector, matrix[i_shape, j_shape])
        return vector

    @staticmethod
    def reshape_vector_to_matrix(vector: cas.MX | cas.DM, matrix_shape: tuple[int, ...]) -> cas.MX | cas.DM:
        matrix = type(vector).zeros(matrix_shape)
        idx = 0
        for i_shape in range(matrix_shape[0]):
            for j_shape in range(matrix_shape[1]):
                matrix[i_shape, j_shape] = vector[idx]
                idx += 1
        return matrix
