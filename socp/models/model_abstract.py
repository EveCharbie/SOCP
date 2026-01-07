from abc import ABC, abstractmethod
import casadi as cas
import numpy as np


class ModelAbstract(ABC):
    """Abstract base class for biomechanics models compatible with the transcriptions suggested."""

    def __init__(self, n_random: int):

        self.n_random = n_random

        self.nb_q: int = None
        self.nb_states: int = None
        self.nb_noised_controls: int = None
        self.nb_references: int = None
        self.nb_k: int = None
        self.nb_noised_states: int = None
        self.nb_noises: int = None

        self.matrix_shape_k: tuple[int, int] = None
        self.matrix_shape_c: tuple[int, int] = None
        self.matrix_shape_a: tuple[int, int] = None
        self.matrix_shape_cov: tuple[int, int] = None
        self.matrix_shape_cov_cholesky: tuple[int, int] = None
        self.matrix_shape_m: tuple[int, int] = None

        self.friction_coefficients: np.ndarray = None

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
