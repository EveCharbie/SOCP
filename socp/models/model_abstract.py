from abc import ABC, abstractmethod
import casadi as cas
import numpy as np


class ModelAbstract(ABC):
    """Abstract base class for biomechanics models compatible with the transcriptions suggested."""

    def __init__(self, nb_random: int):

        self.nb_random = nb_random

        self.nb_q: int = None
        self.nb_states: int = None
        self.nb_controls: int = None
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
    def nb_cholesky_components(nb_components: int):
        nb_cholesky = 0
        for i in range(nb_components):
            nb_cholesky += nb_components - i
        return nb_cholesky

    @staticmethod
    def reshape_matrix_to_vector(matrix: cas.SX | cas.DM) -> cas.SX | cas.DM:
        matrix_shape = matrix.shape
        vector = type(matrix)()
        for i_shape in range(matrix_shape[0]):
            for j_shape in range(matrix_shape[1]):
                vector = cas.vertcat(vector, matrix[i_shape, j_shape])
        return vector

    @staticmethod
    def reshape_vector_to_matrix(vector: cas.SX | cas.DM, matrix_shape: tuple[int, ...]) -> cas.SX | cas.DM:
        matrix = type(vector).zeros(matrix_shape)
        idx = 0
        for i_shape in range(matrix_shape[0]):
            for j_shape in range(matrix_shape[1]):
                matrix[i_shape, j_shape] = vector[idx]
                idx += 1
        return matrix

    @staticmethod
    def reshape_cholesky_matrix_to_vector(matrix: cas.SX | cas.DM) -> cas.SX | cas.DM:
        matrix_shape = matrix.shape
        vector = type(matrix)()
        for i_shape in range(matrix_shape[0]):
            for j_shape in range(i_shape + 1):
                vector = cas.vertcat(vector, matrix[i_shape, j_shape])
        return vector

    @staticmethod
    def reshape_vector_to_cholesky_matrix(vector: cas.SX | cas.DM, matrix_shape: tuple[int, ...]) -> cas.SX | cas.DM:
        matrix = type(vector).zeros(matrix_shape)
        idx = 0
        for i_shape in range(matrix_shape[0]):
            for j_shape in range(i_shape + 1):
                matrix[i_shape, j_shape] = vector[idx]
                idx += 1
        return matrix
