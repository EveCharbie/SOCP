from abc import ABC
import casadi as cas
import numpy as np

from ..transcriptions.variables_abstract import VariablesAbstract


class ModelAbstract(ABC):
    """Abstract base class for biomechanics models compatible with the transcriptions suggested."""

    def __init__(self, nb_random: int):

        self.nb_random = nb_random

        self.nb_q: int = None
        self.nb_states: int = None
        self.nb_controls: int = None
        self.nb_noised_controls: int = None
        self.nb_references: int = 0
        self.nb_k: int = 0
        self.nb_noised_states: int = None
        self.nb_noises: int = None

    """TODO: move this in a utils so that I don't have to deal with multiple levels"""
    @staticmethod
    def transform_to_dm(value: cas.SX | cas.DM | np.ndarray | list) -> cas.DM:
        return VariablesAbstract.transform_to_dm(value)

    @staticmethod
    def reshape_matrix_to_vector(self, matrix):
        return VariablesAbstract.reshape_matrix_to_vector(matrix)

    @staticmethod
    def reshape_vector_to_matrix(
        vector: cas.SX | cas.DM | np.ndarray, matrix_shape: tuple[int, ...]
    ) -> cas.SX | cas.DM | np.ndarray:
        return VariablesAbstract.reshape_vector_to_matrix(vector, matrix_shape)

    @staticmethod
    def reshape_cholesky_matrix_to_vector(matrix: cas.SX | cas.DM | np.ndarray) -> cas.SX | cas.DM | np.ndarray:
        return VariablesAbstract.reshape_cholesky_matrix_to_vector(matrix)

    @staticmethod
    def reshape_vector_to_cholesky_matrix(
        vector: cas.SX | cas.DM | np.ndarray, matrix_shape: tuple[int, ...]
    ) -> cas.SX | cas.DM | np.ndarray:
        return VariablesAbstract.reshape_vector_to_cholesky_matrix(vector, matrix_shape)
