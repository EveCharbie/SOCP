from abc import ABC, abstractmethod
import casadi as cas
import numpy as np

from ..transcriptions.variables_abstract import VariablesAbstract


class ModelAbstract(ABC):
    """Abstract base class for biomechanics models compatible with the transcriptions suggested."""

    def __init__(self, nb_random: int):

        self.nb_random = nb_random
        self.use_sx: bool = False

        self.nb_q: int = None
        self.nb_states: int = None
        self.nb_controls: int = None
        self.nb_noised_controls: int = None
        self.nb_references: int = 0
        self.nb_k: int = 0
        self.nb_noised_states: int = None
        self.nb_noises: int = None

    def nb_sigma_points(self, q_only: bool) -> int:
        if q_only:
            return 1 + 2 * (self.nb_q + self.nb_noises)
        else:
            return 1 + 2 * (self.nb_states + self.nb_noises)

    @property
    @abstractmethod
    def name_dof(self) -> list[str]:
        return []

    @property
    @abstractmethod
    def q_indices(self) -> range:
        return range(0, 1)

    @property
    @abstractmethod
    def qdot_indices(self) -> range:
        return range(0, 1)

    @property
    @abstractmethod
    def state_indices(self) -> dict[str, range]:
        return {}

    @property
    @abstractmethod
    def control_indices(self) -> dict[str, range]:
        return {}

    @property
    @abstractmethod
    def motor_noise_indices(self) -> range:
        return range(0, 1)

    @property
    @abstractmethod
    def sensory_noise_indices(self) -> range:
        return range(0, 1)

    @property
    @abstractmethod
    def noise_indices(self) -> list[range]:
        return [range(0, 1)]

    """TODO: move this in a utils so that I don't have to deal with multiple levels"""

    @staticmethod
    def transform_to_dm(value: cas.MX | cas.SX | cas.DM | np.ndarray | list) -> cas.DM:
        return VariablesAbstract.transform_to_dm(value)

    @staticmethod
    def reshape_matrix_to_vector(self, matrix):
        return VariablesAbstract.reshape_matrix_to_vector(matrix)

    @staticmethod
    def reshape_vector_to_matrix(
        vector: cas.MX | cas.SX | cas.DM | np.ndarray, matrix_shape: tuple[int, ...]
    ) -> cas.MX | cas.SX | cas.DM | np.ndarray:
        return VariablesAbstract.reshape_vector_to_matrix(vector, matrix_shape)

    @staticmethod
    def reshape_cholesky_matrix_to_vector(
        matrix: cas.MX | cas.SX | cas.DM | np.ndarray,
    ) -> cas.MX | cas.SX | cas.DM | np.ndarray:
        return VariablesAbstract.reshape_cholesky_matrix_to_vector(matrix)

    @staticmethod
    def reshape_vector_to_cholesky_matrix(
        vector: cas.MX | cas.SX | cas.DM | np.ndarray, matrix_shape: tuple[int, ...]
    ) -> cas.MX | cas.SX | cas.DM | np.ndarray:
        return VariablesAbstract.reshape_vector_to_cholesky_matrix(vector, matrix_shape)
