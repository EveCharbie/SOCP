from abc import ABC, abstractmethod
import casadi as cas
import numpy as np

from ..models.model_abstract import ModelAbstract


class DiscretizationAbstract(ABC):
    """Abstract base class for the discretization of the optimal control problem."""

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def declare_variables(
        self,
        model: ModelAbstract,
        states_lower_bounds: dict[str, np.ndarray],
        states_upper_bounds: dict[str, np.ndarray],
        states_initial_guesses: dict[str, np.ndarray],
        controls_lower_bounds: dict[str, np.ndarray],
        controls_upper_bounds: dict[str, np.ndarray],
        controls_initial_guesses: dict[str, np.ndarray],
    ) -> tuple[list[cas.MX], list[cas.MX], list[cas.MX], list[float], list[float], list[float]]:
        pass

    @abstractmethod
    def get_variables_from_vector(
        self,
        model: ModelAbstract,
        states_lower_bounds: dict[str, np.ndarray],
        controls_lower_bounds: dict[str, np.ndarray],
        vector: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], cas.DM, cas.DM]:
        pass

    @abstractmethod
    def declare_noises(
        self,
        model: ModelAbstract,
        n_shooting: int,
        n_random: int,
        motor_noise_magnitude: np.ndarray,
        sensory_noise_magnitude: np.ndarray,
    ) -> tuple[np.ndarray, cas.MX]:
        pass

    @abstractmethod
    def get_mean_states(
            self,
            model: ModelAbstract,
            x,
    ):
        pass

    @abstractmethod
    def get_states_variance(
            self,
            model: ModelAbstract,
            x,
    ):
        pass

    @abstractmethod
    def get_reference(
            self,
            model: ModelAbstract,
            x: cas.MX,
            u: cas.MX,
    ):
        pass