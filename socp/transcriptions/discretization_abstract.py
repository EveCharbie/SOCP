from abc import ABC, abstractmethod
import casadi as cas
import numpy as np

from ..examples.example_abstract import ExampleAbstract
from ..models.model_abstract import ModelAbstract


class DiscretizationAbstract(ABC):
    """Abstract base class for the discretization of the optimal control problem."""

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
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
        nb_random: int,
        motor_noise_magnitude: np.ndarray,
        sensory_noise_magnitude: np.ndarray,
    ) -> tuple[np.ndarray, cas.SX]:
        pass

    @abstractmethod
    def get_mean_states(
        self,
        model: ModelAbstract,
        x,
    ):
        pass

    # @abstractmethod
    # def get_states_variance(
    #     self,
    #     model: ModelAbstract,
    #     x,
    # ):
    #     pass

    @abstractmethod
    def get_reference(
        self,
        model: ModelAbstract,
        x: cas.SX,
        u: cas.SX,
    ):
        pass

    @abstractmethod
    def get_ee_variance(
        self,
        model: ModelAbstract,
        x: cas.SX,
        u: cas.SX,
        HAND_FINAL_TARGET: np.ndarray,
    ):
        pass

    @abstractmethod
    def get_mus_variance(
        self,
        model: ModelAbstract,
        x: cas.SX,
    ):
        pass

    @abstractmethod
    def create_state_plots(
        self,
        ocp_example: ExampleAbstract,
        colors,
        axs,
        i_row,
        i_col,
        time_vector: np.ndarray,
    ):
        pass

    @abstractmethod
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
        pass
