from abc import ABC, abstractmethod
import casadi as cas
import numpy as np

from ..examples.example_abstract import ExampleAbstract
from ..models.model_abstract import ModelAbstract
from .variables_abstract import VariablesAbstract


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
        controls_lower_bounds: dict[str, np.ndarray],
    ) -> VariablesAbstract:
        pass

    @abstractmethod
    def declare_bounds_and_init(
        self,
        ocp_example: ExampleAbstract,
        states_lower_bounds: dict[str, np.ndarray],
        states_upper_bounds: dict[str, np.ndarray],
        states_initial_guesses: dict[str, np.ndarray],
        controls_lower_bounds: dict[str, np.ndarray],
        controls_upper_bounds: dict[str, np.ndarray],
        controls_initial_guesses: dict[str, np.ndarray],
        collocation_points_initial_guesses: dict[str, np.ndarray] | None,
    ) -> tuple[list[float], list[float], list[float]]:
        pass

    @abstractmethod
    def declare_noises(
        self,
        model: ModelAbstract,
        n_shooting: int,
        nb_random: int,
        motor_noise_magnitude: np.ndarray,
        sensory_noise_magnitude: np.ndarray,
        seed: int,
    ) -> tuple[np.ndarray, cas.SX]:
        pass

    @abstractmethod
    def get_mean_states(
        self,
        variable_vector: VariablesAbstract,
        node: int,
        squared: bool,
    ):
        pass

    @abstractmethod
    def get_covariance(
        self,
        variables_vector: VariablesAbstract,
        node: int,
        is_matrix: bool = False,
    ):
        pass

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

    @staticmethod
    def interpolate_between_nodes(
        var_pre: np.ndarray,
        var_post: np.ndarray,
        nb_points: int,
        current_point: int,
    ) -> np.ndarray:
        """
        Interpolate between two nodes.
        """
        return var_pre + (var_post - var_pre) * current_point / (nb_points - 1)

    def modify_init(
        self,
        ocp_example: ExampleAbstract,
        w0_vector: VariablesAbstract,
    ):
        """Modify the initial guess if needed."""
        pass
