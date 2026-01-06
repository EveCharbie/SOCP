from abc import ABC, abstractmethod
import numpy as np


class ExampleAbstract(ABC):
    """Abstract base class for optimal control problem examples."""
    def __init__(self) -> None:

        self.n_random: int = None
        self.n_threads: int = None
        self.n_simulations: int = None
        self.seed: int = None
        self.model: object = None

        self.dt: float = None
        self.final_time: float = None
        self.n_shooting: int = None

        self.tol: float = None
        self.max_iter: int = None

    @abstractmethod
    def get_bounds_and_init(
            self,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[
        str, np.ndarray], dict[str, np.ndarray]]:
        pass

    @abstractmethod
    def get_noises_magnitude(self) -> tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def get_specific_constraints(
            self,
            model: object,
            x: list,
            u: list,
            noises_single: list,
            noises_numerical: list,
    ):
        pass

    @abstractmethod
    def get_specific_objectives(
            self,
            model: object,
            x: list,
            u: list,
            noises_single: list,
            noises_numerical: list,
    ):
        pass

