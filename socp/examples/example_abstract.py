from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..models.model_abstract import ModelAbstract
    from ..transcriptions.transcription_abstract import TranscriptionAbstract
    from ..transcriptions.discretization_abstract import DiscretizationAbstract


class ExampleAbstract(ABC):
    """Abstract base class for optimal control problem examples."""

    def __init__(self) -> None:

        self.nb_random: int = None
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
    def name(self) -> str:
        pass

    @abstractmethod
    def get_bounds_and_init(
        self,
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
    ]:
        pass

    @abstractmethod
    def get_noises_magnitude(self) -> tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def get_specific_constraints(
        self,
        model: "ModelAbstract",
        discretization: "DiscretizationAbstract",
        dynamics_transcription: "TranscriptionAbstract",
        x: list,
        u: list,
        noises_single: list,
        noises_numerical: list,
    ):
        pass

    @abstractmethod
    def get_specific_objectives(
        self,
        model: "ModelAbstract",
        discretization: "DiscretizationAbstract",
        dynamics_transcription: "TranscriptionAbstract",
        x: list,
        u: list,
        noises_single: list,
        noises_numerical: list,
    ):
        pass
