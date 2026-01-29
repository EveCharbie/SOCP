from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import numpy as np
import casadi as cas

from ..constraints import Constraints
from ..transcriptions.variables_abstract import VariablesAbstract

if TYPE_CHECKING:
    from ..models.model_abstract import ModelAbstract
    from ..transcriptions.discretization_abstract import DiscretizationAbstract
    from ..transcriptions.noises_abstract import NoisesAbstract
    from ..transcriptions.transcription_abstract import TranscriptionAbstract


class ExampleAbstract(ABC):
    """Abstract base class for optimal control problem examples."""

    def __init__(self) -> None:

        self.nb_random: int = None
        self.n_threads: int = None
        self.n_simulations: int = None
        self.seed: int = None
        self.model: object = None

        self.final_time: float = None
        self.min_time: float = None
        self.max_time: float = None
        self.n_shooting: int = None

        self.tol: float = None
        self.max_iter: int = None

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def get_bounds_and_init(
        self,
        n_shooting: int,
        nb_collocation_points: int,
    ) -> tuple[
        dict[str, np.ndarray],
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
    def set_specific_constraints(
        self,
        model: "ModelAbstract",
        discretization_method: "DiscretizationAbstract",
        dynamics_transcription: "TranscriptionAbstract",
        variables_vector: VariablesAbstract,
        noises_vector: "NoisesAbstract",
        constraints: Constraints,
    ):
        pass

    @abstractmethod
    def get_specific_objectives(
        self,
        model: "ModelAbstract",
        discretization_method: "DiscretizationAbstract",
        dynamics_transcription: "TranscriptionAbstract",
        variables_vector: VariablesAbstract,
        noises_vector: "NoisesAbstract",
    ):
        pass
