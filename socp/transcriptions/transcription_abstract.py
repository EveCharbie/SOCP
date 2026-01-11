from abc import ABC, abstractmethod
import casadi as cas
import numpy as np

from ..models.model_abstract import ModelAbstract
from ..examples.example_abstract import ExampleAbstract
from ..transcriptions.discretization_abstract import DiscretizationAbstract


class TranscriptionAbstract(ABC):
    """Abstract base class for optimal control problem transcription."""

    def __init__(self):
        self.dynamics_func: cas.Function = None
        self.integration_func: cas.Function = None

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def initialize_dynamics_integrator(
        self,
        ocp_example: ExampleAbstract,
        discretization_method: DiscretizationAbstract,
        x: list[cas.SX.sym],
        u: list[cas.SX.sym],
        noises_single: cas.SX.sym,
    ) -> None:
        pass

    @abstractmethod
    def get_dynamics_constraints(
        self,
        model: ModelAbstract,
        n_shooting: int,
        x: list[cas.SX.sym],
        u: list[cas.SX.sym],
        noises_single: cas.SX.sym,
        noises_numerical: np.ndarray,
        dt: float,
        n_threads: int = 8,
    ) -> tuple[list[cas.SX], list[float], list[float], list[str]]:
        pass

    @staticmethod
    def other_internal_constraints(
        model: ModelAbstract,
        x: cas.SX,
        u: cas.SX,
    ) -> tuple[list[cas.SX], list[float], list[float], list[str]]:
        return [], [], [], []
