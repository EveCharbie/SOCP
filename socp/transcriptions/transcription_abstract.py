from abc import ABC, abstractmethod
import casadi as cas
import numpy as np

from ..models.model_abstract import ModelAbstract


class TranscriptionAbstract(ABC):
    """Abstract base class for optimal control problem transcription."""

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def get_dynamics_constraints(
        self,
        model: ModelAbstract,
        n_shooting: int,
        x: list[cas.MX.sym],
        u: list[cas.MX.sym],
        noises_single: cas.MX.sym,
        noises_numerical: np.ndarray,
        dt: float,
        n_threads: int = 8,
    ) -> tuple[list[cas.MX], list[float], list[float], list[str]]:
        pass
