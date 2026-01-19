from abc import ABC, abstractmethod
import casadi as cas
import numpy as np

from ..models.model_abstract import ModelAbstract
from ..examples.example_abstract import ExampleAbstract
from ..transcriptions.discretization_abstract import DiscretizationAbstract
from ..transcriptions.variables_abstract import VariablesAbstract


class TranscriptionAbstract(ABC):
    """Abstract base class for optimal control problem transcription."""

    def __init__(self):
        self.discretization_method = None

        self.dynamics_func: cas.Function = None
        self.integration_func: cas.Function = None
        self.defect_func: cas.Function = None
        self.jacobian_funcs: cas.Function = None

    @abstractmethod
    def name(self) -> str:
        pass

    @property
    def nb_collocation_points(self):
        return 0

    @abstractmethod
    def initialize_dynamics_integrator(
        self,
        ocp_example: ExampleAbstract,
        discretization_method: DiscretizationAbstract,
        variables_vector: VariablesAbstract,
        noises_single: cas.SX.sym,
    ) -> None:
        pass

    @abstractmethod
    def get_dynamics_constraints(
        self,
        model: ModelAbstract,
        n_shooting: int,
        T: cas.SX.sym,
        x: list[cas.SX.sym],
        u: list[cas.SX.sym],
        noises_single: cas.SX.sym,
        noises_numerical: np.ndarray,
        n_threads: int = 8,
    ) -> tuple[list[cas.SX], list[float], list[float], list[str]]:
        pass

    @staticmethod
    def other_internal_constraints(
        ocp_example: ExampleAbstract,
        discretization_method: DiscretizationAbstract,
        T: cas.SX.sym,
        x_single: cas.SX.sym,
        z_single: cas.SX.sym,
        u_single: cas.SX.sym,
        noises_single: cas.SX.sym,
    ) -> tuple[list[cas.SX], list[float], list[float], list[str]]:
        return [], [], [], []
