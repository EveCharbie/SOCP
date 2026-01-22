from abc import ABC, abstractmethod
import casadi as cas
import numpy as np

from ..examples.example_abstract import ExampleAbstract
from ..transcriptions.discretization_abstract import DiscretizationAbstract
from ..transcriptions.noises_abstract import NoisesAbstract
from ..transcriptions.variables_abstract import VariablesAbstract
from ..constraints import Constraints


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
        noises_vector: NoisesAbstract,
    ) -> None:
        pass

    @abstractmethod
    def set_dynamics_constraints(
        self,
        ocp_example: ExampleAbstract,
        discretization_method: DiscretizationAbstract,
        variables_vector: VariablesAbstract,
        noises_vector: NoisesAbstract,
        constraints: Constraints,
        n_threads: int = 8,
    ) -> None:
        pass

    @staticmethod
    def add_other_internal_constraints(
        ocp_example: ExampleAbstract,
        discretization_method: DiscretizationAbstract,
        variables_vector: VariablesAbstract,
        noises_single: cas.SX.sym,
        i_node: int,
        constraints: Constraints,
    ) -> None:
        return
