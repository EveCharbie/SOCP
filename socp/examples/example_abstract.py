from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import numpy as np
import matplotlib.pyplot as plt
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
        self.model: "ModelAbstract" = None

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


    @staticmethod
    def draw_cov_ellipse(cov: np.ndarray, pos: np.ndarray, ax: plt.Axes, **kwargs):
        """
        Draw an ellipse representing the covariance at a given point.
        """

        def eigsorted(cov):
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            return vals[order], vecs[:, order]

        vals, vecs = eigsorted(cov)
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

        # Width and height are "full" widths, not radius
        width, height = 2 * np.sqrt(vals)
        ellip = plt.matplotlib.patches.Ellipse(xy=pos, width=width, height=height, angle=theta, alpha=0.3, **kwargs)

        ax.add_patch(ellip)
        return ellip