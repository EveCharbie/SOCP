from abc import ABC


class ModelAbstract(ABC):
    """Abstract base class for biomechanics models compatible with the transcriptions suggested."""

    def __init__(self, nb_random: int):

        self.nb_random = nb_random

        self.nb_q: int = None
        self.nb_states: int = None
        self.nb_controls: int = None
        self.nb_noised_controls: int = None
        self.nb_references: int = 0
        self.nb_k: int = 0
        self.nb_noised_states: int = None
        self.nb_noises: int = None
