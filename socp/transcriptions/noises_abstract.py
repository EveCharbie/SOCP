from abc import ABC
import casadi as cas
import numpy as np


class NoisesAbstract(ABC):

    @staticmethod
    def transform_to_dm(value: cas.MX | cas.SX | cas.DM | np.ndarray | list) -> cas.DM:
        if isinstance(value, np.ndarray):
            return cas.DM(value.flatten())
        elif isinstance(value, list):
            return cas.DM(np.array(value).flatten())
        else:
            return value

    # --- Add --- #
    def add_motor_noise(self, index: int, value: cas.MX | cas.SX | cas.DM) -> None:
        pass

    def add_sensory_noise(self, index: int, value: cas.MX | cas.SX | cas.DM) -> None:
        pass

    def add_motor_noise_numerical(self, node: int, value: cas.MX | cas.SX | cas.DM) -> None:
        pass

    def add_sensory_noise_numerical(self, node: int, value: cas.MX | cas.SX | cas.DM) -> None:
        pass

    # --- Get vectors --- #
    def get_noise_single(self, index: int) -> cas.MX | cas.SX:
        pass

    def get_sensory_nois(self, index: int) -> cas.MX | cas.SX:
        pass

    def get_motor_noise(self, index: int) -> cas.MX | cas.SX:
        pass

    def get_one_vector_numerical(self, node: int) -> cas.MX | cas.SX:
        pass

    def get_full_matrix_numerical(self) -> cas.MX | cas.SX:
        pass
