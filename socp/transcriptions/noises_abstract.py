from abc import ABC
import casadi as cas


class NoisesAbstract(ABC):

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

    def get_one_vector_numerical(self, node: int) -> cas.MX | cas.SX:
        pass

    def get_full_matrix_numerical(self) -> cas.MX | cas.SX:
        pass
