import casadi as cas
import numpy as np

from .lagrange_utils import LagrangePolynomial


class LobattoPolynomial(LagrangePolynomial):

    def __init__(self, order: int = 5) -> None:

        super().__init__(order=order)

        if self.order == 1:
            points = cas.DM([-1, 1])
            weights = cas.DM([1, 1])

        elif self.order == 2:
            points = cas.DM([-1, 0, 1])
            weights = cas.DM([1 / 3, 4 / 3, 1 / 3])

        elif self.order == 3:
            points = cas.DM([-1, -cas.sqrt(5) / 5, cas.sqrt(5) / 5, 1])
            weights = cas.DM([1 / 6, 5 / 6, 5 / 6, 1 / 6])

        elif self.order == 4:
            points = cas.DM([-1, -cas.sqrt(21) / 7, 0, cas.sqrt(21) / 7, 1])
            weights = cas.DM([1 / 10, 49 / 90, 32 / 45, 49 / 90, 1 / 10])

        elif self.order == 5:
            points = cas.DM(
                [
                    -1,
                    -cas.sqrt((7 + 2 * cas.sqrt(7)) / 21),
                    -cas.sqrt((7 - 2 * cas.sqrt(7)) / 21),
                    cas.sqrt((7 - 2 * cas.sqrt(7)) / 21),
                    cas.sqrt((7 + 2 * cas.sqrt(7)) / 21),
                    1,
                ]
            )
            weights = cas.DM(
                [
                    1 / 15,
                    (14 - cas.sqrt(7)) / 30,
                    (14 + cas.sqrt(7)) / 30,
                    (14 + cas.sqrt(7)) / 30,
                    (14 - cas.sqrt(7)) / 30,
                    1 / 15,
                ]
            )
        else:
            raise ValueError(f"Unsupported order {self.order}. Supported orders are 1 <= order <= 5.")

        self.time_grid = (points + 1) / 2
        self.weights = weights / 2

    def lagrange_polynomial_double_derivative(self, j_collocation: int, time_control_interval: cas.SX) -> cas.SX:

        numer = 0
        denom = 1
        for i_collocation in range(self.nb_collocation_points):
            if i_collocation == j_collocation:
                continue

            term = 0
            for k_collocation in range(self.nb_collocation_points):
                if k_collocation == j_collocation or k_collocation == i_collocation:
                    continue

                fact = 1
                for l_collocation in range(self.nb_collocation_points):
                    if (
                        l_collocation != j_collocation
                        and l_collocation != i_collocation
                        and l_collocation != k_collocation
                    ):
                        fact *= time_control_interval - self.time_grid[l_collocation]
                term += fact

            numer += term
            denom *= self.time_grid[j_collocation] - self.time_grid[i_collocation]

        return numer / denom

    def get_lagrange_coefficients(self) -> np.ndarray:
        coeffs = np.zeros((self.nb_collocation_points, self.nb_collocation_points, 3))

        for i_collocation in range(self.nb_collocation_points):
            for j_collocation in range(self.nb_collocation_points):
                coeffs[i_collocation, j_collocation, 0] = self.lagrange_polynomial(
                    i_collocation,
                    self.time_grid[j_collocation],
                )

                coeffs[i_collocation, j_collocation, 1] = self.lagrange_polynomial_derivative(
                    i_collocation,
                    self.time_grid[j_collocation],
                )

                coeffs[i_collocation, j_collocation, 2] = self.lagrange_polynomial_double_derivative(
                    i_collocation,
                    self.time_grid[j_collocation],
                )

        return coeffs
