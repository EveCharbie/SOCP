import casadi as cas


class LagrangePolynomial:

    def __init__(self, order: int = 5) -> None:

        super().__init__()  # Does nothing
        self.order = order
        self.time_grid = [0] + cas.collocation_points(self.order, "legendre")

    @property
    def nb_collocation_points(self):
        return self.order + 1

    def partial_lagrange_polynomial(
        self, j_collocation: int, time_control_interval: cas.SX, i_collocation: int
    ) -> cas.SX:
        _l = 1
        for r_collocation in range(self.nb_collocation_points):
            if r_collocation != j_collocation and r_collocation != i_collocation:
                _l *= (time_control_interval - self.time_grid[r_collocation]) / (
                    self.time_grid[j_collocation] - self.time_grid[r_collocation]
                )
        return _l

    def lagrange_polynomial(self, j_collocation: int, time_control_interval: cas.SX) -> cas.SX:
        _l = 1
        for r_collocation in range(self.nb_collocation_points):
            if r_collocation != j_collocation:
                _l *= (time_control_interval - self.time_grid[r_collocation]) / (
                    self.time_grid[j_collocation] - self.time_grid[r_collocation]
                )
        return _l

    def lagrange_polynomial_derivative(self, j_collocation: int, time_control_interval: cas.SX) -> cas.SX:

        sum_term = 0
        for k_collocation in range(self.nb_collocation_points):
            if k_collocation == j_collocation:
                continue

            partial_Ljk = self.partial_lagrange_polynomial(j_collocation, time_control_interval, k_collocation)
            sum_term += 1.0 / (self.time_grid[j_collocation] - self.time_grid[k_collocation]) * partial_Ljk

        return sum_term

    def get_states_end(self, z_matrix: cas.SX) -> cas.SX:

        states_end = 0
        for j_collocation in range(self.nb_collocation_points):
            sum_term = self.lagrange_polynomial(
                j_collocation=j_collocation,
                time_control_interval=1.0,
            )
            states_end += z_matrix[:, j_collocation] * sum_term
        return states_end

    def interpolate_first_derivative(self, z_matrix: cas.SX, time_control_interval: cas.SX) -> cas.SX:
        interpolated_value = 0
        for j_collocation in range(self.nb_collocation_points):
            interpolated_value += z_matrix[:, j_collocation] * self.lagrange_polynomial_derivative(
                j_collocation, time_control_interval
            )
        return interpolated_value
