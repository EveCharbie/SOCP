import casadi as cas


class Constraint:
    def __init__(self, g: cas.MX | cas.SX, lbg: cas.DM, ubg: cas.DM, g_names: list[str]):

        self.check_shapes(g, lbg, ubg, g_names)
        self.g = g
        self.lbg = lbg
        self.ubg = ubg
        self.g_names = g_names

    @staticmethod
    def check_shapes(
        g: cas.SX, lbg: cas.DM | list[float] | float, ubg: cas.DM | list[float] | float, g_names: list[str] | str
    ):

        if isinstance(lbg, list):
            len_lbg = len(lbg)
        elif isinstance(lbg, (float, int)):
            len_lbg = 1
        else:
            len_lbg = lbg.shape[0]

        if isinstance(ubg, list):
            len_ubg = len(ubg)
        elif isinstance(ubg, (float, int)):
            len_ubg = 1
        else:
            len_ubg = ubg.shape[0]

        if isinstance(g_names, str):
            len_g_names = 1
        else:
            len_g_names = len(g_names)

        if g.shape[0] != len_lbg or g.shape[0] != len_ubg:
            raise ValueError(f"Shapes of g ({g.shape}), lbg ({len_lbg}) and ubg ({len_ubg}) do not match.")
        if g.shape[1] != 1:
            raise ValueError(f"g should be a vector, but has shape {g.shape}.")
        if len_g_names != g.shape[0]:
            raise ValueError(
                f"Number of g_names ({len_g_names}) does not match the number of constraints ({g.shape[0]})."
            )


class Constraints:
    """
    This class allows to sort the constraints by node to bring the constraint jacobian closer to an identity.
    """

    def __init__(self, n_shooting: int):
        self.n_shooting = n_shooting
        self.constraint_list: list[list[Constraint]] = [[] for _ in range(n_shooting + 1)]

    def add(
        self,
        g: cas.MX | list[cas.MX] | cas.SX | list[cas.SX],
        lbg: cas.DM | list[cas.DM],
        ubg: cas.DM | list[cas.DM],
        g_names: list[str] | list[list[str]],
        node: int,
    ):
        if isinstance(g, list):
            for g_i, lbg_i, ubg_i, g_name_i in zip(g, lbg, ubg, g_names):
                self.constraint_list[node].append(Constraint(g_i, lbg_i, ubg_i, g_name_i))
        else:
            self.constraint_list[node].append(Constraint(g, lbg, ubg, g_names))

    def __getitem__(self, node: int) -> list[Constraint]:
        return self.constraint_list[node]

    def to_list(self) -> tuple[list[cas.MX | cas.SX], list[cas.DM | float], list[cas.DM | float], list[str]]:
        g_list = []
        lbg_list = []
        ubg_list = []
        g_name_list = []
        for node_constraints in self.constraint_list:
            for constraint in node_constraints:
                g_list.append(constraint.g)
                lbg_list.append(constraint.lbg)
                ubg_list.append(constraint.ubg)
                g_name_list.append(constraint.g_names)

        g = cas.vertcat(*g_list)
        lbg = cas.vertcat(*lbg_list)
        ubg = cas.vertcat(*ubg_list)
        g_names = []
        for element in g_name_list:
            if isinstance(element, list):
                g_names.extend(element)
            else:
                g_names.append(element)

        return g, lbg, ubg, g_names
