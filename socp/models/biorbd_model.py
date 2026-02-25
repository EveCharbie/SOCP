from functools import wraps

import casadi as cas
import numpy as np
import biorbd_casadi as biorbd
import pyorerun

from .model_abstract import ModelAbstract



def cache_function(method):
    """Decorator to cache CasADi functions automatically"""

    def make_hashable(value):
        """
        Transforms non-hashable objects (dicts, and lists) into hashable objects (tuple)
        """
        if isinstance(value, list):
            return tuple(make_hashable(v) for v in value)
        elif isinstance(value, dict):
            return tuple(sorted((k, make_hashable(v)) for k, v in value.items()))
        elif isinstance(value, set):
            return frozenset(make_hashable(v) for v in value)
        return value

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        # Create a unique key based on the method name and arguments
        key = method.__name__
        if key in self._cached_functions.keys():
            return self._cached_functions[key]

        # Call the original function to create the CasADi function
        casadi_fun = method(self, *args, **kwargs)

        # Store in the cache
        self._cached_functions[key] = casadi_fun
        return casadi_fun

    return wrapper

class BiorbdModel(ModelAbstract):

    def __init__(self, nb_random: int, model_name: str):

        super().__init__(nb_random=nb_random)

        self._cached_functions = {}
        self.biorbd_model = biorbd.Model(f"socp/models/{model_name}.bioMod")
        self.nb_q = self.biorbd_model.nbQ()

    @property
    def name_dof(self):
        return [m.to_string() for m in self.biorbd_model.nameDof()]

    @cache_function
    def forward_dynamics_biorbd(
            self,
            q: cas.SX | cas.DM | np.ndarray,
            qdot: cas.SX | cas.DM | np.ndarray,
            tau: cas.SX | cas.DM | np.ndarray,
    ) -> cas.SX | cas.DM | np.ndarray:

        q_mx = cas.MX.sym("q", self.nb_q)
        qdot_mx = cas.MX.sym("qdot", self.nb_q)
        tau_mx = cas.MX.sym("tau", self.nb_q)

        fd_func = cas.Function(
            "forward_dynamics",
            [q_mx, qdot_mx, tau_mx],
            [self.biorbd_model.ForwardDynamics(q_mx, qdot_mx, tau_mx).to_mx()],
        )
        return fd_func(q, qdot, tau)

    @cache_function
    def lagrangian_biorbd(
        self,
        q: cas.SX | cas.DM | np.ndarray,
        qdot: cas.SX | cas.DM | np.ndarray,
    ) -> cas.SX | cas.DM | np.ndarray:

        q_mx = cas.MX.sym("q", self.nb_q)
        qdot_mx = cas.MX.sym("qdot", self.nb_q)

        lagrangian_func = cas.Function(
            "lagrangian",
            [q_mx, qdot_mx],
            [self.biorbd_model.Lagrangian(q_mx, qdot_mx).to_mx()],
        )
        return lagrangian_func(q, qdot)

    @cache_function
    def momentum_biorbd(
        self,
        q: cas.SX | cas.DM | np.ndarray,
        qdot: cas.SX | cas.DM | np.ndarray,
        u: cas.SX | cas.DM | np.ndarray,
    ) -> cas.SX | cas.DM | np.ndarray:

        q_mx = cas.MX.sym("q", self.nb_q)
        mass_matrix_func = cas.Function(
            "mass_matrix",
            [q_mx],
            [self.biorbd_model.massMatrix(q_mx).to_mx()],
        )
        p = mass_matrix_func(q) @ qdot
        return p

    def non_conservative_forces_biorbd(
        self,
        tau: cas.SX | cas.DM | np.ndarray,
    ) -> cas.SX | cas.DM | np.ndarray:
        return tau

    def animate(self, q: cas.DM, time_vector: cas.DM):

        # Model
        model = pyorerun.BiorbdModel.from_biorbd_object(self.biorbd_model)
        model.options.transparent_mesh = False
        model.options.show_gravity = True
        model.options.show_floor = False

        # Visualization
        viz = pyorerun.PhaseRerun(time_vector)
        viz.add_animated_model(model, q)
        viz.rerun_by_frame("Optimal solution")