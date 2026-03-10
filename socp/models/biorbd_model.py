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

        self.use_sx = False
        self._cached_functions = {}
        self.biorbd_model = biorbd.Model(f"socp/models/{model_name}.bioMod")
        self.nb_q = self.biorbd_model.nbQ()
        self.nb_root = self.biorbd_model.nbRoot()

    @property
    def name_dof(self):
        return [m.to_string() for m in self.biorbd_model.nameDof()]

    def marker_index(self, name: str) -> int:
        return biorbd.marker_index(self.biorbd_model, name)

    @cache_function
    def marker(self, index: int) -> cas.Function:

        q_mx = cas.MX.sym("q", self.nb_q)

        q_biorbd = biorbd.GeneralizedCoordinates(q_mx)

        marker_func = cas.Function(
            "marker",
            [q_mx],
            [self.model.marker(q_biorbd, index).to_mx()],
        )
        return marker_func

    @cache_function
    def center_of_mass(self) -> cas.Function:

        q_mx = cas.MX.sym("q", self.nb_q)

        q_biorbd = biorbd.GeneralizedCoordinates(q_mx)

        com_func = cas.Function(
            "center_of_mass",
            [q_mx],
            [self.model.CoM(q_biorbd, True).to_mx()],
        )
        return com_func

    @cache_function
    def body_rotation_rate(self) -> cas.Function:

        q_mx = cas.MX.sym("q", self.nb_q)
        qdot_mx = cas.MX.sym("qdot", self.nb_q)

        q_biorbd = biorbd.GeneralizedCoordinates(q_mx)
        qdot_biorbd = biorbd.GeneralizedVelocity(qdot_mx)

        rotation_rate_fun = cas.Function(
            "body_rotation_rate",
            [q_mx, qdot_mx],
            [self.biorbd_model.bodyAngularVelocity(q_biorbd, qdot_biorbd, True).to_mx()],
        )
        return rotation_rate_fun

    @cache_function
    def forward_dynamics_biorbd(
        self,
    ) -> cas.Function:

        q_mx = cas.MX.sym("q", self.nb_q)
        qdot_mx = cas.MX.sym("qdot", self.nb_q)
        tau_mx = cas.MX.sym("tau", self.nb_q)

        q_biorbd = biorbd.GeneralizedCoordinates(q_mx)
        qdot_biorbd = biorbd.GeneralizedVelocity(qdot_mx)
        tau_biorbd = biorbd.GeneralizedTorque(tau_mx)

        fd_func = cas.Function(
            "forward_dynamics",
            [q_mx, qdot_mx, tau_mx],
            [self.biorbd_model.ForwardDynamics(q_biorbd, qdot_biorbd, tau_biorbd).to_mx()],
        )
        return fd_func

    @cache_function
    def lagrangian_biorbd(
        self,
    ) -> cas.Function:

        q_mx = cas.MX.sym("q", self.nb_q)
        qdot_mx = cas.MX.sym("qdot", self.nb_q)

        q_biorbd = biorbd.GeneralizedCoordinates(q_mx)
        qdot_biorbd = biorbd.GeneralizedVelocity(qdot_mx)

        lagrangian_func = cas.Function(
            "lagrangian",
            [q_mx, qdot_mx],
            [self.biorbd_model.Lagrangian(q_biorbd, qdot_biorbd).to_mx()],
        )
        return lagrangian_func

    @cache_function
    def momentum_biorbd(
        self,
    ) -> cas.Function:

        q_mx = cas.MX.sym("q", self.nb_q)
        qdot_mx = cas.MX.sym("qdot", self.nb_q)

        q_biorbd = biorbd.GeneralizedCoordinates(q_mx)

        momentum_func = cas.Function(
            "mass_matrix",
            [q_mx, qdot_mx],
            [self.biorbd_model.massMatrix(q_biorbd).to_mx() @ qdot_mx],
        )
        return momentum_func

    @cache_function
    def mass_matrix(self) -> cas.Function:
        q_mx = cas.MX.sym("q", self.nb_q)

        q_biorbd = biorbd.GeneralizedCoordinates(q_mx)

        mass_matrix_fun = cas.Function(
            "mass_matrix",
            [q_mx],
            [self.biorbd_model.massMatrix(q_biorbd).to_mx()],
        )
        return mass_matrix_fun

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
