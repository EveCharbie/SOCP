
import casadi as cas
import numpy as np
import biorbd_casadi as biorbd
import pyorerun

from .model_abstract import ModelAbstract


class BiorbdModel(ModelAbstract):

    def __init__(self, nb_random: int, model_name: str):

        super().__init__(nb_random=nb_random)

        self.biorbd_model = biorbd.Model(f"socp/models/{model_name}.bioMod")
        self.nb_q = self.biorbd_model.nbQ()

    @property
    def name_dof(self):
        return [m.to_string() for m in self.biorbd_model.nameDof()]

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