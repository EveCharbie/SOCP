import casadi as cas
import numpy as np

from .discretization_abstract import DiscretizationAbstract
from .noises_abstract import NoisesAbstract
from .transcription_abstract import TranscriptionAbstract
from .variables_abstract import VariablesAbstract
from ..examples.example_abstract import ExampleAbstract
from ..constraints import Constraints


class Variational(TranscriptionAbstract):

    def __init__(self) -> None:

        super().__init__()  # Does nothing

    def initialize_dynamics_integrator(
        self,
        ocp_example: ExampleAbstract,
        discretization_method: DiscretizationAbstract,
        variables_vector: VariablesAbstract,
        noises_vector: NoisesAbstract,
    ) -> None:

        # Note: The first and second x and u used to declare the casadi functions, but all nodes will be used during the evaluation of the functions
        self.discretization_method = discretization_method
        initial_defect_func, three_nodes_defect_func, final_defect_func = self.declare_dynamics_integrator(
            ocp_example,
            discretization_method,
            variables_vector,
            noises_vector,
        )
        self.initial_defect_func = initial_defect_func
        self.three_nodes_defect_func = three_nodes_defect_func
        self.final_defect_func = final_defect_func

    @property
    def name(self) -> str:
        return "Variational"

    def declare_dynamics_integrator(
        self,
        ocp_example: ExampleAbstract,
        discretization_method: DiscretizationAbstract,
        variables_vector: VariablesAbstract,
        noises_vector: NoisesAbstract,
    ) -> tuple[cas.Function, cas.Function, cas.Function]:
        """
        Formulate discrete Euler-Lagrange equations and set up a variational integrator.
        We consider that there are no holonomic constraints.
        The following equation as been calculated thanks to the paper "Discrete mechanics and optimal control for
        constrained systems" (https://onlinelibrary.wiley.com/doi/epdf/10.1002/oca.912)
        three_node_defect -> equations (10)
        initial_defect -> equations (14) and (18)
        final_defect -> equations (14) and (18)
        """
        dt = variables_vector.get_time() / ocp_example.n_shooting

        q_previous = variables_vector.get_state("q", 0)
        q_current = variables_vector.get_state("q", 1)
        q_next = variables_vector.get_state("q", 2)

        qdot0 = variables_vector.get_state("qdot", 0)
        qdotN = variables_vector.get_state("qdot", ocp_example.n_shooting)

        control_previous = variables_vector.get_controls(0)
        control_current = variables_vector.get_controls(1)
        control_previous_repeat = None
        control_current_repeat = None
        for i in range(variables_vector.nb_random):
            if control_previous_repeat is None:
                control_previous_repeat = control_previous
                control_current_repeat = control_current
            else:
                control_previous_repeat = cas.vertcat(control_previous_repeat, control_previous)
                control_current_repeat = cas.vertcat(control_current_repeat, control_current)

        noise_previous = noises_vector.get_noise_single(0)
        noise_current = noises_vector.get_noise_single(1)

        # Defects
        qdot_previous = (q_current - q_previous) / dt
        qdot_current = (q_next - q_current) / dt
        f_plus_previous = (
            dt
            / 2
            * discretization_method.get_non_conservative_forces(
                    ocp_example=ocp_example,
                    q=q_previous,
                    qdot=qdot_previous,
                    u=control_previous,
                    noise=noise_previous,
            )
        )
        f_minus_current = (
            dt
            / 2
            * discretization_method.get_non_conservative_forces(
                    ocp_example=ocp_example,
                    q=q_current,
                    qdot=qdot_current,
                    u=control_current,
                    noise=noise_current,
            )
        )

        discrete_lagrangian_previous = (
            discretization_method.get_lagrangian(
                ocp_example=ocp_example,
                q=(q_previous + q_current) / 2,
                qdot=qdot_previous,
                u=control_previous,
            )
            * dt
        )
        discrete_lagrangian_current = (
            discretization_method.get_lagrangian(
                ocp_example=ocp_example,
                q=(q_current + q_next) / 2,
                qdot=qdot_current,
                u=control_current,
            )
            * dt
        )

        # Refers to D_2 L_d(q_{k-1}, q_k) (D_2 is the partial derivative with respect to the second argument, L_d is the
        # discrete Lagrangian)
        p_current = discretization_method.get_lagrangian_jacobian(
            ocp_example,
            discrete_lagrangian_previous,
            q_current,
        )
        # Refers to D_1 L_d(q_{k}, q_{k+1}) (D_2 is the partial derivative with respect to the second argument)
        d1_ld_qcur_qnext = discretization_method.get_lagrangian_jacobian(
            ocp_example,
            discrete_lagrangian_current,
            q_current,
        )

        three_nodes_defect = p_current + d1_ld_qcur_qnext + f_plus_previous + f_minus_current
        three_nodes_defect_func = cas.Function(
            "three_nodes_defects",
            [
                variables_vector.get_time(),
                variables_vector.get_state("q", 0),
                variables_vector.get_state("q", 1),
                variables_vector.get_state("q", 2),
                variables_vector.get_controls(0),
                variables_vector.get_controls(1),
                variables_vector.get_controls(2),
                noises_vector.get_noise_single(0),
                noises_vector.get_noise_single(1),
            ],
            [three_nodes_defect],
        )

        # Refers to D_2 L(q_0, \dot{q_0}) (D_2 is the partial derivative with respect to the second argument)
        discrete_lagrangian_qdot0 = discretization_method.get_lagrangian(
            ocp_example=ocp_example,
            q=q_previous,
            qdot=qdot0,
            u=control_previous,
        )
        d2_l_q0_qdot0 = discretization_method.get_lagrangian_jacobian(
            ocp_example,
            discrete_lagrangian_qdot0,
            qdot0,
        )
        # Refers to D_1 L_d(q_0, q_1) (D1 is the partial derivative with respect to the first argument, Ld is the
        # discrete Lagrangian)
        d1_ld_q0_q1 = discretization_method.get_lagrangian_jacobian(
            ocp_example,
            discrete_lagrangian_previous,
            q_previous,
        )
        initial_defect = d2_l_q0_qdot0 + d1_ld_q0_q1 + f_plus_previous
        initial_defect_func = cas.Function(
            "initial_defects",
            [
                variables_vector.get_time(),
                variables_vector.get_state("q", 0),
                variables_vector.get_state("q", 1),
                variables_vector.get_state("qdot", 0),
                variables_vector.get_controls(0),
                variables_vector.get_controls(1),
                noises_vector.get_noise_single(0),
            ],
            [initial_defect],
        )

        # Refers to D_2 L(q_N, \dot{q_N}) (D_2 is the partial derivative with respect to the second argument)
        discrete_lagrangian_qdotN = discretization_method.get_lagrangian(
            ocp_example=ocp_example, q=q_next, qdot=qdotN, u=control_current
        )
        d2_l_q_ultimate_qdot_ultimate = discretization_method.get_lagrangian_jacobian(
            ocp_example,
            discrete_lagrangian_qdotN,
            qdotN,
        )
        # Refers to D_2 L_d(q_{n-1}, q_1) (Ld is the discrete Lagrangian)
        d2_ld_q_penultimate_q_ultimate = discretization_method.get_lagrangian_jacobian(
            ocp_example,
            discrete_lagrangian_current,
            q_next,
        )
        final_defect = -d2_l_q_ultimate_qdot_ultimate + d2_ld_q_penultimate_q_ultimate + f_minus_current
        final_defect_func = cas.Function(
            "defects",
            [
                variables_vector.get_time(),
                variables_vector.get_state("q", 1),
                variables_vector.get_state("q", 2),
                variables_vector.get_state("qdot", ocp_example.n_shooting),
                variables_vector.get_controls(1),
                variables_vector.get_controls(2),
                noises_vector.get_noise_single(1),
            ],
            [final_defect],
        )

        return initial_defect_func, three_nodes_defect_func, final_defect_func

    def add_other_internal_constraints(
        self,
        ocp_example: ExampleAbstract,
        discretization_method: DiscretizationAbstract,
        variables_vector: VariablesAbstract,
        noises_vector: NoisesAbstract,
        i_node: int,
        constraints: Constraints,
    ) -> None:
        return

    def set_dynamics_constraints(
        self,
        ocp_example: ExampleAbstract,
        discretization_method: DiscretizationAbstract,
        variables_vector: VariablesAbstract,
        noises_vector: NoisesAbstract,
        constraints: Constraints,
        n_threads: int = 8,
    ) -> None:

        nb_states = variables_vector.nb_states
        nb_variables = ocp_example.model.nb_q * variables_vector.nb_random
        n_shooting = variables_vector.n_shooting

        # Multi-thread continuity constraint
        multi_threaded_constraint = self.three_nodes_defect_func.map(n_shooting - 1, "thread", n_threads)
        three_nodes_defects = multi_threaded_constraint(
            variables_vector.get_time(),
            cas.horzcat(*[variables_vector.get_state("q", i_node) for i_node in range(0, n_shooting - 1)]),
            cas.horzcat(*[variables_vector.get_state("q", i_node) for i_node in range(1, n_shooting)]),
            cas.horzcat(*[variables_vector.get_state("q", i_node) for i_node in range(2, n_shooting + 1)]),
            # cas.horzcat(*[variables_vector.get_cov(i_node) for i_node in range(0, n_shooting)]),
            # cas.horzcat(*[variables_vector.get_cov(i_node) for i_node in range(1, n_shooting+1)]),
            # cas.horzcat(*[variables_vector.get_ms(i_node) for i_node in range(0, n_shooting)]),
            # cas.horzcat(*[variables_vector.get_ms(i_node) for i_node in range(1, n_shooting+1)]),
            cas.horzcat(*[variables_vector.get_controls(i_node) for i_node in range(0, n_shooting - 1)]),
            cas.horzcat(*[variables_vector.get_controls(i_node) for i_node in range(1, n_shooting)]),
            cas.horzcat(*[variables_vector.get_controls(i_node) for i_node in range(2, n_shooting + 1)]),
            cas.horzcat(*[noises_vector.get_one_vector_numerical(i_node) for i_node in range(0, n_shooting - 1)]),
            cas.horzcat(*[noises_vector.get_one_vector_numerical(i_node) for i_node in range(1, n_shooting)]),
        )

        if discretization_method.name == "MeanAndCovariance":
            if discretization_method.with_cholesky:
                nb_cov_variables = variables_vector.nb_cholesky_components(nb_states)
            else:
                nb_cov_variables = nb_states * nb_states
        else:
            nb_cov_variables = 0

        for i_node in range(n_shooting - 1):
            constraints.add(
                g=three_nodes_defects[:, i_node],
                lbg=[0] * (nb_variables + nb_cov_variables),
                ubg=[0] * (nb_variables + nb_cov_variables),
                g_names=[f"dynamics_continuity_node_{i_node+1}"] * (nb_variables + nb_cov_variables),
                node=i_node + 1,
            )

        # First node defect
        initial_defect = self.initial_defect_func(
            variables_vector.get_time(),
            variables_vector.get_state("q", 0),
            variables_vector.get_state("q", 1),
            variables_vector.get_state("qdot", 0),
            variables_vector.get_controls(0),
            variables_vector.get_controls(1),
            noises_vector.get_one_vector_numerical(0),
        )
        constraints.add(
            g=initial_defect,
            lbg=[0] * initial_defect.shape[0],
            ubg=[0] * initial_defect.shape[0],
            g_names=[f"dynamics_initial_defect"] * initial_defect.shape[0],
            node=0,
        )

        # Last node defect
        final_defect = self.final_defect_func(
            variables_vector.get_time(),
            variables_vector.get_state("q", n_shooting - 1),
            variables_vector.get_state("q", n_shooting),
            variables_vector.get_state("qdot", n_shooting),
            variables_vector.get_controls(n_shooting - 1),
            variables_vector.get_controls(n_shooting),
            noises_vector.get_one_vector_numerical(n_shooting),
        )
        constraints.add(
            g=final_defect,
            lbg=[0] * final_defect.shape[0],
            ubg=[0] * final_defect.shape[0],
            g_names=[f"dynamics_final_defect"] * final_defect.shape[0],
            node=n_shooting,
        )
