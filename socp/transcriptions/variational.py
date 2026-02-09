import casadi as cas

from .discretization_abstract import DiscretizationAbstract
from .noises_abstract import NoisesAbstract
from .transcription_abstract import TranscriptionAbstract
from .variables_abstract import VariablesAbstract
from ..examples.example_abstract import ExampleAbstract
from ..constraints import Constraints


class Variational(TranscriptionAbstract):

    def __init__(self) -> None:

        super().__init__()  # Does nothing

    @property
    def name(self) -> str:
        return "Variational"

    @property
    def nb_m_points(self):
        return 3

    def initialize_dynamics_integrator(
        self,
        ocp_example: ExampleAbstract,
        discretization_method: DiscretizationAbstract,
        variables_vector: VariablesAbstract,
        noises_vector: NoisesAbstract,
    ) -> None:
        """
        Formulate discrete Euler-Lagrange equations and set up a variational integrator.
        We consider that there are no holonomic constraints.
        The following equation as been calculated thanks to the paper "Discrete mechanics and optimal control for
        constrained systems" (https://onlinelibrary.wiley.com/doi/epdf/10.1002/oca.912)
        three_node_defect -> equations (10)
        initial_defect -> equations (14) and (18)
        final_defect -> equations (14) and (18)
        """

        # Note: The first and second x and u used to declare the casadi functions, but all nodes will be used during the evaluation of the functions
        self.discretization_method = discretization_method


        dt = variables_vector.get_time() / ocp_example.n_shooting

        qdot0 = variables_vector.get_state("qdot", 0)
        qdotN = variables_vector.get_state("qdot", ocp_example.n_shooting)

        # Defects
        f_plus_previous = (
            dt
            / 2
            * discretization_method.get_non_conservative_forces(
                ocp_example=ocp_example,
                q=variables_vector.get_state("q", 0),
                qdot=(variables_vector.get_state("q", 1) - variables_vector.get_state("q", 0)) / dt,
                u=variables_vector.get_controls(0),
                noise=noises_vector.get_noise_single(0),
            )
        )
        f_minus_current = (
            dt
            / 2
            * discretization_method.get_non_conservative_forces(
                ocp_example=ocp_example,
                q=variables_vector.get_state("q", 1),
                qdot=(variables_vector.get_state("q", 2) - variables_vector.get_state("q", 1)) / dt,
                u=variables_vector.get_controls(1),
                noise=noises_vector.get_noise_single(1),
            )
        )

        discrete_lagrangian_previous = (
            discretization_method.get_lagrangian(
                ocp_example=ocp_example,
                q=(variables_vector.get_state("q", 0) + variables_vector.get_state("q", 1)) / 2,
                qdot=(variables_vector.get_state("q", 1) - variables_vector.get_state("q", 0)) / dt,
                u=variables_vector.get_controls(0),
            )
            * dt
        )
        discrete_lagrangian_current = (
            discretization_method.get_lagrangian(
                ocp_example=ocp_example,
                q=(variables_vector.get_state("q", 1) + variables_vector.get_state("q", 2)) / 2,
                qdot=(variables_vector.get_state("q", 2) - variables_vector.get_state("q", 1)) / dt,
                u=variables_vector.get_controls(1),
            )
            * dt
        )

        # Refers to D_2 L_d(q_{k-1}, q_k) (D_2 is the partial derivative with respect to the second argument, L_d is the
        # discrete Lagrangian)
        p_current = discretization_method.get_lagrangian_jacobian(
            ocp_example,
            discrete_lagrangian_previous,
            variables_vector.get_state("q", 1),
        )
        # Refers to D_1 L_d(q_{k}, q_{k+1}) (D_2 is the partial derivative with respect to the second argument)
        d1_ld_qcur_qnext = discretization_method.get_lagrangian_jacobian(
            ocp_example,
            discrete_lagrangian_current,
            variables_vector.get_state("q", 1),
        )

        three_nodes_defect = p_current + d1_ld_qcur_qnext + f_plus_previous + f_minus_current
        self.three_nodes_defect_func = cas.Function(
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
            q=variables_vector.get_state("q", 0),
            qdot=qdot0,
            u=variables_vector.get_controls(0),
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
            variables_vector.get_state("q", 0),
        )
        initial_defect = d2_l_q0_qdot0 + d1_ld_q0_q1 + f_plus_previous
        self.initial_defect_func = cas.Function(
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
            ocp_example=ocp_example,
            q=variables_vector.get_state("q", 2),
            qdot=qdotN,
            u=variables_vector.get_controls(1),
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
            variables_vector.get_state("q", 2),
        )
        final_defect = -d2_l_q_ultimate_qdot_ultimate + d2_ld_q_penultimate_q_ultimate + f_minus_current
        self.final_defect_func = cas.Function(
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

        jacobian_funcs_initial = None
        jacobian_funcs = None
        jacobian_funcs_final = None
        cov_constraint_func_initial = None
        cov_constraint_func = None
        cov_constraint_func_final = None
        if discretization_method.name == "MeanAndCovariance":
            if discretization_method.with_helper_matrix:
                # We consider z = [q_previous, q_current, q_next] temporarily
                m_matrix = variables_vector.get_m_matrix(0)
                sigma_ww = cas.diag(noises_vector.get_noise_single(0))

                z_three = cas.SX.sym("z_three", ocp_example.model.nb_q, 3)
                F = z_three[:, 2]
                G = [z_three[:, 0] - variables_vector.get_state("q", 0)]
                three_node_defect_z = self.three_nodes_defect_func(
                    variables_vector.get_time(),
                    z_three[:, 0],
                    z_three[:, 1],
                    z_three[:, 2],
                    variables_vector.get_controls(0),
                    variables_vector.get_controls(1),
                    variables_vector.get_controls(2),
                    noises_vector.get_noise_single(0),
                    noises_vector.get_noise_single(1),
                )
                G += [three_node_defect_z]
                G += [z_three[:, 2] - variables_vector.get_state("q", 2)]

                dFdz = cas.jacobian(F, z_three)
                dGdz = cas.jacobian(cas.horzcat(*G), z_three)

                dGdx = cas.jacobian(cas.horzcat(*G), variables_vector.get_state("q", 0))

                dFdw = cas.jacobian(F, noises_vector.get_noise_single(0))
                dGdw = cas.jacobian(cas.horzcat(*G), noises_vector.get_noise_single(0))

                self.jacobian_funcs = cas.Function(
                    "jacobian_funcs",
                    [
                        variables_vector.get_time(),
                        variables_vector.get_state("q", 0),
                        variables_vector.get_state("q", 1),
                        variables_vector.get_state("q", 2),
                        z_three,
                        variables_vector.get_controls(0),
                        variables_vector.get_controls(1),
                        variables_vector.get_controls(2),
                        noises_vector.get_noise_single(0),
                        noises_vector.get_noise_single(1),
                    ],
                    [dGdx, dFdz, dGdz, dFdw, dGdw],
                )

                cov_matrix = variables_vector.get_cov_matrix(0)
                cov_integrated = m_matrix @ (dGdx @ cov_matrix @ dGdx.T + dGdw @ sigma_ww @ dGdw.T) @ m_matrix.T

                cov_integrated_vector = variables_vector.reshape_matrix_to_vector(cov_integrated)

                cov_constraint = cov_integrated_vector - variables_vector.get_cov(2)

                self.cov_constraint_func = cas.Function(
                    "covariance_constraint_func",
                    [
                        variables_vector.get_time(),
                        variables_vector.get_state("q", 0),
                        variables_vector.get_state("q", 1),
                        variables_vector.get_state("q", 2),
                        z_three,
                        variables_vector.get_cov(0),
                        variables_vector.get_cov(2),
                        variables_vector.get_ms(0),
                        variables_vector.get_controls(0),
                        variables_vector.get_controls(1),
                        variables_vector.get_controls(2),
                        noises_vector.get_noise_single(0),
                        noises_vector.get_noise_single(1),
                    ],
                    [cov_constraint],
                )

                # Initial
                z_initial = cas.SX.sym("z_initial", ocp_example.model.nb_q, 2)
                F = z_initial[:, 1]
                G = [z_initial[:, 0] - variables_vector.get_state("q", 0)]
                initial_defect_z = self.initial_defect_func(
                    variables_vector.get_time(),
                    z_initial[:, 0],
                    z_initial[:, 1],
                    variables_vector.get_state("qdot", 0),
                    variables_vector.get_controls(0),
                    variables_vector.get_controls(1),
                    noises_vector.get_noise_single(0),
                )

                G += [initial_defect_z]

                dFdz_initial = cas.jacobian(F, z_initial)
                dGdz_initial = cas.jacobian(cas.horzcat(*G), z_initial)

                dGdx_initial = cas.jacobian(cas.horzcat(*G), variables_vector.get_state("q", 0))

                dFdw_initial = cas.jacobian(F, noises_vector.get_noise_single(0))
                dGdw_initial = cas.jacobian(cas.horzcat(*G), noises_vector.get_noise_single(0))

                self.jacobian_funcs_initial = cas.Function(
                    "jacobian_funcs_initial",
                    [
                        variables_vector.get_time(),
                        variables_vector.get_state("q", 0),
                        variables_vector.get_state("q", 1),
                        variables_vector.get_state("qdot", 0),
                        z_initial,
                        variables_vector.get_controls(0),
                        variables_vector.get_controls(1),
                        noises_vector.get_noise_single(0),
                        noises_vector.get_noise_single(1),
                    ],
                    [dGdx_initial, dFdz_initial, dGdz_initial, dFdw_initial, dGdw_initial],
                )

                m_matrix_0 = variables_vector.get_m_matrix(0)[
                    :, : 2 * ocp_example.model.nb_q
                ]  # Only the two fisrt collocations points
                cov_integrated_initial = (
                    m_matrix_0
                    @ (dGdx_initial @ cov_matrix @ dGdx_initial.T + dGdw_initial @ sigma_ww @ dGdw_initial.T)
                    @ m_matrix_0.T
                )

                cov_integrated_vector_initial = variables_vector.reshape_matrix_to_vector(cov_integrated_initial)

                cov_constraint_initial = cov_integrated_vector_initial - variables_vector.get_cov(1)

                self.cov_constraint_func_initial = cas.Function(
                    "covariance_constraint_func",
                    [
                        variables_vector.get_time(),
                        variables_vector.get_state("q", 0),
                        variables_vector.get_state("q", 1),
                        variables_vector.get_state("qdot", 0),
                        z_initial,
                        variables_vector.get_cov(0),
                        variables_vector.get_cov(1),
                        variables_vector.get_ms(0),
                        variables_vector.get_controls(0),
                        variables_vector.get_controls(1),
                        noises_vector.get_noise_single(0),
                        noises_vector.get_noise_single(1),
                    ],
                    [cov_constraint_initial],
                )

                # Final
                z_final = cas.SX.sym("z", ocp_example.model.nb_q, 2)
                F = z_final[:, 1]
                G = [z_final[:, 0] - variables_vector.get_state("q", 1)]
                final_defect_z = self.final_defect_func(
                    variables_vector.get_time(),
                    z_final[:, 0],
                    z_final[:, 1],
                    variables_vector.get_state("qdot", ocp_example.n_shooting),
                    variables_vector.get_controls(1),
                    variables_vector.get_controls(2),
                    noises_vector.get_noise_single(1),
                )
                G += [final_defect_z]

                dFdz_final = cas.jacobian(F, z_final)
                dGdz_final = cas.jacobian(cas.horzcat(*G), z_final)

                dGdx_final = cas.jacobian(cas.horzcat(*G), variables_vector.get_state("q", 1))

                dFdw_final = cas.jacobian(F, noises_vector.get_noise_single(1))
                dGdw_final = cas.jacobian(cas.horzcat(*G), noises_vector.get_noise_single(1))

                self.jacobian_funcs_final = cas.Function(
                    "jacobian_funcs_final",
                    [
                        variables_vector.get_time(),
                        variables_vector.get_state("q", 1),
                        variables_vector.get_state("q", 2),
                        variables_vector.get_state("qdot", ocp_example.n_shooting),
                        z_final,
                        variables_vector.get_controls(1),
                        variables_vector.get_controls(2),
                        noises_vector.get_noise_single(1),
                        noises_vector.get_noise_single(2),
                    ],
                    [dGdx_final, dFdz_final, dGdz_final, dFdw_final, dGdw_final],
                )

                sigma_ww_1 = cas.diag(noises_vector.get_noise_single(1))
                m_matrix_1 = variables_vector.get_m_matrix(1)[
                    :, : 2 * ocp_example.model.nb_q
                ]  # Only the two first collocations points
                cov_matrix_1 = variables_vector.get_cov_matrix(1)
                cov_integrated_final = (
                    m_matrix_1
                    @ (dGdx_final @ cov_matrix_1 @ dGdx_final.T + dGdw_final @ sigma_ww_1 @ dGdw_final.T)
                    @ m_matrix_1.T
                )

                cov_integrated_vector_final = variables_vector.reshape_matrix_to_vector(cov_integrated_final)

                cov_constraint_final = cov_integrated_vector_final - variables_vector.get_cov(2)

                self.cov_constraint_func_final = cas.Function(
                    "covariance_constraint_func",
                    [
                        variables_vector.get_time(),
                        variables_vector.get_state("q", 1),
                        variables_vector.get_state("q", 2),
                        variables_vector.get_state("qdot", ocp_example.n_shooting),
                        z_final,
                        variables_vector.get_cov(1),
                        variables_vector.get_cov(2),
                        variables_vector.get_ms(1),
                        variables_vector.get_controls(1),
                        variables_vector.get_controls(2),
                        noises_vector.get_noise_single(1),
                    ],
                    [cov_constraint_final],
                )

            else:
                raise NotImplementedError(
                    "Covariance dynamics with helper matrix is the only supported method for now."
                )
        return

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

        nb_q = ocp_example.model.nb_q
        nb_variables = ocp_example.model.nb_q * variables_vector.nb_random
        n_shooting = variables_vector.n_shooting

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

        if discretization_method.with_helper_matrix:
            # CoV constraint
            cov_constraint_initial = self.cov_constraint_func_initial(
                variables_vector.get_time(),
                variables_vector.get_state("q", 0),
                variables_vector.get_state("q", 1),
                variables_vector.get_state("qdot", 0),
                cas.horzcat(variables_vector.get_state("q", 0), variables_vector.get_state("q", 1)),
                variables_vector.get_cov(0),
                variables_vector.get_cov(1),
                variables_vector.get_ms(0),
                variables_vector.get_controls(0),
                variables_vector.get_controls(1),
                noises_vector.get_one_vector_numerical(0),
                noises_vector.get_one_vector_numerical(1),
            )

            constraints.add(
                g=cov_constraint_initial,
                lbg=[0] * (nb_q * nb_q),
                ubg=[0] * (nb_q * nb_q),
                g_names=[f"cov_defect_initial"] * (nb_q * nb_q),
                node=0,
            )

            # Constrain M at all collocation points to follow df_integrated/dz.T - dg_integrated/dz @ m.T = 0
            m_matrix_0 = variables_vector.get_m_matrix(0)[
                :, : 2 * ocp_example.model.nb_q
            ]  # Only the two fisrt collocations points
            _, dFdz_initial, dGdz_initial, _, _ = self.jacobian_funcs_initial(
                variables_vector.get_time(),
                variables_vector.get_state("q", 0),
                variables_vector.get_state("q", 1),
                variables_vector.get_state("qdot", 0),
                cas.horzcat(variables_vector.get_state("q", 0), variables_vector.get_state("q", 1)),
                variables_vector.get_controls(0),
                variables_vector.get_controls(1),
                cas.DM.zeros(ocp_example.model.nb_noises * variables_vector.nb_random),
                cas.DM.zeros(ocp_example.model.nb_noises * variables_vector.nb_random),
            )

            constraint = dFdz_initial.T - dGdz_initial.T @ m_matrix_0.T
            constraints.add(
                g=variables_vector.reshape_matrix_to_vector(constraint),
                lbg=[0] * (dFdz_initial.shape[1] * dFdz_initial.shape[0]),
                ubg=[0] * (dFdz_initial.shape[1] * dFdz_initial.shape[0]),
                g_names=[f"helper_matrix_defect_initial"] * (dFdz_initial.shape[1] * dFdz_initial.shape[0]),
                node=0,
            )

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

        for i_node in range(n_shooting - 1):
            constraints.add(
                g=three_nodes_defects[:, i_node],
                lbg=[0] * nb_variables,
                ubg=[0] * nb_variables,
                g_names=[f"dynamics_continuity_node_{i_node+1}"] * nb_variables,
                node=i_node + 1,
            )

            if discretization_method.with_helper_matrix:
                # CoV constraint
                cov_constraint = self.cov_constraint_func(
                    variables_vector.get_time(),
                    variables_vector.get_state("q", i_node),
                    variables_vector.get_state("q", i_node + 1),
                    variables_vector.get_state("q", i_node + 2),
                    cas.horzcat(
                        cas.horzcat(
                            variables_vector.get_state("q", i_node), variables_vector.get_state("q", i_node + 1)
                        ),
                        variables_vector.get_state("q", i_node + 2),
                    ),
                    variables_vector.get_cov(i_node),
                    variables_vector.get_cov(i_node + 2),
                    variables_vector.get_ms(i_node + 1),
                    variables_vector.get_controls(i_node),
                    variables_vector.get_controls(i_node + 1),
                    variables_vector.get_controls(i_node + 2),
                    noises_vector.get_one_vector_numerical(i_node),
                    noises_vector.get_one_vector_numerical(i_node + 1),
                )

                constraints.add(
                    g=cov_constraint,
                    lbg=[0] * (nb_q * nb_q),
                    ubg=[0] * (nb_q * nb_q),
                    g_names=[f"cov_defect"] * (nb_q * nb_q),
                    node=i_node + 1,
                )

                # Constrain M at all collocation points to follow df_integrated/dz.T - dg_integrated/dz @ m.T = 0
                m_matrix = variables_vector.get_m_matrix(i_node)
                _, dFdz, dGdz, _, _ = self.jacobian_funcs(
                    variables_vector.get_time(),
                    variables_vector.get_state("q", i_node),
                    variables_vector.get_state("q", i_node + 1),
                    variables_vector.get_state("q", i_node + 2),
                    cas.horzcat(
                        cas.horzcat(
                            variables_vector.get_state("q", i_node), variables_vector.get_state("q", i_node + 1)
                        ),
                        variables_vector.get_state("q", i_node + 2),
                    ),
                    variables_vector.get_controls(i_node),
                    variables_vector.get_controls(i_node + 1),
                    variables_vector.get_controls(i_node + 2),
                    cas.DM.zeros(ocp_example.model.nb_noises * variables_vector.nb_random),
                    cas.DM.zeros(ocp_example.model.nb_noises * variables_vector.nb_random),
                )

                constraint = dFdz.T - dGdz.T @ m_matrix.T
                constraints.add(
                    g=variables_vector.reshape_matrix_to_vector(constraint),
                    lbg=[0] * (dFdz.shape[1] * dFdz.shape[0]),
                    ubg=[0] * (dFdz.shape[1] * dFdz.shape[0]),
                    g_names=[f"helper_matrix_defect"] * (dFdz.shape[1] * dFdz.shape[0]),
                    node=i_node,
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

        if discretization_method.with_helper_matrix:
            # CoV constraint
            cov_constraint_final = self.cov_constraint_func_final(
                variables_vector.get_time(),
                variables_vector.get_state("q", n_shooting - 1),
                variables_vector.get_state("q", n_shooting),
                variables_vector.get_state("qdot", n_shooting),
                cas.horzcat(
                    variables_vector.get_state("q", n_shooting - 1), variables_vector.get_state("q", n_shooting)
                ),
                variables_vector.get_cov(n_shooting - 1),
                variables_vector.get_cov(n_shooting),
                variables_vector.get_ms(n_shooting - 1),
                variables_vector.get_controls(n_shooting - 1),
                variables_vector.get_controls(n_shooting),
                noises_vector.get_one_vector_numerical(n_shooting - 1),
            )

            constraints.add(
                g=cov_constraint_final,
                lbg=[0] * (nb_q * nb_q),
                ubg=[0] * (nb_q * nb_q),
                g_names=[f"cov_defect_final"] * (nb_q * nb_q),
                node=n_shooting,
            )

            # Constrain M at all collocation points to follow df_integrated/dz.T - dg_integrated/dz @ m.T = 0
            m_matrix_1 = variables_vector.get_m_matrix(1)[
                :, : 2 * ocp_example.model.nb_q
            ]  # Only the two fisrt collocations points
            _, dFdz_final, dGdz_final, _, _ = self.jacobian_funcs_final(
                variables_vector.get_time(),
                variables_vector.get_state("q", n_shooting - 1),
                variables_vector.get_state("q", n_shooting),
                variables_vector.get_state("qdot", n_shooting),
                cas.horzcat(
                    variables_vector.get_state("q", n_shooting - 1), variables_vector.get_state("q", n_shooting)
                ),
                variables_vector.get_controls(n_shooting - 1),
                variables_vector.get_controls(n_shooting),
                cas.DM.zeros(ocp_example.model.nb_noises * variables_vector.nb_random),
                cas.DM.zeros(ocp_example.model.nb_noises * variables_vector.nb_random),
            )

            constraint = dFdz_final.T - dGdz_final.T @ m_matrix_1.T
            constraints.add(
                g=variables_vector.reshape_matrix_to_vector(constraint),
                lbg=[0] * (dFdz_final.shape[1] * dFdz_final.shape[0]),
                ubg=[0] * (dFdz_final.shape[1] * dFdz_final.shape[0]),
                g_names=[f"helper_matrix_defect_final"] * (dFdz_final.shape[1] * dFdz_final.shape[0]),
                node=n_shooting,
            )
