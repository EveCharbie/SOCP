import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


class CovarianceIntegrator:
    """
    Numerical integration of Lyapunov differential equation for uncertainty propagation.

    P_dot(t) = A(t)*P(t) + P(t)*A^T(t) + B(t)*Σ*B^T(t)

    where:
    - A(t) = df/dx: Jacobian w.r.t. state
    - B(t) = df/dw: Jacobian w.r.t. disturbance
    - Σ: Covariance matrix of disturbance w
    """

    def __init__(self, f, n_states, n_disturbances, sigma, epsilon=1e-7):
        """
        Parameters:
        -----------
        f : callable
            System dynamics f(x, u, w, t) returning dx/dt
        n_states : int
            Dimension of state vector x
        n_disturbances : int
            Dimension of disturbance vector w
        sigma : ndarray
            Covariance matrix of disturbance (n_disturbances x n_disturbances)
        epsilon : float
            Finite difference step size for Jacobian computation
        """
        self.f = f
        self.n_x = n_states
        self.n_w = n_disturbances
        self.sigma = np.array(sigma)
        self.epsilon = epsilon

    def compute_jacobian_x(self, x, u, w, t):
        """
        Compute A(t) = df/dx using finite differences.

        Returns:
        --------
        A : ndarray of shape (n_x, n_x)
        """
        A = np.zeros((self.n_x, self.n_x))
        f0 = self.f(x, u, w, t)

        for i in range(self.n_x):
            x_perturbed = x.copy()
            x_perturbed[i] += self.epsilon
            f_perturbed = self.f(x_perturbed, u, w, t)
            A[:, i] = (f_perturbed - f0) / self.epsilon

        return A

    def compute_jacobian_w(self, x, u, w, t):
        """
        Compute B(t) = df/dw using finite differences.

        Returns:
        --------
        B : ndarray of shape (n_x, n_w)
        """
        B = np.zeros((self.n_x, self.n_w))
        f0 = self.f(x, u, w, t)

        for i in range(self.n_w):
            w_perturbed = w.copy()
            w_perturbed[i] += self.epsilon
            f_perturbed = self.f(x, u, w_perturbed, t)
            B[:, i] = (f_perturbed - f0) / self.epsilon

        return B

    def lyapunov_rhs(self, t, P_vec, x, u, w):
        """
        Right-hand side of Lyapunov equation.

        Parameters:
        -----------
        t : float
            Current time
        P_vec : ndarray
            Vectorized P matrix (length n_x^2)
        x : ndarray
            Current state
        u : ndarray or callable
            Control input (or function u(t))
        w : ndarray
            Nominal disturbance (usually zeros)

        Returns:
        --------
        P_dot_vec : ndarray
            Vectorized time derivative of P
        """
        # Reshape vectorized P back to matrix
        P = P_vec.reshape((self.n_x, self.n_x))

        # Get control at current time
        u_t = u(t) if callable(u) else u

        # Compute Jacobians
        A = self.compute_jacobian_x(x, u_t, w, t)
        B = self.compute_jacobian_w(x, u_t, w, t)

        # Lyapunov equation: P_dot = A*P + P*A^T + B*Sigma*B^T
        P_dot = A @ P + P @ A.T + B @ self.sigma @ B.T

        # Vectorize for ODE solver
        return P_dot.flatten()

    def integrate_with_state(self, x_traj, u, w_nominal, P0, time_vector):
        """
        Integrate Lyapunov equation along a state trajectory.

        Parameters:
        -----------
        x_traj : callable
            State trajectory function x(t)
        u : ndarray or callable
            Control input (constant or function u(t))
        w_nominal : ndarray
            Nominal disturbance value (usually zeros)
        P0 : ndarray
            Initial covariance matrix (n_x x n_x)
        time_vector : ndarray
            Time points to evaluate solution

        Returns:
        --------
        result : dict
            Dictionary containing:
            - 't': time points
            - 'P': covariance matrices at each time point
            - 'trace': trace of P(t) (total variance)
        """
        nb_shooting = len(time_vector) - 1
        dt = time_vector[1] - time_vector[0]

        # Initial condition (vectorized P0)
        P0_vec = P0.flatten()

        # Solve ODE
        P_history = []
        for i_node in range(nb_shooting):

            def this_ode_func(t, P_vec):
                u_this_time = u[:, i_node] + (u[:, i_node+1] - u[:, i_node]) * (t - time_vector[i_node]) / dt
                return self.lyapunov_rhs(t, P_vec, x_traj[:, i_node], u_this_time, w_nominal[:, i_node])

            sol = solve_ivp(this_ode_func, (i_node*dt, (i_node+1)*dt), P0_vec, method='RK45', rtol=1e-8, atol=1e-10)

            # Reshape solutions
            P_history += [sol.y[:, i].reshape((self.n_x, self.n_x))
                         for i in range(sol.y.shape[1])]

        # Compute trace (total variance)
        trace_history = [np.trace(P) for P in P_history]

        return {
            't': sol.t,
            'P': P_history,
            'trace': np.array(trace_history),
            'sol': sol
        }