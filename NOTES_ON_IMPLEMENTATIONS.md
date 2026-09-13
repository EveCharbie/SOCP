# Covariance matrix computations
I used the formula  `covariance_matrix = ((X - x_mean) @ (X - x_mean).T) / N` to compute the covariance of sampled states X.
I consider that during the reintegration, I know the real mean of the state samples, since I impose it, so `N = nb_samples`.
I consider that in the NoiseDiscretization approach, mean of the state samples is computed, so `N = nb_samples - 1` due to Bessel's correction.


# Time x noise handling
For now, there are two cases considered in this code base:
1. Discretized-time: In this version, the noise are randomly sampled from a Gaussian distribution at the beginning of each time interval and are kept constant throughout the time interval. In this formulation, the noise in not Brownian, so the duration of the movement simulated cannot be optimized. This formulation is compatible with deterministic integration schemes: DirectCollocationPolynomial, DirectCollocationTrapezoidal, DirectMultipleShooting (RK4 multiple steps), Variational, and VariationalPolynomial. In this version the controls are piecewise linear continuous.
2. Continuous-time: In this version, the noise is Brownian so it follows a fractal structure temporally. In this formulation, the noise is Brownian, so the duration of the movement simulated can be optimized. This formulation is compatible with stochastic integration schemes: DirectCollocationTrapezoidal, DirectMultipleShooting (SRK4 one step). In this version, the controls are piecewise constant.
