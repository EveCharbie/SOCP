# Covariance matrix computations
I used the formula  `covariance_matrix = ((X - x_mean) @ (X - x_mean).T) / N` to compute the covariance of sampled states X.
I consider that during the reintegration, I know the real mean of the state samples, since I impose it, so `N = nb_samples`.
I consider that in the NoiseDiscretization approach, mean of the state samples is computed, so `N = nb_samples - 1` due to Bessel's correction.
