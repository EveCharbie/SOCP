import numpy as np
from scipy.stats import norm, qmc


def exact_gaussian(nb_points: int, mean: float = 0.0, std: float = 1.0):
    """
    Generate a deterministic set of points approximating N(mean, std^2).
    """

    if nb_points % 2 != 0:
        raise ValueError("n must be even.")

    # Equally spaced probabilities
    p = (np.arange(nb_points) + 0.5) / nb_points

    # Gaussian quantiles
    z = norm.ppf(p)

    # Enforce exact zero mean and unit standard deviation
    z -= np.mean(z)
    z /= np.std(z)

    # Transform to desired Gaussian
    x = mean + std * z

    return x


def exact_gaussian_covariance_matrix(
    nb_points: int,
    mean: np.ndarray = np.zeros((2, )),
    covariance: np.ndarray = np.eye(2),
):
    """
    Generate a deterministic finite representation of a Gaussian covariance matrix N(mean, covariance).
    """

    mean = np.asarray(mean, dtype=float)
    covariance = np.asarray(covariance, dtype=float)

    d = len(mean)

    if covariance.shape != (d, d):
        raise ValueError("covariance must have shape (d, d).")

    if nb_points % 2 != 0:
        raise ValueError("nb_points must be even.")

    # Check positive definiteness
    try:
        L = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError:
        raise ValueError(
            "Covariance matrix must be positive definite."
        )

    # 1. Generate deterministic (unscrambled) Sobol low-discrepancy points, best equidistributed when nb_points/2 is a power of 2
    sampler = qmc.Sobol(
        d=d,
        scramble=False,
    )
    U_half = sampler.random(nb_points // 2)

    # Avoid exactly 0 or 1
    U_half = np.clip(U_half, 1e-12, 1 - 1e-12)

    # Transform uniform -> standard Gaussian
    Z_half = norm.ppf(U_half)

    # 2. Antithetic augmentation: pairing every point with its exact negation makes the point
    # cloud symmetric about zero in every direction at once, forcing zero mean, zero skewness,
    # and median == mean, not just the per-axis marginals.
    Z = np.vstack([Z_half, -Z_half])

    # 3. Force identity empirical covariance
    # Empirical covariance using population convention (1/N)
    C = (Z.T @ Z) / nb_points

    # C^(-1/2)
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    if eigenvalues.min() < 1e-10:
        raise ValueError(
            f"The empirical covariance of the {nb_points} generated points is rank-deficient "
            f"(smallest eigenvalue {eigenvalues.min():.3e}) for dimension {d}. Low-discrepancy "
            f"sequences are structured, not generic, so nb_points must be comfortably larger than "
            f"the dimension (nb_points//2 >> d) for the empirical covariance to be full rank; "
            f"increase nb_points."
        )

    C_inv_sqrt = (
        eigenvectors
        @ np.diag(1.0 / np.sqrt(eigenvalues))
        @ eigenvectors.T
    )

    Z = Z @ C_inv_sqrt

    # 4. Transform N(0,I) -> N(mean, covariance)
    X = mean + Z @ L.T

    return X

