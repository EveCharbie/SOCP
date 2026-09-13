import numpy as np
import numpy.testing as npt
import pytest
import matplotlib.pyplot as plt
from scipy.stats import norm

from socp.transcriptions.utils import exact_gaussian, exact_gaussian_covariance_matrix


def test_gaussian():
    PLOT_FLAG = False

    nb_points = 1008
    mean = 10
    std = 3

    x = exact_gaussian(
        nb_points=nb_points,
        mean=mean,
        std=std,
    )

    # ============================================================
    # Check properties
    # ============================================================
    assert mean == np.mean(x)
    assert mean == np.median(x)
    assert std == np.std(x)

    if PLOT_FLAG:
        plt.figure(figsize=(10, 6))
        plt.hist(
            x,
            bins=50,
            density=True,
            alpha=0.7,
            label="Generated points"
        )

        # Theoretical Gaussian PDF
        x_pdf = np.linspace(
            mean - 4 * std,
            mean + 4 * std,
            nb_points * 10
        )

        pdf = norm.pdf(
            x_pdf,
            loc=mean,
            scale=std
        )

        plt.plot(
            x_pdf,
            pdf,
            linewidth=2,
            label="Theoretical Gaussian"
        )

        plt.xlabel("X")
        plt.ylabel("Probability density")
        plt.title(f"Gaussian distribution — N={nb_points:,}, μ={mean}, σ={std}")
        plt.legend()
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.show()


def test_gaussian_covariance():
    PLOT_FLAG = False

    nb_points = 1024
    mean = np.array([
        10.0,
        5.0,
        -2.0
    ])
    covariance = np.array([
        [9.0, 3.0, 1.0],
        [3.0, 4.0, 0.5],
        [1.0, 0.5, 1.0]
    ])

    X = exact_gaussian_covariance_matrix(
        nb_points=nb_points,
        mean=mean,
        covariance=covariance
    )

    d = len(mean)
    npt.assert_almost_equal(mean, np.mean(X, axis=0))
    npt.assert_almost_equal(mean, np.median(X, axis=0))
    npt.assert_almost_equal(covariance, (((X - np.mean(X, axis=0)).T) @ (X - np.mean(X, axis=0))) / nb_points)

    if PLOT_FLAG:
        fig, axes = plt.subplots(1, d, figsize=(15, 4))
        for i, ax in enumerate(axes):
            ax.hist(X[:, i], bins=35, density=True, alpha=0.7, label="Generated points")
            xx = np.linspace(
                mean[i] - 4 * np.sqrt(covariance[i, i]),
                mean[i] + 4 * np.sqrt(covariance[i, i]),
                500
            )
            pdf = norm.pdf(xx, loc=mean[i], scale=np.sqrt(covariance[i, i]))
            ax.plot(xx, pdf, linewidth=2, label="Gaussian PDF")
            ax.set_xlabel(f"$X_{i + 1}$")
            ax.set_ylabel("Density")
            ax.legend()
        plt.tight_layout()
        plt.show()


def test_exact_gaussian_covariance_matrix_is_deterministic():
    """
    Two independent calls with no seed must produce bit-identical output: the whole point of
    replacing seeded LatinHypercube sampling with unscrambled Sobol was to remove randomness
    entirely, not just make it reproducible.
    """
    mean = np.array([1.0, -2.0])
    covariance = np.array([[2.0, 0.3], [0.3, 1.5]])

    X1 = exact_gaussian_covariance_matrix(nb_points=64, mean=mean, covariance=covariance)
    X2 = exact_gaussian_covariance_matrix(nb_points=64, mean=mean, covariance=covariance)

    npt.assert_array_equal(X1, X2)


def test_exact_gaussian_covariance_matrix_rejects_odd_nb_points():
    with pytest.raises(ValueError):
        exact_gaussian_covariance_matrix(nb_points=63, mean=np.zeros(2), covariance=np.eye(2))


def test_exact_gaussian_covariance_matrix_raises_on_rank_deficiency():
    """
    Low-discrepancy sequences are structured, not generic: nb_points too small relative to the
    dimension must raise a clear error instead of silently returning NaNs.
    """
    d = 14
    with pytest.raises(ValueError):
        exact_gaussian_covariance_matrix(nb_points=30, mean=np.zeros(d), covariance=np.eye(d))
