import numpy as np
import casadi as cas
import numpy.testing as npt

from socp.transcriptions.variables_abstract import VariablesAbstract


def test_reshape():

    # Create a matrix of shape (10, 10)
    random_matrix = np.random.rand(10, 10)

    # Reshape to vector
    random_casadi = cas.DM(random_matrix)
    vector = VariablesAbstract.reshape_matrix_to_vector(random_casadi)

    # Reshape back to matrix
    output_matrix = VariablesAbstract.reshape_vector_to_matrix(vector, (10, 10))

    # Check if the reshaped matrix matches the original
    npt.assert_array_almost_equal(np.array(output_matrix), random_matrix, decimal=6)


def test_cholesky_reshape():

    # Create a symmetric positive definite matrix of shape (10, 10)
    random_matrix = np.random.rand(10, 10)
    A = np.dot(random_matrix, random_matrix.T) + 10 * np.eye(10)
    L = np.linalg.cholesky(A)

    # Reshape to vector
    L_cas = cas.DM(L)
    L_vector = VariablesAbstract.reshape_cholesky_matrix_to_vector(L_cas)

    # Reshape back to matrix
    L_matrix = VariablesAbstract.reshape_vector_to_cholesky_matrix(L_vector, (10, 10))

    # Check if the reshaped matrix matches the original
    npt.assert_array_almost_equal(np.array(L_matrix), L, decimal=6)

    # and back to a vector
    L_vector_back = VariablesAbstract.reshape_cholesky_matrix_to_vector(L_matrix)
    npt.assert_array_almost_equal(np.array(L_vector_back), np.array(L_vector), decimal=6)

    # Test that the vector does not contain zeros as only the triangular part is stored
    assert int(np.count_nonzero(np.array(L_vector))) == L_vector.size()[0]
    assert L_vector.size()[0] == VariablesAbstract.nb_cholesky_components(10)


def test_cholesky_decomposition():

    # Create a symmetric positive definite matrix of shape (10, 10)
    random_matrix = np.random.rand(10, 10)
    A = np.dot(random_matrix, random_matrix.T) + 10 * np.eye(10)
    L_np = np.linalg.cholesky(A)

    # Reshape to vector
    L_cas = cas.transpose(cas.chol(A))

    # Check if the cholesky decomposition are equivalent
    npt.assert_array_almost_equal(L_np, np.array(L_cas), decimal=5)


def test_casadi_vs_numpy_implementations():

    # Start with a numpy matrix
    random_matrix = np.random.rand(10, 10)

    # Test reshape_matrix_to_vector
    numpy_vector = VariablesAbstract.reshape_matrix_to_vector(random_matrix)
    casadi_vector = VariablesAbstract.reshape_matrix_to_vector(cas.DM(random_matrix))
    npt.assert_array_almost_equal(np.array(casadi_vector), numpy_vector, decimal=6)

    # Test reshape_vector_to_matrix
    numpy_matrix = VariablesAbstract.reshape_vector_to_matrix(numpy_vector, (10, 10))
    casadi_matrix = VariablesAbstract.reshape_vector_to_matrix(casadi_vector, (10, 10))
    npt.assert_array_almost_equal(np.array(casadi_matrix), numpy_matrix, decimal=6)
    npt.assert_array_almost_equal(random_matrix, numpy_matrix, decimal=6)

    # Test reshape_cholesky_matrix_to_vector
    random_spd_matrix = np.dot(random_matrix, random_matrix.T) + 10 * np.eye(10)
    L_np = np.linalg.cholesky(random_spd_matrix)
    numpy_cholesky_vector = VariablesAbstract.reshape_cholesky_matrix_to_vector(L_np)
    casadi_cholesky_vector = VariablesAbstract.reshape_cholesky_matrix_to_vector(cas.DM(L_np))
    npt.assert_array_almost_equal(np.array(casadi_cholesky_vector), numpy_cholesky_vector, decimal=6)

    # Test reshape_vector_to_cholesky_matrix
    numpy_cholesky_matrix = VariablesAbstract.reshape_vector_to_cholesky_matrix(numpy_cholesky_vector, (10, 10))
    casadi_cholesky_matrix = VariablesAbstract.reshape_vector_to_cholesky_matrix(casadi_cholesky_vector, (10, 10))
    npt.assert_array_almost_equal(np.array(casadi_cholesky_matrix), numpy_cholesky_matrix, decimal=6)
    npt.assert_array_almost_equal(L_np, numpy_cholesky_matrix, decimal=6)
