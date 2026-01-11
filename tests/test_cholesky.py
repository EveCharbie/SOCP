import numpy as np
import casadi as cas
import numpy.testing as npt


def test_reshape():
    from socp.models.model_abstract import ModelAbstract

    # Create a matrix of shape (10, 10)
    random_matrix = np.random.rand(10, 10)

    # Reshape to vector
    random_casadi = cas.DM(random_matrix)
    vector = ModelAbstract.reshape_matrix_to_vector(random_casadi)

    # Reshape back to matrix
    output_matrix = ModelAbstract.reshape_vector_to_matrix(vector, (10, 10))

    # Check if the reshaped matrix matches the original
    npt.assert_array_almost_equal(np.array(output_matrix), random_matrix, decimal=6)


def test_cholesky_reshape():
    from socp.models.model_abstract import ModelAbstract

    # Create a symmetric positive definite matrix of shape (10, 10)
    random_matrix = np.random.rand(10, 10)
    A = np.dot(random_matrix, random_matrix.T) + 10 * np.eye(10)
    L = np.linalg.cholesky(A)

    # Reshape to vector
    L_cas = cas.DM(L)
    L_vector = ModelAbstract.reshape_cholesky_matrix_to_vector(L_cas)

    # Reshape back to matrix
    L_matrix = ModelAbstract.reshape_vector_to_cholesky_matrix(L_vector, (10, 10))

    # Check if the reshaped matrix matches the original
    npt.assert_array_almost_equal(np.array(L_matrix), L, decimal=6)

    # and back to a vector
    L_vector_back = ModelAbstract.reshape_cholesky_matrix_to_vector(L_matrix)
    npt.assert_array_almost_equal(np.array(L_vector_back), np.array(L_vector), decimal=6)

    # Test that the vector does not contain zeros as only the triangular part is stored
    assert int(np.count_nonzero(np.array(L_vector))) == L_vector.size()[0]
    assert L_vector.size()[0] == ModelAbstract.nb_cholesky_components(10)


def test_cholesky_decomposition():
    from socp.models.model_abstract import ModelAbstract

    # Create a symmetric positive definite matrix of shape (10, 10)
    random_matrix = np.random.rand(10, 10)
    A = np.dot(random_matrix, random_matrix.T) + 10 * np.eye(10)
    L_np = np.linalg.cholesky(A)

    # Reshape to vector
    L_cas = cas.transpose(cas.chol(A))

    # Check if the cholesky decomposition are equivalent
    npt.assert_array_almost_equal(L_np, np.array(L_cas), decimal=5)