import numpy as np
import pytest
from dcwb.matrix import identity, from_diag, compose

def test_identity_is_3x3_identity():
    I = identity()
    assert I.shape == (3, 3)
    np.testing.assert_array_equal(I, np.eye(3))

def test_from_diag_creates_diagonal():
    M = from_diag(0.9, 1.0, 1.1)
    expected = np.array([
        [0.9, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.1],
    ])
    np.testing.assert_array_equal(M, expected)

def test_compose_is_matmul_of_first_then_second():
    A = from_diag(2.0, 1.0, 1.0)
    B = from_diag(1.0, 1.0, 3.0)
    C = compose(A, B)
    np.testing.assert_array_equal(C, A @ B)

def test_compose_with_identity_is_noop():
    A = from_diag(0.5, 1.5, 0.7)
    np.testing.assert_array_equal(compose(A, identity()), A)
    np.testing.assert_array_equal(compose(identity(), A), A)

def test_shape_validation_rejects_wrong_shape():
    with pytest.raises(ValueError):
        compose(np.zeros((2, 2)), identity())
