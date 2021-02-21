import numpy as np
import pytest
import linalg


class TestGS:
    def test_invalid_matrix(self):
        with pytest.raises(ValueError):
            a = np.empty((4, 3))
            a[0] = [1, -1, 4]
            a[1] = [1, 4, -2]
            a[2] = [1, 4, 2]
            a[3] = [1, -1, 0]
            linalg.gram_schmidt(a)

    def test_matrix_3_3_float(self):
        a = np.empty((3, 3))
        a[0] = [2.11, 2.79, 3.72]
        a[1] = [10.32, 8.74, 6.88]
        a[2] = [9.97, 4.17, 8.79]

        expected_q = np.empty((3, 3))
        expected_q[0] = [0.15, 0.42, 0.90]
        expected_q[1] = [0.71, 0.59, -0.39]
        expected_q[2] = [0.69, -0.70, 0.21]

        expected_r = np.empty((3, 3))
        expected_r[0] = [14.50, 9.49, 11.48]
        expected_r[1] = [0, 3.39, -0.53]
        expected_r[2] = [0, 0, 2.53]

        q, r = linalg.gram_schmidt(a)

        below_diagonal = np.array([r[1, 0], r[2, 0], r[2, 1]])
        assert (below_diagonal == np.zeros(3)).all()
        assert np.allclose(q, expected_q, atol=0.01)
        assert np.allclose(r, expected_r, atol=0.01)

    def test_matrix_3_3_int(self):
        a = np.empty((3, 3))
        a[0] = [12, -51, 4]
        a[1] = [6, 167, -68]
        a[2] = [-4, 24, -41]

        expected_q = np.empty((3, 3))
        expected_q[0] = [6 / 7, -69 / 175, -58 / 175]
        expected_q[1] = [3 / 7, 158 / 175, 6 / 175]
        expected_q[2] = [-2 / 7, 6 / 35, -33 / 35]

        expected_r = np.empty((3, 3))
        expected_r[0] = [14, 21, -14]
        expected_r[1] = [0, 175, -70]
        expected_r[2] = [0, 0, 35]

        q, r = linalg.gram_schmidt(a)
        assert np.allclose(q, expected_q, atol=0.01)
        assert np.allclose(r, expected_r, atol=0.01)
