import numpy as np


def gram_schmidt(mat_a):
    """
    This function calculates the QR decomposition of the matrix A.
    :param mat_a: The matrix to calculate the composition on
    :return: 2 matrices, of the same order as A, that will be its QR decomposition.
    """
    # NOTE: We will use the same variable names as the one in the
    # pseudo code for clarity
    rows_count = mat_a.shape[0]

    # TODO: Check how much does the order of the matrix matters ('C' or 'F')
    u = mat_a.copy()
    r = np.zeros_like(u)
    q = np.zeros_like(u)
    for i in range(rows_count):
        u_i = u[:, i]
        r[i, i] = np.linalg.norm(u_i)
        q[:, i] = u_i / r[i][i]
        q_i = q[:, i]
        for j in range(i + 1, rows_count):
            r[i, j] = q_i.T.dot(u[:, j])
            u[:, j] -= r[i][j] * q_i

    return q, r