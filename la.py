"""
----- Linear Algebra Module -----
Contains implementation to all algorithms related with Linear Alg.
"""
import numpy as np
from config import epsilon




def qr_iteration(A):
    """
    :param A: 2-D ndarray, with shape (n,n)
    :precondition: input is valid
    :return: A_ - diagonal matrix with A's approximated eigenvalues
             Q_ - orthogonal matrix, each column is approximated eigenvector of A

    """
    n = A.shape[0]
    A_ = A.astype('float64')
    Q_ = np.eye(n)

    for i in range(n):
        # TODO switch next line to our implementation
        Q, R = np.linalg.qr(A_)

        np.matmul(R, Q, out=A_)
        temp_Q = np.matmul(Q_, Q)

        delta_matrix = np.abs(Q_) - np.abs(temp_Q)
        if np.all(np.abs(delta_matrix) <= epsilon):
            # reached convergence
            return A_, Q_

        Q_ = temp_Q

    # reached iterations bound (n)
    return A_, Q_

def eigengap_method(A_):
    """
    :param A_: (ndarray) matrix contains the eigenvalues on its diagonal line
    :return: determines K (by the eigengap method),
             returns an array of the 'first' K *indices* of the appropriate eigenvectors
             Note: the calculated K is the length of the returned array
             Note 2: assumption is that the original order is valuable, hence its indices are to be returned
    """
    n = A_.shape[0]

    eigen_values = np.diagonal(A_)
    # 'stable' sorts eigen-values, and keeps the *indices* of the sorted array
    sorted_indices = np.argsort(eigen_values, kind='stable')
    # calculates the abs difference array for the first half of the eigen-values
    delta_arr = np.diff(eigen_values[sorted_indices][:n//2+1])
    delta_arr = np.abs(delta_arr)
    # gets the first appearance of the maximum difference
    k = np.argmax(delta_arr) + 1
    return sorted_indices[:k]
