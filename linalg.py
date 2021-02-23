"""
----- Linear Algebra Module -----
Contains implementation to all algorithms related with Linear Alg.
"""
import numpy as np

from config import EPSILON


def qr_iteration(a):
    """
    QR Iteration Algorithm for finding EigenValues and EigenVectors
    :param a: 2-D ndarray, with shape (n,n)
    :precondition: input is valid, a is from type 'float64'
    :return: a_ - diagonal matrix with a's approximated eigenvalues
             q_ - orthogonal matrix, each column is approximated eigenvector of a
    """
    # Implementation Note: variables names kept the same as the pseudo code's
    n = a.shape[0]
    a_ = a.copy()
    q_ = np.eye(n)
    temp_q = np.empty((n, n))
    delta_matrix = np.empty((n, n))

    for i in range(n):
        # q, r = gram_schmidt(a_)
        q, r = np.linalg.qr(a_)
        np.matmul(r, q, out=a_)
        np.matmul(q_, q, out=temp_q)
        np.subtract(np.abs(q_), np.abs(temp_q), out=delta_matrix)
        if np.all(np.abs(delta_matrix, out=delta_matrix) <= EPSILON):
            # reached convergence
            return a_, q_
        q_, temp_q = temp_q, q_
    # reached iterations bound (n)
    return a_, q_


def eigengap_method(a_, k=None):
    """
    EigenGap Method - a heuristic for finding the amount of clusters
    :param a_: (ndarray) matrix contains the eigenvalues on its diagonal line
    :param k: if k is not None - skips k-calculation and forces it to be the given k
              Note: this option effectively cancels the eigengap method.
    :return: array of the 'first' K *indices* of the appropriate eigenvectors
             Note: the calculated K is the length of the returned array, which
                   determined  by the eigengap method
             Note 2: assumption is that the original order is valuable, hence
                     its indices are to be returned
    """
    n = a_.shape[0]

    eigen_values = np.diagonal(a_)
    # sorts eigen-values, and keeps the *indices* of the sorted array
    sorted_indices = np.argsort(eigen_values)
    if k is None:
        # calculates the abs difference array for the first half of the  eigen-values
        delta_arr = np.diff(eigen_values[sorted_indices][:n // 2 + 1])
        np.abs(delta_arr, out=delta_arr)
        # gets the first appearance of the maximum difference
        k = np.argmax(delta_arr) + 1
    return sorted_indices[:k]
