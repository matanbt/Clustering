"""
----- Normalized Spectral Clustering -----
Implementation for the Norm. Spectral Clustering Algorithm, using 'linalg' module
Note: all functions assume correctness of the input; in particular an input of
      ndarray with type 'float64'
"""
import numpy as np
from linalg import qr_iteration, eigengap_method
from config import MAX_ITER
from kmeans_pp import k_means  # TODO synchronize with kmeans_pp


def weight_func(x_i, x_j):
    """
    Calculates the Weight of connection of 2 given vectors
    :param x_i, x_j: d-dimensioned vectors
    :return: calculates the weight of connection between x_i to x_j
    """
    diff_vector = x_i - x_j
    return np.exp(-0.5 * np.linalg.norm(diff_vector))


def form_weight(x):
    """
    :param x: an array of n vector from d-dimension; i.e. array of shape (n,d)
    :return: calculates the connection-weight-matrix of x of shape (n,n)
    """
    n = x.shape[0]

    # form Weight Matrix :
    w = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            w[i, j] = weight_func(x[i], x[j])
    w = w + w.T  # adds the symmetric upper triangle
    return w


# TODO either optimize the next function or delete it
def form_weight_np(x):
    """
    numpy alternative to 'form_Weight' (tested to run *slower* than python)
    """
    n = x.shape[0]

    # form Weight Matrix :
    w = np.zeros((n, n))
    u_weight_func = np.frompyfunc(lambda i, j: weight_func(x[i], x[j]), 2, 1)
    triu1, triu2 = np.triu_indices(n, 1)
    upper_triangle = u_weight_func(triu1, triu2)
    w[triu1, triu2] = upper_triangle
    w = w + w.T  # adds the symmetric upper triangle
    return w


def form_laplacian(w):
    """
    :param w: Positive weight-matrix of shape (n,n)
    :return: the laplacian based on the weight-matrix
    """
    n = w.shape[0]

    # form D^-0.5 :
    d = np.zeros((n, n))
    np.fill_diagonal(d, 1 / np.sqrt(np.sum(w, axis=0), dtype=float))

    # calculate L_norm
    return np.eye(n) - np.linalg.multi_dot([d, w, d])


def form_u(l, k=None):
    """
    :param l: l_norm matrix, i.e. laplacian
    :param k: optional - choose k in advance and force it
    :return: forms U, the matrix contains the first K eigenvectors which
             determined by EigenGap
            (U is of shape (n,k), each column is a chosen eigen-vector)
    """

    e_values, e_vectors = qr_iteration(l)
    k_indices = eigengap_method(e_values, k)
    u = e_vectors[:, k_indices]
    return u


def form_t(u):
    t = u / np.linalg.norm(u, axis=1, keepdims=True)
    return t


def run_nsc(points, k=None):
    """
    Normalized Spectral Clustering Algorithm
    :param points: a collection of n points in R^d, given via array of shape (n,d)
    :param k: optional - choose k in advance and force it
    :return: the result of the Normalized Spectral Algorithm:
             res - n-sized array, res[i]=j IFF x_i belongs to cluster c_j
    """
    # Phase 1:
    w = form_weight(points)
    # Phase 2:
    l = form_laplacian(w)
    # Phase 3&4:
    u = form_u(l, k)
    # Phase 5
    t = form_t(u)
    n, k = t.shape
    # Phase 6&7
    # kmeans(points, K, N, d, MAX_ITER)
    res = kmeans(points=t, K=k, N=n, d=k, MAX_ITER=MAX_ITER)
    return res
