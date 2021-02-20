import numpy as np
from la import qr_iteration,eigengap_method
from kmeans_pp import k_means_pp #TODO synchronize with kmeans_pp
from timeit import timeit

def weight_func(x_i, x_j):
    """
    :param x_i, x_j: d-dimensioned vectors
    :return: calculates the weight of connection between x_i to x_j
    """
    diff_vector = x_i - x_j
    #TODO euclidean norm with sqrt or without?
    # return np.exp(-0.5 * (np.inner(diff_vector, diff_vector)))
    return np.exp(-0.5 * np.linalg.norm(diff_vector))



def form_Laplacian(x):
    """
    :param x: an array of n vector from d-dimension; i.e. array of shape (n,d)
    :return: calculates the laplacian based on x
    """
    n = x.shape[0]

    # form Weight Matrix :
    w = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            w[i,j] = weight_func(x[i],x[j])
    w = w + w.T # adds the symmetric upper triangle

    # form D^-0.5 :
    d = np.zeros((n,n))
    np.fill_diagonal(d, 1 / np.sqrt(np.sum(w, axis=0)))

    # calculate L_norm
    return np.eye(n) - np.linalg.multi_dot([d,w,d])

def form_U(l):
    """
    :param l: l_norm matrix, i.e. laplacian
    :return: forms U, the matrix contains the first K eigenvectors which determined by EigenGap
            (U is of shape (n,k), each column is a chosen eigen-vector)
    """

    e_values, e_vectors = qr_iteration(l)
    k_indices = eigengap_method(e_values)
    u = e_vectors[:, k_indices]
    return u

def form_T(u):
    t = u / np.linalg.norm(arr, axis=1, keepdims=True)
    return t

def run_nsc(x):
    """
    :param x: a collection of n points in R^d, given via array of shape (n,d)
    :return: the result of the Normalized Spectral Algorithm:
             res - n-sized array, res[i]=j IFF x_i belongs to cluster c_j
    """
    # Phase 1&2:
    l = form_Laplacian(x)
    # Phase 3%4:
    u = form_U(l)
    # Phase 5
    t = form_T(u)
    k = t.shape[1]
    # Phase 6&7
    res = k_means(k, t)
    # TODO synchronize kmeans.
    #  notes: (1) delivers ndarray each row is obs!
    #         (2) expects clusters indices back! (not just centroids)
    return res


# ------------------------------------ END OF CODE ------------------------
def sanity_check():
    x = np.array(
        [
            [0, 2, 4],
            [2, 4, 6],
            [4, 6, 8],
            [5, 5, 5],
            [2, 8, 99],
        ]
    )
    print(form_laplacian(x))

# deprecated code - pythonic way is surprisingly faster
# def form_laplacian1(x):
#     """
#     :param x: an array of n vector from d-dimension; i.e. array of shape (n,d)
#     :return: calculates the laplacian based on x
#     """
#     n = x.shape[0]
#
#     np_weight = np.frompyfunc(lambda i, j: weight_func(x[int(i)], x[int(j)]), 2, 1)
#     w = np.fromfunction(np_weight, (n, n))
#     np.fill_diagonal(w,0)
#     print(w)