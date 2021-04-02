"""
K-MEANS-PP ALGORITHM IMPLEMENTATION
"""

import numpy as np
import mykmeanssp as km

KMEANS_INIT_RANDOM_SEED = 0


def k_means_pp(k, obs_arr):
    """
    The initialization part of the KMeans++ algorithm.
    :param k: Number of clusters
    :param obs_arr: Observations array
    :return: The indices of the observations to initialize the KMeans clusters
    with.
    """
    np.random.seed(KMEANS_INIT_RANDOM_SEED)

    N = len(obs_arr)
    initial_indices = np.empty(k, dtype=int)

    last_index = np.random.choice(N)
    initial_indices[0] = last_index
    minimal_distances = np.linalg.norm(obs_arr - obs_arr[last_index], axis=1)
    probs = np.empty_like(minimal_distances)
    for i in range(1, k):
        np.divide(minimal_distances, minimal_distances.sum(), out=probs)
        last_index = np.random.choice(N, p=probs)
        initial_indices[i] = last_index
        new_distances = np.linalg.norm(obs_arr - obs_arr[last_index], axis=1)
        np.minimum(minimal_distances, new_distances, out=minimal_distances)

    # Resetting numpy's random seed
    np.random.seed(None)
    return initial_indices


def kmeans(points, K, N, d, MAX_ITER):
    """
    Run the KMeans algorithm (wrapper for the C extension module).
    :param points: Observation points
    :param K: Number of clusters
    :param N: Number of observations
    :param d: Dimensions of points
    :param MAX_ITER: Maximum iterations for the KMeans algorithm
    :return: The cluster of each observation, as a list.
    """
    indices = k_means_pp(K, points)
    c_points = points.tolist()
    indices = indices.tolist()
    clusters = km.kmeans(c_points, indices, K, N, d, MAX_ITER)
    return clusters
