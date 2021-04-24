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

    chosen_index = np.random.choice(N)
    initial_indices[0] = chosen_index
    minimal_distances = np.linalg.norm(obs_arr - obs_arr[chosen_index], axis=1) ** 2
    probs = np.empty_like(minimal_distances)
    for i in range(1, k):
        # Calculate probabilities
        np.divide(minimal_distances, minimal_distances.sum(), out=probs)
        chosen_index = np.random.choice(N, p=probs)
        initial_indices[i] = chosen_index
        # Calculate new minimal distances by calculating the distance from the
        # newly chosen centroid, and getting the minimum from the previous
        # distances and this calculated distance
        new_distances = np.linalg.norm(obs_arr - obs_arr[chosen_index], axis=1) ** 2
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
