"""
K-MEANS-PP ALGORITHM IMPLEMENTATION
"""

import argparse
import pandas as pd
import numpy as np
from time import time
import mykmeanssp as km


def k_means_pp(k, obs_arr):
    """
    TODO
    """
    np.random.seed(0)

    N = len(obs_arr)
    initial_indices = np.empty(k, dtype=int)

    last_index = np.random.choice(N)
    initial_indices[0] = last_index
    minimal_distances = np.linalg.norm(obs_arr - obs_arr[last_index], axis=1)
    for i in range(1, k):
        probs = minimal_distances / minimal_distances.sum()
        last_index = np.random.choice(N, p=probs)
        initial_indices[i] = last_index
        new_distances = np.linalg.norm(obs_arr - obs_arr[last_index], axis=1)
        minimal_distances = np.minimum(minimal_distances, new_distances)

    return initial_indices


def convert_obs_to_c(obs_arr):
    """
    Convert Numpy's array to a list of tuples, for usage by the C API.
    :param obs_arr: Observations.
    :return: Converted observations.
    """
    return [tuple(obs) for obs in obs_arr]


def kmeans(points, K, N, d, MAX_ITER):
    """
    TODO
    """
    t0 = time()
    indices = k_means_pp(K, points)
    t1 = time()
    print(f"1 pp [kmeans] ti_{(t1 - t0) :.10f}")
    t0 = time()
    c_points = convert_obs_to_c(points)
    indices = indices.tolist()
    t1 = time()
    print(f"2 prep_c [kmeans] ti_{(t1 - t0) :.10f}")
    t0 = time()
    clusters = km.kmeans(c_points, indices, K, N, d, MAX_ITER)
    t1 = time()
    print(f"3 c [kmeans] ti_{(t1 - t0) :.10f}")
    return clusters
