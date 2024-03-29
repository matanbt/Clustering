"""
----- Data Initialization Module -----
Deals with accepting, processing and validating the user's input.
Also deals  with generating random points intended to be clustered
"""
import argparse
from dataclasses import dataclass
import numpy as np
from sklearn.datasets import make_blobs

import config


@dataclass
class ProgramParams:
    """
    A data class to hold all of the different parameters of the program.
    """
    n: int
    k: int
    dim: int
    random: bool


def get_args():
    """
    Handles getting the input from the user, and making sure that all of the
    necessary parameters have been passed, and of the correct type.
    :return: The arguments of the program.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('k', help='The number of clusters', type=int)
    parser.add_argument('n', help='The number of observations to generate',
                        type=int)
    parser.add_argument('--random',
                        help='Use random samples and clusters size or not',
                        default=False, action='store_true')

    return parser.parse_args()


def check_user_input(args):
    """
    Check that the user's input is valid.
    :param args: The arguments of the program.
    :return: True if the user's input is valid, False otherwise.
    """
    if args.random:
        return True

    if args.n <= 0 or args.k <= 0:
        print("Please pass positive n, k values")
        return False

    if args.n <= args.k:
        print("The parameter n must be bigger than k")
        return False

    return True


def _generate_data_properties(dimensions):
    """
    Helper function to calculate the sample size and centers count, given the
    dimensions of the data, and assuming these parameters should be random.
    :param dimensions: The dimensions of each point in the data - 2D or 3D.
    :return: The generated sample size and centers count.
    """
    if dimensions == 2:
        n = np.random.randint(config.MAX_N_2D_CAPACITY // 2,
                              config.MAX_N_2D_CAPACITY + 1)
        k = np.random.randint(config.MAX_K_2D_CAPACITY // 2,
                              min(config.MAX_K_2D_CAPACITY + 1, n))
    else: # dimensions == 3
        n = np.random.randint(config.MAX_N_3D_CAPACITY // 2,
                              config.MAX_N_3D_CAPACITY + 1)
        k = np.random.randint(config.MAX_K_3D_CAPACITY // 2,
                              min(config.MAX_K_3D_CAPACITY + 1, n))

    return n, k


def generate_points(args, dimensions=None, random_state=None):
    """
    Generate the points that the program will use.
    :param args: The arguments of the program.
    :param dimensions: sets the dimensions, if None choose randomly from {2,3}
    :param random_state: random-seed to deliver make_blobs API. None means 'pure' random.
    :return: A dataclass that will hold all of the parameters of this program's
    run - (N, K, Dimensions, Random), the points, and the clusters they belong
    to.
    """
    if dimensions is None:
        dimensions = np.random.choice([2, 3])
    if not args.random:
        n, k = args.n, args.k
    else:
        n, k = _generate_data_properties(dimensions)

    params = ProgramParams(n, k, dimensions, args.random)
    points, centers = make_blobs(params.n, params.dim, centers=params.k,
                                 random_state=random_state)
    return params, points, centers
