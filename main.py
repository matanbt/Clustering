import sys
import argparse
import numpy as np
from sklearn.datasets import make_blobs


# TODO: Move this to config.py when it is merged.
MAX_N_CAPACITY = 200
MAX_K_CAPACITY = 10


def handle_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('k', help='The number of clusters', type=int)
    parser.add_argument('n', help='The number of observations to generate',
                        type=int)
    parser.add_argument('--random',
                        help='Use random samples and clusters size or not',
                        default=False, action='store_true')

    args = parser.parse_args()

    if (args.n <= 0 or args.k <= 0) and not args.random:
        print("Please pass positive n, k values")
        sys.exit(-1)

    if args.n < args.k and not args.random:
        print("The parameter n must be bigger than k")
        sys.exit(-1)

    if not args.random:
        return args.n, args.k

    n = np.random.randint(MAX_N_CAPACITY // 2, MAX_N_CAPACITY + 1)
    k = np.random.randint(MAX_K_CAPACITY // 2, MAX_K_CAPACITY + 1)
    return n, k


def generate_points(sample_size, centers_count):
    is_2d = np.random.choice([True, False])
    dimensions = 2 if is_2d else 3

    return make_blobs(sample_size, dimensions, centers=centers_count)


if __name__ == '__main__':
    n, k = handle_args()
    points, clusters = generate_points(n, k)
