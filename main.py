"""
----- Main Module -----
Glues all modules together to provide the desired finished products
"""
import numpy as np
import user_input
from spectral_clustering import run_nsc as nsc
from kmeans_pp import kmeans
from output_data import print_data_txt, print_clusters_txt, \
                        visualization_pdf, calc_jaccard, print_message
from config import MAX_ITER


def run_clustering(params, points, centers):
    """
    Runs the clustering algorithms on a given data {params, points, centers}.
    Can be used as an imported module.
    :return: saves data to `data.txt`, results to `clusters.txt`
             and visualization to `clusters.pdf`
    """
    points = points.astype(np.float32, copy=False)
    # NSC:
    spectral_clusters, spectral_k = nsc(points,
                                        None if params.random else params.k)
    spectral_clusters = spectral_clusters.astype(np.float32, copy=False)
    # KMEANS:
    kmeans_clusters = kmeans(points, spectral_k, params.n, params.dim, MAX_ITER)
    kmeans_clusters = kmeans_clusters.astype(np.float32, copy=False)
    # OUTPUT:
    print_data_txt(points, centers)
    print_clusters_txt(params.k, spectral_clusters, kmeans_clusters)
    spectral_jaccard = calc_jaccard(centers, spectral_clusters)
    kmeans_jaccard = calc_jaccard(centers, kmeans_clusters)
    visualization_pdf(params.k, points, kmeans_clusters, spectral_clusters,
                      spectral_k, spectral_jaccard, kmeans_jaccard)


def main():
    """
    main function, intended when called as a script
    """
    # PRINTS INFORMATIVE MESSAGE:
    print_message()
    # PROCESS PARAMETERS:
    args = user_input.get_args()
    if not user_input.check_user_input(args):
        return None
    params, points, centers = user_input.generate_points(args)
    # RUNS CLUSTERING, SAVES RESULTS FILES
    run_clustering(params, points, centers)


if __name__ == '__main__':
    main()
