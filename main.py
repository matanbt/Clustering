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


def main():
    # PRINTS INFORMATIVE MESSAGE:
    print_message()
    # PROCESS PARAMETERS:
    args = user_input.get_args()
    if not user_input.check_user_input(args):
        return None
    params, points, centers = user_input.generate_points(args)
    points = points.astype(np.float32, copy=False)
    # NSC:
    spectral_clusters, spectral_k = nsc(points,
                                        None if params.random else params.k)
    spectral_clusters = np.array(spectral_clusters, dtype=np.float32)
    # KMEANS:
    kmeans_clusters = kmeans(points, spectral_k, params.n, params.dim, MAX_ITER)
    kmeans_clusters = np.array(kmeans_clusters, dtype=np.float32)
    # OUTPUT:
    print_data_txt(points, centers)
    print_clusters_txt(spectral_k, spectral_clusters, kmeans_clusters)
    spectral_jaccard = calc_jaccard(centers, spectral_clusters)
    kmeans_jaccard = calc_jaccard(centers, kmeans_clusters)
    visualization_pdf(params.k, points, kmeans_clusters, spectral_clusters,
                      spectral_k, spectral_jaccard, kmeans_jaccard)


if __name__ == '__main__':
    main()
