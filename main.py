"""
----- Main Module -----
Glues all modules together to provide the desired finished products
"""
import user_input
from spectral_clustering import run_nsc as nsc
from kmeans_pp import k_means_pp as kmeans  # TODO MAKE SURE IT'S RIGHT
from output_data import print_data_txt, print_clusters_txt, \
                        visualization_pdf, calc_jaccard
from config import MAX_ITER


def main():
    # PROCESS PARAMETERS:
    args = user_input.get_args()
    if not user_input.check_user_input(args):
        return None
    params, points, centers = user_input.generate_points(args)
    # NSC:
    spectral_k, spectral_clusters = nsc(points,
                                        None if params.random else params.k)
    # KMEANS:
    kmeans_clusters = kmeans(points, params.k, MAX_ITER)  # TODO API
    # OUTPUT:
    print_data_txt(points, centers)
    print_clusters_txt(params.k, spectral_clusters, kmeans_clusters)
    spectral_jaccard = calc_jaccard(centers, spectral_clusters)
    kmeans_jaccard = calc_jaccard(centers, kmeans_clusters)
    visualization_pdf(params.k, points, kmeans_clusters, spectral_clusters,
                      spectral_k, spectral_jaccard, kmeans_jaccard)


if __name__ == '__main__':
    main()
