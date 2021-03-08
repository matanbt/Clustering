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
from time import time

def main():
    # PRINTS INFORMATIVE MESSAGE:
    print_message()
    # PROCESS PARAMETERS:
    print("_ input")
    t0 = time()
    args = user_input.get_args()
    if not user_input.check_user_input(args):
        return None
    t1 = time()
    print(f"1 args [input] ti_{(t1 - t0) :.10f}")
    t0 = time()
    params, points, centers = user_input.generate_points(args)
    t1 = time()
    print(f"2 g_points [input] ti_{(t1 - t0) :.10f}")
    # NSC:
    print("_ nsc")
    spectral_clusters, spectral_k = nsc(points,
                                        None if params.random else params.k)
    spectral_clusters = np.array(spectral_clusters, dtype='float32')
    # KMEANS:
    print("_ kmeans")
    kmeans_clusters = kmeans(points, spectral_k, params.n, params.dim, MAX_ITER)
    kmeans_clusters = np.array(kmeans_clusters, dtype='float32')
    # OUTPUT:
    print("_ output")
    t0 = time()
    print_data_txt(points, centers)
    print_clusters_txt(params.k, spectral_clusters, kmeans_clusters)
    t1 = time()
    print(f"1 txts [output] ti_{(t1 - t0) :.10f}")
    t0 = time()
    spectral_jaccard = calc_jaccard(centers, spectral_clusters)
    kmeans_jaccard = calc_jaccard(centers, kmeans_clusters)
    t1 = time()
    print(f"2 jac [output] ti_{(t1 - t0) :.10f}")
    t0 = time()
    visualization_pdf(params.k, points, kmeans_clusters, spectral_clusters,
                      spectral_k, spectral_jaccard, kmeans_jaccard)
    t1 = time()
    print(f"3 pdf [output] ti_{(t1 - t0) :.10f}")



if __name__ == '__main__':
    main()
