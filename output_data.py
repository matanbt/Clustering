"""
----- Output Data & Results -----
module for post-processing and outputting the results of the run
"""

import matplotlib.pyplot as plt
import numpy as np
import config

def print_message():
    """
    Prints informative message about the maximum capacity that has been calculated
    """
    print(f" - Maximum Capacity, 2 Dimensions: n = {config.MAX_N_2D_CAPACITY}, "
          f"k = {config.MAX_K_2D_CAPACITY}")
    print(f" - Maximum Capacity, 3 Dimensions: n = {config.MAX_N_3D_CAPACITY}, "
          f"k = {config.MAX_K_3D_CAPACITY}")


def print_data_txt(points, centers):
    """
        :param points: a numpy-array of points
        :param centers: The REAL center corresponding to each point
        :return: prints formatted data to 'data.txt'
    """
    n, d = points.shape
    format_arr = ["%f"] * d + ["%d"]
    output_array = np.empty((n, d + 1))
    output_array[:, :-1] = points
    output_array[:, -1] = centers
    np.savetxt(config.FNAME_DATA_TXT, output_array,
               fmt=format_arr, delimiter=',')


def print_clusters_txt(k, spectral_clusters, kmeans_spectral):
    """
    :param k: The k given by the user / generated randomly
    :param spectral_clusters: gets np-array, in which each index represent an obs
                            and each element is its cluster
    :param kmeans_spectral: same as above
    :return: prints formatted data to 'clusters.txt'
    """

    s = f"{k}\n"

    def format_clusters(clusters):
        # helper method for formatting the clusters' string
        _s = ""
        clusters_dict = {}
        for i, cluster in enumerate(clusters):
            if cluster not in clusters_dict:
                clusters_dict[cluster] = f"{i},"
            else:
                clusters_dict[cluster] += f"{i},"
        for line in clusters_dict.values():
            _s += line[:-1] + "\n"
        return _s

    s += format_clusters(spectral_clusters)
    s += format_clusters(kmeans_spectral)
    with open(config.FNAME_CLUSTERS_TXT, "w") as f:
        f.write(s)


def calc_jaccard(centers, clusters):
    """
    Calculates the 'Jaccard' distance between original-centers to the
    computed-clusters
    :param centers: array of the *real* centers for each point
    :param clusters: array of the *computed* clusters for each point
    :return: the calculated 'Jaccard' distance (a float in [0,1])
    """
    n = centers.shape[0]
    tri0, tri1 = np.triu_indices(n, 1)
    # for each pair (i,j) s.t. i < j, marks True IFF i and j in the same cluster
    centers_pairs = centers[tri0] == centers[tri1]
    clusters_pairs = clusters[tri0] == clusters[tri1]
    union_count = np.sum(centers_pairs | clusters_pairs)
    intersect_count = np.sum(centers_pairs & clusters_pairs)
    return intersect_count / union_count


def visualization_pdf(k, points, kmeans_clusters, spectral_clusters, spcetral_k,
                      jaccard_spectral, jaccard_kmeans):
    """
    :param k: The k given by the user / generated randomly
    :param points: the points generated
    :param kmeans_clusters: clusters computed by KMeans
    :param spectral_clusters: clusters computed by Spectral Cl.
    :param spcetral_k: the k given by the user / (random case) calculated
    :param jaccard_kmeans: 'Jaccard' distance calculated relative to KMeans run
    :param jaccard_spectral: same as above, for Spectral Cl.
    :return: 'prints' visualization and results summary to 'clusters.pdf'
    """
    n, d = points.shape
    # titles and text
    title_header = ' - Run Results - '
    title_spectral = 'Normalized Spectral Clustering'
    title_kmeans = 'K-means'
    text_footer = "Data was generated from values:\n"
    text_footer += f"n = {n}, k = {k}\n"
    text_footer += f"The k that was used for both algorithms: {spcetral_k}\n "
    text_footer += f"The Jaccard measure for Spectral Clustering: " \
                   f"{jaccard_spectral : .5f}\n"
    text_footer += f"The Jaccard measure for K-means: {jaccard_kmeans : .5f}"
    # figure's prologue
    fig = plt.figure(figsize=(7.5, 6))
    fig.suptitle(title_header, size=30, fontweight='bold')
    # subplots
    if d == 2:
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.set(title=title_spectral, xlabel='X', ylabel='Y')
        ax1.scatter(points[:, 0], points[:, 1], c=spectral_clusters)
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.set(title=title_kmeans, xlabel='X', ylabel='Y')
        ax2.scatter(points[:, 0], points[:, 1], c=kmeans_clusters)
    else:  # d==3
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        ax1.set(title=title_spectral, xlabel='X', ylabel='Y', zlabel='Z')
        ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c=spectral_clusters)
        ax2 = fig.add_subplot(2, 2, 2, projection='3d')
        ax2.set(title=title_kmeans, xlabel='X', ylabel='Y', zlabel='Z')
        ax2.scatter(points[:, 0], points[:, 1], points[:, 2], c=kmeans_clusters)
    # footer text
    fig.text(0.5, 0.15, text_footer, ha='center', size=17)
    # figure's epilogue
    fig.savefig(config.FNAME_VIS_PDF)
