from timeit import timeit
import time
from sklearn.datasets import make_blobs
import output_data as od
import numpy as np

class Test_3d_output:
    def test_vis(self):
        n, k, d = 100, 5, 3
        k_spectral = k + 1
        points, centers = make_blobs(n, d, centers=k)
        od.visualization_pdf(k, points, centers, centers, k_spectral, 0.24, 0.27)

    def test_calc_jaccard(self):
        # Iftah's forum example
        centers = np.array([0,0,1,1,2,2])
        clusters = np.array([0,1,0,1,2,2])
        jac = od.calc_jaccard(centers,clusters)
        assert jac == 0.2


class Test_txt_ouput:
    def test_sanity_check_data(self):
        obs_arr = np.array(
            [(1.234, 2, 3),
             (0, 2.234, 7),
             (100, 11100.11111, 99),
             (1.234, 1, 0),
             (8, 7, 6.234),
             (7, 8.234, 9)], dtype=float)
        centers = [1, 1, 2, 2, 0, 0]
        od.print_data_txt(obs_arr, centers)

    def test_times_data(self):
        obs_arr = np.random.rand(10000, 3)
        centers = np.random.randint(0, 20, 10000)
        t_py = timeit(lambda : od.print_data_txt(obs_arr, centers), number=50)
        print(f"PYTHON: {t_py}")

    def test_sanity_check_clusters(self):
        clusters = np.array([0, 1, 0, 1, 2, 2])
        od.print_clusters_txt(999, clusters, clusters)
        expected_output = [
            '999', '0,2', '1,3', '4,5', '0,2', '1,3', '4,5'
        ] # could be in different order
        with open("clusters.txt") as f:
            for i, line in enumerate(f):
                assert line.strip() == expected_output[i]




# ----------- TRASH (ALL TESTED TO BE SLOWER):
# def add_each_clusters_without_hash(labeled_obs):
#     # helper method for formatting the clusters' string
#     _s = ""
#     sorted_indices = np.argsort(labeled_obs)
#     curr_cluster = -1
#     for i in sorted_indices:
#         if labeled_obs[i] == curr_cluster:
#             # still printing the same cluster --> in the same line
#             _s += ","
#         else:
#             # break line, and set new cluster
#             _s += "\n"
#             curr_cluster = labeled_obs[i]
#         _s += str(i)
#     return _s
#
#
# def print_data_txt(obs_arr, centers):
#     s = ""
#     n = obs_arr.shape[0]
#     obs_lst_str = obs_arr.astype(str)
#     for i in range(n):
#         s += ",".join(obs_lst_str[i])
#         s += "," + str(centers[i])
#         s += "\n"
#
#     with open("data.txt", "w") as f:
#         f.write(s)
