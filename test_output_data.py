from timeit import timeit
import time
from sklearn.datasets import make_blobs
import output_data as od
import numpy as np

class Test_output:
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
