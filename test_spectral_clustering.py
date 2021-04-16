import numpy as np
from timeit import timeit

import spectral_clustering as nsc
import linalg as la
EPSILON = 0.001

# ------------------ LA TESTS: -----------------
def test__qr():
    # tests linalg.qr_iteration for correctness
    a1 = np.array([
        [3, 4, 2],
        [1, 6, 2],
        [1, 4, 4]
    ], dtype=float)
    expected_a1_ = np.array([
        [8.8903, -3.4875, -1.8113],
        [0.2413, 2.1221, -0.0634],
        [0.0473, 0.0239, 1.9876]
    ], dtype=float)
    expected_q1_ = np.array([
        [-0.6061, -0.7806, 0.1529],
        [-0.5624, -0.5565, 0.6115],
        [-0.5624, -0.2847, -0.7763]
    ], dtype=float)
    a1_, q1_ = la.qr_iteration(a1)
    assert np.allclose(abs(a1_), abs(expected_a1_), atol=EPSILON)
    assert np.allclose(abs(q1_), abs(expected_q1_), atol=EPSILON)

    a2 = np.array([
        [2, 1, 3, 4],
        [0, 2, 1, 3],
        [2, 1, 6, 5],
        [1, 2, 4, 8]

    ], dtype=float)
    expected_a2_ = np.array([
        [13.0901, 1.3131, 3.1712, 0.615],
        [0.0672, 2.9191, 0.9679, 0.2144],
        [0.0011, -0.0252, 0.9877, -0.0027],
        [0.0005, 0.0286, 0.0143, 1.0032]
    ], dtype=float)
    expected_q2_ = np.array([
        [0.4156, -0.0775, -0.8795, 0.2187],
        [0.2244, 0.5601, -0.1386, -0.7853],
        [0.6063, -0.6666, 0.2587, -0.3479],
        [0.6398, 0.4857, 0.3747, 0.4631]
    ], dtype=float)
    a2_, q2_ = la.qr_iteration(a2)
    assert np.allclose(abs(q2_), abs(expected_q2_), atol=EPSILON)
    assert np.allclose(abs(a2_), abs(expected_a2_), atol=EPSILON)
    print("TESTED QR ITER SUCCESSFULLY")

def test__eigengap():
    a_ = np.zeros((10,10))
    eigen_vals = [3,4,5,10,11,12,14,16,30,45]
    np.random.shuffle(eigen_vals)
    np.fill_diagonal(a_, eigen_vals)
    expected_clusters = [eigen_vals.index(3),eigen_vals.index(4),eigen_vals.index(5)]
    clusters = la.eigengap_method(a_)
    assert np.all(expected_clusters == clusters)
    print("TESTED EIGENGAP SUCCESSFULLY")


# ------------------ NSC TESTS: -----------------
def test__form_Weight():
    x = np.random.rand(323, 5)
    w_1, w_2 = nsc.form_weight(x), nsc.form_weight_np(x)
    assert w_1.dtype == float and w_2.dtype == float
    assert np.allclose(w_1, w_2, atol=0.001)

def time__form_Weight():
    x = np.random.rand(323, 5)
    t1 = timeit(lambda: nsc.form_weight(x), number=100)
    t2 = timeit(lambda: nsc.form_weight_np(x), number=100)
    return t1, t2

def time_avg__form_Weight():
    k = 7
    times = np.empty((k, 2))
    for i in range(k):
        times[i] = time__form_Weight()
    print(f"- Regular  : {np.mean(times[:, 0])} sec")
    print(f"- With +w.T: {np.mean(times[:, 1])} sec")

def test__phases():
    x = np.array([
        [3, 4, 2],
        [1, 6, 2],
        [21, 0, -2]
    ], dtype=float)
    expected_w = np.array([
        [0, 0.24311, 7.996*(10**-5)],
        [0.24311, 0, 2.4176*(10**-5)],
        [7.996*(10**-5), 2.4176*(10**-5), 0]
    ], dtype=float)
    w = nsc.form_weight(x)
    assert np.allclose(w, expected_w, atol=EPSILON)
    expected_l = np.array([
        [1, -0.99936, -0.01588],
        [-0.99936, 1,-0.0048],
        [-0.01588, -0.0048, 1]
    ], dtype=float)
    l = nsc.form_laplacian(expected_w)
    assert np.allclose(l, expected_l, atol=EPSILON)
    print("TESTED PHASES 1&2 SUCCESSFULLY")

def sanity_check__run():
    x = np.array(
        [(1, 2, 3),
         (0, 2, 7),
         (100, 100, 99),
         (1, 1, 0),
         (8, 7, 6),
         (7, 8, 9)], dtype=float)
    clusters, k = nsc.run_nsc(x,3)
    assert clusters[0] == clusters[1] == clusters[3] and clusters[4] == clusters[5] \
           and len({clusters[0],clusters[4],clusters[2]}) == 3
    print("TESTED RUN FOR SANITY-CHECK SUCCESSFULLY")

def test__res_phase():
    # TODO
    # res = KMeans(n_clusters=k, random_state=0).fit(t)
    # return res.labels_
    pass
def test_laplacian_comparison():
    w = np.random.rand(50,50)
    r1 = nsc.form_laplacian(w)
    r2 = nsc.form_laplacian_imp(w)
    assert np.allclose(r1, r2, atol=0.001)
    print('(1) - ', timeit(lambda: nsc.form_laplacian(w), number=100) * 10000)
    print('(2) - ', timeit(lambda : nsc.form_laplacian_imp(w), number=100)*10000)

# ------------------ TIME COMPARISONS: -----------------
def time__run(n,d):
    np.random.seed(0)
    x = np.random.rand(n, d)
    return timeit(lambda: nsc.run_nsc(x), number=3)

def time_avg__run():
    tests = [
        (20,2),
        (31,3),
        (1729,2),
        (102,5),
        (920,3),
        (53,7),
        (561,3)
    ]
    _sum = 0
    for test in tests:
        t = time__run(*test)
        print(f"-n = {test[0]}, d = {test[1]}: {t}")
        _sum += t
    print(f" -- Average  : {_sum / len(tests)} sec --")

if __name__ == '__main__':
    test__qr()
    test__eigengap()
    test__phases()
    sanity_check__run()
