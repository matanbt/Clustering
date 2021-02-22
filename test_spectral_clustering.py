import numpy as np
from timeit import timeit

import spectral_clustering as nsc


def test__form_Weight(s):
    x = np.random.rand(323, 5)
    t1 = timeit(lambda: nsc.form_Weight(x), number=100)
    t2 = timeit(lambda: nsc.form_Weight_np2(x), number=100)
    t3 = timeit(lambda: nsc.form_Weight_np2_(x), number=100)
    w_1, w_2, w_3 = nsc.form_Weight(x), nsc.form_Weight_np2(x), nsc.form_Weight_np2_(x)
    assert w_1.dtype == float and w_2.dtype == float and w_3.dtype == float
    assert np.allclose(w_1, w_2, atol=0.001)
    assert np.allclose(w_2, w_3, atol=0.001)
    return t1, t2, t3


def time__form_Weight():
    k = 7
    times = np.empty((k, 3))
    for i in range(k):
        times[i] = check_Weight()
    print(times)
    print(f"- Regular  : {np.mean(times[:, 0])} sec")
    print(f"- With +w.T: {np.mean(times[:, 1])} sec")
    print(f"- With ind': {np.mean(times[:, 2])} sec")

def sanity_check__run():
    x = np.array(
        [(1,2,3),
         (0,2,7),
         (100,100,99),
         (1,1,0),
         (8,7,6),
         (7,8,9)], dtype=float)
    return nsc.run_nsc(x)


if __name__ == '__main__':
    res = sanity_check__run()
    print(res)

#----------------------------------
# deprecated code:
# def form_Weight_np(x):
#     """
#     numpy alternative to 'form_Weight', tested to run *slower*
#     """
#     n = x.shape[0]
#
#     # form Weight Matrix :
#     u_weight_func = np.frompyfunc(lambda i,j: 0 if i>=j else weight_func(x[i], x[j]), 2, 1)
#     indices = np.arange(n, dtype=int)
#     w = u_weight_func.outer(indices,indices).astype(float)
#     w = w + w.T  # adds the symmetric upper triangle
#
#     return w