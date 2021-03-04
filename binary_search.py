from invoke import run
from timeit import timeit
from math import ceil
import numpy as np


def avg(n1, n2):
    return ceil((n1 + n2) / 2)

REPEAT = 2


def time_program(n, k):
    avg_time = timeit(lambda: run(f"python3.8.5 main.py {k} {n}"), number=REPEAT)
    return (avg_time / REPEAT) / 60  # averaged and to minutes


def search():
    upper_n, upper_k = 900, 500
    lower_n, lower_k = 700, 300
    while upper_n > lower_n:
        mid_n, mid_k = avg(upper_n, lower_n), avg(upper_k, lower_k)
        avg_time = time_program(mid_n, mid_k)
        print(f"---> for k={mid_k}, n={mid_n} : {avg_time} mins")
        if 4.9 < avg_time < 5.1:
            print("--- SUCCESS ---")
            break
        if avg_time > 5:
            upper_n, upper_k = mid_n, mid_k
        else:
            lower_n, lower_k = mid_n, mid_k

def heatmap(a):
    import seaborn as sns
    import matplotlib.pyplot as plt
    a = np.genfromtxt('times_100.csv', delimiter=',')
    ax = sns.heatmap(a, linewidth=0.2)
    ax.invert_yaxis()
    ax.set(xlabel="K", ylabel="N",
           xticklabels=10 * np.arange(a.shape[1]),
           yticklabels=10 * np.arange(a.shape[0]))
    plt.savefig("times.pdf")
    plt.show()


def times():
    upper_n, upper_k = 300, 300
    upper_n, upper_k = ceil(upper_n / 10), ceil(upper_k / 10)
    mat = np.zeros((upper_n,upper_k))
    for n in range(1, upper_n):
        for k in range(1, upper_k):
            if k < n:
                print(f"n={n}, k={k}")
                mat[n, k] = time_program(n * 10, k * 10)
            else:
                mat[n, k] = 0
    np.savetxt("times.csv", mat, delimiter=",", fmt="%f")


def times_find_max():
    n_arr = np.arange(10) * 10 + 700
    k_arr = np.arange(10) * 40 + 100
    res = np.zeros((len(n_arr),len(k_arr)))
    for i, n in enumerate(n_arr):
        for j, k in enumerate(k_arr):
            if k < n:
                res[i,j] = time_program(n, k)
                print(f"n={n}, k={k}, took: {res[i,j]} mins")
    res = np.vstack((n_arr, k_arr, res))
    np.savetxt("find_max.csv", res, delimiter=",", fmt="%f")
    # row for each N, column for each K
    # first two rows are K,N values

if __name__ == "__main__":
    # times_find_max()
    # times()
    search()