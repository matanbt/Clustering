from invoke import run
from timeit import timeit
from math import ceil
import numpy as np
import sys
import matplotlib.pyplot as plt
from time import time

import spectral_clustering as nsc

REPEAT = 5
REF_K ,REF_N = 350, 200

# ------------- HELPERS --------------:
def avg(n1, n2):
    return ceil((n1 + n2) / 2)


def time_program(n, k):
    avg_time = timeit(lambda: run(f"python3.8.5 main.py {k} {n}"), number=REPEAT)
    return (avg_time / REPEAT) / 60  # averaged and to minutes


def time_reference():
    # measure the chosen n,k to be refrences
    n, k, repeat = REF_N, REF_K, REPEAT
    avg_time = timeit(lambda: run(f"python3.8.5 main.py {k} {n}",hide='stdout'), number=repeat)
    return (avg_time / repeat) / 60  # averaged and to minutes


# ------------------ TIMING ------------
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


def times(n_arr=np.arange(10) * 10 + 400,
          k_arr=np.arange(50) * 10 + 0):
    res = np.zeros((len(n_arr),len(k_arr)))
    for i, n in enumerate(n_arr):
        for j, k in enumerate(k_arr):
            if k < n:
                res[i, j] = time_program(n , k)
                print(f"n={n}, k={k}, took: {res[i,j]} mins")
    # res = np.vstack((n_arr, k_arr, res))
    np.savetxt("times_400_500.csv", res, delimiter=",", fmt="%f")
    # row for each N, column for each K
    # first two rows are K,N values

# ---- BOTTLE NECK LOOK FOR ----
def time_phases(n, k):
    t0 = time()
    res = run(f"python3.8.5 main.py {k} {n}", hide='out')
    t1 = time()
    total_time = t1 - t0
    titles_lst = []
    times_lst = []
    for line in res.stdout.split("\n"):
        try:
            t = float(line.split("ti_")[-1])
            title = line.split(" ")[1]
        except:
            continue
        titles_lst += [title]
        times_lst += [t]
    return titles_lst, np.array(times_lst), total_time


def bottle_neck(n=REF_N, k=REF_K, repeat=REPEAT):
    titles_lst, avg_times, total_time_sum = time_phases(n,k)
    for _ in range(repeat-1):
        _, t, total_time = time_phases(n,k)
        total_time_sum += total_time
        avg_times += t
        # print(f"total={total_time}, sum_phases={np.sum(t)}")
    print(f"AVERAGE RUNTIME: N={n}, K={k}, took: {total_time_sum / repeat / 60} mins")
    avg_times = avg_times / repeat
    fig, ax = plt.subplots()
    titles_lst = [f"{i}{title}" for i, title in enumerate(titles_lst)]
    ax.set(xlabel="Phase", ylabel="Seconds Took")
    ax.bar(titles_lst, avg_times)
    for p in ax.patches:
        ax.annotate("%.5f" % p.get_height(),
                    (p.get_x() + p.get_width() / 2., p.get_height()), ha='center',
                    va='center', xytext=(0, 5), textcoords='offset points',
                    fontsize=4)
    plt.xticks(fontsize=4)
    fig.savefig("bottle_neck_plot.pdf")
    print("Saved: bottle_neck_plot.pdf")

# TIME MEASURES ON TXT FILES:
def time_weight_by_txt(txt_name):
    x = np.genfromtxt(txt_name, delimiter=',')
    avg_time = timeit(lambda: nsc.form_weight(x), number=REPEAT) / REPEAT

# ---- FILES PROCESSING / VISUALIZATION HELPERS -----
def heatmap():
    import seaborn as sns
    import matplotlib.pyplot as plt
    a = np.genfromtxt('times_0_500.csv', delimiter=',')
    a[a == 0] = None
    ax = sns.heatmap(a, linewidth=0.2)
    ax.invert_yaxis()
    ax.set(xlabel="K/10", ylabel="N/10")
    # ax.set_xticks(10 * np.arange(a.shape[1]))
    # ax.set(xticklabels=10 * np.arange(a.shape[1]),
    #        yticklabels=10 * np.arange(a.shape[0]))
    plt.savefig("times.pdf")
    plt.show()


def parse_txt_to_csv():
    res = np.zeros((10, 50))
    from parse import parse
    with open("400_500.txt") as f:
        for line in f:
            try:
                n, k, time_took = parse("n={}, k={}, took: {} mins", line)
            except:
                continue
            n, k, time_took = int(n), int(k), float(time_took)
            res[(n//10)-40, (k//10)] = time_took
    np.savetxt("times_400_500.csv", res, delimiter=",", fmt="%f")


# ----------- MAIN ---------
if __name__ == "__main__":
    if len(sys.argv) == 2 and callable(globals().get(sys.argv[1])):
        func = globals().get(sys.argv[1])
        func()
        print(f"'{func.__name__}' ran successfully")
    else:
        print('Error')