"""
----- Tasks -----
module for invoke tasks
"""

from invoke import task
import numpy as np
from timeit import timeit


# ============ Official Tasks (UI): ============
@task(help={'k': "Amount of centers for the generated data",
            'n': "Amount of points",
            'Random': "for randomized points. Default - True"})
def run(c, k, n, Random=True):
    """
    Setup the program and runs main.py with the given parameters
    """
    c.run("python3.8.5 setup.py build_ext --inplace")
    Random = "--random" if Random else ""
    c.run(f"python3.8.5 main.py {Random} {k} {n}")


# ============ Utilities: ============
@task
def build(c):
    c.run("python3.8.5 setup.py build_ext --inplace")


@task(aliases=["del"])
def delete(c):
    c.run("rm *mykmeanssp*.so")


@task
def clean(c):
    c.run("rm data.txt clusters.txt clusters.pdf")


# ============ Tasks for developer purposes: ============
from time import time

REPEAT = 3
RANDOM_STATE = 0  # SEED TO SET

@task
def time_with_seed(c, _n, _k, _d):
    """
    times non-random case, with given n, k, d
    """
    from user_input import generate_points
    from main import run_clustering

    class args:
        n = _n
        k = _k
        random = False

    t0 = time()
    params, points, centers = generate_points(args, dimensions=_d,
                                              random_state=RANDOM_STATE)
    run_clustering(params, points, centers)
    t1 = time()
    print(f"n={_n}, k={_k}, d={_d}, took: {t1 - t0} secs")
    return t1 - t0

def time_comparisons_with_seed(c):
    """
    runs time_with_seed for bunch of tests
    """
    comparisons_list = [
        # n,   k,  d
        (470, 200, 2),
        (470, 200, 3),
        (312, 215, 2),
        (350, 159, 3),
        (400, 159, 3),
    ]
    for n, k, d in comparisons_list:
        time_with_seed(c, n, k, d)

@task
def time_tests(c):
    """
    runs the PURE project for bunch of tests
    """
    tests_list = [
        # n,   k
        (478, 50),
        (478, 100),
        (478, 150),
        (478, 200),
        (478, 250),
        (478, 300),
        (478, 350),
        (478, 400),
        (478, 450),
        (478, 450),
        (480, 50),
        (480, 100),
        (480, 150),
        (480, 200),
        (480, 250),
        (480, 300),
        (480, 350),
        (480, 400),
        (480, 450),
    ]
    print(f" ---- Timing with fixed n,k , averages on {REPEAT} repeats----")
    for n, k in tests_list:
        sum_t = 0
        for _ in range(REPEAT):
            t0 = time()
            res = c.run(f"python3.8.5 -m invoke run {k} {n} --no-Random", hide="stdout")
            t1 = time()
            is_bad_run = res.stderr or "invalid" in res.stdout.lower() or \
                         "error" in res.stdout.lower()
            if is_bad_run:
                print(f"BAD RUN WITH N={n} K={k}")
                print(res.stdout)
            else:
                sum_t += t1 - t0
        print(f"N={n} K={k} took {sum_t / REPEAT / 60} mins")



#############

@task
def time_func(c, func_name, input_file):
    import spectral_clustering as nsc
    if func_name == 'weight':
        func = nsc.form_weight
    else:
        return
    x = np.genfromtxt(input_file, delimiter=',')
    avg_time = timeit(lambda: func(x), number=REPEAT) / REPEAT
    print(f"-- {func_name} ran for {avg_time} secs")
    print("-- res:")
    print(func(x))


@task
def bottle_neck(c, n, k, repeat=3):
    from tester import bottle_neck
    print("Bottle Neck Check...")
    bottle_neck(n, k, repeat)
    print("... Finished Bottle Neck Check!")
