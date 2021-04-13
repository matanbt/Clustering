"""
----- Tasks -----
module for invoke tasks
"""

from invoke import task

# ============ Official Tasks (UI): ============
@task(help={'k': "Amount of centers for the generated data",
            'n': "Amount of points", 'Random': "for randomized points. Default - True"})
def run(c, k=0, n=0, Random=True):
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
