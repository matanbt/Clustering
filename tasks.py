"""
----- Tasks -----
module for invoke tasks
"""

from invoke import task

# ============ Official Tasks: ============
@task(help={'k': "Amount of centers for the generated data",
            'n': "Amount of points for the generated data",
            'Random': "For randomized n, k. Default - True"})
def run(c, k=0, n=0, Random=True):
    """
    Setup the program and Runs main.py with the given parameters
    """
    c.run("python3.8.5 setup.py build_ext --inplace")
    Random = "--random" if Random else ""
    c.run(f"python3.8.5 main.py {Random} {k} {n}")


# ============ Additional Utilities: ============
@task
def build(c):
    c.run("python3.8.5 setup.py build_ext --inplace")


@task(aliases=["del"])
def delete(c):
    c.run("rm *mykmeanssp*.so")


@task
def clean(c):
    c.run("rm data.txt clusters.txt clusters.pdf")


@task(help={'fname': "Path of the txt containing the data points and centers",
            'k': "K used to create the data",
            'random': "Will use k from eigengap heuristic"})
def run_from_txt(c, fname, k, random=True):
    """
    run the program and cluster the data-set given in 'fname'
    """
    import numpy as np
    from main import run_clustering
    # parameters:
    _k = int(k)
    _random = bool(random)
    # get data from txt:
    data = np.genfromtxt(fname, delimiter=',', dtype=np.float64)
    centers = data[:, -1].astype(np.int32)
    points = data[:, :-1]

    class args:
        n = data.shape[0]
        dim = points.shape[1]
        k = _k
        random = _random

    run_clustering(args, points, centers)
