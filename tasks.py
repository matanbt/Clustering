"""
----- Tasks -----
module for invoke tasks
"""

from invoke import task
import numpy as np
from timeit import timeit

@task(help={'k': "Amount of centers for the generated data",
            'n': "Amount of points", 'Random': "for randomized points. Default - True"})
def run(c, k, n, Random=True):
    """
    Setup the program and runs main.py with the given parameters
    """
    c.run("python3.8.5 setup.py build_ext --inplace")
    Random = "--random" if Random else ""
    c.run(f"python3.8.5 main.py {Random} {k} {n}")


# tasks for developer purposes:
# TESTERS / TIMERS :

REPEAT = 5

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
def build(c):
    c.run("python3.8.5 setup.py build_ext --inplace")


@task(aliases=["del"])
def delete(c):
    c.run("rm *mykmeanssp*.so")

@task
def clean(c):
    c.run("rm data.txt clusters.txt clusters.pdf")
