"""
----- Tasks -----
module for invoke tasks
"""

from invoke import task


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
@task
def build(c):
    c.run("python3.8.5 setup.py build_ext --inplace")


@task(aliases=["del"])
def delete(c):
    c.run("rm *mykmeanssp*.so")

@task
def clean(c):
    c.run("rm data.txt clusters.txt clusters.pdf")
