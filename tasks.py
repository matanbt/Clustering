from invoke import task


@task
def run(c, k, n, Random=False):
    c.run("python3.8.5 setup.py build_ext --inplace")
    Random = "--random" if Random else ""
    c.run(f"python3.8.5 main.py {Random} {k} {n}")
    if not Random:
        print(f"Clustering with: K={k}, N={n}")
    else:
        print(f"Clustering with randomized K,N")


# our tasks:
@task
def build(c):
    c.run("python3.8.5 setup.py build_ext --inplace")


@task(aliases=["del"])
def delete(c):
    c.run("rm *mykmeanssp*.so")

@task
def clean(c):
    c.run("rm data.txt clusters.txt clusters.pdf")
