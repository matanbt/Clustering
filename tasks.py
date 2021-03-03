from invoke import task


@task
def run(c, K, N, random=False):
    c.run("python3.8.5 setup.py build_ext --inplace")
    random = "--random" if random else ""
    c.run(f"python3.8.5 main.py {random} {K} {N}")
    if not random:
        print(f"Clustering with: K={K}, N={N}")
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
