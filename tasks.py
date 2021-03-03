from invoke import task

@task
def build(c):
    c.run("python3.8.5 setup.py build_ext --inplace")

@task(aliases=["del"])
def delete(c):
    c.run("rm *mykmeanssp*.so")

@task
def run(c, K, N, d):
    in_file = "in_{}_{}.txt".format(N, d)
    out_file = "out_{}_{}_{}.txt".format(K, N, d)
    gen_file_command = "python -m gen_file.so {} {} > {}".format(N, d, in_file)
    kmeans_command = "python3.8.5 kmeans_pp.py {} {} {} 1000 {} > {}".format(K, N, d, in_file, out_file)
    tester_command = "python -m kmpp.so {} {} {} 1000 {} {}".format(K, N, d, in_file, out_file)

    print("Running kmeans with: K={}, N={}, d={}".format(K, N, d))

    c.run(gen_file_command)
    c.run(kmeans_command)
    c.run(tester_command)

@task
def clean(c):
    c.run("rm in_*.txt out_*.txt")
