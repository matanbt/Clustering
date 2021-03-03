from invoke import run
from timeit import timeit
from math import ceil
def f(n, k):
    return 0.1 * k + 0.9 * n

def avg(n1, n2):
    return ceil((n1 + n2)/2)


REPEAT = 5


def run_program(n,k):
    run(f"python3.8.5 main.py {k} {n}")


def main():
    upper_n, upper_k = 1000, 500
    lower_n, lower_k = 500, 200
    while upper_n > lower_n:
        mid_n, mid_k = avg(upper_n, lower_n), avg(upper_k, lower_k)
        avg_time = timeit(lambda : run_program(mid_n, mid_k), number=REPEAT) / REPEAT
        avg_time = avg_time / 60  # to minutes
        print(f"---> for k={mid_k}, n={mid_n} : {avg_time} mins")
        if 4.9 < avg_time < 5.1:
            print("--- SUCCESS ---")
            break
        if avg_time > 5:
            upper_n, upper_k = mid_n, mid_k
        else:
            lower_n,lower_k = mid_n, mid_k

if __name__ == "__main__":
    main()