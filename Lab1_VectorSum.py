from numba import cuda
import time
import math
import numpy as np

size = 1000000
cpu_vect = np.random.randint(0, 9, size)
cpu_sum = [0]

gpu_vect = cuda.to_device(cpu_vect)
gpu_sum = cuda.to_device(cpu_sum)


def cpu_vector_sum(v):
    global cpu_sum
    for i in v:
        cpu_sum += i


@cuda.jit
def gpu_vector_sum(v, s):
    for i in range(size):
        s[0] += v[i]


def main():
    tpb = 32
    bpg = math.ceil(size / tpb)
    print(tpb, bpg)

    start_time = time.time()
    cpu_vector_sum(cpu_vect)
    print("--- %s seconds  (CPU)---" % (time.time() - start_time))

    print(cpu_sum)

    start_time = time.time()
    gpu_vector_sum[64, tpb](gpu_vect, gpu_sum)
    print("--- %s seconds  (GPU)---" % (time.time() - start_time))

    rez = gpu_sum.copy_to_host()
    print(rez)
    print(rez == cpu_sum)


if __name__ == "__main__":
    main()