from numba import cuda
import time
import math
import numpy as np

size = 200
cpu_arr1 = np.random.randint(0, 9, (size, size))
cpu_arr2 = np.random.randint(0, 9, (size, size))
cpu_arr_result = np.zeros((size, size), dtype=int)

gpu_arr1 = cuda.to_device(cpu_arr1)
gpu_arr2 = cuda.to_device(cpu_arr2)
gpu_arr_result = cuda.device_array((len(cpu_arr1), len(cpu_arr2)))

def cpu_matmul(a, b, c):
    for i in range(size):
        for j in range(size):
            rez = 0
            for z in range(size):
                rez += a[i,z] * b[z,j]
            c[i,j] = rez

@cuda.jit
def gpu_matmul(a, b, c):
    for i in range(size):
        for j in range(size):
            rez = 0
            for z in range(size):
                rez += a[i,z] * b[z,j]
            c[i,j] = rez

def main():

    # настройка ядра
    threadsperblock = (32, 32)
    blockspergrid_x = int(math.ceil(cpu_arr1.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(math.ceil(cpu_arr2.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    print(blockspergrid, threadsperblock)

    # работа CPU
    print("CPU start")
    start_time = time.time()
    cpu_matmul(cpu_arr1, cpu_arr2, cpu_arr_result)
    print("--- %s seconds  (CPU)---" % (time.time() - start_time))

    # работа GPU
    print("GPU start")
    start_time = time.time()
    gpu_matmul[blockspergrid, threadsperblock](gpu_arr1, gpu_arr2, gpu_arr_result)
    print("--- %s seconds (GPU)---" % (time.time() - start_time))

    # копируем результат работы GPU в оперативную память системы
    cpu_copy_arr = gpu_arr_result.copy_to_host()

    # проверка равенства матриц
    print(np.array_equal(cpu_arr_result, cpu_copy_arr))
    print("end")


if __name__ == "__main__":
    main()