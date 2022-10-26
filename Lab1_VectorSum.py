from numba import cuda
import time
import math
import numpy as np

size = 1000000    # размер вектора 1000 - 1000000
cpu_vect = np.random.randint(0, 10, size)
cpu_sum = [0]

gpu_vect = cuda.to_device(cpu_vect)
gpu_sum = cuda.to_device(cpu_sum)


def cpu_vector_sum(v):
    global cpu_sum  # это конечно мэм, но почему бы и нет?
    for i in v:
        cpu_sum += i


@cuda.jit
def gpu_vector_sum(v, s):
    # накатпливаем сумму элементов вектора "v"
    # в единственном элемнете массива "s"
    for i in range(size):
        s[0] += v[i]


def main():

    # настройка ядра (здесь что то не так!!!)
    tpb = 32
    bpg = math.ceil(size / tpb)
    print(tpb, bpg)

    # работа CPU
    start_time = time.time()
    cpu_vector_sum(cpu_vect)
    print("--- %s seconds  (CPU)---" % (time.time() - start_time))

    print(cpu_sum)

    # работа GPU. Для вектора с размерами 1000 - 1000000 все работает
    # корректно только с ядром [64, 32], что конечно же не правильно(мне так кажется),
    # т.к. гпу обгоняет цпу тоьлко при size = 1000000.
    # Вычисление ядра как в Matmul дает не верный результат.
    start_time = time.time()
    gpu_vector_sum[64, tpb](gpu_vect, gpu_sum)
    print("--- %s seconds  (GPU)---" % (time.time() - start_time))

    # проверка результата
    rez = gpu_sum.copy_to_host()
    print(rez)
    print(rez == cpu_sum)


if __name__ == "__main__":
    main()