import sys
import numpy as np
from timeit import default_timer as timer
import math
from numba import cuda
from threading import Thread


def matrix_mul(A, B, C):
    for row in range(C.shape[0]):
        for col in range(C.shape[1]):
            tmp = 0
            for k in range(A.shape[1]):
                tmp += A[row, k] * B[k, col]
            C[row, col] = tmp


def matrix_parallel_mul(start, end, A, B, C):
    for i in range(start, end):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                C[i][j] += int(A[i][k] * B[k][j])


def thread_function(A, B, C):
    num_of_threads = 10
    thread_handle = []

    for j in range(0, num_of_threads):
        t = Thread(target=matrix_parallel_mul,
                   args=(int((C.shape[0] / num_of_threads) * j), int((C.shape[0] / num_of_threads) * (j + 1)),
                         A, B, C))
        thread_handle.append(t)
        t.start()

    for j in range(0, num_of_threads):
        thread_handle[j].join()


def matrix_generator(size):
    return np.random.randint(0, 10, size=size, dtype='int64')


def get_matrix_size():
    size_x = int(input("Podaj liczbę kolumn macierzy: "))
    size_y = int(input("Podaj liczbę wierszy macierzy: "))
    return size_x, size_y


def validate_matrix_size(size_1, size_2):
    return size_1[1] == size_2[0]


def global_mem(A, B):
    C_global_mem = cuda.device_array((A.shape[0], B.shape[1]))
    threadsperblock = (32, 32)
    blockspergrid_x = int(math.ceil(A.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(math.ceil(B.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    A_global_mem = cuda.to_device(A)
    B_global_mem = cuda.to_device(B)

    cuda_matrix_mul[blockspergrid, threadsperblock](A_global_mem, B_global_mem, C_global_mem)

    C = C_global_mem.copy_to_host()
    return C


@cuda.jit
def cuda_matrix_mul(A, B, C):
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp


if __name__ == '__main__':
    size_1 = get_matrix_size()
    size_2 = get_matrix_size()
    while not validate_matrix_size(size_1, size_2):
        print("Liczba wierszy w macierzy pierwszej nie zgadza się z liczbą kolumn w macierzy 2!", file=sys.stderr)
        size_1 = get_matrix_size()
        size_2 = get_matrix_size()
    result = np.zeros((size_1[0], size_2[1]), dtype='int64')

    martix1 = matrix_generator(size_1)
    matrix2 = matrix_generator(size_2)
    start = timer()
    matrix_mul(martix1, matrix2, result)
    result_time = timer() - start
    print(result)

    print(result_time)
    result2 = np.zeros((size_1[0], size_2[1]), dtype='int64')
    start2 = timer()
    thread_function(martix1, matrix2, result2)
    result_time2 = timer() - start2
    print(result2)
    print(result_time2)

    # start1 = timer()
    # result1 = global_mem(martix1, matrix2)
    # result_time1 = timer() - start1
    #
    # print(result1)
    # print(result_time1)
