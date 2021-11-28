import numpy as np
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


def thread_function(A, B, C, threads):
    thread_handle = []

    for j in range(0, threads):
        t = Thread(target=matrix_parallel_mul,
                   args=(int((C.shape[0] / threads) * j), int((C.shape[0] / threads) * (j + 1)),
                         A, B, C))
        thread_handle.append(t)
        t.start()

    for j in range(0, threads):
        thread_handle[j].join()


def matrix_generator(size):
    return np.random.randint(0, 10, size=size, dtype='int64')


def size_generator():
    return np.random.randint(10, 15)


def get_matrix_size():
    loop_x = True
    loop_y = True
    while loop_x:
        try:
            size_x = int(input("Podaj liczbę kolumn macierzy: "))
            loop_x = False
        except ValueError:
            print('Zła wartość. Możesz wpisać tylko liczbę całkowitą ...')
            loop_x = True
    while loop_y:
        try:
            size_y = int(input("Podaj liczbę wierszy macierzy: "))
            loop_y = False
        except ValueError:
            print('Zła wartość. Możesz wpisać tylko liczbę całkowitą ...')
            loop_y = True

    return size_x, size_y


def validate_matrix_size(size_1, size_2):
    return size_1[1] == size_2[0]


def global_mem(A, B, threads):
    C_global_mem = cuda.device_array((A.shape[0], B.shape[1]))
    threadsperblock = (threads, threads)
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