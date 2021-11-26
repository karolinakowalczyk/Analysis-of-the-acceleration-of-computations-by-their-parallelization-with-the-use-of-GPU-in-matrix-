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
    return np.random.randint(10, 1000)


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


def matrix_multiplication():
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
    print("Czas obliczeń - jednowątkowe CPU: ", result_time, " s")

    result = np.zeros((size_1[0], size_2[1]), dtype='int64')
    start = timer()
    thread_function(martix1, matrix2, result, 32)
    result_time = timer() - start
    print("Czas obliczeń - wielowątkowe CPU: ", result_time, " s")

    start = timer()
    global_mem(martix1, matrix2, 32)
    result_time = timer() - start
    print("Czas obliczeń - GPU: ", result_time, " s")


def test_multiplication():
    repeat_number = 0
    col_1 = size_generator()
    row_1_col_2 = size_generator()
    row_2 = size_generator()

    martix_1 = matrix_generator((col_1, row_1_col_2))
    matrix_2 = matrix_generator((row_1_col_2, row_2))

    loop_input = True
    while loop_input:
        try:
            repeat_number = int(input("Podaj liczbę powtórzeń testu: "))
            loop_input = False
        except ValueError:
            print('Zła wartość. Możesz wpisać tylko liczbę całkowitą ...')
            loop_input = True

    for number in range(repeat_number):
        result = np.zeros((col_1, row_2), dtype='int64')
        matrix_mul(martix_1, matrix_2, result)
        result = np.zeros((col_1, row_2), dtype='int64')
        thread_function(martix_1, matrix_2, result, 8)
        global_mem(martix_1, matrix_2, 8)

    for number in range(repeat_number):
        result = np.zeros((col_1, row_2), dtype='int64')
        matrix_mul(martix_1, matrix_2, result)
        result = np.zeros((col_1, row_2), dtype='int64')
        thread_function(martix_1, matrix_2, result, 8)
        global_mem(martix_1, matrix_2, 16)

    for number in range(repeat_number):
        result = np.zeros((col_1, row_2), dtype='int64')
        matrix_mul(martix_1, matrix_2, result)
        result = np.zeros((col_1, row_2), dtype='int64')
        thread_function(martix_1, matrix_2, result, 8)
        global_mem(martix_1, matrix_2, 32)

    for number in range(repeat_number):
        result = np.zeros((col_1, row_2), dtype='int64')
        matrix_mul(martix_1, matrix_2, result)
        result = np.zeros((col_1, row_2), dtype='int64')
        thread_function(martix_1, matrix_2, result, 16)
        global_mem(martix_1, matrix_2, 8)

    for number in range(repeat_number):
        result = np.zeros((col_1, row_2), dtype='int64')
        matrix_mul(martix_1, matrix_2, result)
        result = np.zeros((col_1, row_2), dtype='int64')
        thread_function(martix_1, matrix_2, result, 16)
        global_mem(martix_1, matrix_2, 16)

    for number in range(repeat_number):
        result = np.zeros((col_1, row_2), dtype='int64')
        matrix_mul(martix_1, matrix_2, result)
        result = np.zeros((col_1, row_2), dtype='int64')
        thread_function(martix_1, matrix_2, result, 16)
        global_mem(martix_1, matrix_2, 32)

    for number in range(repeat_number):
        result = np.zeros((col_1, row_2), dtype='int64')
        matrix_mul(martix_1, matrix_2, result)
        result = np.zeros((col_1, row_2), dtype='int64')
        thread_function(martix_1, matrix_2, result, 32)
        global_mem(martix_1, matrix_2, 8)

    for number in range(repeat_number):
        result = np.zeros((col_1, row_2), dtype='int64')
        matrix_mul(martix_1, matrix_2, result)
        result = np.zeros((col_1, row_2), dtype='int64')
        thread_function(martix_1, matrix_2, result, 32)
        global_mem(martix_1, matrix_2, 16)

    for number in range(repeat_number):
        result = np.zeros((col_1, row_2), dtype='int64')
        matrix_mul(martix_1, matrix_2, result)
        result = np.zeros((col_1, row_2), dtype='int64')
        thread_function(martix_1, matrix_2, result, 32)
        global_mem(martix_1, matrix_2, 32)

menu_options = {
    1: 'Pomnóż macierze',
    2: 'Testuj',
    3: 'Wyjdź',
}


def print_menu():
    for key in menu_options.keys():
        print(key, '--', menu_options[key])


if __name__ == '__main__':
    while True:
        print_menu()
        option = ''
        try:
            option = int(input('Co chcesz zrobić?: '))
        except ValueError:
            print('Zła wartość. Możesz wpisać tylko liczbę całkowitą ...')
        if option == 1:
            matrix_multiplication()
        elif option == 2:
            test_multiplication()
        elif option == 3:
            print('Do zobaczenia!')
            exit()
        else:
            print('Zła oocja.')
