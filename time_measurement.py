import csv
import numpy as np
from timeit import default_timer as timer

from basic_mult import size_generator, matrix_generator, matrix_mul, thread_function, global_mem

thread_sizes = [8, 16, 32]


def test_one_thread(matrix_1, matrix_2, col_1, row_2, repeat_number):
    stats = []
    for number in range(repeat_number):
        result = np.zeros((col_1, row_2), dtype='int64')
        start = timer()
        matrix_mul(matrix_1, matrix_2, result)
        result_time = timer() - start
        stats.append(result_time)
    write_to_file("one_thread_test.csv", stats)


def test_multi_thread(matrix_1, matrix_2, col_1, row_2, repeat_number):
    for num_threads in thread_sizes:
        stats = []
        for number in range(repeat_number):
            result = np.zeros((col_1, row_2), dtype='int64')
            start = timer()
            thread_function(matrix_1, matrix_2, result, num_threads)
            result_time = timer() - start
            stats.append(result_time)
        write_to_file(f"multi_thread_test_{num_threads}.csv", stats)


def gpu_test(matrix_1, matrix_2, repeat_number):
    for num_threads in thread_sizes:
        stats = []
        for number in range(repeat_number):
            start = timer()
            global_mem(matrix_1, matrix_2, num_threads)
            result_time = timer() - start
            stats.append(result_time)
        write_to_file(f"gpu_thread_test_{num_threads}.csv", stats)


def write_to_file(file_name, stats):
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(stats)


def test_multiplication():
    col_1 = size_generator()
    row_1_col_2 = size_generator()
    row_2 = size_generator()

    matrix_1 = matrix_generator((col_1, row_1_col_2))
    matrix_2 = matrix_generator((row_1_col_2, row_2))

    while True:
        try:
            repeat_number = int(input("Podaj liczbę powtórzeń testu: "))
            break
        except ValueError:
            print('Zła wartość. Możesz wpisać tylko liczbę całkowitą ...')

    test_one_thread(matrix_1, matrix_2, col_1, row_2, repeat_number)
    test_multi_thread(matrix_1, matrix_2, col_1, row_2, repeat_number)
    gpu_test(matrix_1, matrix_2, repeat_number)

    # for number in range(repeat_number):
    #     result = np.zeros((col_1, row_2), dtype='int64')
    #     matrix_mul(martix_1, matrix_2, result)
    #     result = np.zeros((col_1, row_2), dtype='int64')
    #     thread_function(martix_1, matrix_2, result, 8)
    #     global_mem(martix_1, matrix_2, 8)
    #
    # for number in range(repeat_number):
    #     result = np.zeros((col_1, row_2), dtype='int64')
    #     matrix_mul(martix_1, matrix_2, result)
    #     result = np.zeros((col_1, row_2), dtype='int64')
    #     thread_function(martix_1, matrix_2, result, 8)
    #     global_mem(martix_1, matrix_2, 16)
    #
    # for number in range(repeat_number):
    #     result = np.zeros((col_1, row_2), dtype='int64')
    #     matrix_mul(martix_1, matrix_2, result)
    #     result = np.zeros((col_1, row_2), dtype='int64')
    #     thread_function(martix_1, matrix_2, result, 8)
    #     global_mem(martix_1, matrix_2, 32)
    #
    # for number in range(repeat_number):
    #     result = np.zeros((col_1, row_2), dtype='int64')
    #     matrix_mul(martix_1, matrix_2, result)
    #     result = np.zeros((col_1, row_2), dtype='int64')
    #     thread_function(martix_1, matrix_2, result, 16)
    #     global_mem(martix_1, matrix_2, 8)
    #
    # for number in range(repeat_number):
    #     result = np.zeros((col_1, row_2), dtype='int64')
    #     matrix_mul(martix_1, matrix_2, result)
    #     result = np.zeros((col_1, row_2), dtype='int64')
    #     thread_function(martix_1, matrix_2, result, 16)
    #     global_mem(martix_1, matrix_2, 16)
    #
    # for number in range(repeat_number):
    #     result = np.zeros((col_1, row_2), dtype='int64')
    #     matrix_mul(martix_1, matrix_2, result)
    #     result = np.zeros((col_1, row_2), dtype='int64')
    #     thread_function(martix_1, matrix_2, result, 16)
    #     global_mem(martix_1, matrix_2, 32)
    #
    # for number in range(repeat_number):
    #     result = np.zeros((col_1, row_2), dtype='int64')
    #     matrix_mul(martix_1, matrix_2, result)
    #     result = np.zeros((col_1, row_2), dtype='int64')
    #     thread_function(martix_1, matrix_2, result, 32)
    #     global_mem(martix_1, matrix_2, 8)
    #
    # for number in range(repeat_number):
    #     result = np.zeros((col_1, row_2), dtype='int64')
    #     matrix_mul(martix_1, matrix_2, result)
    #     result = np.zeros((col_1, row_2), dtype='int64')
    #     thread_function(martix_1, matrix_2, result, 32)
    #     global_mem(martix_1, matrix_2, 16)
    #
    # for number in range(repeat_number):
    #     result = np.zeros((col_1, row_2), dtype='int64')
    #     matrix_mul(martix_1, matrix_2, result)
    #     result = np.zeros((col_1, row_2), dtype='int64')
    #     thread_function(martix_1, matrix_2, result, 32)
    #     global_mem(martix_1, matrix_2, 32)
