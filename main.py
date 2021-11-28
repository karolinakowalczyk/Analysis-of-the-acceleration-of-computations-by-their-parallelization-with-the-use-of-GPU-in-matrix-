import sys
import numpy as np
from timeit import default_timer as timer

from time_measurement import test_multiplication

import basic_mult as bm


def matrix_multiplication():
    size_1 = bm.get_matrix_size()
    size_2 = bm.get_matrix_size()
    while not bm.validate_matrix_size(size_1, size_2):
        print("Liczba wierszy w macierzy pierwszej nie zgadza się z liczbą kolumn w macierzy 2!", file=sys.stderr)
        size_1 = bm.get_matrix_size()
        size_2 = bm.get_matrix_size()

    result = np.zeros((size_1[0], size_2[1]), dtype='int64')
    matrix1 = bm.matrix_generator(size_1)
    matrix2 = bm.matrix_generator(size_2)

    start = timer()
    bm.matrix_mul(matrix1, matrix2, result)
    result_time = timer() - start
    print("Czas obliczeń - jednowątkowe CPU: ", result_time, " s")

    result = np.zeros((size_1[0], size_2[1]), dtype='int64')
    start = timer()
    bm.thread_function(matrix1, matrix2, result, 32)
    result_time = timer() - start
    print("Czas obliczeń - wielowątkowe CPU: ", result_time, " s")

    start = timer()
    bm.global_mem(matrix1, matrix2, 32)
    result_time = timer() - start
    print("Czas obliczeń - GPU: ", result_time, " s")


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
