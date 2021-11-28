import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics
from scipy.stats import norm
from scipy.optimize import curve_fit
from numpy import exp, loadtxt, pi, sqrt


file_names_8 = ['one_thread_test.csv', 'multi_thread_test_8.csv', 'gpu_thread_test_8.csv']


def avg_time():
    avg = []
    for file in file_names_8:
        data = pd.read_csv(file)
        avg.append(data.mean())
    print(avg)