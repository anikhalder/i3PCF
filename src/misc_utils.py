import numpy as np
from scipy import interpolate

import itertools
import multiprocessing as mp

from scipy.special import factorial

def function_interp1d_grid(x_arr, val_func):
    val_grid = np.zeros(x_arr.size)
    for i in range(x_arr.size):
        val_grid[i] = val_func(x_arr[i])

    return interpolate.interp1d(x_arr, val_grid, kind='cubic', fill_value=0.0)

def function_interp2d_grid(x_arr, y_arr, val_func):
    val_grid = np.zeros([x_arr.size, y_arr.size])
    for i in range(x_arr.size):
        for j in range(y_arr.size):
            val_grid[i,j] = val_func(x_arr[i], y_arr[j])

    return interpolate.interp2d(x_arr, y_arr, val_grid.T, kind='cubic', fill_value=0.0)

# serial
def function_interp2d_RectBivariateSpline_grid(x_arr, y_arr, val_func):
    val_grid = np.zeros([x_arr.size, y_arr.size])
    for i in range(x_arr.size):
        for j in range(y_arr.size):
            val_grid[i,j] = val_func(x_arr[i], y_arr[j])

    return interpolate.RectBivariateSpline(x_arr, y_arr, val_grid)

def function_interp2d_RectBivariateSpline_grid_parallel(x_arr, y_arr, val_func):

    global val_func_wrapper

    def val_func_wrapper(x_and_y):
        return val_func(*x_and_y)

    x_and_y_list = list(itertools.product(x_arr, y_arr))
    
    pool = mp.Pool(processes=mp.cpu_count()-2)
    val_grid = pool.map(val_func_wrapper, x_and_y_list)
    val_grid = np.reshape(np.array(val_grid), [x_arr.size, y_arr.size])
    pool.close()
    pool.join()

    return interpolate.RectBivariateSpline(x_arr, y_arr, val_grid)

def num_correlations(n, r):
    if (n==1):
        return int(1)

    # nCr with replacement
    return int(factorial(n+r-1)/factorial(n-1)/factorial(r)) # Note: tgamma(x+1) = factorial(x)