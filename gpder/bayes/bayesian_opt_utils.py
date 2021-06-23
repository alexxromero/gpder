import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.utils.validation import check_random_state
import matplotlib.pyplot as plt
try:
    from termcolor import colored
    color_output = True
except ImportError:
    color_output = False

def print_log(params_keys, params, targets,
              iteration,
              opt_param=False,
              print_header=False):
    col_names = params_keys + ["Target"]
    col_width = max(10, max([len(i) for i in col_names]))

    if print_header:
        header = "| Iter | "
        header += "| ".join("{:{}}".format(
            name, col_width) for name in col_names)
        header += " |"
        print(header)
        print("=" * (len(header)))

    for r, row in enumerate(params):
        iter_column = "| {0:<4} | ".format(iteration[r])
        param_columns = "| ".join(
            "{:{}}".format("{0:4f}".format(param), col_width) for param in row)
        target_column = "| {:{}} |".format(
            "{0:4f}".format(targets[r, 0]), col_width)
        cols = iter_column + param_columns + target_column
        if color_output:
            if print_header:
                print(colored(cols, 'blue'))
            elif opt_param:
                print(colored(cols, 'red'))
            else:
                print(cols)
        else:
            print(cols)

def print_log_mse_uncer(params_keys, params, targets, uncert,
                        iteration,
                        mse_val=None, uncert_val=None,
                        initial_params=False,
                        opt_param=False,
                        print_header=False):
    col_names = params_keys + ["Target"] + ["Uncert"]
    if mse_val is not None:
        col_names += ["MSE val"]
    if uncert_val is not None:
        col_names += ["Uncert val"]
    col_width = max(10, max([len(i) for i in col_names]))

    if print_header:
        header = "| Iter | "
        header += "| ".join("{:{}}".format(
            name, col_width) for name in col_names)
        header += "|"
        print(header)
        print("=" * (len(header)))

    iteration = np.asarray(iteration, dtype=int)
    for r, row in enumerate(params):
        iter_col = "| {0:<4} | ".format(iteration[r])
        param_cols = "| ".join(
            "{:{}}".format("{0:4f}".format(param), col_width) for param in row)
        target_col = "| {:{}}".format("{0:4f}".format(targets[r, 0]), col_width)
        uncert_col = "| {:{}}".format("{0:4f}".format(uncert), col_width)
        cols = iter_col + param_cols + target_col + uncert_col
        if mse_val is not None:
            msev_col = "| {:{}}".format("{0:4f}".format(mse_val), col_width)
            cols += msev_col
        if uncert_val is not None:
            uncert_col = "| {:{}}".format(
                "{0:4f}".format(uncert_val), col_width)
            cols += uncert_col
        cols += "|"
        if color_output:
            if initial_params:
                print(colored(cols, 'blue'))
            else:
                print(cols)
        else:
            print(cols)
