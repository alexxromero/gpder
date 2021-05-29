import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.utils.validation import check_random_state
import matplotlib.pyplot as plt
import pandas as pd
try:
    from termcolor import colored
    color_output = True
except ImportError:
    color_output = False

def print_log(params_keys, params, targets, iteration,
              opt_param=False, print_header=False):
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
        cont = iter_column + param_columns + target_column
        if color_output:
            if print_header:
                print(colored(cont, 'blue'))
            elif opt_param:
                print(colored(cont, 'red'))
            else:
                print(cont)
        else:
            print(cont)


def print_log_uncertainty(params_keys, params, targets, uncertainty,
                          iteration, opt_param=False, print_header=False):
    col_names = params_keys + ["Target"] + ["Uncert"]
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
        target_column = "| {:{}}|".format(
            "{0:4f}".format(targets[r, 0]), col_width)
        uncert_column = " {:{}} |".format(
            "{0:4f}".format(uncertainty[r, 0]), col_width)
        cont = iter_column + param_columns + target_column + uncert_column
        if color_output:
            if print_header:
                print(colored(cont, 'blue'))
            elif opt_param:
                print(colored(cont, 'red'))
            else:
                print(cont)
        else:
            print(cont)
