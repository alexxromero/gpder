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

__all__ = ['plot_gpr']

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


def print_log_mse_uncer(params_keys, params, targets, mse, uncertainty,
                        iteration, opt_param=False, print_header=False):
    col_names = params_keys + ["Target"] + ["MSE"] + ["Uncert"]
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
        mse_column = " {:{}}|".format(
            "{0:4f}".format(mse[r, 0]), col_width)
        uncert_column = " {:{}} |".format(
            "{0:4f}".format(uncertainty[r, 0]), col_width)
        cont = iter_column + param_columns + target_column \
             + mse_column + uncert_column
        if color_output:
            if print_header:
                print(colored(cont, 'blue'))
            elif opt_param:
                print(colored(cont, 'red'))
            else:
                print(cont)
        else:
            print(cont)


def plot_iter(X_train, y_train, X_star, mu_star, std_star, y_true, X_next,
              ax, iter):
    mu_star = mu_star.ravel()
    y_true = y_true.ravel()

    uncert95 = 1.96*std_star
    #ax.figure(figsize=(8, 3), dpi=80)
    ax.fill_between(X_star.reshape(-1,),
                     mu_star+uncert95, mu_star-uncert95, alpha=0.2)
    ax.plot(X_star, mu_star, color='navy', label='Pred')
    ax.plot(X_train, y_train, 'rx', label='Training points')
    ax.plot(X_star, y_true, alpha=0.5, color='blueviolet',
             label='Obj function')
    ax.vlines(X_next,
               ymin=-1,
               ymax=1,
               linestyles='dotted',
               color='black',
               label='Next point')

    ax.set_title("Iteration {}".format(iter))

    # ax.legend(fontsize=12, loc='lower left',
    #            ncol=4, mode="expand")
    #plt.show()


def plot_gpr(X_train, y_train,
             X_star, mu_star, std_star,
             samples=[], F=None, name_save=None):
    """Plotts the data and samples used in the GPR.

    Arguments
    ---------
    X_train: ndarray of shape (n, nfeatures)
        Training data array.

    y_train : ndarray of shape (nx,) or (n, n_targets)
        Training target values.

    X_star : ndarray of shape (nx_star, n_features)
        Data used in the regression.

    mu_star : ndarray of shape (nx_star, n_features)
        Mean of the predicted Gaussian distribution.

    std_star : ndarray of shape (nx_star, n_features)
        Standard deviation of the predicted Gaussian distribution.

    samples : ndarray of shape (n_samples, nx_star)
        Data arrays sampled from a multivariate normal
        distribution with the center and covariance given
        by the gpr.

    F : callable, optional
        True function. If passed, F is evaluated at std_star
        as a reference.

    name_save : string, optional
        File name for saving the plot.
    """

    uncert95 = 1.96*std_star
    plt.figure(figsize=(8, 6), dpi=80)
    plt.fill_between(X_star, mu_star+uncert95, mu_star-uncert95, alpha=0.2)
    plt.plot(X_star, mu_star, color='navy', label='Mean')
    plt.plot(X_train, y_train, 'rx', label='Training points')
    for i, sample in enumerate(samples):
        plt.plot(X_star, sample, ls='--', label='Sample {}'.format(i+1))
    if F:
        plt.plot(X_star, F(X_star), alpha=0.5, color='blueviolet',
                 label='True function')

    plt.legend(fontsize=14)
    plt.xlabel("X", fontsize=14)
    plt.ylabel("F", fontsize=14)
    if name_save:
        plt.savefig(name_save)
    plt.show()
