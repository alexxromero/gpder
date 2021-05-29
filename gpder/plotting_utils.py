"""Plotting functions for GPR."""

import numpy as np
import matplotlib.pyplot as plt

__all__ = ['plot_gpr', 'plot_approximation', 'plot_acquisition',
           'plot_approximation']

def plot_gpr(X_train, y_train, dX_train, dy_train,
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


def plot_approximation(gpr, X, Y, X_sample, Y_sample, X_next=None,
                       show_legend=False, axs=None):
    ax = axs if axs is not None else plt.gca()
    mu, _, std = gpr.predict(X, return_std=True)
    ax.fill_between(X.ravel(),
                    mu.ravel() + 1.96 * std,
                    mu.ravel() - 1.96 * std,
                    alpha=0.1)
    ax.plot(X, Y, 'y--', lw=1, label='Noise-free objective')
    ax.plot(X, mu, 'b-', lw=1, label='Surrogate function')
    ax.plot(X_sample, Y_sample, 'kx', mew=3, label='Noisy samples')
    #if X_next:
    #    ax.axvline(x=X_next, ls='--', c='k', lw=1)
    #if show_legend:
    #    ax.legend()

def plot_acquisition(X, Y, X_next, show_legend=False, axs=None):
    ax = axs if axs is not None else plt.gca()
    ax.plot(X, Y, 'r-', lw=1, label='Acquisition function')
    ax.axvline(x=X_next, ls='--', c='k', lw=1, label='Next sampling location')
    if show_legend:
        ax.legend()

def plot_convergence(X_sample, Y_sample, n_init=2):
    plt.figure(figsize=(12, 3))

    x = X_sample[n_init:].ravel()
    y = Y_sample[n_init:].ravel()
    r = range(1, len(x)+1)

    x_neighbor_dist = [np.abs(a-b) for a, b in zip(x, x[1:])]
    y_max_watermark = np.maximum.accumulate(y)

    plt.subplot(1, 2, 1)
    plt.plot(r[1:], x_neighbor_dist, 'bo-')
    plt.xlabel('Iteration')
    plt.ylabel('Distance')
    plt.title('Distance between consecutive x\'s')

    plt.subplot(1, 2, 2)
    plt.plot(r, y_max_watermark, 'ro-')
    plt.xlabel('Iteration')
    plt.ylabel('Best Y')
    plt.title('Value of best selected sample')
