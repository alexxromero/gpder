"""Plotting functions for GPR."""

import numpy as np
import matplotlib.pyplot as plt

__all__ = ['plot_gpr', 'plot_approximation', 'plot_acquisition',
           'plot_approximation']

def plot_gpr(X_train, y_train, dX_train, dy_train,
             X_star, mu_star, std_star,
             samples=[], F=None, name_save=None):
    """Plotts the data and samples used in the GPR.

    Parameters
        X_train: array of shape (n_samples, n_features)
        Training data array.

        y_train : array of shape (n_samples,) or (n_samples, n_targets)
        Training target values.

        dX : array of shape (n_samples, n_features), default=[]
        Training data array of the derivative observations.

        dy : array of shape (n_samples,) or (n_samples, n_targets),
        default=[]
        Training target values of the derivative observations.

        X_star : array pf shape (n_samples, n_features)
        Data used in the regression.

        mu_star :
        Mean of the predicted Gaussian distribution.

        std_star :
        Standard deviation of the predicted Gaussian distribution.

        samples : []
        Idk

        F : callable, optional
        If given, the true function

        name_save : string, optional
        Name to use when saving the plot
    """

    uncert95 = 1.96*std_star
    plt.fill_between(X_star, mu_star+uncert95, mu_star-uncert95, alpha=0.1)
    plt.plot(X_star, mu_star, color='navy', label='Mean')
    plt.plot(X_train, y_train, 'rx', label='Training points')
    for i, sample in enumerate(samples):
        plt.plot(X_star, sample, ls='--', label='Sample {}'.format(i+1))
    if F:
        plt.plot(X_star, F(X_star), alpha=0.5, color='blueviolet',
                 label='True function')

    plt.legend()
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
