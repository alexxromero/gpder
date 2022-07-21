import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from sklearn.metrics import mean_squared_error

sys.path.append("../..")
import gpder
from gpder.gaussian_process import GaussianProcessRegressor
from gpder.gaussian_process.kernels import GPKernel, GPKernelDerAware


def plot_mse_uncert(bayes, X_test, y_test, save_to=None):
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
    plt.subplots_adjust(wspace=0.3)
    mse_vals = []
    uncert_vals = []
    iters = []
    nsampled = bayes.X_init.shape[0]
    for i, gp in enumerate(bayes._gp_record):
        mu, std = gp.predict(X_test, return_std=True)
        mse = mean_squared_error(y_test, mu)
        mse_vals.append(mse)
        uncert_vals.append(np.sum(std**2))
        iters.append(nsampled+i)
    ax[0].plot(iters, mse_vals)
    ax[0].set_ylabel("MSE")
    ax[1].plot(iters, uncert_vals)
    ax[1].set_ylabel("Uncertainty")
    for i in range(2):
        ax[i].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax[i].set_xlabel("Num. of Sampled Points")
    return mse_vals, uncert_vals


def fill_image(image, extent, ax, title, X_train=None):
    im = ax.imshow(image, origin='upper', extent=extent)
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    if X_train is not None:
        ax.scatter(X_train[:, 0], X_train[:, 1], color='red', s=15,
                   label='Observations')
    pad = 33 if X_train is not None else 10
    ax.set_title(title, pad=pad)
    ax.set_xlabel(r'$\nu^{j_1}$', fontsize=18, labelpad=-0.2)
    ax.set_ylabel(r'$\nu^{j_{2,3}}$', fontsize=18, labelpad=-0.2)
    ax.tick_params(axis='both', which='major')
    if X_train is not None:
        ax.legend(loc=(0.05, 1.02), handletextpad=0.2)


def plot_gp(y_test, y_pred, y_std, X_train, X_lower, X_upper, res,
            dy_test=None, dy_pred=None, dy_std=None):
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = "14"
    if dy_test is None:
        fig, ax = plt.subplots(1, 3, figsize=(12, 4), squeeze=False)
        nderivs = 0
    else:
        nderivs = dy_test.shape[1]
        fig, ax = plt.subplots(1+nderivs, 3,
                               figsize=(12, 4*(1+nderivs)), squeeze=False)
    plt.subplots_adjust(wspace=0.47)
    extent = [X_lower, X_upper, X_upper, X_lower]

    subplots_titles=[r"$\epsilon(\nu^{j_1}, \nu^{j2,3})$",
                     r"${\partial\epsilon(\nu^{j_1}, \nu^{j2,3})} / {\partial \nu^{j_1}}$",
                     r"${\partial\epsilon(\nu^{j_1}, \nu^{j2,3})} / {\partial \nu^{j_{2,3}}}$"]
    # -- efficiency -- #
    fill_image(y_test.reshape(res, res), extent, ax[0][0], title=subplots_titles[0])
    fill_image(y_pred.reshape(res, res), extent, ax[0][1], X_train=X_train,
               title="Prediction")
    fill_image(y_std.reshape(res, res), extent, ax[0][2], X_train=X_train,
               title="Standard deviation")
    for i in range(nderivs):
        fill_image(dy_test[:, i].reshape(res, res), extent, ax[1+i][0], title=subplots_titles[i+1])
        fill_image(dy_pred[:, i].reshape(res, res), extent, ax[1+i][1], X_train=X_train,
                   title="Prediction")
        fill_image(dy_std[:, i].reshape(res, res), extent, ax[1+i][2], X_train=X_train,
                   title="Standard deviation")
    plt.show()


def plot_gp_evolution(bayes, niters, X_test, save_to=None):
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = "14"
    nplots = niters + 1
    fig, ax = plt.subplots(nplots, 3, figsize=(8, 9), sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.25, hspace=0.1)
    res = np.int(np.sqrt(np.shape(X_test)[0]))
    extent = [bayes._X_bounds_vals[0, 0],
              bayes._X_bounds_vals[0, 1],
              bayes._X_bounds_vals[1, 1],
              bayes._X_bounds_vals[1, 0]]
    for i in range(nplots):
        gp = bayes._gp_record[i]
        mu, std = gp.predict(X_test, return_std=True)
        im0 = ax[i][0].imshow(mu.reshape(res, res),
                              origin='upper',
                              extent=extent,
                              vmin=np.min(mu),
                              vmax=np.max(mu))
        cbar0 = fig.colorbar(im0, ax=ax[i][0], fraction=0.046, pad=0.04)
        im1 = ax[i][1].imshow(std.reshape(res, res),
                              origin='upper',
                              extent=extent,
                              vmin=np.min(std),
                              vmax=np.max(std))
        cbar1 = fig.colorbar(im1, ax=ax[i][1], fraction=0.046, pad=0.04)
        X_test_utility = [bayes._acq.utility(
            X, bayes.X_query, gp=gp) for X in X_test]
        acq = np.array(X_test_utility)
        im2 = ax[i][2].imshow(acq.reshape(res, res),
                              origin='upper',
                              extent=extent,
                              vmin=np.min(acq),
                              vmax=np.max(acq))
        cbar2 = fig.colorbar(im2, ax=ax[i][2], fraction=0.046, pad=0.04)
        ninit = bayes.X_init.shape[0]
        for j in [0, 1, 2]:
            ax[i][j].scatter(bayes.X_train[:ninit, 0], bayes.X_train[:ninit, 1],
                             facecolor='red', s=10)
            ax[i][j].scatter(bayes.X_train[ninit:ninit + i, 0], bayes.X_train[ninit:ninit + i, 1],
                             facecolor='darkorange', s=10)
            if i+1 <= niters:
                Xnext = bayes.X_train[ninit + i]
                ax[i][j].scatter(Xnext[0], Xnext[1],
                                facecolor='#FF00FB', s=10)
        ax[i][0].set_ylabel(r"$\nu^{j_{2,3}}$", fontsize=18)
        for j in range(3):
            ax[i][j].set_xlim(0.5, 1.5)
            ax[i][j].set_ylim(1.5, 0.5)
    ax[-1][0].set_xlabel(r"$\nu^{j_{1}}$", fontsize=18)
    ax[-1][1].set_xlabel(r"$\nu^{j_{1}}$", fontsize=18)
    ax[-1][2].set_xlabel(r"$\nu^{j_{1}}$", fontsize=18)
    # ----- legend ----- #
    labels = ['Init. obs.', 'Subsequent obs.', 'Next obs.']
    handles = [Line2D([0], [0], ls=' ', marker='o', ms=4, color='red'),
               Line2D([0], [0], ls=' ', marker='o', ms=4, color='darkorange'),
               Line2D([0], [0], ls=' ', marker='o', ms=4, color='#FF00FB')]
    fig.legend(handles=handles, labels=labels,
               columnspacing=1,
               ncol=3, loc=(0.1, 0.925))
    # ------------------ #
    fig.text(0.18, 0.9, r"Prediction")
    fig.text(0.4, 0.9, r"Standard deviation")
    fig.text(0.76, 0.9, r"Utility")
    if bayes._has_derinfo:
        fig.text(0.44, 0.97, "Derivative GP")
    else:
        fig.text(0.45, 0.97, "Regular GP")
    if save_to is not None:
        plt.savefig(save_to, bbox_inches='tight')
    plt.show()
