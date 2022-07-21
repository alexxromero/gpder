import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle, Arrow, FancyArrow

from sklearn.metrics import mean_squared_error

sys.path.append("../..")
import gpder
from gpder.gaussian_process import GaussianProcessRegressor
from gpder.gaussian_process.kernels import GPKernel, GPKernelDerAware

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = "14"

def yinline(x, m, b):
    return m * x + b

def xincircle(x0, y0, m, b, r):
    A = (1 + m**2)
    B = 2 * (m*b - x0 - y0*m)
    C = x0**2 + y0**2 + b**2 - 2*y0*b - r**2
    res_plus = (-B + np.sqrt(B**2 - 4*A*C)) / (2*A)
    res_neg = (-B - np.sqrt(B**2 - 4*A*C)) / (2*A)
    return np.array((res_plus, res_neg))

def xinelipse(x0, y0, m, b, rx, ry):
    A = (1/rx**2 + m**2/ry**2)
    B = 2 * (m*(b-y0)/ry**2 - x0/rx**2)
    C = x0**2/rx**2 + (b-y0)**2/ry**2 - 1
    res_plus = (-B + np.sqrt(B**2 - 4*A*C)) / (2*A)
    res_neg = (-B - np.sqrt(B**2 - 4*A*C)) / (2*A)
    return np.array((res_plus, res_neg))

def get_gradients_xy(x, y, dy, length_x, length_y=None):
    b = y - dy * x
    if length_y is None:
        x_vals = xincircle(x, y, dy, b, length_x)
    else:
        x_vals = xinelipse(x, y, dy, b, length_x, length_y)
    y_vals = np.array([yinline(x_vals[0], dy, b), yinline(x_vals[1], dy, b)])
    return (x_vals, y_vals)

def plot_gp(gp, X_test, y_test, title, deriv=False):
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    plt.subplots_adjust(wspace=0.4)
    mu, std = gp.predict(X_test, return_std=True)
    _, cov = gp.predict(X_test, return_cov=True)
    lower_conf = mu.ravel()-1.96*std.ravel()
    upper_conf = mu.ravel()+1.96*std.ravel()
    ax.plot(X_test, y_test, zorder=1, color='m', alpha=0.8, label='Noise-free func.')
    ax.plot(X_test, mu, zorder=3, color='navy', label='Mean')
    ax.fill_between(X_test.ravel(), lower_conf, upper_conf, zorder=1,
                    color='lightblue', alpha=0.5, label='95\% confidence')
    nsamples=3
    samples = np.random.multivariate_normal(mu.ravel(), cov, nsamples)
    for i, sample in enumerate(samples):
        label=None if i+1 < len(samples) else 'GP samples'
        ax.plot(X_test, sample, ls='--', zorder=2, color='gray', alpha=0.8,
                linewidth=1, label=label)
    ax.scatter(gp.X_train_, gp.y_train_, marker='o', zorder=4, color='red', s=15)
    if deriv:
        for (x, y, dy) in list(zip(gp.X_train_, gp.y_train_, gp.dy_train_)):
            x_vals, y_vals = get_gradients_xy(x, y, dy, 0.8, 0.8*1.2)
            ax.plot(x_vals, y_vals, color='red',  zorder=5)

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_yticks([-5, -2.5, 0, 2.5, 5])
    ax.set_ylim([-6, 5.2])
    ax.set_xticks([-6, -4, -2, 0, 2, 4, 6])
    ax.tick_params(axis='both', which='major')
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    if deriv:
        obs_handle = Line2D([0], [0], ls=' ', marker='o', ms=4.5, color='#E82400')
    else:
        obs_handle = Line2D([0], [0], ls=' ', marker='o', ms=4.5, color='#E82400')
    handles.append(obs_handle)
    labels.append('Noisy obs.')

    ax.legend(handles=handles, labels=labels, loc=(1.05, 0.2),
              labelspacing=0.2, handlelength=0.8,
              borderaxespad=0.2, handletextpad=0.3, borderpad=0.2,)
    plt.show()

def get_mse_uncert(bayes, niters, X_test, y_test):
    mse, uncert = [], []
    for i in range(niters):
        gp = bayes._gp_record[i]
        mu, cov = gp.predict(X_test, return_cov=True)
        mu = mu.ravel()
        mse.append(mean_squared_error(mu, y_test))
        uncert.append(np.trace(cov))
    return np.array(mse), np.array(uncert)


def plot_gp_evolution(bayes, niters, X_test, y_test, save_to=None):
    nplots = niters + 1
    fig = plt.figure(nplots, figsize=(8.5, 11))
    outer_gs = gridspec.GridSpec(nplots, 1)
    ax = []
    for i in range(nplots):
        gs = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer_gs[i], hspace=0.1, height_ratios=[1, 0.5])
        for cell in gs:
            ax.append(fig.add_subplot(cell))

    res = np.int(np.sqrt(np.shape(X_test)[0]))
    extent = [bayes._X_bounds_vals[0, 0], bayes._X_bounds_vals[0, 1]]
    ninit = bayes.X_init.shape[0]
    nsampled = bayes.X_init.shape[0]
    for i in range(nplots):
        gp = bayes._gp_record[i]
        mu, cov = gp.predict(X_test, return_cov=True)
        _, std = gp.predict(X_test, return_std=True)
        mu = mu.ravel()
        std = std.ravel()
        uncert95 = 1.96*std
        ax[i*2].fill_between(X_test.ravel(), mu-uncert95, mu+uncert95, zorder=1,
                             color='lightblue', alpha=0.5)
        ax[i*2].plot(X_test, mu, color='navy', zorder=3)
        ax[i*2].plot(X_test, y_test, zorder=3, color='m')
        ax[i*2].scatter(bayes.X_train[:ninit], bayes.y_train[:ninit],
                        s=13, marker='o', zorder=50, color='#E82400')
        ax[i*2].scatter(bayes.X_train[ninit:nsampled+i], bayes.y_train[ninit:nsampled+i],
                        s=13, marker='o', zorder=40, color='#FA7A01')
        samples = np.random.multivariate_normal(mu.ravel(), cov, 3)
        for sample in samples:
            ax[i*2].plot(X_test, sample, ls='--', zorder=2, color='gray', alpha=0.8, linewidth=1)

        if gp._has_derinfo:
            for (x, y, dy) in list(zip(bayes.X_train[:ninit],
                                       bayes.y_train[:ninit],
                                       bayes.dy_train[:ninit])):
                x_vals, y_vals = get_gradients_xy(
                    x, y, dy, length_x=0.5, length_y=0.5*2.6
                    )
                ax[i*2].plot(x_vals, y_vals, color='#E82400', zorder=5)

            for (x, y, dy) in list(zip(bayes.X_train[ninit:nsampled+i],
                                        bayes.y_train[ninit:nsampled+i],
                                        bayes.dy_train[ninit:nsampled+i])):
                x_vals, y_vals = get_gradients_xy(
                    x, y, dy, length_x=0.5, length_y=0.5*2.6
                    )
                ax[i*2].plot(x_vals, y_vals, color='#FA7A01', zorder=5)

        ax[i*2].get_xaxis().set_visible(False)
        ax[i*2+1].get_xaxis().set_visible(False)
        # -- and the utility func -- #
        utility = [bayes._acq.utility(X, bayes.X_query, gp=gp) for X in X_test]
        ax[i*2+1].plot(X_test, utility, color='green')

        if i == 0:
            props = dict(boxstyle='round', facecolor='white', alpha=0.5)
            ax[0].text(4, 0.4, "Initial GP", zorder=6, bbox=props)
        if i > 0:
            props = dict(boxstyle='round', facecolor='white', alpha=0.5)
            ax[i*2].text(4, 0.4, "BED iter {}".format(i), zorder=6, bbox=props)

        # if i+1 <= niters:
        if i+1 <= bayes.n_iters:
            ax[i*2].vlines(bayes.X_train[nsampled + i], linestyle='dotted',
                           ymin=np.min(mu-uncert95), ymax=np.max(mu+uncert95))
            ax[i*2+1].vlines(bayes.X_train[nsampled + i], linestyle='dotted',
                             ymin=np.min(utility), ymax=np.max(utility))
        ax[i*2].set_ylabel(r"  $y$")
        ax[i*2+1].set_ylabel("Utility", labelpad=8)
        ax[i*2].set_xlim(-2*np.pi-0.3, 2*np.pi+0.3)
        ax[i*2+1].set_xlim(-2*np.pi-0.3, 2*np.pi+0.3)
        ax[i*2].tick_params(axis='both', which='major')
        ax[i*2+1].tick_params(axis='both', which='major')

    ax[i*2+1].get_xaxis().set_visible(True)
    ax[-1].set_xlabel(r"$x$")

    # ----- legend and title ----- #
    if gp._has_derinfo:
        title = "Derivative GP"
        init_fun_obs_handlde = Line2D([0], [0], ls='-', marker='o', ms=5, color='#E82400')
        later_fun_obs_handlde = Line2D([0], [0], ls='-', marker='o', ms=5, color='#FA7A01')
        init_fun_obs_label = 'Initial func. & deriv. obs.'
        later_fun_obs_label = 'Subsequent func. \n & deriv. obs.'
        legend_loc = (0.01, 0.875)
    else:
        title = "Regular GP"
        init_fun_obs_handlde = Line2D([0], [0], ls=' ', marker='o', ms=5, color='#E82400')
        later_fun_obs_handlde = Line2D([0], [0], ls=' ', marker='o', ms=5, color='#FA7A01')
        init_fun_obs_label = 'Initial func. obs.'
        later_fun_obs_label = 'Subsequent func. obs.'
        legend_loc = (0.05, 0.88)
    labels = ['Noise-free func.', 'Mean', '95% confidence', 'GP samples',
              init_fun_obs_label, later_fun_obs_label, 'Next obs.']
    truth_handle = Line2D([0], [0], color='m', alpha=0.8)
    mean_handle = Line2D([0], [0], color='navy')
    confidence_handle = Patch(color='lightblue', alpha=0.5)
    samples_handle = Line2D([0], [0], ls='--', color='gray', alpha=0.5, linewidth=1)
    next_fun_handle = Line2D([0], [0], ls='dotted', color='black')
    handles = [truth_handle, mean_handle, confidence_handle, samples_handle,
               init_fun_obs_handlde, later_fun_obs_handlde, next_fun_handle]
    fig.legend(handles=handles, labels=labels,
        columnspacing=1.0, labelspacing=0.3, handlelength=1.5,
        ncol=3, loc=legend_loc
        )

    fig.text(0.4, 0.98, title)

    if save_to is not None:
        plt.savefig(save_to, bbox_inches='tight')
    plt.show()
