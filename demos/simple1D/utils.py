import sys
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

sys.path.append("../..")
import gpder
from gpder.gaussian_process import GaussianProcessRegressor
from gpder.gaussian_process.kernels import GPKernel, GPKernelDerAware

def plot_gpr(X_train, y_train,
             X_true, y_true, y_pred, std_pred,
             samples=[], save_to=None):
    uncert95 = 1.96*std_pred
    plt.figure(figsize=(8,6), dpi=80)
    plt.fill_between(X_true, y_pred+uncert95, y_pred-uncert95, alpha=0.2)
    plt.plot(X_train, y_train, '*', color='red', label='Training points')
    plt.plot(X_true, y_pred, color='navy', label='Predictive mean')
    plt.plot(X_true, y_true, color='darkviolet', label='True function')
    for i, sample in enumerate(samples):
        plt.plot(X_true, sample, ls='--',
                 label='Sample {}'.format(i+1), alpha=0.5)
    plt.legend(fontsize=14)
    plt.xlabel("X", fontsize=14)
    plt.ylabel("F", fontsize=14)
    if save_to is not None:
        plt.savefig(save_to, bbox_inches='tight')
    plt.show()


def plot_gpr_iterloss(mse, uncert, save_to=None):
    fig, ax = plt.subplots(1, 2, figsize=(12, 3))
    ax[0].plot(mse)
    ax[0].set_xlabel("Iter", fontsize=14)
    ax[0].set_ylabel("MSE", fontsize=14)
    ax[1].plot(uncert)
    ax[1].set_xlabel("Iter", fontsize=14)
    ax[1].set_ylabel("Uncert", fontsize=14)
    if save_to is not None:
        plt.savefig(save_to, bbox_inches='tight')
    plt.show()


def plot_gpr_evolution(bayes, save_to=None):
    params_test = bayes.params_test
    targets_test = bayes.targets_test

    nbayes = len(bayes.bayes_targets)
    ninit = len(bayes.init_targets)
    nplots = nbayes + 1
    fig, axs = plt.subplots(nplots, figsize=(8, nplots*1.5), sharex=True)

    if len(bayes.init_dtargets) > 0:
        kernel = GPKernelDerAware()
        kernel.theta = bayes.init_theta
        gp = GaussianProcessRegressor(kernel=kernel, optimizer=None)
        gp.fit(X=bayes.init_params, y=bayes.init_targets,
               dX=bayes.init_params, dy=bayes.init_dtargets)
    else:
        kernel = GPKernel()
        kernel.theta = bayes.init_theta
        gp = GaussianProcessRegressor(kernel=kernel, optimizer=None)
        gp.fit(X=bayes.init_params, y=bayes.init_targets)

    mu, cov = gp.predict(params_test, return_cov=True)
    _, std = gp.predict(params_test, return_std=True)
    mse = mean_squared_error(targets_test, mu)
    uncert = np.trace(cov)
    mu = mu.ravel()
    uncert95 = 1.96*std

    axs[0].scatter(bayes.init_params, bayes.init_targets,
                   color='green', label='Initial pts', marker='x')
    axs[0].fill_between(params_test.reshape(-1,),
                        mu-uncert95, mu+uncert95, alpha=0.2)
    axs[0].plot(params_test, mu, color='navy', label='Pred')
    axs[0].plot(params_test, targets_test, color='blueviolet', label='True')
    axs[0].vlines(bayes.bayes_params[0, :], ymin=-1, ymax=1, linestyles='dotted',
                  color='black', label='Next proposed pt')
    axs[0].scatter([], [], color='red', label='Bayes sampled pts', marker='*')
    axs[0].legend(loc='upper center',
                  ncol=3,
                  bbox_to_anchor=(0.5, 1.7))

    for i in range(nbayes):
        if len(bayes.bayes_dtargets) > 0:
            kernel = GPKernelDerAware()
            kernel.theta = bayes.bayes_theta[i]
            gp = GaussianProcessRegressor(kernel=kernel, optimizer=None)
            gp.fit(X=bayes.params[:i+ninit+1, :], y=bayes.targets[:i+ninit+1, :],
                   dX=bayes.params[:i+ninit+1, :], dy=bayes.dtargets[:i+ninit+1, :])
        else:
            kernel = GPKernel()
            kernel.theta = bayes.bayes_theta[i]
            gp = GaussianProcessRegressor(kernel=kernel, optimizer=None)
            gp.fit(X=bayes.params[:i+ninit+1, :], y=bayes.targets[:i+ninit+1, :])
        mu, cov = gp.predict(params_test, return_cov=True)
        _, std = gp.predict(params_test, return_std=True)
        mse = mean_squared_error(targets_test, mu)
        uncert = np.trace(cov)

        mu = mu.ravel()
        uncert95 = 1.96*std
        ii = i+1  # plot axis
        axs[ii].scatter(bayes.init_params, bayes.init_targets,
                        color='green', marker='x', label='Iter {}'.format(i+1))
        axs[ii].scatter(bayes.bayes_params[:i+1, :], bayes.bayes_targets[:i+1, :],
                        color='red', marker='*')
        axs[ii].fill_between(params_test.reshape(-1,),
                             mu-uncert95, mu+uncert95, alpha=0.2)
        axs[ii].plot(params_test, mu, color='navy')
        axs[ii].plot(params_test, targets_test, color='blueviolet')
        if (i < len(bayes.bayes_targets)-1):
            axs[ii].vlines(bayes.bayes_params[i+1, :], ymin=-1, ymax=1,
                           linestyles='dotted', color='black')
        axs[ii].legend(markerscale=0, handletextpad=-2)

    if save_to is not None:
        plt.savefig(save_to, bbox_inches='tight')
    plt.show()


def grid_sampling(niters, x_init, x_val,
                  bound_low, bound_high,
                  fun, dfun=None, save_to=None):
    y_init = fun(x_init)
    if dfun is not None:
        dx_init = x_init
        dy_init = dfun(dx_init)

    fig, axs = plt.subplots(niters+1, figsize=(8, (niters+1)*1.5), sharex=True)

    x_train = x_init
    y_train = fun(x_train)
    if dfun is not None:
        dx_train = x_train
        dy_train = dfun(dx_train)

    params_test = x_val.reshape(-1, 1)
    targets_test = fun(x_val)

    y_val = fun(x_val)

    kernel = GPKernel()
    gpr = GaussianProcessRegressor(kernel=kernel,
                                   ignore_convergence_warnings=True,
                                   n_restarts_optimizer=10,
                                   random_state=123)
    gpr.fit(x_train.reshape(-1, 1), y_train)

    # predict X_star
    mu, cov = gpr.predict(x_val.reshape(-1, 1), return_cov=True)
    _, std = gpr.predict(x_val.reshape(-1, 1), return_std=True)
    mu = mu.ravel()
    uncert95 = 1.96*std

    axs[0].scatter(x_train, y_train,
                   color='green', label='Initial pts', marker='x')
    axs[0].fill_between(params_test.reshape(-1,),
                        mu-uncert95, mu+uncert95, alpha=0.2)
    axs[0].plot(params_test, mu, color='navy', label='Pred')
    axs[0].plot(params_test, targets_test, color='blueviolet', label='True')
    axs[0].legend(loc='upper center',
                  ncol=3,
                  bbox_to_anchor=(0.5, 1.7))

    mse_grid = [mean_squared_error(y_val, mu)]
    uncert_grid = [np.trace(cov)]

    for i in range(niters):
        bound_low = 0
        bound_high = 1
        dpts = (bound_high - bound_low) / (i + 2)
        #x_grid = np.array([bound_low + dpts*(j) for j in range(1, i + 2)]).reshape(-1, 1)
        x_grid = np.linspace(bound_low, bound_high, i+1).reshape(-1, 1)
        y_grid = fun(x_grid)

        x_train = np.concatenate((x_init, x_grid))
        y_train = fun(x_train)
        if dfun is not None:
            dx_train = x_train
            dy_train = dfun(dx_train)

        if dfun is not None:
            kernel = GPKernelDerAware()
        else:
            kernel = GPKernel()
        gpr = GaussianProcessRegressor(kernel=kernel,
                                       ignore_convergence_warnings=True,
                                       n_restarts_optimizer=10,
                                       random_state=123)
        if dfun is not None:
            gpr.fit(x_train.reshape(-1, 1), y_train,
                    dx_train.reshape(-1, 1), dy_train)
        else:
            gpr.fit(x_train.reshape(-1, 1), y_train)

        # predict X_star
        mu, cov = gpr.predict(x_val.reshape(-1, 1), return_cov=True)
        _, std = gpr.predict(x_val.reshape(-1, 1), return_std=True)
        mu = mu.ravel()
        uncert95 = 1.96*std

        mse_grid.append(mean_squared_error(y_val, mu))
        uncert_grid.append(np.trace(cov))

        ii = i+1  # plot axis
        axs[ii].scatter(x_init, y_init,
                        color='green', marker='x', label='Iter {}'.format(i+1))
        axs[ii].scatter(x_grid, y_grid,
                        color='red', marker='*')
        axs[ii].fill_between(params_test.reshape(-1,),
                             mu-uncert95, mu+uncert95, alpha=0.2)
        axs[ii].plot(params_test, mu, color='navy')
        axs[ii].plot(params_test, targets_test, color='blueviolet')
        axs[ii].legend(markerscale=0, handletextpad=-2)

    if save_to is not None:
        plt.savefig(save_to, bbox_inches='tight')
    plt.show()

    return mse_grid, uncert_grid
