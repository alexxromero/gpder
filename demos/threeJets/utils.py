import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

sys.path.append("../..")
from gpder.gaussian_process import GaussianProcessRegressor
from gpder.gaussian_process.kernels import GPKernelDerAware, GPKernel

from matplotlib.patches import Circle

def plot_gpr_iterloss(bayes, saveto=None):
    fig, ax = plt.subplots(1, 2, figsize=(12, 3))
    ax[0].plot(bayes.mse_val)
    ax[0].set_xlabel("Iter", fontsize=14)
    ax[0].set_ylabel("MSE", fontsize=14)
    ax[1].plot(bayes.uncert_val)
    ax[1].set_xlabel("Iter", fontsize=14)
    ax[1].set_ylabel("Uncert", fontsize=14)
    if saveto is not None:
        plt.savefig(saveto + ".png", bbox_inches='tight')

def plot_gpr_evolution(bayes, X_lower, X_upper, saveto=None):

    nbayes = len(bayes.bayes_targets)
    ninit = len(bayes.init_targets)
    nplots = nbayes + 1
    fig, axs = plt.subplots(3, 3, figsize=(12, 12), sharex=True, sharey=True)
    ax = axs.ravel()

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

    mu, cov = gp.predict(bayes.params_val, return_cov=True)
    _, std = gp.predict(bayes.params_val, return_std=True)
    mse = mean_squared_error(bayes.targets_val, mu)
    uncert = np.trace(cov)
    mu = mu.ravel()
    uncert95 = 1.96*std

    res = int(np.sqrt(len(mu)))
    im0 = ax[0].imshow(mu.reshape(res, res),
                     origin='upper',
                     extent=[X_lower, X_upper, X_upper, X_lower],
                     vmin=min(mu), vmax=max(mu))
    patches_init = [Circle((bayes.init_params[i, 0], bayes.init_params[i, 1]),
                       radius=0.02, color='cyan') for i in range(ninit)]
    for patch in patches_init:
        ax[0].add_patch(patch)
    ax[0].add_patch(Circle((bayes.bayes_params[0, 0],
                                bayes.bayes_params[0, 1]),
                                radius=0.02, color='red'))
    fig.colorbar(im0, ax=ax[0])

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
        mu, cov = gp.predict(bayes.params_val, return_cov=True)
        _, std = gp.predict(bayes.params_val, return_std=True)
        mse = mean_squared_error(bayes.targets_val, mu)
        uncert = np.trace(cov)

        ii = i+1
        im = ax[ii].imshow(mu.reshape(res, res),
                          origin='upper',
                          extent=[X_lower, X_upper, X_upper, X_lower])
        patches_init = [Circle((bayes.init_params[i, 0], bayes.init_params[i, 1]),
                           radius=0.02, color='cyan') for i in range(ninit)]
        for patch in patches_init:
            ax[ii].add_patch(patch)
        patches_bayes = [Circle((bayes.bayes_params[j, 0], bayes.bayes_params[j, 1]),
                                 radius=0.02, color='orange') for j in range(i+1)]
        for patch in patches_bayes:
            ax[ii].add_patch(patch)
        if (i < len(bayes.bayes_targets)-1):
            ax[ii].add_patch(Circle((bayes.bayes_params[i+1, 0], bayes.bayes_params[i+1, 1]),
                                         radius=.02, color='red'))
        fig.colorbar(im, ax=ax[ii])

    if saveto is not None:
        plt.savefig(saveto + ".png", bbox_inches='tight')
    plt.show()



# def plot_gpr_evolution(bayes, X_lower, X_upper):
#
#     nbayes = len(bayes.bayes_targets)
#     ninit = len(bayes.init_targets)
#     nplots = nbayes + 1
#     fig, axs = plt.subplots(nplots, 2, figsize=(6, nplots*4), sharex=True)
#
#     if len(bayes.init_dtargets) > 0:
#         kernel = GPKernelDerAware()
#         kernel.theta = bayes.init_theta
#         gp = GaussianProcessRegressor(kernel=kernel, optimizer=None)
#         gp.fit(X=bayes.init_params, y=bayes.init_targets,
#                dX=bayes.init_params, dy=bayes.init_dtargets)
#     else:
#         kernel = GPKernel()
#         kernel.theta = bayes.init_theta
#         gp = GaussianProcessRegressor(kernel=kernel, optimizer=None)
#         gp.fit(X=bayes.init_params, y=bayes.init_targets)
#
#     mu, cov = gp.predict(bayes.params_val, return_cov=True)
#     _, std = gp.predict(bayes.params_val, return_std=True)
#     mse = mean_squared_error(bayes.targets_val, mu)
#     uncert = np.trace(cov)
#     mu = mu.ravel()
#     uncert95 = 1.96*std
#
#     min_val = np.min((bayes.targets_val))
#     max_val = np.max((bayes.targets_val))
#
#     res = int(np.sqrt(len(mu)))
#     axs[0][0].imshow(mu.reshape(res, res),
#                      origin='upper',
#                      extent=[X_lower, X_upper, X_upper, X_lower],
#                      vmin=min_val, vmax=max_val)
#     patches_init = [Circle((bayes.init_params[i, 0], bayes.init_params[i, 1]),
#                        radius=0.04, color='cyan') for i in range(ninit)]
#     for patch in patches_init:
#         axs[0][0].add_patch(patch)
#     axs[0][0].add_patch(Circle((bayes.bayes_params[0, 0],
#                                 bayes.bayes_params[0, 1]),
#                                 radius=.04, color='red'))
#
#     axs[0][1].imshow(std.reshape(res, res),
#                      origin='upper',
#                      extent=[X_lower, X_upper, X_upper, X_lower])
#
#
#     for i in range(nbayes):
#         if len(bayes.bayes_dtargets) > 0:
#             kernel = GPKernelDerAware()
#             kernel.theta = bayes.bayes_theta[i]
#             gp = GaussianProcessRegressor(kernel=kernel, optimizer=None)
#             gp.fit(X=bayes.params[:i+ninit+1, :], y=bayes.targets[:i+ninit+1, :],
#                    dX=bayes.params[:i+ninit+1, :], dy=bayes.dtargets[:i+ninit+1, :])
#         else:
#             kernel = GPKernel()
#             kernel.theta = bayes.bayes_theta[i]
#             gp = GaussianProcessRegressor(kernel=kernel, optimizer=None)
#             gp.fit(X=bayes.params[:i+ninit+1, :], y=bayes.targets[:i+ninit+1, :])
#         mu, cov = gp.predict(bayes.params_val, return_cov=True)
#         _, std = gp.predict(bayes.params_val, return_std=True)
#         mse = mean_squared_error(bayes.targets_val, mu)
#         uncert = np.trace(cov)
#
#         ii = i+1
#         axs[ii][0].imshow(mu.reshape(res, res),
#                           origin='upper',
#                           extent=[X_lower, X_upper, X_upper, X_lower],
#                           vmin=min_val, vmax=max_val)
#         patches_init = [Circle((bayes.init_params[i, 0], bayes.init_params[i, 1]),
#                            radius=0.04, color='cyan') for i in range(ninit)]
#         for patch in patches_init:
#             axs[ii][0].add_patch(patch)
#         patches_bayes = [Circle((bayes.bayes_params[j, 0], bayes.bayes_params[j, 1]),
#                                  radius=0.04, color='orange') for j in range(i+1)]
#         for patch in patches_bayes:
#             axs[ii][0].add_patch(patch)
#         if (i < len(bayes.bayes_targets)-1):
#             axs[ii][0].add_patch(Circle((bayes.bayes_params[i+1, 0], bayes.bayes_params[i+1, 1]),
#                                          radius=.04, color='red'))
#
#         axs[ii][1].imshow(std.reshape(res, res),
#                           origin='upper',
#                           extent=[X_lower, X_upper, X_upper, X_lower])
#
#     plt.show()
