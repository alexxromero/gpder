import itertools
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np


def download_dataset():
    # Download here or directly from zenodo. DOI: 10.5281/zenodo.10971439
    # url: https://zenodo.org/records/10971439
    # by default, the data is saved in the current directory
    try:
        import zenodo_get

        os.system("zenodo_get -d 10.5281/zenodo.10971439")
    except:
        print(
            "Downloading the dataset failed. Please download the dataset manually from zenodo. DOI: 10.5281/zenodo.10971439"
        )


def load_dataset():
    filename = "./three_jets_30k.h5"
    with h5py.File(filename, "r") as f:
        J1_threeM = np.array(f["j1_threeM"])
        J2_threeM = np.array(f["j2_threeM"])
        J3_threeM = np.array(f["j3_threeM"])
        threeM = np.stack((J1_threeM, J2_threeM, J3_threeM), axis=1)
    return threeM


def calculate_mse_uncertainty(gp_model, X_test, y_test, batch_size=512):
    """Calculate the mean squared error and the net predictive variance of
    the GP model.
    """
    n_batches = int(np.ceil(len(y_test) / batch_size))
    y_pred = np.zeros(len(y_test))
    cov_trace = 0.0
    for i in range(n_batches):
        low = i * batch_size
        high = (i + 1) * batch_size
        X_test_batch = X_test[low:high]
        y_pred_batch, cov_batch = gp_model.predict(X_test_batch, return_cov=True)
        y_pred[low:high] = y_pred_batch.ravel()
        cov_trace += np.trace(cov_batch)
    mse = np.mean((y_test - y_pred) ** 2)
    return mse, cov_trace


def predict_mean_std(gp_model, X_test, batch_size=512):
    n_batches = int(np.ceil(len(X_test) / batch_size))
    y_pred = np.zeros(len(X_test))
    std = np.zeros(len(X_test))
    for i in range(n_batches):
        low = i * batch_size
        high = (i + 1) * batch_size
        X_test_batch = X_test[low:high]
        y_pred_batch, std_batch = gp_model.predict(X_test_batch, return_std=True)
        y_pred[low:high] = y_pred_batch.ravel()
        std[low:high] = std_batch.ravel()
    return y_pred, std


def plot_regression(gp_model, X_test, y_test, X_train_init=9):
    res_test = int(len(y_test) ** (1 / 4))
    X_lower = np.min(X_test[:, 0])
    X_upper = np.max(X_test[:, 0])
    y_pred_mean, y_pred_std = predict_mean_std(gp_model, X_test)
    X_train = gp_model.X_train

    param_combinations = list(itertools.combinations(np.arange(4), 2))
    params_labels = [
        "$\\nu_{J_1 in}$",
        "$\\nu_{J_1 out}$",
        "$\\nu_{J_{23} in}$",
        "$\\nu_{J_{23} out}$",
    ]
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = "10"
    fig, axs = plt.subplots(6, 3, figsize=(6, 9), sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.8, hspace=0.8)
    val = [0.5]

    for i, fixed_params in enumerate(param_combinations):
        ix = np.where((X_test[:, fixed_params[0]] == val) & (X_test[:, fixed_params[1]] == val))
        # -- truth -- #
        vmin = np.min(y_test[ix])
        vmax = np.max(y_test[ix])
        im_truth = axs[i][0].imshow(
            y_test[ix].reshape(res_test, res_test),
            extent=[X_lower, X_upper, X_upper, X_lower],
            origin="upper",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        fig.colorbar(im_truth, ax=axs[i][0])
        # -- mean prediction -- #
        im_pred = axs[i][1].imshow(
            y_pred_mean[ix].reshape(res_test, res_test),
            extent=[X_lower, X_upper, X_upper, X_lower],
            origin="upper",
            cmap="viridis",
        )
        fig.colorbar(im_pred, ax=axs[i][1])

        # -- std prediction -- #
        im_std = axs[i][2].imshow(
            y_pred_std[ix].reshape(res_test, res_test),
            extent=[X_lower, X_upper, X_upper, X_lower],
            origin="upper",
            cmap="Oranges",
        )
        fig.colorbar(im_std, ax=axs[i][2])

        # -- labels -- #
        varying_params = [x for x in range(4) if x not in fixed_params]
        axs[i][0].set_ylabel(params_labels[varying_params[1]])
        axs[i][0].set_xlabel(params_labels[varying_params[0]])
        axs[i][0].set_title(
            f"{params_labels[fixed_params[0]]}, {params_labels[fixed_params[1]]} = {val[0]}",
        )

        axs[i][1].set_ylabel(params_labels[varying_params[1]])
        axs[i][1].set_xlabel(params_labels[varying_params[0]])
        axs[i][1].set_title(
            f"{params_labels[fixed_params[0]]}, {params_labels[fixed_params[1]]} = {val[0]}",
        )

        axs[i][2].set_ylabel(params_labels[varying_params[1]])
        axs[i][2].set_xlabel(params_labels[varying_params[0]])
        axs[i][2].set_title(
            f"{params_labels[fixed_params[0]]}, {params_labels[fixed_params[1]]} = {val[0]}",
        )

        if len(X_train) > X_train_init:
            axs[i][1].scatter(
                X_train[:X_train_init, fixed_params[0]],
                X_train[:X_train_init, fixed_params[1]],
                c="red",
                s=10,
                label="Initial training data",
            )
            axs[i][1].scatter(
                X_train[X_train_init:, fixed_params[0]],
                X_train[X_train_init:, fixed_params[1]],
                c="hotpink",
                s=10,
                label="BED-selected training data",
            )
            axs[i][2].scatter(
                X_train[:X_train_init, fixed_params[0]],
                X_train[:X_train_init, fixed_params[1]],
                c="red",
                s=10,
            )
            axs[i][2].scatter(
                X_train[X_train_init:, fixed_params[0]],
                X_train[X_train_init:, fixed_params[1]],
                c="hotpink",
                s=10,
            )
        else:
            axs[i][1].scatter(
                X_train[:, fixed_params[0]],
                X_train[:, fixed_params[1]],
                c="red",
                s=10,
                label="Training data",
            )
            axs[i][2].scatter(
                X_train[:, fixed_params[0]], X_train[:, fixed_params[1]], c="red", s=10
            )

    # -- titles -- #
    axs[0][0].text(0.6, 0, "Truth", fontsize=12)
    axs[0][1].text(0.2, 0, "Mean Prediction", fontsize=12)
    axs[0][2].text(0.8, 0, "Std", fontsize=12)

    # -- legend -- #
    if len(X_train) > X_train_init:
        axs[0][1].legend(loc=(-2.5, 1.8), ncol=2)
    else:
        axs[0][1].legend(loc=(2, 1.8), ncol=1)
    plt.show()
