import os
import numpy as np
import matplotlib.pyplot as plt
import h5py


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


def plot_regression(gp_model, X_test, y_test, X_train_init=5):
    y_pred, y_std = gp_model.predict(X_test, return_std=True)

    X_train = gp_model.X_train

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = "10"

    fig, ax = plt.subplots(1, 3, figsize=(8, 2))
    plt.subplots_adjust(wspace=0.7)

    res_test = int(np.sqrt(len(y_test)))
    X_lower = np.min(X_test[:, 0])
    X_upper = np.max(X_test[:, 0])

    im0 = ax[0].imshow(
        y_test.reshape(res_test, res_test),
        origin="upper",
        extent=(X_lower, X_upper, X_lower, X_upper),
        vmin=0.0,
        vmax=1.1,
    )
    fig.colorbar(im0, ax=ax[0], shrink=0.73)
    ax[0].set_title("True efficiency")
    ax[0].set_xlabel(r"$\nu_{J1}$")
    ax[0].set_ylabel(r"$\nu_{J23}$")

    im1 = ax[1].imshow(
        y_pred.reshape(res_test, res_test),
        origin="upper",
        extent=(X_lower, X_upper, X_lower, X_upper),
        vmin=0.0,
        vmax=1.1,
    )
    fig.colorbar(im1, ax=ax[1], shrink=0.73)
    ax[1].set_title("Predicted mean")
    ax[1].set_xlabel(r"$\nu_{J1}$")
    ax[1].set_ylabel(r"$\nu_{J23}$")

    im2 = ax[2].imshow(
        y_std.reshape(res_test, res_test),
        origin="upper",
        extent=(X_lower, X_upper, X_lower, X_upper),
        cmap="Oranges",
    )
    fig.colorbar(im2, ax=ax[2], shrink=0.73)
    ax[2].set_title("Standard deviation")
    ax[2].set_xlabel(r"$\nu_{J1}$")
    ax[2].set_ylabel(r"$\nu_{J23}$")

    if len(X_train) > X_train_init:
        ax[1].scatter(
            X_train[:X_train_init, 0],
            X_train[:X_train_init, 1],
            color="r",
            s=10,
            label="Initial training data",
        )
        ax[1].scatter(
            X_train[X_train_init:, 0],
            X_train[X_train_init:, 1],
            color="hotpink",
            s=10,
            label="BED-selected training data",
        )

        ax[2].scatter(X_train[:X_train_init, 0], X_train[:X_train_init, 1], color="r", s=10)
        ax[2].scatter(X_train[X_train_init:, 0], X_train[X_train_init:, 1], color="hotpink", s=10)

        ax[1].legend(loc=(-0.5, 1.3), ncol=2)
    else:
        ax[1].scatter(X_train[:, 0], X_train[:, 1], color="r", s=10, label="Training data")
        ax[2].scatter(X_train[:, 0], X_train[:, 1], color="r", s=10)

    plt.show()
