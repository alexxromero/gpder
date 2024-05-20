import numpy as np
from sklearn.utils import check_random_state
import scipy
import time

# Functions from: http://krasserm.github.io/2018/03/21/bayesian-optimization/
# https://github.com/thuijskens/bayesian-optimization/blob/master/python/gp.py

__all__ = ["GPUncertaintyOptimizer", "NetVarianceLoss"]


class PrintLog:
    """The PrintLog class prints a table with the input/output values
    selected by the GPUncertaintyOptimizer's minimize_variance method.
    """

    def __init__(self, keys):
        self.keys = keys
        col_width = max(10, max(len(k) for k in keys))
        self.header = "| Iter | "
        self.header += " | ".join([f"{k:^{col_width}}" for k in keys])
        self.header += " | {} |".format("Target".center(10, " "))
        self.header += "\n" + "-" * (len(self.header))
        print(self.header)

    def update_log(self, X, y, iter):
        for input, target in zip(X, y):
            row = f"| {iter:^4} | "
            row += " | ".join([f"{x:^10.2f}" for x in input])
            row += f" | {target[0]:^10.2f} |"
            print(row)


class NetVarianceLoss:
    """The NetVarianceLoss class computes the net loss in the
    predictive variance of a GP model if a new training sample were
    to be added at input X.

    Parameters
    ----------
    gp : GaussianProcessRegressor object
        The Gaussian process model.

    X_util : array-like, shape (n_samples_util, n_features)
        Utility input.

    norm : float
        Normalization factor.
    """

    def __init__(self, gp, X_util, norm):
        self.gp = gp
        self.X_util = X_util
        self.norm = norm

    def utility(self, X, batch_size=512):
        """Utility function. Computes the net loss in the predictive variance.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input samples.

        batch_size : int, default=512
            Prediction batch size.

        Returns
        -------
        float
            The net loss in the predictive variance.
        """
        X = np.atleast_2d(X)
        n_batches = int(np.ceil(len(self.X_util) / batch_size))
        cov_trace = 0.0
        for i in range(n_batches):
            low = i * batch_size
            high = (i + 1) * batch_size
            X_util_batch = self.X_util[low:high]
            if self.gp._has_gradients:
                cov = self.gp.predict_using_query(
                    X_predict=X_util_batch,
                    X_train_query=X,
                    dX_train_query=X,
                    return_mean=False,
                    return_cov=True,
                )
            else:
                cov = self.gp.predict_using_query(
                    X_predict=X_util_batch,
                    X_train_query=X,
                    return_mean=False,
                    return_cov=True,
                )
            cov_trace += np.trace(cov)
        return 1 - cov_trace / self.norm


class GPUncertaintyOptimizer:
    """The GPUncertaintyOptimizer class performs bayesian optimization
    for experimental design with the goal of minizing the net predictive
    variance of a GP model.

    Parameters
    ----------
    gp_model : GaussianProcessRegressor object
        The Gaussian process model.

    bounds : dict
        The parameter bounds.

    function : callable
        The objective function.

    der_function : callable, default=None
        The derivative function.

    random_state : int, RandomState instance or None, default=None
        State of the random number generator.

    verbose : bool, default=True
        Whether to print the input/output values selected by the optimizer.
    """

    def __init__(
        self,
        gp_model,
        bounds,
        function,
        der_function=None,
        random_state=None,
        verbose=True,
    ):
        self.gp_model = gp_model
        self.bounds = bounds
        self.function = function
        self.der_function = der_function
        self.random_state = check_random_state(random_state)
        self.verbose = verbose
        self._has_gradients = False if der_function is None else True
        self._param_keys = list(self.bounds.keys())
        self._param_bounds = np.array(list(self.bounds.values()))
        self._param_dim = len(self._param_keys)

    def minimize_variance(
        self,
        X_util,
        n_iters,
        added_noise="gaussian",
        gamma=0,
        acquisition_function=NetVarianceLoss,
        acquisition_function_args=(),
        optimizer="L-BFGS-B",
        n_restarts_optimizer=3,
        batch_size=512,
    ):
        """Minimize the net predictive variance of the GP model.

        Parameters
        ----------
        X_util : array-like, shape (n_samples_util, n_features)
            Utility input.

        n_iters : int
            Number of iterations.

        added_noise : 'gaussian', None or callable
            Noise added to the utility input. If 'gaussian', the noise
            is Gaussian with mean 0 and variance gamma^2. If None, no noise
            is added. Else if callable, the noise is added using the callable.

        gamma : float, default=0
            Squared root of the noise variance.

        acquisition_function : callable, default=NetVarianceLoss
            The acquisition function.

        acquisition_function_args : tuple, default=()
            Additional arguments to the acquisition function.

        optimizer : 'L-BFGS-B' or callable, default='L-BFGS-B'
            The optimizer.

        n_restarts_optimizer : int, default=3
            The number of times to restart for the optimizer.

        batch_size : int, default=512
            Prediction batch size.

        Returns
        -------
        GaussianProcessRegressor object
            The updated GP model.
        """
        self.X_util = X_util
        self.n_iters = n_iters
        self.added_noise = added_noise
        self.gamma = gamma
        self.acquisition_function = acquisition_function
        self.acquisition_function_args = acquisition_function_args
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.batch_size = batch_size

        self.X_init = self.gp_model.X_train
        self.y_init = self.gp_model.y_train
        self.n_init = len(self.y_init)
        if self._has_gradients:
            self.dX_init = self.gp_model.dX_train
            self.dy_init = self.gp_model.dy_train
            self.nd_init = len(self.dy_init)

        if self.verbose:
            plog = PrintLog(self._param_keys)
            plog.update_log(X=self.X_init, y=self.y_init, iter=0)

        for i in range(self.n_iters):
            X_next, _ = self._find_next()
            y_next = self.function(X_next).reshape(1, 1)
            if self.gp_model._has_gradients:
                dy_next = self.der_function(X_next).reshape(1, -1)
                self.gp_model.update(X_next, y_next, X_next, dy_next)
            else:
                self.gp_model.update(X_next, y_next)
            if self.verbose:
                plog.update_log(X=X_next, y=y_next, iter=i + 1)
        return self.gp_model

    def _current_net_variance(self):
        n_batches = int(np.ceil(len(self.X_util) / self.batch_size))
        cov_trace = 0.0
        for i in range(n_batches):
            low = i * self.batch_size
            high = (i + 1) * self.batch_size
            X_util_batch = self.X_util[low:high]
            _, cov = self.gp_model.predict(X_util_batch, return_cov=True)
            cov_trace += np.trace(cov)
        return cov_trace

    def _find_next(self):
        if self.added_noise is None:
            X_util = self.X_util
        elif self.added_noise == "gaussian":
            X_util = self.X_util + self.random_state.normal(
                scale=self.gamma**2, size=self.X_util.shape
            )
        elif callable(self.added_noise):
            X_util = self.X_util + self.added_noise(self.X_util)
        else:
            raise ValueError("Invalid noise type.")

        for i in range(self._param_dim):
            lt_ix = X_util[:, i] < self._param_bounds[i, 0]
            X_util[lt_ix, i] = self._param_bounds[i, 1] - np.abs(
                X_util[lt_ix, i] - self._param_bounds[i, 0]
            )
            gt_ix = X_util[:, i] > self._param_bounds[i, 1]
            X_util[gt_ix, i] = self._param_bounds[i, 0] + np.abs(
                X_util[gt_ix, i] - self._param_bounds[i, 1]
            )

        self._X_util_temp = X_util

        norm = self._current_net_variance()

        if self.acquisition_function == NetVarianceLoss:
            acq = NetVarianceLoss(self.gp_model, X_util, norm)

            def neg_acq_fun(X):
                return -acq.utility(X)

        else:

            def neg_acq_fun(*args, **kwargs):
                return -self.acquisition_function(*args, **kwargs)

        X0 = self.random_state.uniform(
            self._param_bounds[:, 0],
            self._param_bounds[:, 1],
            size=(1, self._param_dim),
        )
        X_opt, min_neg_util = self._optimize_acq_fun(
            neg_acq_fun, X0, self.acquisition_function_args
        )
        if self.n_restarts_optimizer > 0:
            for j in range(self.n_restarts_optimizer):
                X0 = self.random_state.uniform(
                    self._param_bounds[:, 0],
                    self._param_bounds[:, 1],
                    size=(1, self._param_dim),
                )
                X, neg_util = self._optimize_acq_fun(
                    neg_acq_fun, X0, self.acquisition_function_args
                )
                if neg_util < min_neg_util:
                    X_opt, min_neg_util = X, neg_util
        return X_opt.reshape(1, self._param_dim), -min_neg_util

    def _optimize_acq_fun(self, neg_acq_fun, X0, args):
        if self.optimizer == "L-BFGS-B":
            res = scipy.optimize.minimize(
                neg_acq_fun, X0, args=args, bounds=self._param_bounds, method="L-BFGS-B"
            )
            X_opt, min_neg_util = res.x, res.fun
        elif callable(self.optimizer):
            X_opt, min_neg_util = self.optimizer(neg_acq_fun, X0, args)
        else:
            raise ValueError("Invalid optimizer.")
        return X_opt, min_neg_util
