import numpy as np
import scipy
from sklearn.utils import check_random_state

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
        for input, target in zip(X, y, strict=False):
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

    X_util : array-like, shape (n_samples_util, n_feat)
        Utility input.

    norm : float
        Normalization factor.
    """

    def __init__(self, gp, X_util, norm):
        self.gp = gp
        self.X_util = X_util
        self.norm = norm

        self._util_cache = {
            "X_util": X_util,
            "gp": gp,
            "norm": norm,
            "kernel_star_trace": None,
            "posterior_covariance": None,
        }

    def utility(self, X, batch_size=512):
        """Utility function. Computes the net loss in the predictive variance.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_feat)
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

    def approx_utility(self, X, batch_size=512):
        """Approximate utility function using cached values.
        It relies on the trace property Tr(A+B) = Tr(A) + Tr(B).
        This function is faster than utility() in higher dimensions.

        See utility() for parameters and returns.
        """
        X_new = np.atleast_2d(X)
        dX_new = X_new if self.gp._has_gradients else None

        train_cov_exp = self.gp._expand_training_covariance(X_new=X_new, dX_new=dX_new)

        try:
            L_chol_exp = scipy.linalg.cholesky(
                train_cov_exp + self.gp.kappa * np.eye(train_cov_exp.shape[0]),
                lower=True,
            )
        except scipy.linalg.LinAlgError as err:
            raise ValueError("The covariance matrix is not positive definite.") from err

        post_cov = self._calculate_posterior_covariance()
        post_cov_exp = self._expand_posterior_covariance(post_cov, X_new, dX_new)
        v = scipy.linalg.cho_solve((L_chol_exp, True), post_cov_exp.T)

        kernel_star_trace = self._calculate_kernel_star_trace(X_new)
        pred_cov_trace = kernel_star_trace - np.trace(post_cov_exp.dot(v))

        return 1 - pred_cov_trace / self.norm

    def utility_quick(self, X, batch_size=512):
        X_new = np.atleast_2d(X)
        dX_new = X_new if self.gp._has_gradients else None

        train_cov_exp = self.gp._expand_training_covariance(X_new, dX_new)

        try:
            L_chol_exp = scipy.linalg.cholesky(
                train_cov_exp + self.gp.kappa * np.eye(train_cov_exp.shape[0]),
                lower=True,
            )
        except scipy.linalg.LinAlgError as err:
            raise ValueError("The covariance matrix is not positive definite.") from err

        post_cov = self._calculate_posterior_covariance()
        post_cov_exp = self._expand_posterior_covariance(post_cov, X_new, dX_new)
        v = scipy.linalg.cho_solve((L_chol_exp, True), post_cov_exp.T)

        kernel_star_trace = self._calculate_kernel_star_trace(X_new)
        pred_cov_trace = kernel_star_trace - np.trace(post_cov_exp.dot(v))

        return 1 - pred_cov_trace / self.norm

    def _calculate_kernel_star_trace(self, X):
        if self._util_cache["kernel_star_trace"] is None:
            kernel_star = self.gp.kernel._cov_yy(self._util_cache["X_util"], add_noise=False)
            self._util_cache["kernel_star_trace"] = np.trace(kernel_star)
        return self._util_cache["kernel_star_trace"]

    def _expand_posterior_covariance(self, post_cov, new_X, new_dX=None):
        n = self.gp.X_train.shape[0]
        post_cov_yy = post_cov[:, :n]
        post_cov_yy_exp = np.block(
            [
                post_cov_yy,
                self.gp.kernel._cov_yy(X=self.X_util, Y=new_X, add_noise=False),
            ]
        )
        if self.gp._has_gradients and new_dX is not None:
            n_odX = self.gp.dX_train.shape[0]
            n_ndX = new_dX.shape[0]
            post_cov_wy = post_cov[:, n:].T
            post_cov_wy_new = self.gp.kernel._cov_wy(dX=new_dX, Y=self.X_util, idx=self.gp.idx)
            post_cov_wy_exp = np.zeros(
                ((n_odX + n_ndX) * self.gp._n_dims_der, self.X_util.shape[0])
            )
            for i in range(self.gp._n_dims_der):
                low = i * (n_odX + n_ndX)
                post_cov_wy_exp[low : low + n_odX, :] = post_cov_wy[i * n_odX : (i + 1) * n_odX, :]
                post_cov_wy_exp[low + n_odX : low + n_odX + n_ndX, :] = post_cov_wy_new[
                    i * n_ndX : (i + 1) * n_ndX, :
                ]
            posterior_cov = np.block([post_cov_yy_exp, post_cov_wy_exp.T])
        else:
            posterior_cov = post_cov_yy_exp
        return posterior_cov

    def _calculate_posterior_covariance(self):
        if self._util_cache["posterior_covariance"] is None:
            post_cov_yy = self.gp.kernel._cov_yy(X=self.X_util, Y=self.gp.X_train, add_noise=False)
            if self.gp._has_gradients:
                post_cov_wy = self.gp.kernel._cov_wy(
                    dX=self.gp.dX_train, Y=self.X_util, idx=self.gp.idx
                )
                posterior_cov = np.block([post_cov_yy, post_cov_wy.T])
            else:
                posterior_cov = post_cov_yy
            self._util_cache["posterior_covariance"] = posterior_cov
        return self._util_cache["posterior_covariance"]


class GPUncertaintyOptimizer:
    """The GPUncertaintyOptimizer class performs bayesian optimization
    for experimental design with the goal of minimizing the net predictive
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
        self._has_gradients = der_function is not None
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
        optimizer="L-BFGS-B",
        n_restarts_optimizer=3,
        use_approx_acq=False,
        batch_size=512,
    ):
        """Minimize the net predictive variance of the GP model.

        Parameters
        ----------
        X_util : array-like, shape (n_samples_util, n_feat)
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

        optimizer : 'L-BFGS-B' or callable, default='L-BFGS-B'
            The optimizer.

        n_restarts_optimizer : int, default=3
            The number of times to restart for the optimizer.

        use_approx_acq : bool, default=False
            Whether to use the approximate acquisition function.
            Faster in higher dimensions.

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
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.use_approx_acq = use_approx_acq
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

        acq = NetVarianceLoss(self.gp_model, X_util, norm)

        def neg_acq_fun(X):
            nac = (
                -acq.approx_utility(X, self.batch_size)
                if self.use_approx_acq
                else -acq.utility(X, self.batch_size)
            )
            return nac

        X0 = self.random_state.uniform(
            self._param_bounds[:, 0],
            self._param_bounds[:, 1],
            size=(self._param_dim,),
        )
        X_opt, min_neg_util = self._optimize_acq_fun(neg_acq_fun, X0)
        if self.n_restarts_optimizer > 0:
            for j in range(self.n_restarts_optimizer):
                X0 = self.random_state.uniform(
                    self._param_bounds[:, 0],
                    self._param_bounds[:, 1],
                    size=(self._param_dim,),
                )
                X, neg_util = self._optimize_acq_fun(neg_acq_fun, X0)
                if neg_util < min_neg_util:
                    X_opt, min_neg_util = X, neg_util
        return X_opt.reshape(1, self._param_dim), -min_neg_util

    def _optimize_acq_fun(self, neg_acq_fun, X0):
        if self.optimizer == "L-BFGS-B":
            res = scipy.optimize.minimize(
                neg_acq_fun, X0, bounds=self._param_bounds, method="L-BFGS-B"
            )
            X_opt, min_neg_util = res.x, res.fun
        elif callable(self.optimizer):
            X_opt, min_neg_util = self.optimizer(neg_acq_fun, X0)
        else:
            raise ValueError("Invalid optimizer.")
        return X_opt, min_neg_util
