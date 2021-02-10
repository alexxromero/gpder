"""Gaussian process regressor (GPR) with derivative observations."""

import warnings
from operator import itemgetter

import numpy as np
from sklearn import clone
from sklearn.base import BaseEstimator, MultiOutputMixin, RegressorMixin
from sklearn.utils.optimize import _check_optimize_result
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_random_state
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.optimize import minimize

from .kernel import DerivativeAwareKernel

__all__ = ['GaussianProcessRegressor']

class GaussianProcessRegressor(MultiOutputMixin,
                               RegressorMixin, BaseEstimator):
    """Gaussian process regressor (GPR) w/derivative observations.

    The GPR implementation is based on scikit-learn GaussianProcessRegressor
    and Algorithm 2.1 of Gaussian Processes for Machine Learning (GPML)
    by Rasmussen and Williams.

    The modification of the GPR to include derivative observations is
    based on the use of the DerivativeAwareKernel, as described in
    “Derivative observations in Gaussian process models of
    dynamic systems,” by E. Solak, R. Murray-Smith, W. E. Leithead,
    D. J. Leith, and C. E. Rasmussen, in Advances in Neural Information
    Processing Systems 15, (Vancouver, British Columbia, Canada), 2002.

    Args:
        kernel: kernel instance, default = DerivativeAwareKernel
            The kernel of the GPR. DerivativeAwareKernel is recommended.
            The parameters of the kernel (theta) are optimized during fitting
            unless the bounds are marked as "fixed".

        alpha: float or ndarray of shape (n_samples, ), default=1e-10
            Value added to the diagonal of the kernel matrix during fitting
            to prevent numerical issues by ensuring that the matrtix is
            positive definitte.

        optimizer: "fmin_l_bfgs_b" or callable, default="fmin_l_bfgs_b"
            Optimizing function for the kernel's parameters (theta).
            If "fmin_l_bfgs_b", the 'L-BGFS-B' algorithm from
            scipy.optimize.minimize is used. If None is passed, the
            theta values are kept fixed and no optimization is done.
            Alternatively, custom optimizer can be passed as a callable.
            Note that the current optimization method is based on
            minimizing the negative log marginal likelihood of the kernel
            without derivative observations.

        n_restarts_optimizer: int, default=0
            Number of times the optimizer is restarted. The first run of
            the optimizer is calculated on the kernel's initial theta.
            Subsequent runs (if n_restarts_optimizer > 0) are performed
            on parameter sets sampled from a log-uniform distribution
            randomly within the parameter bounds.
            If n_restarts_optimizer > 0, all parameter bounds must be finite.

        normalize_y: bool, default=False
            If true, the target values 'y' are normalized by setting the
            mean and standard deviation to one and zero respectively.
            Normalization is recommended for cases where zero-mean and
            unit-variance are used. The normalization is reversed before
            the GP predictions are returned.

        copy_data: bool, default=True
            If true, a copy of the training data is stored int he object.
            Else, a reference to the training data is stored.

        random_state: int, RandomState instance or None, default=None
            Determines the random number generator used to initialize
            the centers. For reproducible results, pass an int to be used as
            the random state seeed.

    Attributes:
        kernel_: kernel instance
            The kernel used to prediction.

        X_train_: array of shape (n_samples, n_features)
            Training data.

        y_train_: array of shape (n_samples,) or (n_samples, n_targets)
            Training target values.

        has_derinfo_: bool
            True if derivative observations are used in the GP regression.

        dX_train_: array of shape (n_samples, n_features)
            Training derivative data. Only present if 'dX' and 'dy' are
            passed during training.

        dy_train_: array of shape (n_samples,) or (n_samples, n_targets)
            Training derivative target values. Only present if 'dX' and 'dy'
            are passed during training.

        log_marginal_likelihood_value_: float
            Log marginal likelihood of the kernel's theta values.

        L_chol_: array of shape (n_samples, n_samples)
            Lower-triangular Cholesky decomposition of the kernel.

        alpha_chol_: array of shape (n_samples,)
            Dual coefficients of training samples in the kernel space.
    """

    def __init__(self, kernel, alpha=1e-10,
                 optimizer="fmin_l_bfgs_b", n_restarts_optimizer=0,
                 normalize_y=False, copy_data=True,
                 random_state=None):
        self.kernel = kernel
        self.alpha = alpha
        self.optimizer = optimizer
        self.n_restarts_opt = n_restarts_optimizer
        self.normalize_y = normalize_y
        self.copy_data = copy_data
        self._rng = check_random_state(random_state)

    def fit(self, X, y, dX=None, dy=None):
        """Fit the Gaussian process regression model.

        Args:
            X: array of shape (n_samples, n_features)
                Training data.

            y: array of shape (n_samples,) or (n_samples, n_targets)
                Training target values.

            dX: array of shape (n_samples, n_features), optional
                Training derivative data
                If None, no derivative information is included in the GP.

            dy: array of shape (n_samples,) or (n_samples, n_targets), optional
                Training derivative target values.
                If None, no derivative information is included in the GP.

        Returns:
            self: instance of self.
        """
        if self.kernel is None:
            self.kernel_ = DerivativeAwareKernel()
        else:
            self.kernel_ = clone(self.kernel)

        self.has_derinfo_ = False if (dX is None or dy is None) else True

        X, y = self._validate_data(X, y,
                                   multi_output=True, y_numeric=True,
                                   ensure_2d=True, dtype="numeric")

        self.X_train_ = np.copy(X) if self.copy_data else X
        self.y_train_ = np.copy(y) if self.copy_data else y

        self.alpha = self._validate_alpha(self.alpha)

        if self.has_derinfo_:
            dX, dy = self._validate_data(dX, dy,
                                         multi_output=True, y_numeric=True,
                                         ensure_2d=True, dtype="numeric")

            self.dX_train_ = np.copy(dX) if self.copy_data else dX
            self.dy_train_ = np.copy(dy) if self.copy_data else dy

        if self.normalize_y:
            self._y_train_mean = np.mean(self.y_train_ , axis=0)
            self._y_train_std = np.std(self.y_train_ , axis=0)
            self.y_train_ = self.y_train_  - self._y_train_mean
            self.y_train_ = self.y_train_ / self._y_train_std
            if self.has_derinfo_:
                self._dy_train_mean = np.mean(self.dy_train_ , axis=0)
                self._dy_train_std = np.std(self.dy_train_ , axis=0)
                self.dy_train_ = self.dy_train_  - self._dy_train_mean
                self.dy_train_ = self.dy_train_ / self._dy_train_std
        else:
            self._y_train_mean = np.zeros(1)
            self._y_train_std = 1
            self._dy_train_mean = np.zeros(1)
            self._dy_train_std = 1

        if self.optimizer is not None and self.kernel_.n_dims > 0:
            def obj_func(theta, eval_gradient=True):
                if eval_gradient:
                    lml, grad = self.log_marginal_likelihood(
                        theta, eval_gradient=True, clone_kernel=False)
                    return -lml, -grad
                else:
                    lml = self.log_marginal_likelihood(
                        theta, eval_gradient=False, clone_kernel=False)
                    return -lml

            # First optimization is evaluated on initial parameters.
            optima = [(self._constrained_optimization(obj_func,
                                                      self.kernel_.theta,
                                                      self.kernel_.bounds))]

            # Subsequent optimization runs are sampled from a
            # log-uniform randomly from the theta values within their bounds
            if self.n_restarts_opt > 0:
                if not np.isfinite(self.kernel_.bounds).all():
                    raise ValueError(
                        "n_restarts_optimizer > 0 requires that"
                        " all parameter bounds are finite.")
                bounds = self.kernel_.bounds
                for it in range(self.n_restarts_opt):
                    theta_init = self._rng.uniform(bounds[:, 0], bounds[:, 1])
                    optima.append(self._constrained_optimization(
                        obj_func, theta_init, bounds))

            # select the run which results in the minimal neg. log-marginal
            # likelihood
            lml_values = list(map(itemgetter(1), optima))
            self.kernel_.theta = optima[np.argmin(lml_values)][0]
            #self.kernel_._check_bounds_params()
            self.log_marginal_likelihood_value_ = -np.min(lml_values)
        else:
            self.log_marginal_likelihood_value_ = \
                self.log_marginal_likelihood(self.kernel_.theta,
                                             clone_kernel=False)

        if self.has_derinfo_:
            kernel_train = self.kernel_(X=self.X_train_, dX=self.dX_train_)
            kernel_train += self.alpha * np.eye((kernel_train.shape[0]))
            y_chol = np.concatenate((self.dy_train_, self.y_train_))
        else:
            kernel_train = self.kernel_(X=self.X_train_)
            kernel_train += self.alpha * np.eye((kernel_train.shape[0]))
            y_chol = self.y_train_

        try:
            self.L_chol_ = cholesky(kernel_train, lower=True)
        except np.linalg.LinAlgError as exc:
            print("The kernel is not positive definite. Try increasing alpha.")
            raise
        self.alpha_chol_ = cho_solve((self.L_chol_, True), y_chol)
        return self

    def predict(self, X, return_std=False):
        """Predict using the Gaussian process regression model.

        Predictions can also be made with an unfitted model by using the
        GP prior.

        Args:
            X: array of shape (n_samples, n_features)
                Query points where the GP is evaluated.

            return_std: bool, default=False
                If true, the standard deviation of the predictive
                distribution at the query points is returned, along with the
                mean and covariance.

        Returns:
            mu: array of shape (n_samples, [n_output_dims])
                Mean of the predictive distribution at the query points.

            cov: array of shape (n_samples, n_samples)
                Covariance of the predictive distribution at the query points.

            std: array of shape (n_samples,) optional
                Standard deviation of the predictive distribution at the
                query points. Only returned if 'return_std' is True.
        """

        X_star = check_array(X, ensure_2d=True, dtype="numeric")

        # If the model is not fitted, predict based on the GP prior
        if not hasattr(self, "X_train_"):
            if self.kernel is None:
                kernel = DerivativeAwareKernel()
            else:
                kernel = self.kernel

            mu = np.zeros(X_star.shape[0])
            cov = kernel(X=X_star)
            if return_std:
                std = np.sqrt(kernel.diag(X_star))
                return mu, cov, std
            else:
                return mu, cov
        else:
            # Predict based on the training kernel
            if self.has_derinfo_:
                k_xstardx = self.kernel_._kernel_rbf_wder(
                    X=X_star, dX=self.dX_train_, return_Kdxx=True).T
                k_xstarx = self.kernel_(X=X_star, Y=self.X_train_)
                kernel_post = np.block([k_xstardx, k_xstarx])
            else:
                kernel_post = self.kernel_(X=X_star, Y=self.X_train_)

            mu = kernel_post.dot(self.alpha_chol_)

            v = cho_solve((self.L_chol_, True), kernel_post.T)
            kernel_star = self.kernel_(X=X_star)
            cov = kernel_star - kernel_post.dot(v)

            # Undo normalization
            mu = self._y_train_std * mu + self._y_train_mean
            cov = cov * self._y_train_std**2

            if return_std:
                L_inv = solve_triangular(self.L_chol_.T,
                                         np.eye(self.L_chol_.shape[0]))
                K_inv = L_inv.dot(L_inv.T)
                std2 = np.copy(np.diagonal(self.kernel_(X=X_star)))
                std2 -= np.einsum("ij,ij->i",
                                        np.dot(kernel_post, K_inv),
                                        kernel_post)
                # If any std2 vals are neg, set them to 0.
                std2_neg = std2 < 0
                if np.any(std2_neg):
                    warnings.warn("Predicted variances smaller than 0. "
                                  "Setting those variances to 0.")
                    std2[std2_neg] = 0.0

                # Undo normalization
                std2 = std2 * self._y_train_std**2
                return mu, cov, np.sqrt(std2)
            else:
                return mu, cov

    def sample_y(self, X, nsamples=1, random_state=0):
        """Draw samples from a GP and evaluate at X.

        Args:
            X: array of shape (n_samples, n_features)
                Query points where the GP is evaluated.

            n_samples: int, default = 1
                Number of samples to draw from the GP.

            random_state: int, RandomState instance or None, default=0
                Determines the random number generator to use to randomly
                draw samples.

        Returns:
            y_samples: array of shape (n_samples_X, [n_output_dims], n_samples)
                Values of the n_samples drawn from the GP process and
                evaluated at 'X'.
        """
        rng = check_random_state(random_state)

        mu, cov = self.predict(X)
        if mu.ndim == 1:
            y_samples = rng.multivariate_normal(mu, cov, n_samples).T
        else:
            y_samples = [rng.multivariate_normal(mu[:, i], cov,
                n_samples).T[:, np.newaxis] for i in range(y_mean.shape[1])]
            y_samples = np.hstack(y_samples)
        return y_samples

    def log_marginal_likelihood(self, theta=None, eval_gradient=False,
                                clone_kernel=True):
        """Log marginal likelihood (lml) of the theta values used for training.
        Note that the lml is evaluated only for training data without
        derivative information.

        Args:
            theta: array of shape (n_kernel_params,) default=None
                Kernel parameters on which the lml is evaluated.

            eval_gradient: bool, default=False
                If true, the gradient of the lml with respect to the
                parameters is also returned. Can only be true if theta is None.

            clone_kernel: bool, default=True
                If true, the kerneel attribute is copied.

        Returns:
            log_likelihood: float
                Log-marginal likelihood of theta.

            log_likelihood_gradient: ndarray of shape (n_kernel_params,),
                optional
                Gradient of the lml. Only evaluated and returned if
                eval_gradient is true.
        """
        if theta is None:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated for "
                                 " theta!=None")
            return self.log_marginal_likelihood_value_

        if clone_kernel:
            kernel = self.kernel_.clone_with_theta(theta)
        else:
            kernel = self.kernel_
            kernel.theta = theta

        if eval_gradient:
            #if self.has_derinfo_:
            #    K, K_gradient = kernel(X=self.X_train_, dX=self.dX_train_,
            #                           eval_gradient=True)
            #else:
            #    K, K_gradient = kernel(X=self.X_train_, eval_gradient=True)
            K, K_gradient = kernel(X=self.X_train_, eval_gradient=True)
        else:
            #if self.has_derinfo_:
            #    K = kernel(X=self.X_train_, dX=self.dX_train_,
            #               eval_gradient=False)
            #else:
            #    K = kernel(X=self.X_train_, eval_gradient=False)
            K = kernel(X=self.X_train_, eval_gradient=False)

        K[np.diag_indices_from(K)] += self.alpha
        try:
            L = cholesky(K, lower=True)
        except np.linalg.LinAlgError:
            return (-np.inf, np.zeros_like(theta)) \
                if eval_gradient else -np.inf

        #if self.has_derinfo_:
        #    y_chol = np.concatenate((self.dy_train_, self.y_train_))
        #else:
        #    y_chol = self.y_train_
        y_chol = self.y_train_
        if y_chol.ndim == 1:
            y_chol = y_chol[:, np.newaxis]

        alpha = cho_solve((L, True), y_chol)

        log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", y_chol, alpha)
        log_likelihood_dims -= np.log(np.diag(L)).sum()
        log_likelihood_dims -= K.shape[0] / 2 * np.log(2 * np.pi)
        log_likelihood = log_likelihood_dims.sum(-1)  # sum over dimensions

        if eval_gradient:  # compare Equation 5.9 from GPML
            tmp = np.einsum("ik,jk->ijk", alpha, alpha)  # k: output-dimension
            tmp -= cho_solve((L, True), np.eye(K.shape[0]))[:, :, np.newaxis]
            log_likelihood_gradient_dims = \
                0.5 * np.einsum("ijl,jik->kl", tmp, K_gradient)
            log_likelihood_gradient = log_likelihood_gradient_dims.sum(-1)

        if eval_gradient:
            return log_likelihood, log_likelihood_gradient
        else:
            return log_likelihood

    def _validate_alpha(self, alpha):
        if np.iterable(alpha) and alpha.shape[0] != self.y_train_.shape[0]:
            if alpha.shape[0] == 1:
                alpha = alpha[0]
            else:
                raise ValueError("alpha must be a scalar or an array"
                                 " with same number of entries as y.(%d != %d)"
                                 % (alpha.shape[0], self.y_train_.shape[0]))
        return alpha

    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        if self.optimizer == "fmin_l_bfgs_b":
            opt_res = minimize(
                obj_func,
                initial_theta,
                method="L-BFGS-B",
                jac=True,
                options={'maxiter':1000},
                bounds=bounds)
            _check_optimize_result("lbfgs", opt_res)
            theta_opt, func_min = opt_res.x, opt_res.fun
        elif callable(self.optimizer):
            theta_opt, func_min = \
                self.optimizer(obj_func, initial_theta, bounds=bounds)
        else:
            raise ValueError("Unknown optimizer %s." % self.optimizer)
        return theta_opt, func_min
