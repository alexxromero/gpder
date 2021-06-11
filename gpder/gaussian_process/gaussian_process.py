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

from .deraware_kernel import GPKernelDerAware
from .kernel import GPKernel
from .utils import _atleast2d

__all__ = ['GaussianProcessRegressor']

class GaussianProcessRegressor(MultiOutputMixin, RegressorMixin, BaseEstimator):
    """Gaussian process regressor (GPR) w/derivative observations.

    The GPR implementation is based on scikit-learn
    GaussianProcessRegressor and Algorithm 2.1 of Gaussian Processes
    for Machine Learning (GPML) by Rasmussen and Williams [1].

    The modification of the GPR to include derivative observations is
    based on the use of the hybrid kernel DerAwareKernel, which
    combines the covariances of function and derivative observations,
    as described in [2] and [3].

    Parameters
    ----------
    kernel: kernel instance, default=None
        The kernel used in the GPR.
        GPKernel is recommended for regular GPR.
        GPKernelDerAware is recommended for GPR with derivative info.
        The hyperparameters of the kernel (theta) are optimized during
        fitting unless the bounds are marked as "fixed".
        The log of the kernel's hyperparameters are stored in the
        vector 'theta'.

    alpha: float or ndarray of shape (n_samples,), default=1e-6
        Value added to the diagonal of the kernel matrix during
        fitting to prevent numerical issues by ensuring that the
        matrtix is positive definitte.

    optimizer: "fmin_l_bfgs_b" or callable, default="fmin_l_bfgs_b"
        Optimizing function for the kernel's parameters (theta).
        If "fmin_l_bfgs_b", the 'L-BGFS-B' method from
        scipy.optimize.minimize is used. If None is passed, the
        theta values are kept fixed and no optimization is done.
        Alternatively, a custom optimizer can be passed as a callable.

    n_restarts_optimizer: int, default=0
        Number of times to restart the optimizer which finds the set of
        'theta' parameters that minimize the neg log-marginal
        likelihood. The first run of the optimizer is initialized
        on the kernel's initial 'theta' values. Subsequent runs
        (if n_restarts_optimizer > 0) are initialized on values
        sampled from a log-uniform distribution randomly within the
        parameter bounds. All parameter bounds must be finite.

    normalize_y: bool, default=False
        If true, the target values 'y' are normalized by setting the
        mean and standard deviation to one and zero respectively.
        Normalization is recommended for cases where zero-mean and
        unit-variance are used. The normalization is reversed before
        the GP predictions are returned.

    copy_data: bool, default=True
        If true, a copy of the training data is stored int the object.
        Else, a reference to the training data is stored.

    random_state: int, RandomState instance or None, default=None
        Determines the random number generator used to initialize
        the centers. For reproducible results, pass an int to be used
        as the random state seeed.

    Attributes
    ----------
    kernel_: kernel instance
        The kernel used in the GPR.

    X_train_: ndarray of shape (nsamples, nfeatures)
        Coordinates of the training observation points.

    y_train_: ndarray of shape (nsamples,)
        Target values observed at 'X_train_'.

    dX_train_: ndarray of shape (dnsamples, nfeatures), optional
        Coordinates of the training derivative observation points.

    dy_train_: ndarray of shape (dnsamples, ndfeatures), optional
        Partial derivatives of the target values wrt to the
        hyperparameters, observed at 'dX_train_'.

    has_derinfo_: bool
        True if derivative information is used when fitting the GPR.

    log_marginal_likelihood_value_: float
        Log marginal likelihood of the hyperparameters.

    L_chol_: ndarray of shape (nsamples, nsamples)
        Lower-triangular Cholesky decomposition of the kernel.

    alpha_chol_: ndarray of shape (nsamples,)
        Dual coefficients of the training samples in the kernel space.

    References
    ----------
    [1] https://github.com/scikit-learn/scikit-learn/blob/95119c13a/sklearn/gaussian_process/_gpr.py#L23

    [2] E. Solak, R. Murray-Smith, W. E. Leithead, D. J. Leith,
        and C. E. Rasmussen, “Derivative observations in Gaussian
        process models of dynamic systems,” in Advances in Neural
        Information Processing Systems 15, (Vancouver,
        British Columbia, Canada), 2002.

    [3] X. Yang, B. Peng, H. Zhou, and L. Yang, "State Estimation
        for Nonlinear Dynamic Systems using Gaussian Processes and
        Pre-computed Local Linear Models"
        (http://ieeexplore.ieee.org/document/7829090/ "IEEE Xplore"),
        2016 IEEE Chinese Guidance, Navigation and Control
        Conference (CGNCC), Nanjing, 2016, pp. 1963-1968.
    """

    def __init__(self, kernel=None, alpha=1e-10,
                 optimizer="fmin_l_bfgs_b", n_restarts_optimizer=0,
                 normalize_y=False, copy_data=True,
                 random_state=None):
        self.kernel = kernel
        self.alpha = alpha
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.normalize_y = normalize_y
        self.copy_data = copy_data
        self.random_state = random_state

    def fit(self, X, y, dX=None, dy=None):
        """Fit the Gaussian process regression model.

        Parameters
        ----------
        X_train_: ndarray of shape (nsamples, nfeatures)
            Coordinates of the training observation points.

        y_train_: ndarray of shape (nsamples,)
            Target values observed at 'X_train_'.

        dX_train_: ndarray of shape (dnsamples, nfeatures),
            default=None
            Coordinates of the training derivative observation points.

        dy_train_: ndarray of shape (dnsamples, ndfeatures),
            default=None
            Partial derivatives of the target values wrt to the
            hyperparameters, observed at 'dX_train_'.

        Returns:
            self: instance of self.
        """
        self._rng = check_random_state(self.random_state)

        X = _atleast2d(X)
        X, y = self._validate_data(X, y,
                                   multi_output=True, y_numeric=True,
                                   ensure_2d=True, dtype="numeric")
        self.X_train_ = np.copy(X) if self.copy_data else X
        self.y_train_ = np.copy(y) if self.copy_data else y
        self.alpha = self._validate_alpha(self.alpha)

        self.has_derinfo_ = False
        if (dX is not None) and (dy is not None):
            dX = _atleast2d(dX)
            dy = _atleast2d(dy)
            dX, dy = self._validate_data(dX, dy,
                                        multi_output=True, y_numeric=True,
                                        ensure_2d=True, dtype="numeric")
            self.dX_train_ = np.copy(dX) if self.copy_data else dX
            self.dy_train_ = np.copy(dy) if self.copy_data else dy
            self._dy_train_flat = np.zeros((dy.shape[0] * dy.shape[1],))
            for i in range(dy.shape[1]):
                self._dy_train_flat[i * y.shape[0] : (i + 1) * y.shape[0]] = \
                    self.dy_train_[:, i]
            self.has_derinfo_ = True
        elif (dX is not None) or (dy is not None):
            raise ValueError("Both 'dX' and 'dy' must be passed.")

        if self.kernel is None:
            if self.has_derinfo_:
                self.kernel_ = DerAwareKernel()
            else:
                self.kernel_ = Kernel()
        else:
            self.kernel_ = clone(self.kernel)

        if self.normalize_y:
            self._y_train_mean = np.mean(self.y_train_ , axis=0)
            self._y_train_std = np.std(self.y_train_ , axis=0)
            self.y_train_ -= self._y_train_mean
            self.y_train_ /= self._y_train_std
        else:
            self._y_train_mean = np.zeros(1)
            self._y_train_std = 1

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
            # Subsequent optimization runs are evaluated on thetas
            # sampled from log-uniform distributions
            if self.n_restarts_optimizer > 0:
                if not np.isfinite(self.kernel_.bounds).all():
                    raise ValueError("n_restarts_optimizer > 0 requires that"
                                     " all parameter bounds are finite.")
                bounds = self.kernel_.bounds
                for it in range(self.n_restarts_optimizer):
                    theta_init = self._rng.uniform(bounds[:, 0], bounds[:, 1])
                    optima.append(self._constrained_optimization(
                        obj_func, theta_init, bounds))
            # select the run which results in the minimal neg. log-marginal
            # likelihood
            lml_values = list(map(itemgetter(1), optima))
            self.kernel_.theta = optima[np.argmin(lml_values)][0]

            self.log_marginal_likelihood_value_ = -np.min(lml_values)
        else:
            self.log_marginal_likelihood_value_ = \
                self.log_marginal_likelihood(self.kernel_.theta,
                                             clone_kernel=False)

        if self.has_derinfo_:
            # for predicting function observations
            kernel_train = self.kernel_(X=self.X_train_, dX=self.dX_train_)
            kernel_train += self.alpha * np.eye((kernel_train.shape[0]))
            y_chol = np.concatenate((self.y_train_.reshape(-1,),
                                     self._dy_train_flat))
            # for predicting derivative observations
            kernel_train_ders = self.kernel_._cov_ww(self.dX_train_)
            kernel_train_ders += self.alpha \
                               * np.eye((kernel_train_ders.shape[0]))
            try:
                self.L_chol_ = cholesky(kernel_train, lower=True)
                self._L_chol_der_ = cholesky(kernel_train_ders, lower=True)
            except np.linalg.LinAlgError as exc:
                raise("The kernel is not positive definite. "
                      "Try increasing alpha.")
            self.alpha_chol_ = cho_solve((self.L_chol_, True), y_chol)
            self._alpha_chol_der_ = cho_solve((self._L_chol_der_, True),
                                               self._dy_train_flat)
        else:
            kernel_train = self.kernel_(X=self.X_train_)
            kernel_train += self.alpha * np.eye((kernel_train.shape[0]))
            y_chol = self.y_train_
            try:
                self.L_chol_ = cholesky(kernel_train, lower=True)
            except np.linalg.LinAlgError as exc:
                raise("The kernel is not positive definite. "
                      "Try increasing alpha.")
            self.alpha_chol_ = cho_solve((self.L_chol_, True), y_chol)

        return self


    def predict(self, X, return_cov=False, return_std=False):
        """Predict using the Gaussian process regression model.
        Predictions can also be made with an unfitted model by using
        the GP prior.

        Parameters
        ----------
        X: array of shape (npred, nfeatures)
            Coordinates of the query points where the function
            observations are predicted.

        return_cov: bool, default=False
            If True, the covariance of the joint predictive
            distribution at the quety points is returned along
            with the mean.

        return_std: bool, default=False
            If True, the standard deviation of the predictive
            distribution at the query points is also returned along
            with the mean.

        Returns
        -------
        y_mean: array of shape (npred,)
            Mean of the predictive distribution at the query
            points.

        y_cov: array of shape (npred, npred)
            Covariance of the predictive distribution at the
            query points. Only returned if 'return_cov'
            is True.

        y_std: array of shape (npred,) optional
            Standard deviation of the predictive distribution
            at the query points. Only returned if 'return_std'
            is True.
        """

        if return_std and return_cov:
            raise RuntimeError(
                "Only one of return_std or return_cov can be true at the"
                " same time.")

        X_star = check_array(_atleast2d(X), ensure_2d=True, dtype="numeric")

        # If the model is not fitted, predict based on the GP prior
        if not hasattr(self, "X_train_"):
            if self.kernel is None:
                kernel = GPKernel()
            else:
                kernel = self.kernel
            y_mean = np.zeros(X_star.shape[0])
            if return_cov:
                y_cov = kernel(X=X_star)
                return y_mean, y_cov
            elif return_std:
                y_std = np.sqrt(kernel.diag(X_star))
                return y_mean, y_std
            else:
                return y_mean
        else:
            # Predict based on the trained kernel
            if self.has_derinfo_:
                kernel_post = self.kernel_(X=self.X_train_, dX=self.X_train_,
                                           X_pred=X_star)
            else:
                kernel_post = self.kernel_(X=X_star, Y=self.X_train_)

            y_mean = kernel_post.dot(self.alpha_chol_)
            y_mean = self._y_train_std * y_mean + self._y_train_mean

            if return_cov:
                v = cho_solve((self.L_chol_, True), kernel_post.T)
                kernel_star = self.kernel_._cov_yy(X=X_star)
                y_cov = kernel_star - kernel_post.dot(v)
                y_cov = y_cov * self._y_train_std**2
                return y_mean, y_cov
            elif return_std:
                L_inv = solve_triangular(self.L_chol_.T,
                                         np.eye(self.L_chol_.shape[0]))
                K_inv = L_inv.dot(L_inv.T)
                std2 = np.copy(np.diagonal(self.kernel_._cov_yy(X=X_star)))
                std2 -= np.einsum("ij,ij->i",
                                  np.dot(kernel_post, K_inv),
                                  kernel_post)
                # If any std2 vals are neg, set them to 0.
                std2_neg = std2 < 0
                if np.any(std2_neg):
                    warnings.warn("Predicted variances smaller than 0. "
                                  "Setting those variances to 0.")
                    std2[std2_neg] = 0.0
                std2 = std2 * self._y_train_std**2
                y_std = np.sqrt(std2)
                return y_mean, y_std
            else:
                return y_mean


    def predict_der(self, dX, return_cov=False, return_std=False):
        """Predict the gradients at X using the Gaussian process
        regression model.

        Parameters
        ----------
        dX: array of shape (ndpred, nfeatures)
            Coordinates of the query points where the
            derivative observations are predicted.

        return_cov: bool, default=False
            If True, the covariance of the joint predictive
            distribution at the quety points is returned along
            with the mean.

        return_std: bool, default=False
            If True, the standard deviation of the predictive
            distribution at the query points is also returned along
            with the mean.

        Returns
        -------
        dy_mean: array of shape (ndpred,)
            Mean of the predictive distribution of the
            gradients at the query points.

        dy_cov: array of shape (ndpred, ndpred)
            Covariance of the predictive distribution
            of the gradients at the query points.
            Only returned if 'return_cov' is True.

        dy_std: array of shape (ndpred,) optional
            Standard deviation of the predictive distribution
            of the gradients at the query points.
            Only returned if 'return_std' is True.
        """

        if return_std and return_cov:
            raise RuntimeError(
                "Only one of return_std or return_cov can be true at the"
                " same time.")

        dX_star = check_array(_atleast2d(dX), ensure_2d=True, dtype="numeric")

        # Requires a fitted GP model
        if not hasattr(self, "dX_train_"):
            raise ValueError(
                "Prediction of derivative observations is only supported "
                "for trained GP models.")

        # Predict based on the trained kernel
        kernel_post = self.kernel_._cov_ww(dX_star, self.dX_train_)

        dy_mean = kernel_post.dot(self._alpha_chol_der_)

        if return_cov:
            v = cho_solve((self._L_chol_der_, True), kernel_post.T)
            kernel_star = self.kernel_._cov_ww(dX_star)
            dy_cov = kernel_star - kernel_post.dot(v)
            return dy_mean, dy_cov
        elif return_std:
            L_inv = solve_triangular(self._L_chol_der_.T,
                                     np.eye(self._L_chol_der_.shape[0]))
            K_inv = L_inv.dot(L_inv.T)
            std2 = np.copy(np.diagonal(self.kernel_._cov_ww(dX_star)))
            std2 -= np.einsum("ij,ij->i",
                              np.dot(kernel_post, K_inv),
                              kernel_post)
            # If any std2 vals are neg, set them to 0.
            std2_neg = std2 < 0
            if np.any(std2_neg):
                warnings.warn("Predicted variances smaller than 0. "
                              "Setting those variances to 0.")
                std2[std2_neg] = 0.0
            dy_std = np.sqrt(std2)
            return dy_mean, dy_std
        else:
            return dy_mean


    def sample_y(self, X, ndraws=1, random_state=0):
        """Draw samples from a GP and evaluate at X.

        Parameters
        ----------
        X: array of shape (nsamples, nfeatures)
            Coordinates of the query points to sample.

        ndraws: int, default = 1
            Number of samples to draw from the GP.

        random_state: int, RandomState instance or None, default=0
            Determines the random number generator to use to randomly
            draw samples.

        Returns
        -------
        y_samples: array of shape (nsamples, ndraws)
            Values of the nsamples drawn from the GP process and
            evaluated at 'X'.
        """
        rng = check_random_state(random_state)

        y_mean, y_cov = self.predict(X)
        y_samples = rng.multivariate_normal(y_mean, y_cov, ndraws).T
        return y_samples


    def sample_dy(self, dX, ndraws=1, random_state=0):
        """Draw samples from a GP and evaluate at X.

        Parameters
        ----------
        dX: array of shape (nsamples, nfeatures)
            Coordinates of the query derivative points to sample.

        ndraws: int, default = 1
            Number of samples to draw from the GP.

        random_state: int, RandomState instance or None, default=0
            Determines the random number generator to use to randomly
            draw samples.

        Returns
        -------
        dy_samples: array of shape (nsamples, ndraws)
            Derivative values of the nsamples drawn from the GP
            process and evaluated at 'dX'.
        """
        rng = check_random_state(random_state)

        dy_mean, dy_cov = self.predict_dX(dX)
        dy_samples = rng.multivariate_normal(dy_mean, dy_cov, ndraws).T
        return dy_samples


    def log_marginal_likelihood(self, theta=None, eval_gradient=False,
                                clone_kernel=True):
        """Log marginal likelihood (lml) of the theta values used for training.
        Note that the lml is evaluated only for training data without
        derivative information.

        Parameters
        ----------
        theta: array of shape (n_kernel_params,) default=None
            Kernel parameters on which the lml is evaluated.

        eval_gradient: bool, default=False
            If true, the gradient of the lml with respect to the
            parameters is also returned. Can only be true if theta is None.

        clone_kernel: bool, default=True
            If true, the kerneel attribute is copied.

        Returns
        -------
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
            if self.has_derinfo_:
                K, K_gradient = kernel(self.X_train_, dX=self.X_train_,
                                       eval_gradient=True)
            else:
                K, K_gradient = kernel(self.X_train_, eval_gradient=True)
        else:
            if self.has_derinfo_:
                K = kernel(self.X_train_, dX=self.X_train_,
                           eval_gradient=False)
            else:
                K = kernel(self.X_train_, eval_gradient=False)

        K[np.diag_indices_from(K)] += self.alpha
        try:
            L = cholesky(K, lower=True)
        except np.linalg.LinAlgError:
            return (-np.inf, np.zeros_like(theta)) \
                if eval_gradient else -np.inf

        if self.has_derinfo_:
            y_chol = np.concatenate((self.y_train_.reshape(-1,),
                                     self._dy_train_flat))
        else:
            y_chol = self.y_train_

        if y_chol.ndim==1:
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
