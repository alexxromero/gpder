import numpy as np
import scipy
from sklearn.base import BaseEstimator, MultiOutputMixin, RegressorMixin, clone
from sklearn.utils.optimize import _check_optimize_result
from sklearn.utils.validation import check_random_state

__all__ = ["GaussianProcessRegressor"]


def atleast_2d(arr):
    arr = np.asarray(arr)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    elif arr.ndim == 1:
        return arr[:, np.newaxis]
    else:
        return arr


class GaussianProcessRegressor(MultiOutputMixin, RegressorMixin, BaseEstimator):
    """Gaussian process regression (GPR).

    This implementation is based on scikit-learn's GaussianProcessRegressor and
    Algorithm 2.1 of Gaussian Processes for Machine Learning (GPML) by
    Rasmussen and Williams [1].

    For regular GP regression, use RegularKernel. To incorporate derivatrive
    information, use DerivativeKernel [2, 3].

    Parameters
    ----------
    kernel : RegularKernel or DerivativeKernel
        Kernel.

    kappa : float, default=1e-10
        Value added to the diagonal of the kernel matrix during fitting to
        prevent numerical instability by ensuring that it is posisive definite.

    optimizer : 'L-BFGS-B', None, or callable, default='L-BFGS-B'
        Optimization method to use. If 'L-BFGS-B', scipy's L-BFGS-B method
        is called. If None, no optimization is performed and the kernel's
        hyperparameters are kept fixed.

    n_restarts_optimizer : int, default=3
        Number of times to restart the optimizer.
        The first run uses the initial values of the kernel's hyperparameters.
        Subsequent runs are initialized on values drawn from a log-uniform
        distribution with bounds equal to the hyperparameters' bounds.

    random_state : int, RandomState instance or None, default=None
        State of the random number generator used by the optimizer.

    mean: float or callable, default=0
        Mean of the GP. If float, the mean is assumed to be constant.

    derivative_mean: float or callable, default=0
        Mean of the derivative GP. If float, the mean is assumed to be constant.

    Attributes
    ----------
    kernel_opt : RegularKernel or DerivativeKernel
        Optimized kernel.

    X_train: ndarray, shape (n_samp, n_feat)
        Training input.

    y_train: ndarray, shape (n_samp,)
        Training output.

    dX_train: ndarray, shape (n_samp_der, n_feat)
        Training derivative input.

    dy_train: ndarray, shape (n_samp_der, n_feat_der)
        Training derivative output.

    idx: ndarray, shape (n_feat_der)
        Indices along which the derivative output is observed.

    lml: float
        Log-marginal likelihood.

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

    def __init__(
        self,
        kernel,
        kappa=1e-10,
        optimizer="L-BFGS-B",
        n_restarts_optimizer=3,
        random_state=None,
        mean=0.0,
        derivative_mean=0.0,
    ):
        self.kernel = kernel
        self.kappa = kappa
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.random_state = random_state
        self.mean = mean
        self.derivative_mean = derivative_mean

    def fit(self, X, y, dX=None, dy=None, idx=None):
        """Fit the Gaussian process regression model.

        Parameters
        ----------
        X : ndarray, shape (n_samp, n_feat)
            Training input.

        y : ndarray, shape (n_samp,)
            Training output.

        dX : ndarray, shape (n_samp_der, n_feat), default=None
            Training derivative input.

        dy : ndarray, shape (n_samp_der, n_feat_der), default=None
            Training derivative output.

        idx : ndarray, shape (n_feat_der), default=None
            Indices along which the derivative output is observed.
            For example, if only the gradients along the 0th and 2nd dimension
            are observed, pass idx=[0, 2]. In None, all dimensions are assumed.

        Returns
        -------
        self : returns an instance of self.
        """
        self._rng = check_random_state(self.random_state)

        self.X_train = atleast_2d(X)
        self.y_train = atleast_2d(y)
        (self._n_samp_X, self._n_dims) = self.X_train.shape
        if self.y_train.shape[0] != self._n_samp_X:
            raise ValueError("X and y must have the same number of samples.")

        if dX is None and dy is None:
            self._has_gradients = False
            self.dX_train, self.dy_train = None, None
        elif dX is None or dy is None:
            raise ValueError("Both dX and dy must be provided.")
        else:
            self._has_gradients = True
            self.dX_train = atleast_2d(dX)
            self.dy_train = atleast_2d(dy)
            self._n_samp_der = len(self.dX_train)
            if len(self.dX_train) != len(self.dy_train):
                raise ValueError("dX and dy must have the same number of samples.")
            self.idx = np.arange(self._n_dims) if idx is None else idx
            self._n_dims_der = len(self.idx)

        self._y_cholesky = self.y_train - self._mean_func(self.X_train)
        if self._has_gradients:
            dy_chol = self.dy_train - self._der_mean_func(self.dX_train)
            if self._n_dims_der > 1:
                dy_chol = dy_chol.reshape(-1, self.y_train.shape[1], order="F")
            self._y_cholesky = np.concatenate((self._y_cholesky, dy_chol), axis=0)

        kernel_opt = clone(self.kernel)
        if self.optimizer is not None and self.kernel.n_dims > 0:
            kernel_opt.theta, self.lml = self._optimizer(self.n_restarts_optimizer)
        else:
            self.lml = self._calculate_lml(kernel_opt.theta)

        if self._has_gradients:
            self._training_cov = kernel_opt(
                X=self.X_train, dX=self.dX_train, idx=self.idx, add_noise=True
            )
        else:
            self._training_cov = kernel_opt(self.X_train, add_noise=True)

        try:
            # adding small alpha value to the diagonal for numerical stability
            self._L_cholevsky = scipy.linalg.cholesky(
                self._training_cov + self.kappa * np.eye(len(self._training_cov)),
                lower=True,
            )
        except scipy.linalg.LinAlgError as err:
            raise ValueError("The kernel is not positive definite. Try increasing kappa.") from err
        self._alpha_cholevsky = scipy.linalg.cho_solve((self._L_cholevsky, True), self._y_cholesky)

        self.kernel = kernel_opt
        return self

    def update(self, X_train=None, y_train=None, dX_train=None, dy_train=None):
        """Add new training data to the GP model. This function expands
        the training covariance wiith new data and updates the
        cholesky decomposition. No optimization of the kernel's
        hyperparameters is performed.

        Parameters
        ----------
        X_train : ndarray, shape (n_samp, n_feat), default=None
            New training input.

        y_train : ndarray, shape (n_samp,), default=None
            New training output.

        dX_train : ndarray, shape (n_samp_der, n_feat), default=None
            New training derivative input.

        dy_train : ndarray, shape (n_samp_der, n_feat_der), default=None
            New training derivative output.

        Returns
        -------
        self : returns an instance of self.
        """
        if X_train is not None and y_train is not None:
            X_train = atleast_2d(X_train)
            y_train = atleast_2d(y_train)
        if dX_train is not None and dy_train is not None:
            dX_train = atleast_2d(dX_train)
            dy_train = atleast_2d(dy_train)

        training_cov = self._expand_training_covariance(X_new=X_train, dX_new=dX_train)

        y_cholesky = self._expand_y_cholesky(
            X_new=X_train, y_new=y_train, dX_new=dX_train, dy_new=dy_train
        )

        try:
            L_cholevsky = scipy.linalg.cholesky(
                training_cov + self.kappa * np.eye(training_cov.shape[0]), lower=True
            )
        except scipy.linalg.LinAlgError as err:
            raise ValueError("The kernel is not positive definite. Try increasing kappa.") from err
        alpha_cholevsky = scipy.linalg.cho_solve((L_cholevsky, True), y_cholesky)

        if X_train is not None and y_train is not None:
            self.X_train = np.concatenate((self.X_train, X_train), axis=0)
            self.y_train = np.concatenate((self.y_train, y_train), axis=0)
        if dX_train is not None and dy_train is not None:
            self.dX_train = np.concatenate((self.dX_train, dX_train), axis=0)
            self.dy_train = np.concatenate((self.dy_train, dy_train), axis=0)

        self._training_cov = training_cov
        self._y_cholesky = y_cholesky
        self._L_cholevsky = L_cholevsky
        self._alpha_cholevsky = alpha_cholevsky
        self._n_samp_X = len(self.X_train)
        return self

    def predict(self, X, return_std=False, return_cov=False):
        """Predict using the Gaussian process regression model.

        Parameters
        ----------
        X : ndarray, shape (n_samp, n_feat)
            Test input.

        return_std : bool, default=False
            Whether to return the standard deviation of the predictive
            distribution. Cannot be True if return_cov is True.

        return_cov : bool, default=False
            Whether to return the covariance matrix of the predictive
            distribution. Cannot be True if return_std is True.

        Returns
        -------
        y_mean : ndarray, shape (n_samp,)
            Mean of predictive distribution.

        y_std : ndarray, shape (n_samp,)
            Standard deviation of predictive distribution. Only returned if
            return_std is True.

        y_cov : ndarray, shape (n_samp, n_samp)
            Covariance matrix of predictive distribution. Only returned if
            return_cov is True.
        """
        X_test = atleast_2d(X)

        if self._has_gradients:
            posterior_cov = np.block(
                [
                    self.kernel._cov_yy(X=X_test, Y=self.X_train, add_noise=False),
                    self.kernel._cov_wy(dX=self.dX_train, Y=X_test, idx=self.idx).T,
                ]
            )
        else:
            posterior_cov = self.kernel._cov_yy(X=X_test, Y=self.X_train, add_noise=False)
        self._posterior_cov = posterior_cov

        y_mean = self._mean_func(X_test)
        y_mean += np.dot(posterior_cov, self._alpha_cholevsky)

        if return_std and return_cov:
            raise ValueError("return_std and return_cov cannot both be True.")

        if return_std or return_cov:
            kernel_star = self.kernel._cov_yy(X=X_test, add_noise=False)

        if return_std:
            L_inv = scipy.linalg.solve_triangular(
                self._L_cholevsky.T, np.eye(self._L_cholevsky.shape[0])
            )
            K_inv = L_inv.dot(L_inv.T)
            std2 = np.copy(np.diag(kernel_star))
            std2 -= np.einsum("ij,ij->i", np.dot(posterior_cov, K_inv), posterior_cov)
            std2[std2 < 0] = 0.0
            y_std = atleast_2d(np.sqrt(std2))
        if return_cov:
            v = scipy.linalg.cho_solve((self._L_cholevsky, True), posterior_cov.T)
            y_cov = kernel_star - posterior_cov.dot(v)

        if return_std:
            return y_mean, y_std
        elif return_cov:
            return y_mean, y_cov
        else:
            return y_mean

    def predict_gradients(self, dX, return_std=False, return_cov=False):
        """Predict the gradients using the gaussian process regression model.

        Parameters
        ----------
        dX : ndarray, shape (n_samp, n_feat)
            Test derivative input.

        return_std : bool, default=False
            Whether to return the standard deviation of the predictive distribution.
            Cannot be True if return_cov is True.

        return_cov : bool, default=False
            Whether to return the covariance matrix of the predictive distribution.
            Cannot be True if return_std is True.

        Returns
        -------
        dy_mean : ndarray, shape (n_samp, n_targets)
            Mean of predictive distribution.

        dy_std : ndarray, shape (n_samp, n_targets)
            Standard deviation of predictive distribution. Only returned if return_std is True.

        dy_cov : ndarray, shape (n_samp, n_samp)
            Covariance matrix of predictive distribution. Only returned if return_cov is True.
        """

        dX_test = atleast_2d(dX)
        posterior_cov = np.block(
            [
                self.kernel._cov_wy(Y=self.X_train, dX=dX_test, idx=self.idx),
                self.kernel._cov_ww(dX=dX_test, dy=self.dX_train, idx=self.idx, add_noise=False),
            ]
        )

        dy_mean = self._der_mean_func(dX_test).reshape(-1, self.y_train.shape[1], order="F")
        dy_mean += posterior_cov.dot(self._alpha_cholevsky)

        if return_std and return_cov:
            raise ValueError("return_std and return_cov cannot both be True.")

        if return_std or return_cov:
            kernel_star = self.kernel._cov_ww(dX=dX_test, idx=self.idx, add_noise=False)

        if return_std:
            L_inv = scipy.linalg.solve_triangular(
                self._L_cholevsky.T, np.eye(self._L_cholevsky.shape[0])
            )
            K_inv = L_inv.dot(L_inv.T)
            std2 = np.copy(np.diag(kernel_star))
            std2 -= np.einsum("ij,ij->i", np.dot(posterior_cov, K_inv), posterior_cov)
            std2[std2 < 0] = 0.0
            dy_std = atleast_2d(np.sqrt(std2))
        if return_cov:
            v = scipy.linalg.cho_solve((self._L_cholevsky, True), posterior_cov.T)
            dy_cov = kernel_star - posterior_cov.dot(v)

        if return_std:
            return dy_mean, dy_std
        elif return_cov:
            return dy_mean, dy_cov
        else:
            return dy_mean

    def predict_using_query(
        self,
        X_predict,
        X_train_query=None,
        dX_train_query=None,
        return_mean=True,
        return_std=False,
        return_cov=False,
    ):
        """Prediction using query training input.
        The GP's training data is NOT modified during this process.
        This function is useful for the Bayesian Experimental Design
        (BED) algorithm.

        Parameters
        ----------
        X_predict : ndarray, shape (n_samp_test, n_feat)
            Test input.

        X_train_query : ndarray, shape (n_samp, n_feat), default=None
            This input is temporarily added to the training data when predicting 'X_predict'.

        dX_train_query : ndarray, shape (n_samp_der, n_feat), default=None
            This derivative input is temporarily added to the training data when predicting 'X_predict'.

        return_mean : bool, default=True
            Whether to return the mean of the predictive distribution.

        return_std : bool, default=False
            Whether to return the standard deviation of the predictive distribution.

        return_cov : bool, default=False
            Whether to return the covariance matrix of the predictive distribution.

        Returns
        -------
        y_mean : ndarray, shape (n_samp_test,)
            Mean of predictive distribution.

        y_std : ndarray, shape (n_samp_test,)
            Standard deviation of predictive distribution. Only returned if
            return_std is True.

        y_cov : ndarray, shape (n_samp_test, n_samp_test)
            Covariance matrix of predictive distribution. Only returned if
            return_cov is True.
        """
        X_test = atleast_2d(X_predict)

        if return_std and return_cov:
            raise ValueError("return_std and return_cov cannot both be True.")

        if X_train_query is None:
            X_train = self.X_train
        else:
            X_train_query = atleast_2d(X_train_query)
            X_train = np.concatenate((self.X_train, X_train_query), axis=0)

        if dX_train_query is None:
            dX_train = self.dX_train
        else:
            dX_train_query = atleast_2d(dX_train_query)
            dX_train = np.concatenate((self.dX_train, dX_train_query), axis=0)

        training_cov = self._expand_training_covariance(
            X_new=X_train_query,
            dX_new=dX_train_query,
        )

        try:
            L_cholevsky = scipy.linalg.cholesky(
                training_cov + self.kappa * np.eye(training_cov.shape[0]), lower=True
            )
        except scipy.linalg.LinAlgError:
            raise ValueError("The kernel is not positive definite. Try increasing kappa.")

        if self._has_gradients:
            posterior_cov = np.block(
                [
                    self.kernel._cov_yy(X=X_test, Y=X_train, add_noise=False),
                    self.kernel._cov_wy(dX=dX_train, Y=X_test, idx=self.idx).T,
                ]
            )
        else:
            posterior_cov = self.kernel._cov_yy(X=X_test, Y=X_train, add_noise=False)

        if return_mean:
            # the outputs of y_train_query and dy_train_query are
            # assumed to be the model's predictions at the query inputs
            y_train_query, dy_train_query = None, None
            if X_train_query is not None:
                y_train_query = self.predict(X_train_query)
            if dX_train_query is not None:
                dy_train_query = self.predict_gradients(dX_train_query)

            y_cholesky = self._expand_y_cholesky(
                X_new=X_train_query,
                y_new=y_train_query,
                dX_new=dX_train_query,
                dy_new=dy_train_query,
            )

            alpha_cholevsky = scipy.linalg.cho_solve((L_cholevsky, True), y_cholesky)

            y_mean = self._mean_func(X_test)
            y_mean += np.dot(posterior_cov, alpha_cholevsky)

            if not return_std and not return_cov:
                return y_mean

        kernel_star = self.kernel._cov_yy(X=X_test, add_noise=False)
        if return_std:
            L_inv = scipy.linalg.solve_triangular(L_cholevsky.T, np.eye(L_cholevsky.shape[0]))
            K_inv = L_inv.dot(L_inv.T)
            std2 = np.copy(np.diag(kernel_star))
            std2 -= np.einsum("ij,ij->i", np.dot(posterior_cov, K_inv), posterior_cov)
            std2[std2 < 0] = 0.0
            y_std = np.sqrt(std2)
        if return_cov:
            v = scipy.linalg.cho_solve((L_cholevsky, True), posterior_cov.T)
            y_cov = kernel_star - posterior_cov.dot(v)

        if return_mean and return_std:
            return y_mean, y_std
        elif return_mean and return_cov:
            return y_mean, y_cov
        elif return_std:
            return y_std
        elif return_cov:
            return y_cov

    def sample(self, X, n_draws=1, random_state=None):
        """Draw samples from the predictive distribution.

        Parameters
        ----------
        X : ndarray, shape (n_samp, n_feat)
            Test input.

        n_draws : int, default=1
            Number of samples to draw.

        random_state : int, RandomState instance or None, default=None
            State of the random number generator.

        Returns
        -------
        y_samples : ndarray, shape (n_draws, n_samp)
            Samples from the predictive distribution.
        """
        rng = check_random_state(random_state)
        y_mean, y_cov = self.predict(X, return_cov=True)
        y_samples = rng.multivariate_normal(y_mean.ravel(), y_cov, n_draws)
        return y_samples

    def sample_gradients(self, dX, n_draws=1, random_state=None):
        """Draw samples from the predictive distribution of the gradients.'

        Parameters
        ----------
        dX : ndarray, shape (n_samp, n_feat)
            Test derivative input.

        n_draws : int, default=1
            Number of samples to draw.

        random_state : int, RandomState instance or None, default=None
            State of the random number generator.

        Returns
        -------
        dy_samples : ndarray, shape (n_draws, n_samp * n_targets)
            Samples from the predictive distribution of the gradients.
        """
        rng = check_random_state(random_state)
        dy_mean, dy_cov = self.predict_gradients(dX, return_cov=True)
        dy_samples = np.zeros((n_draws, len(dX) * self._n_dims_der))
        for i in range(self._n_dims_der):
            start = i * len(dX)
            end = (i + 1) * len(dX)
            dy_samples[:, start:end] = rng.multivariate_normal(
                dy_mean[start:end].ravel(), dy_cov[start:end, start:end], n_draws
            )
        return dy_samples

    def _expand_cov_yy(self, X_new):
        n = self.X_train.shape[0]

        cov_yy_oX = self._training_cov[:n, :n]
        cov_yy_oX_nX = self.kernel._cov_yy(X=self.X_train, Y=X_new, add_noise=False)
        cov_yy_nX = self.kernel._cov_yy(X=X_new, add_noise=True)
        return np.block([[cov_yy_oX, cov_yy_oX_nX], [cov_yy_oX_nX.T, cov_yy_nX]])

    def _expand_cov_ww(self, dX_new):
        n_odX = self.dX_train.shape[0]
        n_ndX = dX_new.shape[0]

        cov_ww_oX = self._training_cov[-n_odX * self._n_dims_der :, -n_odX * self._n_dims_der :]
        cov_ww_odX_ndX = self.kernel._cov_ww(dX=self.dX_train, dy=dX_new, add_noise=False)
        cov_ww_ndX_odX = cov_ww_odX_ndX.T
        cov_ww_ndX = self.kernel._cov_ww(dX=dX_new, add_noise=True)

        cov_ww = np.zeros(
            (
                (n_odX + n_ndX) * self._n_dims_der,
                (n_odX + n_ndX) * self._n_dims_der,
            )
        )
        for i in range(self._n_dims_der):
            for j in range(self._n_dims_der):
                low_i = i * (n_odX + n_ndX)
                low_j = j * (n_odX + n_ndX)
                cov_ww[low_i : low_i + n_odX, low_j : low_j + n_odX] = cov_ww_oX[
                    i * n_odX : (i + 1) * n_odX, j * n_odX : (j + 1) * n_odX
                ]
                cov_ww[low_i : low_i + n_odX, low_j + n_odX : low_j + n_odX + n_ndX] = (
                    cov_ww_odX_ndX[i * n_odX : (i + 1) * n_odX, j * n_ndX : (j + 1) * n_ndX]
                )
                cov_ww[low_i + n_odX : low_i + n_odX + n_ndX, low_j : low_j + n_odX] = (
                    cov_ww_ndX_odX[i * n_ndX : (i + 1) * n_ndX, j * n_odX : (j + 1) * n_odX]
                )
                cov_ww[
                    low_i + n_odX : low_i + n_odX + n_ndX,
                    low_j + n_odX : low_j + n_odX + n_ndX,
                ] = cov_ww_ndX[i * n_ndX : (i + 1) * n_ndX, j * n_ndX : (j + 1) * n_ndX]
        return cov_ww

    def _expand_cov_wy(self, X_new, dX_new):
        n_oX = self.X_train.shape[0]
        n_nX = X_new.shape[0]

        n_odX = self.dX_train.shape[0]
        n_ndX = dX_new.shape[0]

        cov_wy_odX_oX = self._training_cov[n_oX:, :n_oX]
        cov_wy_odX_nX = self.kernel._cov_wy(self.dX_train, X_new)
        cov_wy_ndX_oX = self.kernel._cov_wy(dX_new, self.X_train)
        cov_wy_ndX_nX = self.kernel._cov_wy(dX_new, X_new)

        cov_wy = np.zeros(
            (
                n_odX * self._n_dims_der + n_ndX * self._n_dims_der,
                n_oX + n_nX,
            )
        )

        for i in range(self._n_dims_der):
            low = i * (n_odX + n_ndX)
            cov_wy[low : low + n_odX, :-n_nX] = cov_wy_odX_oX[i * n_odX : (i + 1) * n_odX, :]
            cov_wy[low + n_odX : low + n_odX + n_ndX, :-n_nX] = cov_wy_ndX_oX[
                i * n_ndX : (i + 1) * n_ndX, :
            ]
            cov_wy[low : low + n_odX, -n_nX:] = cov_wy_odX_nX[i * n_odX : (i + 1) * n_odX, :]
            cov_wy[low + n_odX : low + n_odX + n_ndX, -n_nX:] = cov_wy_ndX_nX[
                i * n_ndX : (i + 1) * n_ndX, :
            ]
        return cov_wy

    def _expand_training_covariance(self, X_new=None, dX_new=None):
        cov_yy_exp = self._expand_cov_yy(X_new)
        if self._has_gradients and dX_new is not None:
            cov_ww_exp = self._expand_cov_ww(dX_new)
            cov_wy_exp = self._expand_cov_wy(X_new, dX_new)
            training_cov = np.block([[cov_yy_exp, cov_wy_exp.T], [cov_wy_exp, cov_ww_exp]])
        else:
            training_cov = cov_yy_exp
        return training_cov

    def _expand_y_cholesky(self, X_new=None, y_new=None, dX_new=None, dy_new=None):
        if self._has_gradients:
            n_oX, n_odX = self.X_train.shape[0], self.dX_train.shape[0]
            if X_new is not None and dX_new is not None:
                n_nX, n_ndX = X_new.shape[0], dX_new.shape[0]
                y_chol_new = y_new - self._mean_func(X_new)
                dy_chol_new = dy_new - self._der_mean_func(dX_new)
                dy_chol_new = dy_chol_new.reshape(-1, self.y_train.shape[1], order="F")
                y_cholesky = np.zeros(
                    (
                        len(self._y_cholesky) + n_nX + n_ndX * self._n_dims_der,
                        self.y_train.shape[1],
                    )
                )
                y_cholesky[:n_oX] = self._y_cholesky[:n_oX]
                y_cholesky[n_oX : n_oX + n_nX] = y_chol_new
                for i in range(self._n_dims_der):
                    low = n_oX + n_nX + i * (n_odX + n_ndX)
                    y_cholesky[low : low + n_odX] = self._y_cholesky[
                        n_oX + i * n_odX : n_oX + (i + 1) * n_odX
                    ]
                    y_cholesky[low + n_odX : low + n_odX + n_ndX] = dy_chol_new[
                        i * n_ndX : (i + 1) * n_ndX
                    ]
            elif X_new is not None:
                n_nX = X_new.shape[0]
                y_chol_new = y_new - self._mean_func(X_new)
                y_cholesky = np.zeros((len(self._y_cholesky) + len(X_new), self.y_train.shape[1]))
                y_cholesky[:n_oX] = self._y_cholesky[:n_oX]
                y_cholesky[n_oX : n_oX + n_nX] = y_chol_new
                y_cholesky[n_oX + n_nX :] = self._y_cholesky[n_oX:]
            elif dX_new is not None:
                n_ndX = dX_new.shape[0]
                dy_chol_new = dy_new - self._der_mean_func(dX_new)
                dy_chol_new = dy_chol_new.reshape(-1, self.y_train.shape[1], order="F")
                y_cholesky = np.zeros(
                    (
                        len(self._y_cholesky) + n_ndX * self._n_dims_der,
                        self.y_train.shape[1],
                    )
                )
                y_cholesky[:n_oX] = self._y_cholesky[:n_oX]
                for i in range(self._n_dims_der):
                    low = n_oX + i * (n_odX + n_ndX)
                    y_cholesky[low : low + n_odX] = self._y_cholesky[
                        n_oX + i * n_odX : n_oX + (i + 1) * n_odX
                    ]
                    y_cholesky[low + n_odX : low + n_odX + n_ndX] = dy_chol_new[
                        i * n_ndX : (i + 1) * n_ndX
                    ]
        else:
            y_chol_new = y_new - self._mean_func(X_new)
            y_cholesky = np.concatenate((self._y_cholesky, y_chol_new), axis=0)
        return y_cholesky

    def _mean_func(self, X):
        """Mean function."""
        if callable(self.mean):
            return self.mean(X)
        else:
            return np.full((X.shape[0], 1), self.mean)

    def _der_mean_func(self, dX):
        """Mean derivative function."""
        if callable(self.derivative_mean):
            return self.derivative_mean(dX)
        else:
            return np.full((dX.shape[0], self._n_dims_der), self.derivative_mean)

    def _optimizer(self, n_restarts):
        """Optimize the kernel's hyperparameters."""

        def obj_func(theta, eval_gradient=True):
            if eval_gradient:
                lml, grad = self._calculate_lml(theta, eval_gradient=True)
                return -lml, -grad
            else:
                return -self._calculate_lml(theta, eval_gradient=False)

        # First optimization is evaluated on the initial parameters
        optima = [self._constrained_optimization(obj_func, self.kernel.theta, self.kernel.bounds)]
        # Subsequent optimizations are evaluated on parameters drawn from random log-uniform distributions
        if n_restarts > 0:
            if not np.isfinite(self.kernel.bounds).all():
                raise ValueError("Parameter bounds should be finite.")
            bounds = self.kernel.bounds
            theta0 = self._rng.uniform(
                bounds[:, 0], bounds[:, 1], size=(n_restarts, bounds.shape[0])
            )
            optima.extend(
                [self._constrained_optimization(obj_func, theta, bounds) for theta in theta0]
            )
        # Select the theta with minimal neg log marginal likelihood
        lml_values = list(map(lambda x: x[1], optima))
        best_theta = optima[np.argmin(lml_values)][0]
        best_lml = -np.min(lml_values)
        return best_theta, best_lml

    def _calculate_lml(self, theta, eval_gradient=False):
        """Log-marginal likelihood."""

        kernel = self.kernel
        kernel.theta = theta

        if eval_gradient:
            if self._has_gradients:
                K, K_gradient = kernel(
                    self.X_train, self.dX_train, idx=self.idx, eval_gradient=True
                )
            else:
                K, K_gradient = kernel(self.X_train, eval_gradient=True)
        else:
            if self._has_gradients:
                K = kernel(self.X_train, self.dX_train, idx=self.idx, eval_gradient=False)
            else:
                K = kernel(self.X_train, eval_gradient=False)

        K += self.kappa * np.eye(K.shape[0])
        try:
            L = scipy.linalg.cholesky(K, lower=True)
        except scipy.linalg.LinAlgError:
            return -np.inf, np.zeros_like(theta) if eval_gradient else -np.inf

        alpha_chol = scipy.linalg.cho_solve((L, True), self._y_cholesky)
        log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", self._y_cholesky, alpha_chol)
        log_likelihood_dims -= np.log(np.diag(L)).sum()
        log_likelihood_dims -= K.shape[0] / 2.0 * np.log(2 * np.pi)
        log_likelihood = log_likelihood_dims.sum(-1)

        if eval_gradient:
            tmp = np.einsum("ik,jk->ijk", alpha_chol, alpha_chol)
            tmp -= scipy.linalg.cho_solve((L, True), np.eye(K.shape[0]))[:, :, np.newaxis]
            log_likelihood_grad_dims = 0.5 * np.einsum("ijl,ijk->kl", tmp, K_gradient)
            log_likelihood_grad = log_likelihood_grad_dims.sum(-1)
            return log_likelihood, log_likelihood_grad

        return log_likelihood

    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        """Constrained optimization."""
        initial_fun = obj_func(initial_theta, eval_gradient=False)
        res = scipy.optimize.minimize(
            obj_func, initial_theta, method="L-BFGS-B", jac=True, bounds=bounds
        )
        _check_optimize_result("lbfgs", res)
        if not res.success:
            print("Optimization failed:", res.message)
            print("Returning initial parameters and function value.")
            return initial_theta, initial_fun
        else:
            return res.x, res.fun
