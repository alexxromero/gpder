import numpy as np
import scipy
from scipy.stats import norm
import sklearn
from sklearn.utils.validation import check_random_state
from sklearn.metrics import mean_squared_error
from multiprocessing import Pool

from ..gaussian_process import GaussianProcessRegressor
from ..gaussian_process.kernels import GPKernel, GPKernelDerAware
from .bayesian_opt_utils import print_log, print_log_mse_uncer

from scipy.spatial.distance import cdist

# Functions from: http://krasserm.github.io/2018/03/21/bayesian-optimization/
# https://github.com/thuijskens/bayesian-optimization/blob/master/python/gp.py

__all__ = ['BayesUncertaintyOptimization']


class BayesUncertaintyOptimization():
    """BayesUncertaintyOptimization performs a Gaussian Process
    Regression (GPR) of the objective function and minimizing the
    uncertainty of the regression. The uncertainty minimization is
    done using a Bayesian optimization routine with a GPR as the
    surrogate model.
    If derivative information of the objective function is available,
    it is used to aid the GPR.

    Parameters
    ----------
    fun: callable
        Objective function.

    params_bounds: dict
        Dictionary with the parameter names as the keys and their
        upper/lower bounds as the values.
        Example: {'param1_name': (0, 1), 'param2_name': (-1, 1)}

    dfun: callable, default=None
        Function returning the partial derivatives of the
        objective function 'fun' wrt the nuisance parameters.

    gp: GaussianProcessRegressor instance, default=None
        Instance of the class GaussianProcessRegressor.
        If None, the GPR class is initiated with its default values.

    random_state: RandomState instance, int, or None, default=None
        Determines the random number generator used in the GPR
        optimization.

    verbose: bool
        If True, a table is printed with the nuisance parameters
        selected in every iteration.
    """

    def __init__(self, fun, param_bounds, dfun=None, gp=None,
                 random_state=None, verbose=True):
        self.fun = fun
        self.params_bounds = param_bounds
        self.dfun = dfun
        self.gp = gp
        self.random_state = check_random_state(random_state)
        self.verbose = verbose
        self._has_derinfo = False if self.dfun is None else True
        self._X_keys = self._get_param_keys(param_bounds)
        self._X_bounds_vals = self._get_param_vals(param_bounds)
        self._nparams = len(self._X_keys)

    def _get_param_keys(self, param_bounds):
        return list(param_bounds.keys())

    def _get_param_vals(self, param_bounds):
        return np.array([param_bounds.get(p) for p in self._X_keys])

    def minimize_uncertainty(self,
                             X_train, y_train,
                             dX_train=None, dy_train=None,
                             X_test=None, y_test=None,
                             nX_grid=10,
                             n_iters=10,
                             minimizer='fmin_l_bfgs_b',
                             n_minimizer_restarts=10,
                             sigma=0,
                             sigma_tol=0,
                             workers=1):
        """Minimize the uncertainty of the GPR model used to regress the
        objective function 'fun'.

        Parameters
        ----------
        X_train: ndarray of shape (ntr, nparams)
            Coordinates of the initial function observations.

        y_train: ndarray of shape (ntr,)
            Values of the initial function observations
            evaluated at 'X_train'.

        dX_train: ndarray of shape (ndtr, ndparams), optional
            Coordinates of the initial function derivative observations.

        dy_train: ndarray of shape (ndtr, ndparams), optional
            Values of the initial function derivative observations
            evaluated at 'dX_train'.

        X_test: ndarray of shape (ntst, ntparams), optional
            Coordinates of the test function observations.

        y_test: ndarray of shape (ntst,), optional
            Values of the test function observations
            evaluated at 'X_test'.

        nX_grid: int, default=10
            Number of points used to infer the uncertainty of the GPR.
            The points are sampled uniformly along each parameter
            dimension.

        n_iters: int, default=10
            Number of iterations. At every iteration, a Bayesian
            optimization search is performed over the parameters.

        minimizer: 'fmin_l_bfgs_b' or callable, default='fmin_l_bfgs_b'
            Minimizing function for the neg. acquisition function.
            If 'fmin_l_bfgs_b', scipy's minimizer function with the
            fmin_l_bfgs_b method is used.
            Alternatively, a minimzer can be provided as a callable
            with the signature
                x, fval = minimizer(fun, bounds, random_state)
            See minimizers.py for brute-force and random search
            minimizing functions.

        n_minimizer_restarts: int, default=10
            Number of times to restart the 'minimizer' algorithm
            at every iteration.

        sigma: float, default=0
            Exploitation-exploration parameter.
        """

        self.X_train, self.y_train = X_train, y_train
        self.dX_train, self.dy_train = dX_train, dy_train
        self.X_test, self.y_test = X_test, y_test
        self.n_iters = n_iters
        self.nX_grid = nX_grid
        self.minimizer = minimizer
        self.n_minimizer_restarts = n_minimizer_restarts
        self.sigma = sigma
        self.sigma_tol = sigma_tol
        self.workers = workers

        if self._has_derinfo:
            if (self.dX_train is not None) and (self.dy_train is not None):
                self.dX_train = self.dX_train
                self.dy_train = self.dy_train
            else:
                raise ValueError(
                    "dX_train and dy_train must be passed if using "
                    "derivative information..")
            if self.gp is None:
                self.gp = GaussianProcessRegressor(
                    kernel=GPKernelDerAware(), random_state=self.random_state)
            self.gp.fit(X=self.X_train, y=self.y_train,
                        dX=self.dX_train, dy=self.dy_train)
            self._kernel_hparams = [self.gp.kernel_.constant_value,
                                    self.gp.kernel_.length_scale,
                                    self.gp.kernel_.noise_level,
                                    self.gp.kernel_.noise_level_dX]
        else:
            if self.gp is None:
                self.gp = GaussianProcessRegressor(
                    kernel=GPKernel(), random_state=self.random_state)
            self.gp.fit(X=self.X_train, y=self.y_train)
            self._kernel_hparams = [self.gp.kernel_.constant_value,
                                    self.gp.kernel_.length_scale,
                                    self.gp.kernel_.noise_level]
        self._n_init = np.shape(self.X_train)[0]

        # -- uniform sample grid -- #
        n = len(self._X_keys)
        grid_coords = np.linspace(
            self._X_bounds_vals[:, 0], self._X_bounds_vals[:, 1], self.nX_grid)
        X_grid = np.meshgrid(*grid_coords.T)
        self.X_grid = np.asarray(X_grid).reshape(n, self.nX_grid**n).T
        # initial uncertainty as evaluated on the uniform sample grid
        mu, cov = self.gp.predict(self.X_grid, return_cov=True)
        self.uncert = np.trace(cov)

        # if test data is given, calculate the MSE
        if (self.X_test is not None) and (self.y_test is not None):
            mu_test, cov_test = self.gp.predict(X=self.X_test, return_cov=True)
            uncert_test = np.trace(cov_test)
            mse_test = mean_squared_error(self.y_test, mu_test)
            self.mse_test = mse_test
            self.uncert_test = uncert_test
        else:
            self.mse_test = None
            self.uncert_test = None

        self._record_init_values()

        if self.verbose:
            print_log_mse_uncer(self._X_keys,
                                self.X_train,
                                self.y_train,
                                uncert=self.uncert,
                                mse_test=self.mse_test,
                                uncert_test=self.uncert_test,
                                iteration_ids=[0]*self._n_init,
                                initial_params=True,
                                print_header=True)

        for i in range(self.n_iters):
            X_next = self._find_next_X(n_restarts=self.n_minimizer_restarts)
            y_next = self._eval_fun(X_next)
            self.X_train = np.vstack((self.X_train, X_next))
            self.y_train = np.concatenate((self.y_train, y_next), axis=-1)
            if self._has_derinfo:
                dy_next = self._eval_dfun(X_next)
                self.dX_train = np.vstack((self.dX_train, X_next))
                self.dy_train = np.vstack((self.dy_train, dy_next))
                self.gp.fit(X=self.X_train, y=self.y_train,
                            dX=self.dX_train, dy=self.dy_train)
                kernel_hyperparams = [self.gp.kernel_.constant_value,
                                      self.gp.kernel_.length_scale,
                                      self.gp.kernel_.noise_level,
                                      self.gp.kernel_.noise_level_dX]
            else:
                self.gp.fit(X=self.X_train, y=self.y_train)
                kernel_hyperparams = [self.gp.kernel_.constant_value,
                                      self.gp.kernel_.length_scale,
                                      self.gp.kernel_.noise_level]
            self._kernel_hparams = np.vstack(
                (self._kernel_hparams, kernel_hyperparams))
            mu, cov = self.gp.predict(X=self.X_grid, return_cov=True)
            uncert = np.trace(cov)
            self.uncert = np.vstack((self.uncert , uncert))

            if self.X_test is not None:
                mu_test, cov_test = self.gp.predict(X=self.X_test,
                                                    return_cov=True)
                mse_test = mean_squared_error(self.y_test, mu_test)
                mse_test = mse_test
                uncert_test = np.trace(cov_test)
                self.mse_test = np.vstack((self.mse_test, mse_test))
                self.uncert_test = np.vstack((self.uncert_test, uncert_test))
            else:
                mse_test = None
                uncert_test = None

            if self.verbose:
                print_log_mse_uncer(
                    self._X_keys, X_next, y_next,
                    uncert=uncert,
                    mse_test=mse_test,
                    uncert_test=uncert_test,
                    iteration_ids=[i+1],
                    print_header=False)

        self._record_bayes_values()

    def _record_init_values(self):
        self._X_init = self.X_train
        self._y_init = self.y_train
        if self._has_derinfo:
            self._dX_init = self.dX_train
            self._dy_init = self.dy_train
        else:
            self._dtargets_init = []
        self._uncert_init = self.uncert
        self._kernel_hparams_init = self._kernel_hparams
        if self.X_test is not None:
            self._mse_test_init = self.mse_test
            self._uncert_test_init = self.uncert_test

    def _record_bayes_values(self):
        self._X_bayes = self.X_train[self._n_init:, :]
        self._y_bayes = self.y_train[self._n_init:]
        if self._has_derinfo:
            self._dX_bayes = self.dX_train[self._n_init:, :]
            self._dy_bayes = self.dy_train[self._n_init:, :]
        else:
            self._dtargets_bayes = []
        self._uncert_bayes = self.uncert[1:]
        self._kernel_hparams_bayes = self._kernel_hparams[1:]
        if self.X_test is not None:
            self._mse_test_bayes = self.mse_test[1:]
            self._uncert_test_bayes = self.uncert_test[1:]

    def _eval_fun(self, X_arr):
        targets = []
        for X in X_arr:
            X_dict = dict(zip(self._X_keys, X))
            targets.append(self.fun(**X_dict))
        return np.asarray(targets).reshape(-1,)

    def _eval_dfun(self, dX_arr):
        y_arr = []
        for X in dX_arr:
            X_dict = dict(zip(self._X_keys, X))
            y_arr.append(self.dfun(**X_dict))
        return np.asarray(y_arr).reshape(-1, self._nparams)

    def _find_next_X(self, n_restarts):
        def uncert_fun(X):
            return self._expected_uncertainty_improvement(X)

        best_x = None
        min_uncert = 1e30
        for i in range(n_restarts):
            if self.minimizer=='fmin_l_bfgs_b':
                x0 = self.random_state.uniform(self._X_bounds_vals[:, 0],
                                               self._X_bounds_vals[:, 1],
                                               size=(1, self._nparams))
                res = scipy.optimize.minimize(
                    fun=uncert_fun, x0=x0, method='L-BFGS-B',
                    bounds=self._X_bounds_vals)
                fval = res.fun[0]
                x = res.x
            elif callable(self.minimizer):
                x, fval = self.minimizer(fun=uncert_fun,
                                         bounds=self._X_bounds_vals)
            else:
                raise ValueError("Unknown minimizer %s." % self.minimizer)

            if fval < min_uncert:
                min_uncert = fval
                best_x = x
        best_x = np.array(best_x).reshape(1, self._nparams)
        return best_x


    def _expected_uncertainty_improvement(self, X):
        # The GPR's hyperparameters are frozen for efficiency, and to
        # prevent any odd results from fitting the GPR on an estimated
        # y and dy
        gp = GaussianProcessRegressor(kernel=self.gp.kernel_,
                                      optimizer=None,
                                      random_state=self.random_state)
        if self._has_derinfo:
            gp.fit(X=self.X_train, y=self.y_train,
                   dX=self.dX_train, dy=self.dy_train)
        else:
            gp.fit(X=self.X_train, y=self.y_train)
        X = np.reshape(X, (-1, self._nparams))
        y = gp.predict(X=X).reshape(-1,)
        X_temp = np.vstack((self.X_train, X))
        y_temp = np.concatenate((self.y_train, y), axis=-1)
        if self._has_derinfo:
            dy = gp.predict_der(dX=X).reshape(-1, self._nparams)
            dy_temp = np.vstack((self.dy_train, dy))
            dX_temp = np.vstack((self.dX_train, X))
            gp.fit(X=X_temp, y=y_temp, dX=dX_temp, dy=dy_temp)
        else:
            gp.fit(X=X_temp, y=y_temp)
        mu, cov = gp.predict(self.X_grid, return_cov=True)
        uncert = np.trace(cov)
        dist_min = np.min(cdist(X, self.X_train))
        if dist_min <= self.sigma_tol:
            uncert += self.sigma
        return uncert.reshape(-1, 1)
