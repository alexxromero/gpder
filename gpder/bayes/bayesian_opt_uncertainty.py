import numpy as np
import warnings
import scipy
from scipy.stats import norm
from scipy import optimize
from scipy.spatial.distance import cdist
import sklearn
from sklearn.utils.validation import check_random_state
from sklearn.metrics import mean_squared_error
from sklearn.exceptions import ConvergenceWarning
from copy import deepcopy
from multiprocessing import Pool

from ..gaussian_process import GaussianProcessRegressor
from ..gaussian_process.kernels import GPKernel, GPKernelDerAware
from .bayesian_opt_utils import print_log, print_log_mse_uncer

import matplotlib.pyplot as plt

# Functions from: http://krasserm.github.io/2018/03/21/bayesian-optimization/
# https://github.com/thuijskens/bayesian-optimization/blob/master/python/gp.py

__all__ = ['UncertaintyOptimization']

class UncertaintyOptimization():
    """UncertaintyOptimization minimizes the uncertainty of the
    objective function by using a Bayesian optimization routine with
    Gaussian Process Regression (GPR) as the surrogate model.
    If derivative information of the objective function is available,
    it is used to aid the gpr.

    Parameters
    ----------
    GaussianProcessRegressor: GaussianProcessRegressor
        Instance of GaussianProcessRegressor.

    fun: callable
        Objective function.

    param_bounds: dict
        Dictionary with the parameter names as the keys and a
        tuple with their correspondent bounds as the values.
        Example: {'x': (0, 1), 'y': (-1, 1)}

    dfun: callable, default=None
        Function returning the partial derivatives of the
        objective function with respect to the parameters.
        Optional.

    random_state: RandomState instance or None, default=None
        Determines the random number generator used in the GPR
        optimization.

    verbose: bool
        If True, print a table with the selected hyperparameters
        per iter.
    """

    def __init__(self,
                 GaussianProcessRegressor,
                 param_bounds,
                 fun,
                 dfun=None,
                 random_state=None,
                 ignore_convergence_warnings=False,
                 verbose=True):
        self.gp = GaussianProcessRegressor
        self.params_bounds = param_bounds
        self.fun = fun
        self.dfun = dfun
        self.ignore_convergence_warnings = ignore_convergence_warnings
        self.random_state = check_random_state(random_state)
        self.verbose = verbose
        self._has_derinfo = False if self.dfun is None else True
        self._params_keys = self._get_param_keys(param_bounds)
        self._params_bounds_vals = self._get_param_vals(param_bounds)
        self._params_dict = param_bounds
        self._nparams = len(self._params_keys)

    def _get_param_keys(self, param_bounds):
        return list(param_bounds.keys())

    def _get_param_vals(self, param_bounds):
        return np.array([param_bounds.get(p) for p in self._params_keys])

    def minimize_uncertainty(self,
                             params_train=None,
                             nrand_train=0,
                             params_test=None,
                             minimizer='fmin_l_bfgs_b',
                             niters=10,
                             minimizer_restarts=10):
        """Minimizes the uncertainty of the GPR model.

        Parameters
        ----------
        params_train: ndarray of shape (ntr, nparams)
            Initial training parameters. If None, random parameters
            are used to initialize the fitting of the GPR.

        nrand_train: int, default=0
            Number of random parameters to use as the initial training
            parameters for the GPR.

        params_test: ndarray of shape (nval, nparams), optional
            test parameters.

        minimizer: 'fmin_l_bfgs_b', or callable, default='fmin_l_bfgs_b'
            Minimizing function for the neg. acquisition function.
            If 'fmin_l_bfgs_b', scipy's minimizer function with the
            fmin_l_bfgs_b method is used.
            Alternatively, a minimzer can be provided as a callable
            with the signature:
                x, fval = minimizer(fun, bounds)
            See minimizers.py for brute-force, random search,
            and hybrid search minimizer functions.

        niters: int, default=10
            Number of Bayesian optimization iterations to do.

        minimizer_restarts: int, default=10
            Number of times to restart the 'minimizer' per Bayesian
            iteration.
        """

        self.params_train = params_train
        self.nrand_train = nrand_train
        self.params_test = params_test
        self.minimizer = minimizer
        self.niters = niters
        self.minimizer_restarts = minimizer_restarts

        if (self.params_train is None) and (self.nrand_train==0):
            raise ValueError(
                "Please enter either params_train and/or nrand_train > 0.")

        if self._has_derinfo:
            # get initial trainig data
            param_init, target_init, dtarget_init = \
                self._get_initial_training_points(include_ders=True)
            param_rand, target_rand, dtarget_rand = \
                self._get_random_training_points(include_ders=True)
            self.params = np.vstack((param_init, param_rand))
            self.targets = np.vstack((target_init, target_rand))
            self.dtargets = np.vstack((dtarget_init, dtarget_rand))
            self.gp.fit(X=self.params, y=self.targets,
                        dX=self.params, dy=self.dtargets)
            uncert_train = np.trace(self.gp.kernel_._cov_yy(self.params))
        else:
            # get initial trainig data
            param_init, target_init = \
                self._get_initial_training_points(include_ders=False)
            param_rand, target_rand = \
                self._get_random_training_points(include_ders=False)
            self.params = np.vstack((param_init, param_rand))
            self.targets = np.vstack((target_init, target_rand))
            self.gp.fit(X=self.params, y=self.targets)
            uncert_train = np.trace(self.gp.kernel_(self.params))

        # save kernel's uncertainty and hyperparameters at each stage
        # to see the evolution of the GP
        self.uncert_train = uncert_train
        self.theta = self.gp.kernel_.theta

        # if test data is given, calculate the MSE
        if len(self.params_test) > 0:
            self.params_test, self.targets_test = self._get_test_points()
            if self._has_derinfo:
                mu_test, cov_test = self.gp.predict(X=self.params_test,
                                                    full_covariance=False,
                                                    return_cov=True)
            else:
                mu_test, cov_test = self.gp.predict(X=self.params_test,
                                                    return_cov=True)
            mse_test = mean_squared_error(self.targets_test, mu_test)
            mse_test = mse_test
            uncert_test = np.trace(cov_test)
            self.mse_test = mse_test
            self.uncert_test = uncert_test
        else:
            self.mse_test = None
            self.uncert_test = None

        self._record_init_values()

        itsamples = np.shape(self.params)[0]
        if self.verbose:
            print_log_mse_uncer(self._params_keys,
                                self.params,
                                self.targets,
                                uncert_train=uncert_train,
                                mse_test=mse_test,
                                uncert_test=uncert_test,
                                iteration=list(range(1, itsamples+1)),
                                initial_params=True,
                                print_header=True)

        for i in range(niters):
            # propose the next sampling location
            param_next, uncert_next = self._find_next_location(
                n_restarts=self.minimizer_restarts)

            # add proposed location to the training set
            target_next = self._eval_fun(param_next)
            self.params = np.vstack((self.params, param_next))
            self.targets = np.vstack((self.targets, target_next))
            if self._has_derinfo:
                dtarget_next = self._eval_dfun(param_next)
                self.dtargets = np.vstack((self.dtargets, dtarget_next))

            # refit the GP
            if self._has_derinfo:
                self.gp.fit(X=self.params, y=self.targets,
                            dX=self.params, dy=self.dtargets)
                uncert_train = np.trace(self.gp.kernel_._cov_yy(self.params))
            else:
                self.gp.fit(X=self.params, y=self.targets)
                uncert_train = np.trace(self.gp.kernel_(self.params))
            self.uncert_train = np.vstack((self.uncert_train , uncert_train))
            self.theta = np.vstack((self.theta, self.gp.kernel_.theta))

            if len(self.params_test) > 0:
                if self._has_derinfo:
                    mu_test, cov_test = self.gp.predict(X=self.params_test,
                                                        full_covariance=False,
                                                        return_cov=True)
                else:
                    mu_test, cov_test = self.gp.predict(X=self.params_test,
                                                        return_cov=True)

                mse_test = mean_squared_error(self.targets_test, mu_test)
                mse_test = mse_test
                uncert_test = np.trace(cov_test)
                self.mse_test = np.vstack((self.mse_test, mse_test))
                self.uncert_test = np.vstack((self.uncert_test, uncert_test))

            itsamples+=1
            if self.verbose:
                print_log_mse_uncer(
                    self._params_keys, param_next, target_next,
                    uncert_train=uncert_train,
                    mse_test=mse_test,
                    uncert_test=uncert_test,
                    iteration=[itsamples],
                    print_header=False)

        self._record_bayes_values()

    def _record_init_values(self):
        if self._has_derinfo:
            self.init_params = self.params
            self.init_targets = self.targets
            self.init_dtargets = self.dtargets
        else:
            self.init_params = self.params
            self.init_targets = self.targets
            self.init_dtargets = []
        self.init_uncert_train = self.uncert_train
        self.init_theta = self.theta
        self.init_mse_test = self.mse_test
        self.init_uncert_test = self.uncert_test

    def _record_bayes_values(self):
        if self._has_derinfo:
            self.bayes_params = self.params[len(self.init_params):, :]
            self.bayes_targets = self.targets[len(self.init_params):, :]
            self.bayes_dtargets = self.dtargets[len(self.init_params):, :]
        else:
            self.bayes_params = self.params[len(self.init_params):, :]
            self.bayes_targets = self.targets[len(self.init_params):, :]
            self.bayes_dtargets = []
        self.bayes_uncert_train = self.uncert_train[1:]
        self.bayes_theta = self.theta[1:]
        self.bayes_mse_test = self.mse_test[1:]
        self.bayes_uncert_test = self.uncert_test[1:]

    def _eval_fun(self, params):
        targets = []
        for X in params:
            X_dict = dict(zip(self._params_keys, X))
            targets.append(self.fun(**X_dict))
        return np.asarray(targets).reshape(-1, 1)

    def _eval_dfun(self, params):
        targets = []
        for X in params:
            X_dict = dict(zip(self._params_keys, X))
            targets.append(self.dfun(**X_dict))
        return np.asarray(targets).reshape(-1, self._nparams)

    def _get_initial_training_points(self, include_ders=False):
        if self.params_train is not None:
            if self.params_train.ndim < 1:
                raise ValueError(
                    "Expected 2D array for params_train, got 1D array instead. "
                    "Reshape using array.reshape(-1, 1) if your data has only "
                    "one pameter.")
            params = np.copy(self.params_train)
            targets = self._eval_fun(params)
            if include_ders:
                dtargets = self._eval_dfun(params)
                return params, targets, dtargets
            return params, targets

        else:
            params = np.empty(shape=(0, self._nparams))
            targets = np.empty(shape=(0, 1))
            if include_ders:
                dtargets = np.empty(shape=(0, self._nparams))
                return params, targets, dtargets
            return params, targets

    def _get_random_training_points(self, include_ders):
        if self.nrand_train > 0:
            params = self.random_state.uniform(self._params_bounds_vals[:, 0],
                                               self._params_bounds_vals[:, 1],
                                               size=(self.nrand_train,
                                                     self._nparams))
            targets = self._eval_fun(params)
            if include_ders:
                dtargets = self._eval_dfun(params)
                return params, targets, dtargets
            return params, targets

        else:
            params = np.empty(shape=(0, self._nparams))
            targets = np.empty(shape=(0, 1))
            if include_ders:
                dtargets = np.empty(shape=(0, self._nparams))
                return params, targets, dtargets
            return params, targets

    def _get_test_points(self):
        if self.params_test is not None:
            if self.params_test.ndim < 1:
                raise ValueError(
                    "Expected 2D array for params_test, got 1D array instead. "
                    "Reshape using array.reshape(-1, 1) if your data has only "
                    "one pameter.")
            params = np.copy(self.params_test)
            targets = self._eval_fun(params)
            return params, targets
        else:
            return [], []

    def _get_log(self):
        return [{'target': target,
                 'parameters': dict(zip(self._params_keys, param))}
                for target, param in zip(self.target_samples,
                                         self.parameter_samples)]

    @property
    def optimal(self):
        return self._get_optimal_val()

    @property
    def log(self):
        return self._get_log()

    def _find_next_location(self, n_restarts):
        def global_uncert(X):
            return self._expected_uncertainty(X, gp=deepcopy(self.gp))

        best_x = None
        min_uncert = 1e5
        for i in range(n_restarts):
            if self.minimizer=='fmin_l_bfgs_b':
                x0 = self.random_state.uniform(self._params_bounds_vals[:, 0],
                                               self._params_bounds_vals[:, 1],
                                               size=(1, self._nparams))
                if self.ignore_convergence_warnings:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore",
                                              category=ConvergenceWarning)
                        res = optimize.minimize(
                            fun=global_uncert, x0=x0, method='L-BFGS-B',
                            bounds=self._params_bounds_vals)
                        fval = res.fun[0]
                        x = res.x
                else: # print all warnings
                    res = optimize.minimize(
                        fun=global_uncert, x0=x0, method='L-BFGS-B',
                        bounds=self._params_bounds_vals)
                    fval = res.fun[0]
                    x = res.x
            elif callable(self.minimizer):
                x, fval = self.minimizer(fun=global_uncert,
                                         bounds=self._params_bounds_vals)
            else:
                raise ValueError("Unknown minimizer %s." % self.minimizer)

            if fval < min_uncert:
                min_uncert = fval
                best_x = x

        best_x = np.array(best_x).reshape(1, self._nparams)

        return best_x, min_uncert


    def _expected_uncertainty(self, X, gp):
        X = np.reshape(X, (-1, self._nparams))
        y = gp.predict(X=X).reshape(-1, 1)
        X_temp = np.concatenate((self.params, X))
        y_temp = np.concatenate((self.targets, y))
        if self._has_derinfo:
            dy = gp.predict_der(dX=X).reshape(-1, self._nparams)
            dy_temp = np.concatenate((self.dtargets, dy))
            gp.fit(X=X_temp, y=y_temp, dX=X_temp, dy=dy_temp)
            uncert = np.trace(gp.kernel_._cov_yy(X_temp))
        else:
            gp.fit(X=X_temp, y=y_temp)
            uncert = np.trace(gp.kernel_(X_temp))

        return uncert.reshape(-1, 1)
