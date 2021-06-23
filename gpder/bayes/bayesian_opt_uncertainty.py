import numpy as np
import scipy
from scipy.stats import norm
from scipy import optimize
from scipy.spatial.distance import cdist
import sklearn
from sklearn.utils.validation import check_random_state
from sklearn.metrics import mean_squared_error
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
    objective function using a Bayesian optimization routine with
    Gaussian Process Regression (GPR) as the surrogate model.
    If derivative information of the objective function is available,
    it is used to aid the gpr.

    Parameters
    ----------
    fun: callable
        Objective function.

    param_bounds: dict
        Dictionary with the parameter names as the keys and a
        tuple with their correspondent bounds as the values.
        Example: {'x': (0, 1), 'y': (-1, 1)}

    dfun: callable
        Function returning the derivative of the objective
        function with respect to the hyperparameters.

    random_state: RandomState instance or None, default=None
        Determines the random number generator used in the GPR
        optimization.

    verbose: bool
        If True, print a table with the selected hyperparameters
        per iter.
    """

    def __init__(self, fun, param_bounds, dfun=None,
                 random_state=None,
                 verbose=True):
        self.fun = fun
        self.params_bounds = param_bounds
        self.dfun = dfun
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
                             params_val=None,
                             minimizer='fmin_l_bfgs_b',
                             gamma=0.01,
                             niters=10,
                             nminimizer_restarts=10,
                             workers=1):
        """Minimizes the uncertainty of the GPR model.

        Parameters
        ----------
        params_train: ndarray of shape (ntr, nparams)
            Initial training parameters. If None, random parameters
            are used to initialize the fitting of the GPR.

        nrand_train: int, default=0
            Number of random parameters to use as the initial training
            parameters for the GPR.

        params_val: ndarray of shape (nts, nparams), optional
            Validation parameters.

        minimizer: 'fmin_l_bfgs_b', or callable, default='fmin_l_bfgs_b'
            Minimizing function for the neg. acquisition function.
            If 'fmin_l_bfgs_b', scipy's minimizer function with the
            fmin_l_bfgs_b method is used.
            Alternatively, a minimzer can be provided as a callable
            with the signature:
                x, fval = minimizer(fun, bounds)
            See minimizers.py for brute-force and random search
            minimizer functions.

        gamma: float > 0
            Weight of the proximity penalty. If 0, there is no
            penalty for sampling next/on a previously smapled parameter.

        niters: int, default=10
            Number of Bayesian optimization iterations to do.

        nminimizer_restarts: int, default=10
            Number of times to restart the 'minimizer' per Bayesian
            iteration.
        """

        self.params_train = params_train
        self.nrand_train = nrand_train
        self.params_val = params_val
        self.minimizer = minimizer
        self.gamma = gamma
        self.niters = niters
        self.nminimizer_restarts = nminimizer_restarts
        self.workers = workers

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
            # setup and fit GP
            self._kernel = GPKernelDerAware()
            self._gp = GaussianProcessRegressor(kernel=self._kernel,
                                                n_restarts_optimizer=10,
                                                random_state=self.random_state)
            self._gp.fit(X=self.params, y=self.targets,
                         dX=self.params, dy=self.dtargets)
        else:
            # get initial trainig data
            param_init, target_init = \
                self._get_initial_training_points(include_ders=False)
            param_rand, target_rand = \
                self._get_random_training_points(include_ders=False)
            self.params = np.vstack((param_init, param_rand))
            self.targets = np.vstack((target_init, target_rand))
            # setup and fit GP
            self._kernel = GPKernel()
            self._gp = GaussianProcessRegressor(kernel=self._kernel,
                                                n_restarts_optimizer=10,
                                                random_state=self.random_state)
            self._gp.fit(X=self.params, y=self.targets)

        itsamples = np.shape(self.params)[0]

        # calculate the covariance of the initial parameters
        # mu, cov = self._gp.predict(X=self.params, return_cov=True)
        cov = self._gp.kernel_(self.params, self.params)
        uncert = np.trace(cov)
        self.uncert = uncert
        # also save kernel's parameters at each stage to see the
        # evolution of the GP
        self.theta = self._gp.kernel_.theta

        # if validation data is given, calculate the MSE
        if len(self.params_val) > 0:
            self.params_val, self.targets_val = self._get_validation_points()
            mu_val, cov_val = self._gp.predict(X=self.params_val,
                                               return_cov=True)
            mse_val = mean_squared_error(self.targets_val, mu_val)
            mse_val = mse_val
            uncert_val = np.trace(cov_val)
            self.mse_val = mse_val
            self.uncert_val = uncert_val
        else:
            self.mse_val = None
            self.uncert_val = None

        self._record_init_values()

        if self.verbose:
            print_log_mse_uncer(self._params_keys,
                                self.params,
                                self.targets,
                                uncert=uncert,
                                mse_val=mse_val,
                                uncert_val=uncert_val,
                                iteration=list(range(1, itsamples+1)),
                                initial_params=True,
                                print_header=True)

        for i in range(niters):
            # propose the next sampling location
            param_next, uncert_next = self._find_next_location(
                n_restarts=self.nminimizer_restarts)

            # add proposed location to the training set
            target_next = self._eval_fun(param_next)
            self.params = np.vstack((self.params, param_next))
            self.targets = np.vstack((self.targets, target_next))
            if self._has_derinfo:
                dtarget_next = self._eval_dfun(param_next)
                self.dtargets = np.vstack((self.dtargets, dtarget_next))

            # refit the GP
            if self._has_derinfo:
                self._gp.fit(X=self.params, y=self.targets,
                             dX=self.params, dy=self.dtargets)
            else:
                self._gp.fit(X=self.params, y=self.targets)
            # mu, cov = self._gp.predict(X=self.params, return_cov=True)
            cov = self._gp.kernel_(self.params, self.params)
            uncert = np.trace(cov)
            self.uncert = np.vstack((self.uncert , uncert))
            self.theta = np.vstack((self.theta, self._gp.kernel_.theta))

            if len(self.params_val) > 0:
                mu_val, cov_val = self._gp.predict(X=self.params_val,
                                                   return_cov=True)
                mse_val = mean_squared_error(self.targets_val, mu_val)
                mse_val = mse_val
                uncert_val = np.trace(cov_val)
                self.mse_val = np.vstack((self.mse_val, mse_val))
                self.uncert_val = np.vstack((self.uncert_val, uncert_val))

            itsamples+=1
            if self.verbose:
                print_log_mse_uncer(
                    self._params_keys, param_next, target_next,
                    uncert=uncert,
                    mse_val=mse_val,
                    uncert_val=uncert_val,
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
        self.init_uncert = self.uncert
        self.init_theta = self.theta
        self.init_mse_val = self.mse_val
        self.init_uncert_val = self.uncert_val

    def _record_bayes_values(self):
        if self._has_derinfo:
            self.bayes_params = self.params[len(self.init_params):, :]
            self.bayes_targets = self.targets[len(self.init_params):, :]
            self.bayes_dtargets = self.dtargets[len(self.init_params):, :]
        else:
            self.bayes_params = self.params[len(self.init_params):, :]
            self.bayes_targets = self.targets[len(self.init_params):, :]
            self.bayes_dtargets = []
        self.bayes_uncert = self.uncert[1:]
        self.bayes_theta = self.theta[1:]
        self.bayes_mse_val = self.mse_val[1:]
        self.bayes_uncert_val = self.uncert_val[1:]

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

    def _get_validation_points(self):
        if self.params_val is not None:
            if self.params_val.ndim < 1:
                raise ValueError(
                    "Expected 2D array for params_val, got 1D array instead. "
                    "Reshape using array.reshape(-1, 1) if your data has only "
                    "one pameter.")
            params = np.copy(self.params_val)
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
        def neg_acq(X):
            return -self._global_uncert_search(X, gp=deepcopy(self._gp))

        best_x = None
        neg_uncert = 1e5
        for i in range(n_restarts):
            if self.minimizer=='fmin_l_bfgs_b':
                x0 = self.random_state.uniform(self._params_bounds_vals[:, 0],
                                               self._params_bounds_vals[:, 1],
                                               size=(1, self._nparams))
                res = optimize.minimize(
                    fun=neg_acq, x0=x0, method='L-BFGS-B',
                    bounds=self._params_bounds_vals)
                fval = res.fun[0]
                x = res.x
            elif callable(self.minimizer):
                x, fval = self.minimizer(fun=neg_acq,
                                         bounds=self._params_bounds_vals)
            else:
                raise ValueError("Unknown minimizer %s." % self.minimizer)

            if fval < neg_uncert:
                neg_uncert = fval
                best_x = x

        best_x = np.array(best_x).reshape(1, self._nparams)

        return best_x, -neg_uncert

    def _quick_uncert_search(self, X, gp):  # deprecated
        X = np.reshape(X, (-1, self._nparams))
        mu, cov = gp.predict(X=X, return_cov=True)

        # distance between X and closest previous samples
        min_edist = np.min(cdist(X, self.params))

        uncert = np.trace(cov) + self.gamma * min_edist

        return uncert.reshape(-1, 1)

    def _global_uncert_search(self, X, gp):
        X = np.reshape(X, (-1, self._nparams))
        y = gp.predict(X=X).reshape(-1, 1)
        X_temp = np.concatenate((self.params, X))
        y_temp = np.concatenate((self.targets, y))
        if self._has_derinfo:
            dy = gp.predict_der(dX=X).reshape(-1, self._nparams)
            dy_temp = np.concatenate((self.dtargets, dy))
            gp.fit(X=X_temp, y=y_temp, dX=X_temp, dy=dy_temp)
        else:
            gp.fit(X=X_temp, y=y_temp)

        # distance between X and closest previous sample
        min_edist = np.min(cdist(X, self.params))

        # mu, cov = gp.predict(X_temp, return_cov=True)
        cov = gp.kernel_(X_temp, X_temp)
        neg_uncert = -(np.trace(cov) - self.gamma * min_edist)

        return neg_uncert.reshape(-1, 1)
