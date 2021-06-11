import numpy as np
import scipy
from scipy.stats import norm
from scipy import optimize
from scipy.spatial.distance import cdist
import sklearn
from sklearn.utils.validation import check_random_state
from sklearn.metrics import mean_squared_error
from copy import deepcopy

from .gaussian_process import GaussianProcessRegressor
from .gaussian_process import GPKernel, GPKernelDerAware
from .bayesian_opt_utils import print_log, print_log_mse_uncer
from .bayesian_opt_utils import plot_iter
from .minimizers import brute_minimizer, random_minimizer, hybrid_minimizer

import matplotlib.pyplot as plt

# Functions from: http://krasserm.github.io/2018/03/21/bayesian-optimization/
# https://github.com/thuijskens/bayesian-optimization/blob/master/python/gp.py

__all__ = ['BayesUncertOptimization']

class BayesUncertOptimization():
    """BayesUncertOptimization minimizes the uncertainty of the
    objective function using Gaussian Process Regression (GPR) as
    the surrogate model.
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
                 plot_iterations=False,
                 verbose=True):
        self.fun = fun
        self.params_bounds = param_bounds
        self.dfun = dfun
        self.random_state = check_random_state(random_state)
        self.plot_iterations = plot_iterations
        self.verbose = verbose
        self._has_derinfo = False if self.dfun is None else True
        self._params_keys = self._get_param_keys(param_bounds)
        self._params_bounds_vals = self._get_param_vals(param_bounds)
        self._nparams = len(self._params_keys)

    def _get_param_keys(self, param_bounds):
        return list(param_bounds.keys())

    def _get_param_vals(self, param_bounds):
        return np.array([param_bounds.get(p) for p in self._params_keys])

    def minimize_uncertainty(self,
                             params_train=None,
                             nrand_train=0,
                             params_val=None,
                             quick_search=True,
                             minimizer='hybrid',
                             gamma=0.01,
                             niters=10):
        """Minimizes the uncertainty of the GPR model.

        Parameters
        ----------
        params_train: ndarray of shape (ntr, nparams)
            Initial parameters to sample the objective function at.
            These parameters are used as the initial training points
            of the GPR. If None, random parameters are used to initiate
            the training of the GPR.

        nrand_train: int, default=0
            Number of initial random parameters to sample the objective
            function at. These parameters are used as the initial
            training points of the GPR if no params_train are given.

        params_val: ndarray of shape (nts, nparams), optional
            Validation parameters.

        quick_search: Bool, default=True
            If true, a quick search of the parameters is conducted
            which proposes the next sampling location based on the
            areas which highest uncertianty.
            Else, the next sampling location is proposed based on
            minimizing the global uncertainty.

        minimizer: 'hybrid', 'fmin_l_bfgs_b', or callable,
            default='hybrid'
            Minimizing function for the neg. acquisition function.
            'hybrid' is a hybrid approach which first performs a
            random search, followed by a grid search near the minimum
            value found by the random search.
            'fmin_l_bfgs_b' uses scipy's minimizer with the
            fmin_l_bfgs_b method.
            Alternatively, a minimzer can be provided as a callable
            with the signature:
                x0, fval = minimizer(fun, bounds)

        gamma: float > 0
            Weight of the proximity penalty. If 0, there is no
            penalty for proposing the next sampling value close
            to a value sampled before.

        niters: int, default=10
            Number of Bayesian optimization iterations to do.
        """

        self.params_train = params_train
        self.nrand_train = nrand_train
        self.params_val = params_val
        self.quick_search = quick_search
        self.minimizer = minimizer
        self.niters = niters
        self.gamma = gamma

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
            # setup GP
            self._kernel = GPKernelDerAware()
            self._gp = GaussianProcessRegressor(kernel=self._kernel,
                                                n_restarts_optimizer=10,
                                                random_state=self.random_state)
            self._gp.fit(X=self.params, y=self.targets, dy=self.dtargets)
        else:
            # get initial trainig data
            param_init, target_init = \
                self._get_initial_training_points(include_ders=False)
            param_rand, target_rand = \
                self._get_random_training_points(include_ders=False)
            self.params = np.vstack((param_init, param_rand))
            self.targets = np.vstack((target_init, target_rand))
            # setup GP
            self._kernel = GPKernel()
            self._gp = GaussianProcessRegressor(kernel=self._kernel,
                                                n_restarts_optimizer=10,
                                                random_state=self.random_state)
            self._gp.fit(X=self.params, y=self.targets)

        # evaluate the GP performance given the initial training data
        self.params_val, self.targets_val = self._get_validation_points()
        mu, cov = self._gp.predict(X=self.params_val, return_cov=True)
        mse = mean_squared_error(self.targets_val, mu).reshape(-1, 1)
        uncert = np.trace(cov).reshape(-1, 1)

        itsamples = np.shape(self.params)[0]

        if self.verbose:
            print_log_mse_uncer(self._params_keys,
                                self.params, self.targets,
                                mse=np.ones((itsamples, 1))*mse,
                                uncertainty=np.ones((itsamples, 1))*uncert,
                                iteration=list(range(1, itsamples+1)),
                                print_header=True)

        self.uncertainty = uncert
        self.mse = mse

        if self.plot_iterations:
            fig, axs = plt.subplots(5, 2, figsize=(15, 12))
            ax = axs.ravel()
            plt.subplots_adjust(hspace=0.5, wspace=0.3)

        for i in range(niters):
            # propose the next sampling location
            param_next = self._find_next_location(n_restarts=10)

            if self.plot_iterations:
                if self._nparams > 1:
                    raise ValueError("plot_iterations is only supported for 1D.")
                mu_tmp, std_tmp = self._gp.predict(
                    X=self.params_val, return_std=True)
                plot_iter(X_train=self.params,
                          y_train=self.targets,
                          X_star=self.params_val,
                          mu_star=mu_tmp,
                          std_star=std_tmp,
                          y_true=self.targets_val,
                          X_next=param_next,
                          ax=ax[i], iter=i)

            # add proposed location to the training set
            target_next = self._eval_fun(param_next)
            self.params = np.vstack((self.params, param_next))
            self.targets = np.vstack((self.targets, target_next))
            if self._has_derinfo:
                dtarget_next = self._eval_dfun(param_next)
                self.dtargets = np.vstack((self.dtargets, dtarget_next))

            # evaluate the performance
            if self._has_derinfo:
                self._gp.fit(X=self.params, y=self.targets, dy=self.dtargets)
            else:
                self._gp.fit(X=self.params, y=self.targets)
            mu, cov = self._gp.predict(X=self.params_val, return_cov=True)
            uncert = np.trace(cov).reshape(-1, 1)
            mse = mean_squared_error(self.targets_val, mu).reshape(-1, 1)
            self.mse = np.vstack((self.mse, mse))
            self.uncertainty = np.vstack((self.uncertainty, uncert))

            itsamples+=1
            if self.verbose:
                print_log_mse_uncer(
                    self._params_keys, param_next, target_next,
                    mse=mse,
                    uncertainty=uncert,
                    iteration=[itsamples],
                    print_header=False)
        if plot_iter:
            plt.show()

    def _eval_fun(self, params):
        targets = []
        for X in params:
            X_dict = dict(zip(self._params_keys,
                              np.reshape(X, (-1, self._nparams))))
            targets.append(self.fun(**X_dict))
        return np.asarray(targets).reshape(-1, 1)

    def _eval_dfun(self, params):
        targets = []
        for X in params:
            X_dict = dict(zip(self._params_keys,
                              np.reshape(X, (-1, self._nparams))))
            targets.append(self.dfun(**X_dict))
        return np.asarray(targets).reshape(-1, self._nparams)

    def _get_initial_training_points(self, include_ders=False):
        if self.params_train is not None:
            if self.params_train.dim < 1:
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
                                               size=(self.nrand_train,self._nparams))
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
        if self.quick_search:
            def neg_acq(X):
                return -self._quick_uncert_search(X, gp=deepcopy(self._gp))
        else:
            def neg_acq(X):
                return -self._global_uncert_search(X, gp=deepcopy(self._gp))

        best_x = None
        neg_uncert = 1e5
        for i in range(n_restarts):
            if self.minimizer=='hybrid':
                x0, fval = hybrid_minimizer(fun=neg_acq,
                                            N_rand=15,
                                            N_brute=5,
                                            bounds=self._params_bounds_vals,
                                            random_state=self.random_state,
                                            workers=1)
            elif self.minimizer=='fmin_l_bfgs_b':
                res = optimize.minimize(
                    fun=neg_acq, x0=x0, method='L-BFGS-B',
                    bounds=self._params_bounds_vals)
                fval = res.fun[0]
                x0 = res.x
            else:
                x0, fval = self.minimizer(fun=neg_acq,
                                          bounds=self._params_bounds_vals)

            if fval < neg_uncert:
                neg_uncert = fval
                best_x = x0

        best_x = np.array(best_x).reshape(1, self._nparams)

        return best_x

    def _quick_uncert_search(self, X, gp):
        X = np.reshape(X, (-1, self._nparams))
        mu, cov = gp.predict(X=X, return_cov=True)

        # distance between X and closest previous samples
        min_edist = np.min(cdist(X, self.params))

        uncert = np.trace(cov) + self.gamma * min_edist

        return uncert.reshape(-1, 1)

    def _global_uncert_search(self, X, gp):
        X = np.reshape(X, (-1, self._nparams))
        y = gp.predict(X=X)
        X_temp = np.concatenate((self.params, X))
        y_temp = np.concatenate((self.targets, y))
        if self._has_derinfo:
            dy = gp.predict_dX(dX=X)
            dy_temp = np.concatenate((self.dtargets, dy))
            gp.fit(X=X_temp, y=y_temp, dy=dy_temp)
        else:
            gp.fit(X=X_temp, y=y_temp)

        # distance between X and closest previous sample
        min_edist = np.min(cdist(X, self.params))

        mu, cov = gp.predict(X_temp, return_cov=True)
        neg_uncert = -(np.trace(cov) - self.gamma * min_edist)

        return neg_uncert.reshape(-1, 1)
