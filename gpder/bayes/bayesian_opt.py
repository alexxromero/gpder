import numpy as np
import scipy
from scipy.stats import norm
import sklearn
from sklearn.utils.validation import check_random_state

from ..gaussian_process import GaussianProcessRegressor
from ..gaussian_process.kernels import GPKernel, GPKernelDerAware
from .bayesian_opt_utils import print_log, print_log_mse_uncer

# Functions from: http://krasserm.github.io/2018/03/21/bayesian-optimization/
# https://github.com/thuijskens/bayesian-optimization/blob/master/python/gp.py

__all__ = ['BayesianOptimization']

def expected_improvement(X, y_samples, gp,
                         xi=0.01,
                         maximize=True):
    """Expected improvement (EI) at X.

    Arguments
    ---------
    X: array of shape (1, nparams)
        Point where the EI is computed.

    y_samples: array of shape (n, 1)
        Previous sample target values.

    gpr: GaussianProcessRegressor
        Instance of GaussianProcessRegressor.

    xi: float
        Exploitation-exploration parameter.

    maximize: Bool
        True if the optimization is to find a maximum of the
        objective function. False if it is to find a minimum.

    Returns:
        ei: float
            Expected improvement at X.
    """
    mu, std = gp.predict(X=X, return_std=True)

    if maximize:
        sample_opt_val = np.max(y_samples)
        scaling_factor = 1
    else:
        sample_opt_val = np.min(y_samples)
        scaling_factor = -1

    with np.errstate(divide='warn'):
        imp = scaling_factor * (mu - sample_opt_val - xi)
        Z = imp / std
        ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
        ei[std == 0.0] = 0.0

    return ei


class BayesianOptimization():
    """BayesianOptimization of an objective function's nuisance
    parameters using Gaussian Process Regression (GPR) as the
    surrogate model. If derivative information of the objective
    function to optimize is available, it can be used to aid
    in the GPR.

    Parameters
    ----------
    fun: callable
        Function to be optimized.

    param_bounds: dict
        Dictionary with the parameter names as the keys and a
        tuple with their correspondent bounds as the values.
        Example: {'x': (0, 1), 'y': (-1, 1)}

    dfun: callable
        Function returning the partial derivatives of the
        objective function with respect to the parameters.

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

    def optimize(self,
                 params_train=None,
                 nrand_train=0,
                 minimizer='fmin_l_bfgs_b',
                 niters=10,
                 xi=0.01,
                 minimizer_restarts=10,
                 gp_optimizer_restarts=10,
                 maximize=True,
                 workers=1):
        """Optimize the objective function.

        Parameters
        ----------
        params_train: ndarray of shape (n, nparams)
            Initial training parameters. If None, random parameters
            are used to initialize the fitting of the GPR.

        nrand_train: int, default=0
            Number of random parameters to use as the initial training
            parameters for the GPR.

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

        xi: float, default=0.01
            Exploitation-exploration trade-off parameter.

        maximize: float, default=True
            If true, find the maximum of the objective function.
            Else, find the minimum.

        minimizer_restarts: int, default=10
            Number of times to restart the 'minimizer' per Bayesian
            iteration.

        gp_optimizer_restarts: int, default=10
            Number of times to restart the optimizer of the GP's
            hyperparameters.
        """

        self.params_train = params_train
        self.nrand_train = nrand_train
        self.minimizer = minimizer
        self.niters = niters
        self.xi = xi
        self.maximize = maximize
        self.minimizer_restarts = minimizer_restarts
        self.gp_restarts = gp_optimizer_restarts

        if (self.params_train is None) and (self.nrand_train==0):
            raise ValueError(
                "Please enter either params_train and/or nrand_train > 0.")

        if self._has_derinfo:
            param_init, target_init, dtarget_init = \
                self._get_initial_training_points(include_ders=True)
            param_rand, target_rand, dtarget_rand = \
                self._get_random_training_points(include_ders=True)
            self.params = np.vstack((param_init, param_rand))
            self.targets = np.vstack((target_init, target_rand))
            self.dtargets = np.vstack((dtarget_init, dtarget_rand))
            # setup and fit GP
            self._kernel = GPKernelDerAware()
            self._gp = GaussianProcessRegressor(
                kernel=self._kernel, n_restarts_optimizer=self.gp_restarts,
                random_state=self.random_state)
            self._gp.fit(X=self.params, y=self.targets,
                         dX=self.params, dy=self.dtargets)
        else:
            param_init, target_init = \
                self._get_initial_training_points(include_ders=False)
            param_rand, target_rand = \
                self._get_random_training_points(include_ders=False)
            self.params = np.vstack((param_init, param_rand))
            self.targets = np.vstack((target_init, target_rand))
            # setup and fit GP
            self._kernel = GPKernel()
            self._gp = GaussianProcessRegressor(
                kernel=self._kernel, n_restarts_optimizer=self.gp_restarts,
                random_state=self.random_state)
            self._gp.fit(X=self.params, y=self.targets)

        # save kernel's hyperparameters at each stage
        # to see the evolution of the GP
        self.theta = self._gp.kernel_.theta
        self._record_init_values()

        itsamples = np.shape(self.params)[0]
        if self.verbose:
            print_log(self._params_keys, self.params, self.targets,
                      iteration=list(range(1, itsamples+1)),
                      print_header=True)

        if maximize:
            opt_val = np.max(self.targets)
        else:
            opt_val = np.min(self.targets)

        for i in range(niters):
            is_opt = False

            param_next = self.sample_next_location_by_EI(
                X_samples=self.params,
                target_samples=self.targets,
                gp=self._gp,
                params_bounds=self._params_bounds_vals)

            target_next = self._eval_fun(param_next)
            self.params = np.vstack((self.params, param_next))
            self.targets = np.vstack((self.targets, target_next))
            if self._has_derinfo:
                dtarget_next = self._eval_dfun(param_next)
                self.dtargets = np.vstack((self.dtargets, dtarget_next))

            if self._has_derinfo:
                self._gp.fit(X=self.params, y=self.targets,
                             dX=self.params, dy=self.dtargets)
            else:
                self._gp.fit(X=self.params, y=self.targets)
            self.theta = np.vstack((self.theta, self._gp.kernel_.theta))

            itsamples+=1
            if maximize:
                if target_next > opt_val:
                    is_opt = True
                    opt_val = target_next
            else:
                if target_next < opt_val:
                    is_opt = True
                    opt_val = target_next

            if self.verbose:
                print_log(self._params_keys, param_next, target_next,
                          iteration=[itsamples], opt_param=is_opt,
                          print_header=False)

        if self.maximize:
            self.target_optimal = np.max(self.targets)
            self.parameter_optimal = self.params[np.argmax(self.targets)]
        else:
            self.target_optimal = np.min(self.targets)
            self.parameter_optimal = self.params[np.argmin(self.targets)]

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
        self.init_theta = self.theta

    def _record_bayes_values(self):
        if self._has_derinfo:
            self.bayes_params = self.params[len(self.init_params):, :]
            self.bayes_targets = self.targets[len(self.init_params):, :]
            self.bayes_dtargets = self.dtargets[len(self.init_params):, :]
        else:
            self.bayes_params = self.params[len(self.init_params):, :]
            self.bayes_targets = self.targets[len(self.init_params):, :]
            self.bayes_dtargets = []
        self.bayes_theta = self.theta[1:]

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

    def _get_optimal_val(self):
        param_dic = dict(zip(self._params_keys, self.parameter_optimal.T))
        desc = {'target': self.target_optimal,
                'parameters': param_dic}
        return desc

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

    def evaluate(self, params, update_log=True):
        keys = self._get_param_keys(params)
        X = self._get_param_vals(params)
        target = self._eval_fun(X)
        if update_log:
            self.params = np.vstack((self.params, X))
            self.targets = np.vstack((self.targets, target))
        return target

    def sample_next_location_by_EI(self,
                                   X_samples,
                                   target_samples,
                                   gp,
                                   params_bounds,
                                   minimizer='fmin_l_bfgs_b',
                                   n_restarts=25):
        """Location of the next proposed sample according to
        its Expected improvement (ei).

        Args:
            X_samples: array of shape (n_samples, d)
                Previous parameters.
            target_samples: array of shape (n_samples, 1)
                Previous target values.
            gp: GaussianProcessRegressor
                Instance of GaussianProcessRegressor fitted to
                (X_samples, target_samples).
            params_bounds: dictionary
                Dictionary with the lower and upper bounds of the
                parameters.
            minimizer: 'fmin_l_bfgs_b' or 'brute'
                Minimization function.
            n_restarts: int
                Number of times to restart the minimizer.

        Returns:
            Location of the acquisition function maximum.
            This location is taken to be the next proposed
            sampling point.
        """
        def neg_acq(X):
            return -expected_improvement(X=X.reshape(-1, self._nparams),
                                         y_samples=target_samples,
                                         gp=gp,
                                         xi=self.xi,
                                         maximize=self.maximize)
        best_x = None
        min_val = 1e5
        for i in range(n_restarts):
            if minimizer=='fmin_l_bfgs_b':
                x0 = self.random_state.uniform(self._params_bounds_vals[:, 0],
                                               self._params_bounds_vals[:, 1],
                                               size=(1, self._nparams))
                res = scipy.optimize.minimize(
                        fun=neg_acq,
                        x0=x0,
                        method='L-BFGS-B',
                        bounds=self._params_bounds_vals)
                fval = res.fun[0]
                x = res.x
            elif callable(self.minimizer):
                x, fval = self.minimizer(fun=neg_acq,
                                         bounds=self._params_bounds_vals)
            else:
                raise ValueError("Unknown minimizer %s." % self.minimizer)

            if fval < min_val:
                min_val = fval
                best_x = x

        return np.array(best_x).reshape(1, self._nparams)
