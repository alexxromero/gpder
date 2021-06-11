import numpy as np
import scipy
from scipy.stats import norm
import sklearn
from sklearn.utils.validation import check_random_state

from .gaussian_process import GaussianProcessRegressor
from .gaussian_process import GPKernel, GPKernelDerAware
from .bayesian_opt_utils import print_log, print_log_mse_uncer
from .bayesian_opt_utils import plot_iter

# Functions from: http://krasserm.github.io/2018/03/21/bayesian-optimization/
# https://github.com/thuijskens/bayesian-optimization/blob/master/python/gp.py

__all__ = ['BayesianOptimization']

def expected_improvement(X, y_samples, gp,
                         xi=0.01,
                         maximize=True):
    """Expected improvement (EI) at X according to the
    GaussianProcessRegressor (gp) fit to X_samples and y_samples.

    Arguments
    ---------
    X: array of shape (n, d)
        Points where the EI is computed.

    y_samples: array of shape (ny, 1)
        Sample target values for the gpr.

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
    """BayesianOptimization of the objective function's nuisance
    parameters using Gaussian Process Regression (gpr) as the
    surrogate model. If derivative information of the objective
    function to optimize is available, it can be used to aid the gpr.

    Parameters
    ----------
    fun: callable
        Function to be optimized.

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

    def __init__(self, fun, param_bounds, dfun=None, random_state=None,
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

        if self._has_derinfo:
            self._kernel = GPKernelDerAware()
        else:
            self._kernel = GPKernel()
        self._gp = GaussianProcessRegressor(kernel=self._kernel,
                                            n_restarts_optimizer=10,
                                            random_state=self.random_state)

    def _get_param_keys(self, param_bounds):
        return list(param_bounds.keys())

    def _get_param_vals(self, param_bounds):
        return np.array([param_bounds.get(p) for p in self._params_keys])

    def optimize(self, params_init=None, n_rand=0, n_iters=10,
                 xi=0.01, maximize=True):
        """Optimize the objective function.

        Parameters
        ----------
        params_init: ndarray of shape (n, nparams)
            Locations of the initial points to sample the objective
            function at. If None, random points are used to sample
            the objective function and initialize the optimization.

        n_rand: int, default=0
            Number of initial random points to sample from the
            objective function.

        n_iters: int, default=10
            Number of iterations to run the search algorithm for.

        xi: float, default=0.01
            Exploitation-exploration trade-off parameter.

        maximize: float, default=True
            If true, find the maximum of the objective function.
            Else, find the minimum.
        """

        self.params_init = params_init
        self.n_rand = n_rand
        self.n_iters = n_iters
        self.xi = xi
        self.maximize = maximize

        if self._has_derinfo:
            param_init, target_init, dtarget_init = \
                self._get_initial_training_points(include_ders=True)
            param_rand, target_rand, dtarget_rand = \
                self._get_random_training_points(include_ders=True)
            self.params = np.vstack((param_init, param_rand))
            self.targets = np.vstack((target_init, target_rand))
            self.dtargets = np.vstack((dtarget_init, dtarget_rand))
        else:
            param_init, target_init = \
                self._get_initial_training_points(include_ders=False)
            param_rand, target_rand = \
                self._get_random_training_points(include_ders=False)
            self.params = np.vstack((param_init, param_rand))
            self.targets = np.vstack((target_init, target_rand))

        itsamples = np.shape(self.params)[0]
        if self.verbose:
            print_log(self._params_keys, self.params, self.targets,
                      iteration=list(range(1, itsamples+1)),
                      print_header=True)

        if maximize:
            opt_val = np.max(self.targets)
        else:
            opt_val = np.min(self.targets)
        for i in range(n_iters):
            is_opt = False
            if self._has_derinfo:
                self._gp.fit(X=self.params, y=self.targets,
                             dy=self.dtargets)
            else:
                self._gp.fit(X=self.params, y=self.targets)

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
            self.target_opt = np.max(self.targets)
            self.parameter_opt = self.params[np.argmax(self.targets)]
        else:
            self.target_opt = np.min(self.targets)
            self.parameter_opt = self.params[np.argmin(self.targets)]

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
        if self.params_init is not None:
            if self.params_init.dim < 1:
                raise ValueError(
                    "Expected 2D array for params_init, got 1D array instead. "
                    "Reshape using array.reshape(-1, 1) if your data has only "
                    "one pameter.")
            params = np.copy(self.params_init)
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
        if self.n_rand > 0:
            params = self.random_state.uniform(self._params_bounds_vals[:, 0],
                                               self._params_bounds_vals[:, 1],
                                               size=(self.n_rand,self._nparams))
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
        param_dic = dict(zip(self._params_keys, self.parameter_opt.T))
        desc = {'target': self.target_opt,
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
        """Location of the proposed next sampling according to
        its Expected improvement (ei).

        Args:
            X_samples: array of shape (n_samples, d)
                Sample location points for the gp.
            target_samples: array of shape (n_samples, 1)
                Sample target values for the gp.
            gp: GaussianProcessRegressor
                Instance of GaussianProcessRegressor fitted to the samples.
            params_bounds: dictionary
                Dictionary with the lower and upper bounds of the
                parameters.
            minimizer: 'fmin_l_bfgs_b' or 'brute'
                Minimization algorithm.
            n_restarts: int
                Number of times to restart the minimization algo.

        Returns:
            Location of the acquisition function maximum.
        """
        def neg_acq(X):
            return -expected_improvement(X=X.reshape(-1, self._nparams),
                                         y_samples=target_samples,
                                         gp=gp,
                                         xi=self.xi,
                                         maximize=self.maximize)
        best_x = None
        min_val = 1
        for i in range(n_restarts):
            x0 = self.random_state.uniform(self._params_bounds_vals[:, 0],
                                           self._params_bounds_vals[:, 1],
                                           size=(1, self._nparams))
            if minimizer=='fmin_l_bfgs_b':
                res = scipy.optimize.minimize(
                        fun=neg_acq,
                        x0=x0,
                        method='L-BFGS-B',
                        bounds=self._params_bounds_vals)
                if res.fun[0] < min_val:
                    min_val = res.fun[0]
                    best_x = res.x
            elif minimizer=='brute':
                x0, fval, grid, _ = scipy.optimize.brute(
                                        func=neg_acq,
                                        ranges=self._params_bounds_vals,
                                        full_output=True,
                                        finish=None)
                if fval < min_val:
                    min_val = fval
                    best_x = x0
            else:
                raise ValueError("minimizer must be 'L-BFGS-B' or 'brute'.")

        best_x = np.array(best_x).reshape(1, self._nparams)

        if self.plot_iterations:
            if self._nparams > 1:
                raise ValueError("plot_iterations is only supported for 1D.")
            X_star = np.linspace(self._params_bounds_vals[0][0],
                                 self._params_bounds_vals[0][1],
                                 num=100)
            mu, std = gp.predict(X=X_star, return_std=True)
            y_true = self.fun(X_star)
            plot_iter(X_train=X_samples, y_train=target_samples,
                      X_star=X_star, mu_star=mu.ravel(), std_star=std,
                      y_true=y_true,
                      X_next=best_x)

        return best_x
