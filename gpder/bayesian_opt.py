import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.utils.validation import check_random_state

from .gaussian_process import GaussianProcessRegressor
from .gaussian_process import GPKernel, GPKernelDerAware
from .plotting_utils import plot_approximation, plot_acquisition
from .bayesian_opt_utils import print_log, print_log_uncertainty

# Functions from: http://krasserm.github.io/2018/03/21/bayesian-optimization/
# https://github.com/thuijskens/bayesian-optimization/blob/master/python/gp.py

__all__ = ['BayesianOptimization']

def expected_improvement(X, X_samples, y_samples, gp,
                         uncert_samples=None,
                         xi=0.01, maximize=True):
    """Expected improvement (EI) at X according to the
    GP regressor model fit on X_samples and y_samples.

    Args:
        X: array of shape (n, d)
            Points where the EI is computed.
        X_sample: array of shape (n_samples, d)
            Sample location points for the gpr.
        y_sample: array of shape (n_samples, 1)
            Sample target values for the gpr.
        gpr: GaussianProcessRegressor
            An instance of GaussianProcessRegressor fitted to the
            the sample points.

    Returns:
        Expected improvement at points X.
    """
    mu, std = gp.predict(X=X, return_std=True)

    if maximize:
        sample_opt_val = np.max(y_sample)
        scaling_factor = 1
    else:
        sample_opt_val = np.min(y_sample)
        scaling_factor = -1

    with np.errstate(divide='warn'):
        imp = scaling_factor * (mu - sample_opt_val - xi)
        Z = imp / std
        ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
        ei[std == 0.0] = 0.0

    return ei


def expected_uncertainty(X, X_samples, y_samples, gp,
                         uncert_samples,
                         return_uncert=False):
    """Expected max uncertainty (EU) at X according to the
    GP regressor model fit on X_samples and y_samples, and the
    record of the model's uncertainty on y_samples.

    Args:
        X: array of shape (n, d)
            Points where the EI is computed.
        X_sample: array of shape (n_samples, d)
            Sample location points for the gpr.
        y_sample: array of shape (n_samples, 1)
            Sample target values for the gpr.
        uncert_sample: array of shape (n_samples, 1)
            Uncertainty at previous X_sample.
        gpr: GaussianProcessRegressor
            An instance of GaussianProcessRegressor fitted to the
            the sample points.

    Returns:
        Difference between the expected uncertainty at X and
        the previous maximum uncertainty at X_samples.
    """
    mu, std = gp.predict(X=X, return_std=True)
    _, cov = gp.predict(X=X, return_cov=True)

    #uncert = np.sqrt(np.linalg.det(cov))
    uncert = np.trace(cov)

    if return_uncert:
        return uncert

    uncert_max = np.max(uncert_samples)

    #util = np.trace(cov) - uncert_max  # og util
    util = np.trace(cov)  # uncert util

    return util


class BayesianOptimization():
    """Bayesian optimization of an objective function with
    Gaussian Process. The Gaussian Process Regression allows for
    derivative observations to improve the regression model.

    Parameters
    ----------
    fun: callable
        Objective function to be optimized.

    dfun: callable
        Function returning the derivative of the objective
        function with respect to the parameters.

    param_bounds: dict
        Dictionary with the parameter names as the keys and a
        tuple with their correspondent bounds as the values.
        Example: {'x': (0, 1), 'y': (-1, 1)}

    random_state: RandomState instance or None, default=None
        Determines the random number generator used in the GPR
        optimization.
    """

    def __init__(self, fun, param_bounds, dfun=None, random_state=None):
        self.fun = fun
        self.dfun = dfun
        self._has_derinfo = True if self.dfun is not None else False
        self.params_bounds = param_bounds
        self._params_keys = self._get_param_keys(param_bounds)
        self._params_bounds_vals = self._get_param_vals(param_bounds)
        self._nparams = len(self._params_keys)
        self.random_state = check_random_state(random_state)

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

    def optimize(self, params_init=None, n_rand=0, n_iters=10,
                 xi=0.01, maximize=True):
        """Optimize the objective function.

        Parameters
        ----------
        params_init: ndarray of shape (n, nparams)
            Locations of the initial points to sample the objective
            function at. If None, random points are used to sample
            the objective function.

        n_rand: int, default=0
            Number of random points to sample from the objective
            function.

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
        print_log(self._params_keys, self.params, self.targets,
                  iteration=list(range(itsamples)),
                  print_header=True)

        if maximize:
            opt_val = np.max(self.targets)
        else:
            opt_val = np.min(self.targets)
        for i in range(n_iters):
            is_opt = False
            if self._has_derinfo:
                self._gp.fit(X = self.params, y = self.targets,
                             dy = self.dtargets)
            else:
                self._gp.fit(X=self.params, y=self.targets)

            param_next = self.sample_next_location_by_EI(
                X_samples=self.params,
                target_samples=self.targets,
                gp=self._gp,
                params_bounds=self._params_bounds_vals
                )

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

            print_log(self._params_keys, param_next, target_next,
                      iteration=[itsamples], opt_param=is_opt,
                      print_header=False)

        if self.maximize:
            self.target_opt = np.max(self.targets)
            self.parameter_opt = self.params[np.argmax(self.targets)]
        else:
            self.target_opt = np.min(self.targets)
            self.parameter_opt = self.params[np.argmin(self.targets)]


    def minimize_uncertainty(self, params_init=None, n_rand=0, n_iters=10):
        """Minimize the uncertainty of the GPR on the objective
        function.

        Parameters
        ----------
        params_init: ndarray of shape (n, nparams)
            Locations of the initial points to sample the objective
            function at. If None, random points are used to sample
            the objective function.

        n_rand: int, default=0
            Number of random points to sample from the objective
            function.

        n_iters: int, default=10
            Number of iterations to run the search algorithm for.
        """

        self.params_init = params_init
        self.n_rand = n_rand
        self.n_iters = n_iters
        self.n_restarts = 25

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
        # print_log(self._params_keys, self.params, self.targets,
        #           iteration=list(range(itsamples)),
        #           print_header=True)
        print_log_uncertainty(self._params_keys, self.params, self.targets,
                  uncertainty=np.zeros((itsamples, 1)),
                  iteration=list(range(itsamples)),
                  print_header=True)

        # dummy number to kick off the uncertainty search
        self.uncertainties = [0]

        for i in range(n_iters):
            # (1) fit the GP
            if self._has_derinfo:
                self._gp.fit(X=self.params, y=self.targets,
                             dy=self.dtargets)
            else:
                self._gp.fit(X=self.params, y=self.targets)
            # (2) propose next sampling location
            param_next, uncert_next = self.sample_next_location_by_EU(
                X_samples=self.params,        # previous sampling locations (X)
                target_samples=self.targets,  # previous function evaluations
                gp=self._gp,                  # GP trained on previous X
                params_bounds=self._params_bounds_vals,  # bounds of X
                uncertainty_samples=self.uncertainties,  # pred uncert
                )

            target_next = self._eval_fun(param_next)
            self.params = np.vstack((self.params, param_next))
            self.targets = np.vstack((self.targets, target_next))
            self.uncertainties = np.vstack((self.uncertainties, uncert_next))
            if self._has_derinfo:
                dtarget_next = self._eval_dfun(param_next)
                self.dtargets = np.vstack((self.dtargets, dtarget_next))

            itsamples+=1
            # print_log(self._params_keys, param_next, target_next,
            #           iteration=[itsamples], print_header=False)
            print_log_uncertainty(self._params_keys, param_next, target_next,
                      uncertainty=uncert_next,
                      iteration=[itsamples], print_header=False)

    def _get_initial_training_points(self, include_ders=False):
        if self.params_init is not None:
            params = np.copy(self.params_init).reshape(-1, self._nparams)
            targets = self._eval_fun(params)
            if include_ders:
                dtargets = self._eval_dfun(params)
        else:
            params = np.empty(shape=(0, self._nparams))
            targets = np.empty(shape=(0, 1))
            if include_ders:
                dtargets = np.empty(shape=(0, self._nparams))

        if include_ders:
            return params, targets, dtargets
        else:
            return params, targets


    def _get_random_training_points(self, include_ders):
        if self.n_rand > 0:
            params = np.random.uniform(self._params_bounds_vals[:, 0],
                                       self._params_bounds_vals[:, 1],
                                       size=(n_rand, self._nparams))
            targets = self._eval_fun(params)
            if include_ders:
                dtargets = self._eval_dfun(params)
        else:
            params = np.empty(shape=(0, self._nparams))
            targets = np.empty(shape=(0, 1))
            if include_ders:
                dtargets = np.empty(shape=(0, self._nparams))

        if include_ders:
            return params, targets, dtargets
        else:
            return params, targets


    def _get_optimal_val(self):
        param_dic = dict(zip(self._params_keys, self.parameter_opt.T))
        desc = {'target': self.target_opt,
                'parameters': param_dic}
        return desc

    def _get_log(self):
        return [{'target': target, 'parameters': dict(zip(self._params_keys, param))}
                for target, param in zip(self.target_samples, self.parameter_samples)]

    @property
    def optimal(self):
        return self._get_optimal_val()

    @property
    def log(self):
        return self._get_log()

    def evaluate(self, params, update_log=True):
        keys = self._get_param_keys(params)
        if keys != self._params_keys:
            print("assert here")
        X = self._get_param_vals(params)
        target = self._eval_fun(X)
        if update_log:
            self.param_samples = np.vstack((self.param_samples, X))
            self.target_samples = np.vstack((self.target_samples, target))
        return target


    def sample_next_location_by_EI(self,
                                   X_samples,
                                   target_samples,
                                   gp,
                                   params_bounds,
                                   n_restarts=25):
        """Location of the proposed next sampling by optimizing the
        acquisition function (Expected improvement).

        Args:
            X_samples: array of shape (n_samples, d)
                Sample location points for the gp.
            target_samples: array of shape (n_samples, 1)
                Sample target values for the gp.
            uncertainty_samples: array of shape (n_samples, 1)
                Uncertainty of the target values for the gp.
            gp: GaussianProcessRegressor
                Instance of GaussianProcessRegressor fitted to the samples.
            params_bounds: dictionary
                Dictionary with the lower and upper bounds of the
                parameters.
            n_restarts: int
                Number of times to restart the something...

        Returns:
            Location of the acquisition function maximum.
        """
        def neg_acq(X):
            return -expected_improvement(X=X.reshape(-1, self._nparams),
                                         X_samples=X_samples,
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
            res = minimize(fun=neg_acq, x0=x0, method='L-BFGS-B',
                           bounds=self._params_bounds_vals)
            if res.fun < min_val:
                min_val = res.fun[0]
                best_x = res.x

        best_x = np.array(best_x).reshape(1, self._nparams)

        return best_x


    def sample_next_location_by_EU(self,
                                   X_samples,
                                   target_samples,
                                   gp,
                                   params_bounds,
                                   uncertainty_samples,
                                   n_restarts=25):
        """Location of the proposed next sampling by picking
        points according to the model's uncertainty
        (expected uncertainty).

        Args:
            acquisition_func: Callable
                Acquisition function to optimize.
            X_samples: array of shape (n_samples, d)
                Sample location points for the gp.
            target_samples: array of shape (n_samples, 1)
                Sample target values for the gp.
            gp: GaussianProcessRegressor
                Instance of GaussianProcessRegressor fitted to the samples.
            params_bounds: dictionary
                Dictionary with the lower and upper bounds of the
                parameters
            uncertainty_samples: array of shape (n_samples, 1)
                Uncertainty of the target values for the gp.
            return_uncertainty: bool, default=False
                If true, also return the expected uncertainty at the
                proposed sample location.
            n_restarts: int
                Number of times to restart the something...

        Returns:
            Location of the acquisition function maximum.
        """
        def neg_acq(X):
            return -expected_uncertainty(X=X.reshape(-1, self._nparams),
                                         X_samples=X_samples,
                                         y_samples=target_samples,
                                         gp=gp,
                                         uncert_samples=uncertainty_samples)

        best_x = None
        min_val = 1e5
        for i in range(n_restarts):
            x0 = self.random_state.uniform(self._params_bounds_vals[:, 0],
                                           self._params_bounds_vals[:, 1],
                                           size=(1, self._nparams))
            res = minimize(fun=neg_acq, x0=x0, method='L-BFGS-B',
                           bounds=self._params_bounds_vals)
            if res.fun < min_val:
                min_val = res.fun
                best_x = res.x

        best_x = np.array(best_x).reshape(1, self._nparams)
        uncert = expected_uncertainty(X=best_x,
                                      X_samples=X_samples,
                                      y_samples=target_samples,
                                      uncert_samples=uncertainty_samples,
                                      gp=gp,
                                      return_uncert=True)
        uncert = np.array(uncert).reshape(-1, 1)
        return best_x, uncert
