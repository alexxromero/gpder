import numpy as np
import scipy
import sklearn
from sklearn.utils.validation import check_random_state

from ..gaussian_process import GaussianProcessRegressor
from ..gaussian_process.kernels import GPKernel, GPKernelDerAware
from .bayesian_opt_utils import print_log, print_log_mse_uncer

# Functions from: http://krasserm.github.io/2018/03/21/bayesian-optimization/
# https://github.com/thuijskens/bayesian-optimization/blob/master/python/gp.py

__all__ = ['BayesOptimization']

def expected_improvement(X, y_samples, gp, xi=0.01, maximize=True):
    """Expected improvement (EI) at X according to the
    GaussianProcessRegressor (gp) fit to X_samples and y_samples.

    Arguments
    ---------
    X: array of shape (n, 1)
        Point where the EI is computed.

    y_samples: array of shape (ny, 1)
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
        ei = imp * scipy.stats.norm.cdf(Z) + std * scipy.stats.norm.pdf(Z)
        ei[std == 0.0] = 0.0

    return ei


class BayesOptimization():
    """BayesOptimization minimizes (or maximazes) the the objective
    function by using a Bayesian optimization routine with
    Gaussian Process Regression (GPR) as the surrogate model.
    If derivative information of the objective function is available,
    it is used to aid the GPR.

    Parameters
    ----------
    fun: callable
        Objective function.

    param_bounds: dict
        Dictionary with the parameter names as the keys and their
        upper/lower bounds as the values.
        Example: {'param1_name': (0, 1), 'param2_name': (-1, 1)}

    dfun: callable
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
        self._X_keys = self._get_X_keys(param_bounds)
        self._X_bounds_vals = self._get_X_bounds_vals(param_bounds)
        self._nparams = len(self._X_keys)

    def _get_X_keys(self, param_bounds):
        return list(param_bounds.keys())

    def _get_X_bounds_vals(self, param_bounds):
        return np.array([param_bounds.get(p) for p in self._X_keys])

    def optimize(self,
                 X_train, y_train,
                 dX_train=None, dy_train=None,
                 n_iters=10,
                 xi=0.01,
                 minimizer='fmin_l_bfgs_b',
                 n_minimizer_restarts=10,
                 maximize=False,
                 workers=1):
        """Optimize the objective function 'fun'.

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

        n_iters: int, default=10
            Number of Bayesian optimization iterations to do.

        xi: float, default=0.01
            Exploitation-exploration trade-off parameter.

        minimizer: 'fmin_l_bfgs_b' or callable, default='fmin_l_bfgs_b'
            Minimizing function for the neg. acquisition function.
            If 'fmin_l_bfgs_b', scipy's minimizer function with the
            fmin_l_bfgs_b method is used.
            Alternatively, a minimzer can be provided as a callable
            with the signature:
                x, fval = minimizer(fun, bounds)
            See minimizers.py for brute-force and random search
            minimizer functions.

        n_minimizer_restarts: int, default=10
            Number of times to restart the 'minimizer' algorithm
            at every iteration.

        maximize: bool, default=True
            If True, the objective function is maximized.
            If False, the objective function is minimized.
        """

        self.X_train, self.y_train = X_train, y_train
        self.dX_train, self.dy_train = dX_train, dy_train
        self.n_iters = n_iters
        self.xi = xi
        self.n_minimizer_restarts = n_minimizer_restarts
        self.minimizer = minimizer
        self.maximize = maximize
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
        self._record_init_values()
        if self.verbose:
            print_log(self._X_keys, self.X_train, self.y_train,
                      iteration_ids=[0]*self._n_init,
                      print_header=True)

        if maximize:
            opt_val = np.max(self.y_train)
        else:
            opt_val = np.min(self.y_train)

        for i in range(self.n_iters):
            is_opt = False
            X_next = self.sample_next_X_by_EI(
                X_samples=self.X_train,
                target_samples=self.y_train,
                gp=self.gp,
                params_bounds=self._X_bounds_vals
                )
            y_next = self._eval_fun(X_next)
            # update the sample set to include the new sample
            self.X_train = np.vstack((self.X_train, X_next))
            self.y_train = np.concatenate((self.y_train, y_next))
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
            # check if a new minimum/maximum is found
            if maximize:
                if y_next > opt_val:
                    is_opt = True
                    opt_val = y_next
            else:
                if y_next < opt_val:
                    is_opt = True
                    opt_val = y_next

            if self.verbose:
                print_log(self._X_keys, X_next, y_next,
                          iteration_ids=[i+1],
                          opt_param=is_opt,
                          print_header=False)

        if self.maximize:
            self.y_optimal = np.max(self.y_train)
            self.X_optimal = self.X_train[np.argmax(self.y_train)]
        else:
            self.y_optimal = np.min(self.y_train)
            self.X_optimal = self.X_train[np.argmin(self.y_train)]

        self._record_bayes_values()

    def _record_init_values(self):
        self._X_init = self.X_train
        self._y_init = self.y_train
        if self._has_derinfo:
            self._dX_init = self.dX_train
            self._dy_init = self.dy_train
        else:
            self._dX_init = []
            self._dy_init = []
        self._kernel_hparams_init = self._kernel_hparams

    def _record_bayes_values(self):
        self._X_bayes = self.X_train[self._n_init:, :]
        self._y_bayes = self.y_train[self._n_init:]
        if self._has_derinfo:
            self._dX_bayes = self.dX_train[self._n_init:, :]
            self._dy_bayes = self.dy_train[self._n_init:, :]
        else:
            self._dX_bayes = []
            self._dy_bayes = []
        self._kernel_hparams_bayes = self._kernel_hparams[1:]

    def _eval_fun(self, X_arr):
        targets = []
        for X in X_arr:
            X_dict = dict(zip(self._X_keys, X))
            targets.append(self.fun(**X_dict))
        return np.asarray(targets).reshape(-1,)

    def _eval_dfun(self, dX_arr):
        targets = []
        for X in dX_arr:
            X_dict = dict(zip(self._X_keys, X))
            targets.append(self.dfun(**X_dict))
        return np.asarray(targets).reshape(-1, self._nparams)

    def _get_optimal_val(self):
        param_dic = dict(zip(self._X_keys, self.X_optimal.T))
        desc = {'target': self.y_optimal, 'parameters': param_dic}
        return desc

    def _get_log(self):
        return [{'y': y, 'X': dict(zip(self._X_keys, X))}
                for y, X in zip(self.y_train, self.X_train)]

    @property
    def optimal(self):
        return self._get_optimal_val()

    @property
    def log(self):
        return self._get_log()

    def sample_next_X_by_EI(self,
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
                x0 = self.random_state.uniform(self._X_bounds_vals[:, 0],
                                               self._X_bounds_vals[:, 1],
                                               size=(1, self._nparams))
                res = scipy.optimize.minimize(
                        fun=neg_acq,
                        x0=x0,
                        method='L-BFGS-B',
                        bounds=self._X_bounds_vals)
                fval = res.fun[0]
                x = res.x
            elif callable(self.minimizer):
                x, fval = self.minimizer(fun=neg_acq,
                                         bounds=self._X_bounds_vals)
            else:
                raise ValueError("Unknown minimizer %s." % self.minimizer)

            if fval < min_val:
                min_val = fval
                best_x = x

        return np.array(best_x).reshape(1, self._nparams)
