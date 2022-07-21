import numpy as np
import scipy
from scipy.stats import norm
import sklearn
from sklearn.utils.validation import check_random_state
from sklearn.metrics import mean_squared_error
from multiprocessing import Pool
import copy

from ..gaussian_process import GaussianProcessRegressor
from ..gaussian_process.kernels import GPKernel, GPKernelDerAware
from ..gaussian_process.kernels.utils import _atleast2d
from .bayesian_opt_utils import print_log
from .acquisition_functions import AcquisitionFunction

from scipy.spatial.distance import cdist

# Functions from: http://krasserm.github.io/2018/03/21/bayesian-optimization/
# https://github.com/thuijskens/bayesian-optimization/blob/master/python/gp.py

__all__ = ['BayesUncertaintyOptimization']

class GPlog():
    def __init__(self):
        self.instances = []

    def add_gp(self, gp):
        self.instances.append(copy.deepcopy(gp))

    def __getitem__(self, items):
        return self.instances[items]

class BayesUncertaintyOptimization():
    """BayesUncertaintyOptimization performs Bayesian Optimization
    to find the next optimal sampling point to minimize the overall
    predictive variance of a Gaussian Process model.

    Parameters
    ----------
    fun: callable
        Objective function.

    params_bounds: dict
        Dictionary with the parameter names as the keys and a puple with
        their (lower, upper) bounds as the values.
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
                             X_init, y_init,
                             X_query,
                             dX_init=None, dy_init=None,
                             acquisition_opt='trace',
                             batch_size=None,
                             n_iters=10,
                             minimizer='fmin_l_bfgs_b',
                             n_minimizer_restarts=10,
                             workers=1):
        """Minimize the uncertainty of the GPR model used to regress the
        objective function 'fun'.

        Parameters
        ----------
        X_init: ndarray of shape (ninit, nparams)
            Coordinates of the initial function observations.

        y_init: ndarray of shape (ninit,)
            Values of the initial function observations evaluated at 'X_train'.

        X_query: ndarray of shape(nopt, nparams)
            Coordinates of the function observations that will be used to
            estimate the expected uncertainty of the model.

        dX_init: ndarray of shape (ndinit, ndparams), optional
            Coordinates of the initial function derivative observations.

        dy_train: ndarray of shape (ndinit, ndparams), optional
            Values of the initial function derivative observations
            evaluated at 'dX_train'.

        acquisition_opt: string, default='trace'
            Measure of the global uncertainty to minimize.
             Options: {'trace', 'determinant'}

        batch_size: int, default=None
            Batch size

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
        """

        self.X_init, self.y_init = X_init, y_init
        self.dX_init, self.dy_init = dX_init, dy_init
        self.X_query = X_query
        self.acquisition_opt = acquisition_opt
        self.batch_size = batch_size
        self.n_iters = n_iters
        self.minimizer = minimizer
        self.n_minimizer_restarts = n_minimizer_restarts
        self.workers = workers

        self.X_train = self.X_init
        self.y_train = np.array(self.y_init).reshape(-1,)
        if self._has_derinfo:
            if (self.dX_init is not None) and (self.dy_init is not None):
                self.dX_train = self.dX_init
                self.dy_train = _atleast2d(self.dy_init)
            else:
                raise ValueError(
                    "dX_train and dy_train must be passed if using "
                    "derivative information..")
            if self.gp is None:
                self.gp = GaussianProcessRegressor(
                    kernel=GPKernelDerAware(), random_state=self.random_state
                    )
            self.gp.fit(X=self.X_train, y=self.y_train,
                        dX=self.dX_train, dy=self.dy_train)
        else:
            if self.gp is None:
                self.gp = GaussianProcessRegressor(
                    kernel=GPKernel(), random_state=self.random_state
                    )
            self.gp.fit(X=self.X_train, y=self.y_train)
        self._gp_record = GPlog()
        self._gp_record.add_gp(self.gp)
        self._n_init = np.shape(self.X_train)[0]
        self._acq = AcquisitionFunction(kind=self.acquisition_opt)

        if self.verbose:
            print_log(self._X_keys, self.X_init, self.y_init,
                      iteration_ids=[0]*self._n_init, print_header=True)

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
            else:
                self.gp.fit(X=self.X_train, y=self.y_train)
            self._gp_record.add_gp(self.gp)

            if self.verbose:
                print_log(self._X_keys, X_next, y_next,
                          iteration_ids=[i+1], print_header=False)

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
        #return np.asarray(y_arr).reshape(-1, self._nparams
        return _atleast2d(y_arr)


    def _find_next_X(self, n_restarts):
        def neg_acquisition_fun(X):
            return -1 * self._acq.utility(X, X_query=self.X_query, gp=self.gp,
                                          batch_size=self.batch_size)

        best_x = None
        min_val = 1e30  # dummy large number
        for i in range(n_restarts):
            if self.minimizer=='fmin_l_bfgs_b':
                x0 = self.random_state.uniform(self._X_bounds_vals[:, 0],
                                               self._X_bounds_vals[:, 1],
                                               size=(1, self._nparams))
                res = scipy.optimize.minimize(
                    fun=neg_acquisition_fun, x0=x0, method='L-BFGS-B',
                    bounds=self._X_bounds_vals)
                if not res.success:
                    continue
                val = res.fun
                x = res.x
            elif callable(self.minimizer):
                x, val = self.minimizer(fun=neg_acquisition_fun,
                                         bounds=self._X_bounds_vals)
            else:
                raise ValueError("Unknown minimizer %s." % self.minimizer)

            if val < min_val:
                min_val = val
                best_x = x
        best_x = np.array(best_x).reshape(1, self._nparams)
        return best_x


class BasinHoppingBounds:
    def __init__(self, X_bounds_vals):
        self.xmax = X_bounds_vals[:, 1]
        self.xmin = X_bounds_vals[:, 0]

    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin
