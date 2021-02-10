import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.utils.validation import check_random_state
import matplotlib.pyplot as plt

from .gaussian_process import GaussianProcessRegressor
from .kernel import DerivativeAwareKernel
from .plotting_utils import plot_approximation, plot_acquisition

# Functions from: http://krasserm.github.io/2018/03/21/bayesian-optimization/

__all__ = ['BayesianOptimization']

def expected_improvement(X, X_sample, y_sample, gpr, xi=0.01):
    """ Expected improvement (ei) at points X according to the GP regressor
    model trained on X_sample and y_sample.

    Args:
        X: array of shape (n, d)
            Points where the ei is computed.
        X_sample: array of shape (n_samples, d)
            Sample location points for the gpr.
        y_sample: array of shape (n_samples, 1)
            Sample target values for the gpr.
        gpr: GaussianProcessRegressor
            An instance of GaussianProcessRegressor fitted to the given samples.
        xi: float
            Exploitation-exploration trade-off parameter.

    Returns:
        Expected improvement at points X.
    """
    X = np.atleast_2d(X)
    X_sample = np.atleast_2d(X_sample)
    mu, _, std = gpr.predict(X=X, return_std=True)
    mu_sample, _ = gpr.predict(X=X_sample)

    std = std.reshape(-1, 1)
    mu_sample_max = np.max(mu_sample)

    with np.errstate(divide='warn'):
        imp = mu - mu_sample_max - xi
        Z = imp / std
        ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
        ei[std == 0.0] = 0.0

    return ei


class BayesianOptimization():

    def __init__(self, n_iters,
                 X, y,
                 f, X_sample, y_sample,
                 df=None, dX_sample=None, dy_sample=None,
                 bounds=(1e-5, 1e5),
                 random_state=None):

        self.random_state = check_random_state(random_state)

        kernel = DerivativeAwareKernel()
        model = GaussianProcessRegressor(kernel=kernel,
                                         n_restarts_optimizer=10,
                                         random_state=self.random_state)

        X_sample = np.copy(X_sample)
        y_sample = np.copy(y_sample)
        self.has_derinfo_ = False
        if ((dX_sample is not None) and \
            (dy_sample is not None) and \
            (df is not None)):
            dX_sample = np.copy(dX_sample)
            dy_sample = np.copy(dy_sample)
            self.has_derinfo_ = True
        else:
            dX_sample = None
            dy_sample = None

        # plot the regression at every iteration
        fig, axs = plt.subplots(n_iters, 2, figsize=(12, 2*n_iters))

        for i in range(n_iters):
            if self.has_derinfo_:
                model.fit(X=X_sample, y=y_sample,
                          dX=dX_sample, dy=dy_sample)
            else:
                model.fit(X=X_sample, y=y_sample)

            X_next = self.sample_next_location(expected_improvement,
                                               X_sample, dy_sample,
                                               model, bounds)
            y_next = f(X_next)
            if self.has_derinfo_:
                dX_next = X_next
                dy_next = df(dX_next)

            plot_approximation(model, X, y, X_sample, y_sample, X_next,
                               axs=axs[i][0])
            plot_acquisition(X,
                             expected_improvement(X, X_sample, y_sample, model),
                             X_next, axs=axs[i][1])

            # update sampled points
            X_sample = np.vstack((X_sample, X_next))
            y_sample = np.vstack((y_sample, y_next))
            if self.has_derinfo_:
                dX_sample = np.vstack((dX_sample, dX_next))
                dy_sample = np.vstack((dy_sample, dy_next))


    def sample_next_location(self, acquisition_func, X_sample, y_sample, gpr,
                             bounds, n_restarts=25):
        """ Location of the proposed next sampling by optimizing the
        acquisition function.

        Args:
            acquisition_func: Callable
                Acquisition function to optimize.
            X_sample: array of shape (n_samples, d)
                Sample location points for the gpr.
            y_sample: array of shape (n_samples, 1)
                Sample target values for the gpr.
            gpr: GaussianProcessRegressor
                Instance of GaussianProcessRegressor fitted to the samples.

        Returns:
            Location of the acquisition function maximum.
        """
        best_x = None
        min_val = 1
        dim = X_sample.shape[1]

        def min_obj(X):
            return -acquisition_func(X.reshape(-1, dim),
                                     X_sample, y_sample, gpr)

        # random search
        for x0 in self.random_state.uniform(bounds[:, 0], bounds[:, 1],
                                            size=(n_restarts, dim)):
            res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')
            if res.fun < min_val:
                min_val = res.fun[0]
                best_x = res.x

        return best_x.reshape(-1, 1)
