import unittest 
from unittest.mock import MagicMock
import numpy as np

import gpder 
from gpder.bayes import NetVarianceLoss, GPUncertaintyOptimizer
from gpder.gaussian_process import GaussianProcessRegressor
from gpder.gaussian_process.kernels import RegularKernel

class TestNetVarianceLoss(unittest.TestCase):
    def setUp(self):
        # dummy data for testing
        self.X_train = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.y_train = np.array([5.0, 6.0])
        self.X_test = np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0], [13.0, 14.0]])
        self.X_util = np.random.uniform(0, 1, (10, 2))
        self.gp = GaussianProcessRegressor(RegularKernel(), optimizer=None)
        self.gp.fit(self.X_train, self.y_train)

    def test_utility(self):
        X = np.random.uniform(0, 1, (1, 2))
        # testing the utility function
        nvl = NetVarianceLoss(self.gp, self.X_util, 1.0)
        utility = nvl.utility(X)
        # and retraining the GP
        gp_updated = self.gp.fit(np.vstack([self.X_train, X]), np.hstack([self.y_train, self.gp.predict(X).ravel()]))
        _, cov_exp_updated = gp_updated.predict(self.X_util, return_cov=True)
        self.assertAlmostEqual(utility, 1 - np.trace(cov_exp_updated))

class TestGPUncertaintyOptimizer(unittest.TestCase):
    def setUp(self):
        self.mock_gp = MagicMock()
        self.mock_gp.predict.return_value = (np.array([1.0]), np.array([[1.0]]))
        self.bounds = {'x1': (0, 1), 'x2': (-1, 1)}
        self.function = MagicMock(return_value=np.array([1.0]))
        self.bayes_opt = GPUncertaintyOptimizer(self.mock_gp, self.bounds, self.function)

    def test_minimize_variance_setup(self):
        X_util = np.random.uniform(0, 1, (10, 2))
        # self.bayes_opt._current_net_variance = 1.0
        self.bayes_opt.minimize_variance(X_util, n_iters=1)
    #     self.mock_gp.fit.assert_called_once()

