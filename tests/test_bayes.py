import unittest
from unittest.mock import MagicMock, patch

import numpy as np

import gpder
from gpder.bayes import GPUncertaintyOptimizer, NetVarianceLoss
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
        gp_updated = self.gp.fit(
            np.vstack([self.X_train, X]),
            np.hstack([self.y_train, self.gp.predict(X).ravel()]),
        )
        _, cov_exp_updated = gp_updated.predict(self.X_util, return_cov=True)
        self.assertAlmostEqual(utility, 1 - np.trace(cov_exp_updated))


class TestGPUncertaintyOptimizer(unittest.TestCase):
    def setUp(self):
        self.mock_gp = MagicMock()
        self.mock_gp._has_gradients = False
        self.bounds = {"x1": (0, 1), "x2": (-1, 1)}
        self.X_util = np.random.uniform(0, 1, (10, 2))
        self.X_next_dummy = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.neg_util_dummy = np.array([1.0])
        self.function = MagicMock(return_value=np.array([1.0]))
        self.bayes_opt = GPUncertaintyOptimizer(self.mock_gp, self.bounds, self.function)
        self.bayes_opt.verbose = False

    def test_minimize_variance_setup(self):
        self.bayes_opt._find_next = MagicMock(return_value=(self.X_next_dummy, self.neg_util_dummy))
        self.bayes_opt.minimize_variance(self.X_util, n_iters=1)
        self.mock_gp.update.assert_called_once()

    def test_acq_fun_usage(self):
        self.bayes_opt._find_next = MagicMock(return_value=(self.X_next_dummy, self.neg_util_dummy))
        self.bayes_opt.minimize_variance(self.X_util, n_iters=1)
        self.function.assert_called_once()

    def test_find_next(self):
        with patch.object(
            GPUncertaintyOptimizer,
            "_find_next",
            return_value=(self.X_next_dummy, self.neg_util_dummy),
        ) as mock_find_next:
            self.bayes_opt._find_next(self.X_util)
            mock_find_next.assert_called_once()


if __name__ == "__main__":
    unittest.main()
