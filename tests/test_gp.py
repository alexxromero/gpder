import unittest

import numpy as np

import gpder
from gpder.gaussian_process import GaussianProcessRegressor
from gpder.gaussian_process.kernels import DerivativeKernel, RegularKernel


class TestRegularKernel(unittest.TestCase):
    def setUp(self):
        self.kernel = RegularKernel()

    def test_initialization(self):
        default_kernel = RegularKernel()
        self.assertEqual(default_kernel.amplitude, 1.0)
        self.assertEqual(default_kernel.length_scale, 1.0)
        self.assertEqual(default_kernel.noise_level, 0.01)

        custom_kernel = RegularKernel(amplitude=2.0, length_scale=3.0, noise_level=4.0)
        self.assertEqual(custom_kernel.amplitude, 2.0)
        self.assertEqual(custom_kernel.length_scale, 3.0)
        self.assertEqual(custom_kernel.noise_level, 4.0)

    def test_hyperparameter_amplitude(self):
        self.assertEqual(self.kernel.hyperparameter_amplitude.name, "amplitude")
        self.assertEqual(self.kernel.hyperparameter_amplitude.value_type, "numeric")
        self.assertEqual(self.kernel.hyperparameter_amplitude.bounds[0][0], 1e-5)
        self.assertEqual(self.kernel.hyperparameter_amplitude.bounds[0][1], 1e5)

    def test_hyperparameter_length_scale(self):
        self.assertEqual(self.kernel.hyperparameter_length_scale.name, "length_scale")
        self.assertEqual(self.kernel.hyperparameter_length_scale.value_type, "numeric")
        self.assertEqual(self.kernel.hyperparameter_length_scale.bounds[0][0], 1e-5)
        self.assertEqual(self.kernel.hyperparameter_length_scale.bounds[0][1], 1e5)

    def test_hyperparameter_noise_level(self):
        self.assertEqual(self.kernel.hyperparameter_noise_level.name, "noise_level")
        self.assertEqual(self.kernel.hyperparameter_noise_level.value_type, "numeric")
        self.assertEqual(self.kernel.hyperparameter_noise_level.bounds[0][0], 1e-2)
        self.assertEqual(self.kernel.hyperparameter_noise_level.bounds[0][1], 1e4)

    def test_call(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        Y = np.array([[5.0, 6.0], [7.0, 8.0]])
        K = self.kernel(X=X, Y=Y, eval_gradient=False)
        _, grad = self.kernel(X, eval_gradient=True)
        self.assertEqual(K.shape, (2, 2))
        self.assertEqual(grad.shape, (2, 2, 3))


class TestDerivativeKernel(unittest.TestCase):
    def setUp(self):
        self.kernel = DerivativeKernel()

    def test_initialization(self):
        default_kernel = DerivativeKernel()
        self.assertEqual(default_kernel.amplitude, 1.0)
        self.assertEqual(default_kernel.length_scale, 1.0)
        self.assertEqual(default_kernel.noise_level, 0.01)
        self.assertEqual(default_kernel.noise_level_der, 0.01)

        custom_kernel = DerivativeKernel(
            amplitude=2.0, length_scale=3.0, noise_level=4.0, noise_level_der=5.0
        )
        self.assertEqual(custom_kernel.amplitude, 2.0)
        self.assertEqual(custom_kernel.length_scale, 3.0)
        self.assertEqual(custom_kernel.noise_level, 4.0)
        self.assertEqual(custom_kernel.noise_level_der, 5.0)

    def test_hyperparameter_amplitude(self):
        self.assertEqual(self.kernel.hyperparameter_amplitude.name, "amplitude")
        self.assertEqual(self.kernel.hyperparameter_amplitude.value_type, "numeric")
        self.assertEqual(self.kernel.hyperparameter_amplitude.bounds[0][0], 1e-5)
        self.assertEqual(self.kernel.hyperparameter_amplitude.bounds[0][1], 1e5)

    def test_hyperparameter_length_scale(self):
        self.assertEqual(self.kernel.hyperparameter_length_scale.name, "length_scale")
        self.assertEqual(self.kernel.hyperparameter_length_scale.value_type, "numeric")
        self.assertEqual(self.kernel.hyperparameter_length_scale.bounds[0][0], 1e-5)
        self.assertEqual(self.kernel.hyperparameter_length_scale.bounds[0][1], 1e5)

    def test_hyperparameter_noise_level(self):
        self.assertEqual(self.kernel.hyperparameter_noise_level.name, "noise_level")
        self.assertEqual(self.kernel.hyperparameter_noise_level.value_type, "numeric")
        self.assertEqual(self.kernel.hyperparameter_noise_level.bounds[0][0], 1e-2)
        self.assertEqual(self.kernel.hyperparameter_noise_level.bounds[0][1], 1e4)

    def test_hyperparameter_noise_level_der(self):
        self.assertEqual(self.kernel.hyperparameter_noise_level_der.name, "noise_level_der")
        self.assertEqual(self.kernel.hyperparameter_noise_level_der.value_type, "numeric")
        self.assertEqual(self.kernel.hyperparameter_noise_level_der.bounds[0][0], 1e-2)
        self.assertEqual(self.kernel.hyperparameter_noise_level_der.bounds[0][1], 1e4)

    def test_call(self):
        X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        DX = np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
        K = self.kernel(X, DX, eval_gradient=False)
        _, grad = self.kernel(X, eval_gradient=True)
        self.assertEqual(K.shape, (2 + 2 * 3, 2 + 2 * 3))
        self.assertEqual(grad.shape, (2 + 2 * 3, 2 + 2 * 3, 4))


class TestGaussianProcessRegressor(unittest.TestCase):
    def setUp(self):
        self.regular_gp = GaussianProcessRegressor(kernel=RegularKernel())
        self.derivative_gp = GaussianProcessRegressor(kernel=DerivativeKernel())
        # dummy data for testing
        self.X_train = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.y_train = np.array([5.0, 6.0])
        self.dy_train = np.array([[11.0, 12.0], [13.0, 14.0]])
        self.X_test = np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0], [13.0, 14.0]])

    def test_initialization(self):
        self.assertEqual(self.regular_gp.kernel, RegularKernel())
        self.assertEqual(self.derivative_gp.kernel, DerivativeKernel())

    def test_fit(self):
        self.regular_gp.fit(self.X_train, self.y_train)
        self.assertTrue(hasattr(self.regular_gp, "X_train"))
        self.assertTrue(hasattr(self.regular_gp, "y_train"))
        self.assertTrue(hasattr(self.regular_gp, "_L_cholevsky"))
        self.assertTrue(hasattr(self.regular_gp, "_alpha_cholevsky"))

        self.derivative_gp.fit(self.X_train, self.y_train, self.X_train, self.dy_train)
        self.assertTrue(hasattr(self.derivative_gp, "X_train"))
        self.assertTrue(hasattr(self.derivative_gp, "y_train"))
        self.assertTrue(hasattr(self.derivative_gp, "dX_train"))
        self.assertTrue(hasattr(self.derivative_gp, "dy_train"))
        self.assertTrue(hasattr(self.derivative_gp, "_L_cholevsky"))
        self.assertTrue(hasattr(self.derivative_gp, "_alpha_cholevsky"))

    def test_update(self):
        # dummy data for testing
        X_new = np.array([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
        y_new = np.array([11.0, 12.0, 13.0])
        dy_new = np.array([[15.0, 16.0], [17.0, 18.0], [19.0, 20.0]])
        self.regular_gp.fit(self.X_train, self.y_train)
        self.regular_gp.update(X_new, y_new)
        self.assertEqual(self.regular_gp.X_train.shape, (self.X_train.shape[0] + X_new.shape[0], 2))
        self.assertEqual(self.regular_gp.y_train.shape, (self.X_train.shape[0] + X_new.shape[0], 1))

        self.derivative_gp.fit(self.X_train, self.y_train, self.X_train, self.dy_train)
        self.derivative_gp.update(X_new, y_new, X_new, dy_new)
        self.assertEqual(
            self.derivative_gp.X_train.shape,
            (self.X_train.shape[0] + X_new.shape[0], 2),
        )
        self.assertEqual(
            self.derivative_gp.y_train.shape,
            (self.X_train.shape[0] + X_new.shape[0], 1),
        )
        self.assertEqual(
            self.derivative_gp.dX_train.shape,
            (self.X_train.shape[0] + X_new.shape[0], 2),
        )
        self.assertEqual(
            self.derivative_gp.dy_train.shape,
            (self.X_train.shape[0] + X_new.shape[0], 2),
        )

    def test_predict(self):
        self.regular_gp.fit(self.X_train, self.y_train)
        y_pred, y_std = self.regular_gp.predict(self.X_test, return_std=True)
        _, y_cov = self.regular_gp.predict(self.X_test, return_cov=True)
        self.assertEqual(y_pred.shape, (4, 1))
        self.assertEqual(y_std.shape, (4, 1))
        self.assertEqual(y_cov.shape, (4, 4))

        self.derivative_gp.fit(self.X_train, self.y_train, self.X_train, self.y_train)
        y_pred, y_std = self.derivative_gp.predict(self.X_test, return_std=True)
        _, y_cov = self.derivative_gp.predict(self.X_test, return_cov=True)
        self.assertEqual(y_pred.shape, (4, 1))
        self.assertEqual(y_std.shape, (4, 1))
        self.assertEqual(y_cov.shape, (4, 4))

    def test_predict_gradients(self):
        self.derivative_gp.fit(self.X_train, self.y_train, self.X_train, self.dy_train)
        dy_pred, dy_std = self.derivative_gp.predict_gradients(self.X_test, return_std=True)
        _, dy_cov = self.derivative_gp.predict_gradients(self.X_test, return_cov=True)
        self.assertEqual(dy_pred.shape, (4 * 2, 1))
        self.assertEqual(dy_std.shape, (4 * 2, 1))
        self.assertEqual(dy_cov.shape, (4 * 2, 4 * 2))

    def test_sample(self):
        self.regular_gp.fit(self.X_train, self.y_train)
        y_sample = self.regular_gp.sample(self.X_test, n_draws=10)
        self.assertEqual(y_sample.shape, (10, 4))

        self.derivative_gp.fit(self.X_train, self.y_train, self.X_train, self.dy_train)
        y_sample = self.derivative_gp.sample(self.X_test, n_draws=10)
        self.assertEqual(y_sample.shape, (10, 4))

    def tets_sample_gradients(self):
        self.derivative_gp.fit(self.X_train, self.y_train, self.X_train, self.dy_train)
        dy_sample = self.derivative_gp.sample_gradients(self.X_test, n_draws=10)
        self.assertEqual(dy_sample.shape, (10, 4 * 2))


class TestGPIntegration(unittest.TestCase):
    def setUp(self):
        reg_kernel = RegularKernel(amplitude=1, length_scale=2, noise_level=3)
        der_kernel = DerivativeKernel(amplitude=1, length_scale=2, noise_level=3, noise_level_der=4)
        self.regular_gp = GaussianProcessRegressor(kernel=reg_kernel)
        self.derivative_gp = GaussianProcessRegressor(kernel=der_kernel)
        # dummy data for testing
        self.X_train = np.random.uniform(0, 1, (5, 3))
        self.y_train = np.random.normal(0, 1, (5, 1))
        self.dy_train = np.random.uniform(0, 1, (5, 3))
        self.X_test = np.random.uniform(0, 1, (10, 3))

    def test_optimizer(self):
        self.regular_gp.n_restarts_optimizer = 10
        self.regular_gp.fit(self.X_train, self.y_train)
        amplitude = self.regular_gp.kernel.amplitude
        length_scale = self.regular_gp.kernel.length_scale
        noise_level = self.regular_gp.kernel.noise_level
        self.assertNotEqual(amplitude, 1)
        self.assertNotEqual(length_scale, 2)
        self.assertNotEqual(noise_level, 3)

        self.derivative_gp.n_restarts_optimizer = 10
        self.derivative_gp.fit(self.X_train, self.y_train, self.X_train, self.dy_train)
        amplitude = self.derivative_gp.kernel.amplitude
        length_scale = self.derivative_gp.kernel.length_scale
        noise_level = self.derivative_gp.kernel.noise_level
        noise_level_der = self.derivative_gp.kernel.noise_level_der
        self.assertNotEqual(amplitude, 1)
        self.assertNotEqual(length_scale, 2)
        self.assertNotEqual(noise_level, 3)
        self.assertNotEqual(noise_level_der, 4)

    def test_noise_level(self):
        kernel_high_noise = RegularKernel(amplitude=1, length_scale=1, noise_level=10.0)
        gp_high_noise = GaussianProcessRegressor(kernel=kernel_high_noise, optimizer=None)
        gp_high_noise.fit(self.X_train, self.y_train)
        _, y_std_high_noise = gp_high_noise.predict(self.X_test, return_std=True)

        kernel_low_noise = RegularKernel(amplitude=1, length_scale=1, noise_level=0.001)
        gp_low_noise = GaussianProcessRegressor(kernel=kernel_low_noise, optimizer=None)
        gp_low_noise.fit(self.X_train, self.y_train)
        _, y_std_low_noise = gp_low_noise.predict(self.X_test, return_std=True)

        self.assertTrue(np.all(y_std_high_noise >= y_std_low_noise))


if __name__ == "__main__":
    unittest.main()
