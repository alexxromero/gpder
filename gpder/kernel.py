from abc import ABCMeta, abstractmethod
from collections import namedtuple
import math
from inspect import signature

import numpy as np
from scipy.special import kv, gamma
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.gaussian_process.kernels import StationaryKernelMixin
from sklearn.gaussian_process.kernels import NormalizedKernelMixin
from sklearn.gaussian_process.kernels import Kernel
from sklearn.gaussian_process.kernels import Hyperparameter
from sklearn.utils.validation import _num_samples

__all__ = ['DerivativeAwareKernel']

def dist(X1, X2): # pairwise distance
    distance = np.zeros((X1.shape[0], X2.shape[0]))
    for i in range(X1.shape[0]):
        for j in range(X2.shape[0]):
            distance[i][j] = X1[i]-X2[j]
    return distance

class DerivativeAwareKernel(StationaryKernelMixin,
                            NormalizedKernelMixin, Kernel):
    """Kernel for Gaussian process regression with derivative observations.

    The DerivativeAwareKernel is a modification of the radial-basis function
    (RBF) kernel to include derivative observations, weighted by a
    constant kernel, in addition to a white noise kernel. The kernel is
    given by:

    .. math::
        k(x, dx) = \\alpha * RBF(x, dx, l) + \\sigma \\delta

    where :math:`\\alpha` is the magnitude of the constant kernel,
    :math:`\\sigma` the magnitude of the noise level of the white noise kernel,
    and :math:`RBF` is the modified RBF kernel with length scale :math:`l`.

    Args:
        constant_value: float, default=1.0
            Magnitude of the constant kernel multiplying the modified
            RBF kernel.

        constant_value_bounds: pair of floats >= 0 or "fixed", \
            default=(1e-5, 1e5)
            The lower and upper bounds of 'constant_value'. If "fixed", the
            'constant_value' parameter is not changed during hyperparameter
            tunning.

        length_scale: float or ndarray of shape (n_features,), default=1.0
            Length scale of the modified RBF kernel. If a float, an isotropic
            kernel is used. If an array, and anisotropic kernel is used where
            each dimension of l defines the length-scale of the respective
            feature dimension.

        length_scale_bounds: pair of floats >= 0 or "fixed", default=(1e-5, 1e5)
            The lower and upper bounds of 'length_scale'. If "fixed", the
            'length_scale' parameter is not changed during hyperparameter
            tunning.

        noise_level: float or None, default=1.0
            Parameter controlling the noise level (variance) of the white
            noise kernel. If None, the noise level is set to 1e-12 to with
            "fixed" bounds to approximate non-noisy kernel.

        noise_level_bounds: pair of floats >= 0 or "fixed", default=(1e-5, 1e5)
            The lower and upper bounds of 'noise_level_bounds'. If "fixed", the
            'noise_level' parameter is not changed during hyperparameter
            tunning.
    """
    def __init__(self,
                 constant_value=1.0, constant_value_bounds=(1e-5, 1e5),
                 length_scale=1.0, length_scale_bounds=(1e-5, 1e5),
                 noise_level=1.0, noise_level_bounds=(1e-5, 1e5)):
        self.constant_value = constant_value
        self.constant_value_bounds = constant_value_bounds
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds
        if noise_level is None:
            self._noisy = False
            self.noise_level = 1e-12
            self.noise_level_bounds = "fixed"
        else:
            self._noisy = True
            self.noise_level = noise_level
            self.noise_level_bounds = noise_level_bounds

    @property
    def anisotropic(self):
        return np.iterable(self.length_scale) and len(self.length_scale) > 1

    @property
    def hyperparameter_length_scale(self):
        if self.anisotropic:
            return Hyperparameter("length_scale", "numeric",
                                  self.length_scale_bounds,
                                  len(self.length_scale))
        return Hyperparameter(
            "length_scale", "numeric", self.length_scale_bounds)

    @property
    def hyperparameter_constant_value(self):
        return Hyperparameter(
            "constant_value", "numeric", self.constant_value_bounds)

    @property
    def hyperparameter_noise_level(self):
        return Hyperparameter(
            "noise_level", "numeric", self.noise_level_bounds)

    def __call__(self, X, Y=None, dX=None,
                 eval_gradient=False):
        """Return the kernel k and optionally its gradient.

        Args:
            X: array of shape (n_samples_X, n_features)
                Argument of the kernel. If only X is passed, k(X, X)
                is returned.

            Y: array of shape (n_samples_X, n_features), default=None
                Argument of the kernel. Can only be passed if X is
                also passed, returning k(X, Y).

            dX: array of shape (n_samples_X, n_features), default=None
                Argument of the kernel which includes derivative
                observations. Can only be passed if X is
                also passed, returning the hybrid kernel k(dX, X).

            eval_gradient: bool, default=False
                If true, the gradient with respect to the log of the kernel
                is computed, with respect to the hyperparameters. Only
                the gradients of k(X, X) or k(dX, X) are supported.

        Returns:
            K: darray of shape (n_samples_X, n_samples_X) if k(X, X),
                (n_samples_X, n_samples_Y) if k(X, Y), or
                (n_samples_X + n_samples_dX, n_samples_Y + n_samples_Y)
                if k(X, dX)
                Returned kernel.

            K_gradient: darray of shape (n_samples_X, n_samples_X, n_dims)
                if k(X, X) or
                (n_samples_X + n_samples_dX, n_samples_Y + n_samples_Y, n_dims)
                if k(X, dX), optional
                If true, the gradient of k(X) or kk(dX, X) with respect to
                the log of the kernel hyperparameters is also returned.
        """
        X = np.atleast_2d(X)
        length_scale = self._check_length_scale(X, self.length_scale)

        if dX is not None:  # compute k(dX, X) -- hybrid kernel
            if Y is not None:
                raise ValueError("Y and dX cannot be passed at the same time.")

            if eval_gradient:
                kernel_constant, kernel_constant_grad = \
                    self._kernel_constant_wder(X, dX, eval_gradient=True)
                kernel_rbf, kernel_rbf_grad = \
                    self._kernel_rbf_wder(X, dX, eval_gradient=True)
                kernel_noise, kernel_noise_grad = \
                    self._kernel_noise_wder(X, dX, eval_gradient=True)
                K = kernel_constant * kernel_rbf + kernel_noise
                grad_K = np.concatenate([kernel_constant_grad,
                                         kernel_rbf_grad,
                                         kernel_noise_grad], -1)
                return K, grad_K
            else:
                K = self._kernel_constant_wder(X, dX) * \
                    self._kernel_rbf_wder(X, dX) + \
                    self._kernel_noise_wder(X, dX)
                return K

        else:
            if Y is not None:  # compute k(X, Y)
                K = self._kernel_constant(X, Y) * \
                    self._kernel_rbf(X, Y) + \
                    self._kernel_noise(X, Y)
                return K
            else:  # compute k(X)
                if eval_gradient:
                    kernel_constant, kernel_constant_grad = \
                        self._kernel_constant(X, Y=None, eval_gradient=True)
                    kernel_rbf, kernel_rbf_grad = \
                        self._kernel_rbf(X, Y=None, eval_gradient=True)
                    kernel_noise, kernel_noise_grad = \
                        self._kernel_noise(X, Y=None, eval_gradient=True)
                    K = kernel_constant * kernel_rbf + kernel_noise
                    grad_K = np.concatenate([kernel_constant_grad,
                                             kernel_rbf_grad,
                                             kernel_noise_grad], -1)
                    return K, grad_K
                else:
                    K = self._kernel_constant(X) * \
                        self._kernel_rbf(X) + \
                        self._kernel_noise(X)
                    return K

    def __repr__(self):
        if self.anisotropic:
            desc = "Constant({0:.3g}**2) * RBF(length_scale={2:.3g})".format(
                np.sqrt(self.constant_value),
                map("{0:.3g}".format, self.length_scale))
            if self._noisy:
                desc += " + WhiteKernel(noise_level={0:.3g})".format(
                    self.noise_level)
            return desc
        else:
            desc = "Constant({0:.3g}**2) * RBF(length_scale={1:.3g})".format(
                np.sqrt(self.constant_value),
                np.ravel(self.length_scale)[0])
            if self._noisy:
                desc += " + WhiteKernel(noise_level={0:.3g})".format(
                    self.noise_level)
            return desc

    def _kernel_constant(self, X, Y=None, eval_gradient=False):
        if Y is None:
            Y = X
        elif eval_gradient:
            raise ValueError("Gradient can only be evaluated when Y is None.")

        K = np.full((_num_samples(X), _num_samples(Y)),
                     self.constant_value,
                     dtype=np.array(self.constant_value).dtype)
        if eval_gradient:
            if not self.hyperparameter_constant_value.fixed:
                K_gradient = self.constant_value * self._kernel_rbf(X)
                return K, K_gradient[..., np.newaxis]
            else:
                return K, np.empty((_num_samples(X), _num_samples(X), 0))
        else:
            return K

    def _kernel_constant_wder(self, X, dX, eval_gradient=False):
        X_mixed = np.concatenate((dX, X))
        K = np.full((_num_samples(X_mixed), _num_samples(X_mixed)),
                    self.constant_value,
                    dtype=np.array(self.constant_value).dtype)
        if eval_gradient:
            if not self.hyperparameter_constant_value.fixed:
                length_scale = self.length_scale
                gamma = 1 / length_scale**2
                # grad kernel(dX)
                dl = np.ones((dX.shape[0], dX.shape[0]))
                dists2_dX = squareform(pdist(dX, metric='sqeuclidean'))
                coeff_dx = gamma * (dl - gamma * dists2_dX)
                Kdx_gradient = 2 * coeff_dx * self._kernel_rbf(dX)
                dx_gradient = self.constant_value * Kdx_gradient
                # grad kernel(dX, X)
                coeff_dxx = -gamma * dist(dX, X)
                Kdxx_gradient = 2 * coeff_dxx * self._kernel_rbf(dX, X)
                Kdxx_gradient = self.constant_value * Kdxx_gradient
                # grad kernel(X)
                Kx_gradient = 2 * self.constant_value * self._kernel_rbf(dX, X)
                K_gradient = np.block([[Kdx_gradient, Kdxx_gradient],
                                       [Kdxx_gradient.T, Kx_gradient]])
                return K, K_gradient[..., np.newaxis]
            else:
                return K, np.empty((_num_samples(X_mixed),
                                    _num_samples(X_mixed), 0))
        else:
            return K

    def _kernel_rbf(self, X=None, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        length_scale = self._check_length_scale(X, self.length_scale)
        if Y is None:
            dists2 = pdist(X / length_scale, metric='sqeuclidean')
            K = squareform(np.exp(-.5 * dists2))
            np.fill_diagonal(K, 1)
            if eval_gradient:  # gradient wrt 1/length_scale**2
                if self.hyperparameter_length_scale.fixed:
                    return K, np.empty((X.shape[0], X.shape[0], 0))
                elif not self.anisotropic or length_scale.shape[0] == 1:
                    K_gradient = self.constant_value * K * squareform(dists2)
                    return K, K_gradient[:, :, np.newaxis]
                elif self.anisotropic:
                    K_gradient = (X[:, np.newaxis, :] \
                    - X[np.newaxis, :, :])** 2 \
                    / (length_scale ** 2)
                    K_gradient *= K[..., np.newaxis]
                    K_gradient *= self.constant_value
                return K, K_gradient
            else:
                return K
        else:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated when Y is None.")
            Y = np.atleast_2d(Y)
            dists = cdist(X / length_scale, Y / length_scale,
                          metric='sqeuclidean')
            K = np.exp(-.5 * dists)
            return K

    def _kernel_rbf_wder(self, X, dX, eval_gradient=False, return_Kdxx=False):
        X = np.atleast_2d(X)
        dX = np.atleast_2d(dX)
        length_scale = self._check_length_scale(dX, self.length_scale)
        gamma = 1/length_scale**2
        # kernel(dX, X)
        coeff_dxx = -gamma * dist(dX, X)
        Kdxx = coeff_dxx * self._kernel_rbf(dX, X)
        if return_Kdxx:
            return self.constant_value * Kdxx
        # kernel(dX)
        coeff_dx = gamma - gamma * cdist(dX, dX, metric='sqeuclidean')  * gamma
        Kdx = coeff_dx * self._kernel_rbf(dX)
        # kernel(X)
        Kx = self._kernel_rbf(X)
        # mixed kernel
        K = np.block([[Kdx, Kdxx], [Kdxx.T, Kx]])
        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                K_gradient = np.empty((dX.shape[0]+X.shape[0],
                                       dX.shape[0]+X.shape[0], 0))
                return K, K_gradient
            elif not self.anisotropic or length_scale.shape[0] == 1:
                # grad kernel(dX)
                dists2_dX = squareform(pdist(dX, metric='sqeuclidean'))
                dl = np.ones((dX.shape[0], dX.shape[0]))
                coef = gamma - gamma * dists2_dX * gamma
                coeff1 = dl - dl * dists2_dX * gamma - gamma * dists2_dX * dl
                coeff2 = coef * (-0.5 * dists2_dX)
                K1 = coeff1 * self._kernel_rbf(dX)
                K2 = coeff2 * self._kernel_rbf(dX)
                Kdx_gradient = -2 * self.constant_value * gamma * (K1 + K2)
                # grad kernel(dX, X)
                dist2_dxx = cdist(dX, X, metric='sqeuclidean')
                coef1 = -np.ones((dX.shape[0], X.shape[0])) * (dist(dX, X))
                coef2 = -gamma * dist(dX, X) * (-0.5 * dist2_dxx)
                K1 = coef1 * self._kernel_rbf(dX, X)
                K2 = coef2 * self._kernel_rbf(dX, X)
                Kdxx_gradient = -2 * self.constant_value * (K1 + K2) * gamma
                # grad kernel(X)
                dists2_x = pdist(X, metric='sqeuclidean')
                Kx_gradient = self._kernel_rbf(X) * squareform(dists2_x)
                K_gradient = np.block([[Kdx_gradient, Kdxx_gradient],
                                       [Kdxx_gradient.T, Kx_gradient]])
                return K, K_gradient[..., np.newaxis]
            elif self.anisotropic:
                # grad kernel(dX)
                dists2_dX = squareform(pdist(dX / length_scale,
                                             metric='sqeuclidean'))
                dl = np.ones((dX.shape[0], dX.shape[0]))
                coef = gamma - gamma * dists2_dX * gamma
                coeff1 = dl - dl * dists2_dX * gamma - gamma * dists2_dX * dl
                coeff2 = coef * (-0.5 * dists2_dX)
                K1 = coeff1 * self._kernel_rbf(dX)
                K2 = coeff2 * self._kernel_rbf(dX)
                Kdx_gradient = -2 * gamma * (K1 + K2)
                # grad kernel(dX, X)
                dist2_dxx = cdist(dX / length_scale, X / length_scale,
                              metric='sqeuclidean')
                coef1 = -np.ones((dX.shape[0], X.shape[0])) * (dist(dX, X))
                coef2 = -gamma * dist(dX, X) * (-0.5 * dist2_dxx)
                K1 = coef1 * self._kernel_rbf(dX, X)
                K2 = coef2 * self._kernel_rbf(dX, X)
                Kdxx_gradient = -2 * (K1 + K2) * gamma
                # grad kernel(X)
                dists2_x = pdist(X / length_scale, metric='sqeuclidean')
                Kx_gradient = self._kernel_rbf(X) * squareform(dists2_x)
                K_gradient = np.block([[Kdx_gradient, Kdxx_gradient],
                                      [Kdx_gradient.T, Kx_gradient]])
                return K, K_gradient
        else:
            return K

    def _kernel_noise(self, X, Y=None, eval_gradient=False):
        if Y is not None and eval_gradient:
            raise ValueError("Gradient can only be evaluated when Y is None.")

        if Y is None:
            K = self.noise_level * np.eye(_num_samples(X))
            if eval_gradient:
                if not self.hyperparameter_noise_level.fixed:
                    return (K, self.noise_level
                            * np.eye(_num_samples(X))[:, :, np.newaxis])
                else:
                    return K, np.empty((_num_samples(X), _num_samples(X), 0))
            else:
                return K
        else:
            return np.zeros((_num_samples(X), _num_samples(Y)))

    def _kernel_noise_wder(self, X, dX, eval_gradient=False):
        X_mixed = np.concatenate((dX, X))
        K_dim = _num_samples(X_mixed)
        K = np.full((K_dim, K_dim), 0,
                    dtype=np.array(self.noise_level).dtype)
        X_dim = _num_samples(X)
        Kx = np.eye(X_dim, dtype=np.array(self.noise_level).dtype)
        Kx *= self.noise_level
        K[X_dim:, X_dim:] = Kx
        if eval_gradient:
            if not self.hyperparameter_noise_level.fixed:
                Kdx_gradient = np.full(
                    (_num_samples(dX), _num_samples(dX)), 0,
                    dtype=np.array(self.noise_level).dtype)
                Kdxx_gradient = np.full(
                    (_num_samples(dX), _num_samples(X)), 0,
                    dtype=np.array(self.noise_level).dtype)
                Kx_gradient = np.full(
                    (_num_samples(X), _num_samples(X)), 0,
                    dtype=np.array(self.noise_level).dtype)
                np.fill_diagonal(Kx_gradient, self.noise_level)
                K_gradient = np.block([[Kdx_gradient, Kdxx_gradient],
                                       [Kdx_gradient.T, Kx_gradient]])
                return K, K_gradient[..., np.newaxis]
            else:
                return K, np.empty((_num_samples(X_mixed),
                                    _num_samples(X_mixed), 0))
        else:
            return K

    def _check_length_scale(self, X, length_scale):
        length_scale = np.squeeze(length_scale).astype(float)
        if np.ndim(length_scale) > 1:
            raise ValueError("length_scale cannot be of dimension" \
                             " greater than 1")
        if np.ndim(length_scale) == 1 and X.shape[1] != length_scale.shape[0]:
            raise ValueError("Anisotropic kernel must have the same number of "
                            "dimensions as data (%d!=%d)"
                            % (length_scale.shape[0], X.shape[1]))
        return length_scale
