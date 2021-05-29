from abc import ABCMeta, abstractmethod
from collections import namedtuple
import math
from inspect import signature

import numpy as np
from scipy.special import kv
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.gaussian_process.kernels import StationaryKernelMixin
from sklearn.gaussian_process.kernels import NormalizedKernelMixin
from sklearn.gaussian_process.kernels import Kernel
from sklearn.gaussian_process.kernels import Hyperparameter
from sklearn.utils.validation import _num_samples

__all__ = ['GPKernel']

from .utils import _atleast2d

class GPKernel(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    """Kernel for Gaussian Process Eegression (GPR).

    Kernel is an application of the radial-basis function
    (RBF) weighted by a constant kernel, in addition to a
    white noise kernel. The Kernel is summarized as :

    .. math::
        k(x, dx) = \\alpha * RBF(X, Y, \\sigma) + \\nu \\delta

    where :math:`\\alpha` is the magnitude of the constant kernel,
    :math:`\\nu` is the magnitude of the noise level of the white
    noise kernel, and :math:`RBF` is the modified RBF kernel with
    length scale :math:`\\sigma`.

    The kernel implementation is based on SKlearn's ConstantKernel,
    RBF, and WhiteKernel. See [3].

    Parameters
    ----------
    constant_value: float, default=1.0
        Magnitude of the constant kernel.

    constant_value_bounds: "fixed" or pair of floats >= 0,
        default=(1e-5, 1e5)
        The lower and upper bounds of constant_value.
        If "fixed", constant_value parameter is not changed during
        the hyperparameter tunning.

    length_scale: float or ndimarray of shape (n_features,), default=1.0
        Length scale of the modified RBF kernel.
        If a float, an isotropic kernel is used.
        If an array, and anisotropic_length_scale kernel is used where the mag. of
        each entry in the array defines the length-scale of the
        respective feature dimension.

    length_scale_bounds: "fixed" or pair of floats >= 0,
        default=(1e-5, 1e5)
        The lower and upper bounds of length_scale.
        If "fixed", the length_scale parameter is not changed during
        the hyperparameter tunning.

    noise_level: float or None, default=1.0
        Parameter controlling the noise level (variance) of the white
        noise kernel. If None, the noise level is set to 1e-12 to with
        "fixed" bounds to approximate non-noisy kernel.

    noise_level_bounds: "fixed" or pair of floats >= 0,
        default=(1e-5, 1e5)
        The lower and upper bounds of noise_level_bounds.
        If "fixed", the noise_level parameter is not changed during
        the hyperparameter tunning.

    References
    ----------
    [1] Solak, E., Murray-Smith, R., Leithead, W.E., Leith, D.J.
    and Rasmussen, C.E. (2003) Derivative observations in Gaussian
    Process models of dynamic systems. In: Conference on Neural
    Information Processing Systems, Vancouver, Canada,
    9-14 December 2002, ISBN 0262112450

    [2] Carl Edward Rasmussen, Christopher K. I. Williams (2006).
    "Gaussian Processes for Machine Learning". The MIT Press.
    <http://www.gaussianprocess.org/gpml/>`_

    [3] https://github.com/scikit-learn/scikit-learn/blob/95119c13a/sklearn/gaussian_process/kernels.py#L1379
    """
    def __init__(self, constant_value=1.0, constant_value_bounds=(1e-5, 1e5),
                 length_scale=1.0, length_scale_bounds=(1e-5, 1e5),
                 noise_level=1.0, noise_level_bounds=(1e-5, 1e5)):
        self.constant_value = constant_value
        self.constant_value_bounds = constant_value_bounds
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds
        if noise_level is (None or 0):
            self.noisy = False
            self.noise_level = 1e-12
            self.noise_level_bounds = "fixed"
        else:
            self.noisy = True
            self.noise_level = noise_level
            self.noise_level_bounds = noise_level_bounds

    @property
    def anisotropic_length_scale(self):
        return np.iterable(self.length_scale) and len(self.length_scale) > 1

    @property
    def hyperparameter_length_scale(self):
        if self.anisotropic_length_scale:
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

    def __call__(self, X, Y=None,
                 eval_gradient=False):
        """Return the kernel and optionally its gradient.

        Parameters
        ----------
        X: ndimarray of shape (nsamples_X, ndimims_X)
            Left argument of the kernel. If only X is passed,
            k(X, X) is returned.

        Y: ndimarray of shape (nsamples_Y, ndimims_Y), default=None
            Right argument of the kernel. Can only be passed if X
            is also passed, returning k(X, Y).

        eval_gradient: bool, default=False
            If True, the gradient with respect to the log of the
            kernel hyperparameters is computed.
            Only supported if Y=None.

        Returns
        -------
        K: ndimarray of shape
            Evaluated kernel.

        K_grad: optional ndimarray of shape (nsamp, nsamp).
            Gradient of the kernel wrt the hyperparameters.
            Only returned if eval_gradient is True.
        """
        self._check_length_scale(X, self.length_scale)

        if Y is None:
            return self._cov_yy(X, eval_gradient=eval_gradient)
        else:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated when Y is None.")
            return self._cov_yy(X, Y, eval_gradient=False)

    def _rbf(self, X, Y=None):
        if Y is None:
            dists2 = pdist(X / self.length_scale, metric='sqeuclidean')
            K = np.exp(-0.5 * dists2)
            K = squareform(K)
            np.fill_diagonal(K, 1)
            return K
        else:
            dists2 = cdist(X / self.length_scale, Y  / self.length_scale, metric='sqeuclidean')
            K = np.exp(-0.5 * dists2)
            return K

    def _cov_yy(self, X, Y=None, eval_gradient=False):
        # covariance between the locations of the func at X (and Y).
        if Y is None:
            K = self.constant_value * self._rbf(X)
            K += self.noise_level * np.eye(_num_samples(X))
            if eval_gradient:
                # -- wrt the constant value -- #
                if self.hyperparameter_constant_value.fixed:
                    K_grad_const = np.empty((_num_samples(X), _num_samples(X), 0))
                else:
                    K_grad_const = self.constant_value * self._rbf(X)
                    K_grad_const = K_grad_const[..., np.newaxis]
                # -- wrt the length scale -- #
                if self.hyperparameter_length_scale.fixed:
                    K_grad_lensc = np.empty((_num_samples(X), _num_samples(X), 0))
                else:
                    dists2 = pdist(X / self.length_scale, metric='sqeuclidean')
                    dists2 = squareform(dists2)
                    K_grad_lensc = self.constant_value * dists2 * self._rbf(X)
                    K_grad_lensc = K_grad_lensc[..., np.newaxis]
                # -- wrt the noise level -- #
                if self.hyperparameter_noise_level.fixed:
                    K_grad_noise = np.empty((_num_samples(X), _num_samples(X), 0))
                else:
                    K_grad_noise = self.noise_level * np.eye(_num_samples(X))
                    K_grad_noise = K_grad_noise[..., np.newaxis]

                return K, np.concatenate((K_grad_const, K_grad_lensc, K_grad_noise), axis=-1)
            else:
                return K
        else:
            if eval_gradient:
                raise ValueError("Grad can only be evaluated when Y is None.")
            K = self.constant_value * self._rbf(X, Y)
            return K

    def __repr__(self):
        if self.anisotropic_length_scale:
            desc = "Constant({0:.3g}**2) * RBF(length_scale={2:.3g})".format(
                np.sqrt(self.constant_value),
                map("{0:.3g}".format, self.length_scale))
            if self.noisy:
                desc += " + WhiteKernel(noise_level={0:.3g})".format(
                    self.noise_level)
            return desc
        else:
            desc = "Constant({0:.3g}**2) * RBF(length_scale={1:.3g})".format(
                np.sqrt(self.constant_value),
                np.ravel(self.length_scale)[0])
            if self.noisy:
                desc += " + WhiteKernel(noise_level={0:.3g})".format(
                    self.noise_level)
            return desc

    def _check_length_scale(self, X, length_scale):
        if np.ndim(length_scale) > 1:
            raise ValueError("length_scale cannot be of dimension"
                             " greater than 1.")
        if np.ndim(length_scale) == 1 and X.shape[1] != length_scale.shape[0]:
            raise ValueError("anisotropic_length_scale kernel must have the same number of"
                             " dimensions as data (%d!=%d)"
                            % (length_scale.shape[0], X.shape[1]))
