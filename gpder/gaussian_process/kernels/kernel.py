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
    """Kernel for Gaussian Process Regression (GPR).

    GPKernel is an application of the radial-basis function
    (RBF) weighted by a constant kernel, in addition to a
    white noise kernel. GPKernel is summarized as :

    .. math::
        k(X, Y) = \\alpha * RBF(X, Y, \\ell) + \\nu \\delta

    where :math:`\\alpha` is the magnitude of the constant kernel,
    :math:`\\nu` is the magnitude of the noise level of the white
    noise kernel, and :math:`RBF` is the modified RBF kernel with
    length scale :math:`\\ell`.

    The kernel implementation is based on SKlearn's ConstantKernel,
    RBF, and WhiteKernel. See [3].

    Parameters
    ----------
    constant_value: float, default=1.0
        Magnitude of the constant kernel.

    constant_value_bounds: "fixed" or pair of floats > 0, default=(1e-5, 1e5)
        The lower and upper bounds of 'constant_value'.
        If "fixed", constant_value parameter is not changed during
        the hyperparameter tunning.

    length_scale: float or ndarray of shape (ndims,), default=1.0
        Length scale of the RBF kernel.

    length_scale_bounds: "fixed" or pair of floats > 0, default=(1e-5, 1e5)
        The lower and upper bounds of 'length_scale'.
        If "fixed", the length_scale parameter is not changed during
        the hyperparameter tunning.

    noise_level: float or None, default=1.0
        Parameter controlling the noise level of X.

    noise_level_bounds: "fixed" or pair of floats > 0, default=(1e-5, 1e5)
        The lower and upper bounds of 'noise_level'.
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
        if noise_level is None:
            self.noisy = False
            self.noise_level = np.array(0)
            self.noise_level_bounds = "fixed"
        else:
            self.noisy = True
            self.noise_level = noise_level
            self.noise_level_bounds = noise_level_bounds

    @property
    def hyperparameter_constant_value(self):
        return Hyperparameter(
        "constant_value", "numeric", self.constant_value_bounds)

    @property
    def anisotropic_length_scale(self):
        return np.iterable(self.length_scale) and len(self.length_scale) > 1

    @property
    def hyperparameter_length_scale(self):
        if self.anisotropic_length_scale:
            return Hyperparameter("length_scale", "numeric",
                                   self.length_scale_bounds,
                                   len(self.length_scale))
        else:
            return Hyperparameter("length_scale", "numeric",
                                  self.length_scale_bounds)
    @property
    def hyperparameter_noise_level(self):
        return Hyperparameter("noise_level", "numeric",
                              self.noise_level_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        """Returns the kernel and optionally its gradient.

        Parameters
        ----------
        X: ndimarray of shape (nsamples_X, ndimims_X)
            Left argument of the kernel.
            If only X is passed, k(X, X) is returned.

        Y: ndimarray of shape (nsamples_Y, ndimims_Y), default=None
            Right argument of the kernel.
            Can only be passed if X is also passed. Then, k(X, Y) is returned.

        eval_gradient: bool, default=False
            If True, the gradient with respect to the log of the
            kernel hyperparameters is also returned.
            Only supported if Y is 'None'.

        Returns
        -------
        K: array of shape (nsamples_X, nsamples_X)
            Kernel.

        K_grad: array of shape (nsamples_X, nsamples_X, nparams)
            Gradient of the kernel wrt the log of the hyperparameters.
            Only returned if eval_gradient is True.
        """
        self._check_length_scale(X, self.length_scale)

        return self._cov_yy(X, Y, noisy=self.noisy, eval_gradient=eval_gradient)

    def _rbf(self, X, Y=None):
        if Y is None:
            dists2 = pdist(X / self.length_scale, metric='sqeuclidean')
            K = np.exp(-0.5 * dists2)
            K = squareform(K)
            np.fill_diagonal(K, 1)
            return K
        else:
            dists2 = cdist(X / self.length_scale,
                           Y / self.length_scale,
                           metric='sqeuclidean')
            K = np.exp(-0.5 * dists2)
            return K

    def _cov_yy(self, X, Y=None, noisy=False, eval_gradient=False):
        if Y is None:
            (nX, ndim) = X.shape
            K = self.constant_value * self._rbf(X)
            if noisy:
                K += self.noise_level * np.eye(nX)

            if eval_gradient:
                # -- wrt the log of the constant value -- #
                if self.hyperparameter_constant_value.fixed:
                    dK_dconst = np.empty((_num_samples(X), _num_samples(X), 0))
                else:
                    dK_dconst = self._rbf(X)[:, :, np.newaxis]
                    dK_dconst *= self.constant_value
                # -- wrt the log of the length scale -- #
                if self.hyperparameter_length_scale.fixed:
                    dK_dls = np.empty((_num_samples(X), _num_samples(X), 0))
                else:
                    if not self.anisotropic_length_scale:
                        dists2 = pdist(X / self.length_scale, metric='sqeuclidean')
                        dists2 = squareform(dists2)
                        dK_dls = self.constant_value * dists2 * self._rbf(X)
                        dK_dls = dK_dls[:, :, np.newaxis]
                    else:
                        dists2 = (X[:, np.newaxis, :] - X[np.newaxis, :, :])**2
                        dists2 /= self.length_scale**2
                        dK_dls = self.constant_value * dists2 * self._rbf(X)[..., np.newaxis]
                # -- wrt the log of the noise level -- #
                if noisy and not self.hyperparameter_noise_level.fixed:
                    dK_dnoise = np.eye(_num_samples(X))[..., np.newaxis]
                    dK_dnoise *= self.noise_level
                else:
                    dK_dnoise = np.empty((_num_samples(X), _num_samples(X), 0))

                return K, np.concatenate((dK_dconst, dK_dls, dK_dnoise), axis=-1)
            else:
                return K
        else:
            if eval_gradient:
                raise ValueError("Grad can only be evaluated when Y is None.")
            K = self.constant_value * self._rbf(X, Y)
            if noisy:
                K += self.noise_level * np.eye(nX)
            return K

    def __repr__(self):
        if not self.anisotropic_length_scale:
            desc = "{0:.3g} * RBF(length_scale={1:.3g})".format(
                self.constant_value, np.ravel(self.length_scale)[0])
        else:
            desc = "{0:.3g} * RBF(length_scale=[{1}])".format(
                self.constant_value,
                ", ".join(map("{0:.3g}".format, self.length_scale)))
        if self.noisy:
            desc += " + WhiteKernel(noise_level={0:.3g})".format(
                self.noise_level)
        return desc

    def _check_length_scale(self, X, length_scale):
        if np.ndim(length_scale) > 1:
            raise ValueError(
                "length_scale cannot be of dimension greater than 1.")
        if np.ndim(length_scale) == 1 and X.shape[1] != length_scale.shape[0]:
            raise ValueError(
                "anisotropic_length_scale kernel must have the same number of"
                 " dimensions as data (%d!=%d)"%(length_scale.shape[0],
                                                 X.shape[1]))
