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

__all__ = ['GPKernelDerAware']

from .utils import _atleast2d

class GPKernelDerAware(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    """Kernel for Gaussian Process Regression (GPR) with derivative
       observations.

    GPKernelDerAware is a modification of the the radial-basis
    function (RBF) kernel which allows for the integration
    of derivative observations, resulting in a hybrid kernel
    based on [1, 2]. The hyrbid RBF kernel is weighted by a
    constant kernel to regulate its magnitude and added to a
    white noise kernel to regulate noisy inputs.
    GPKernelDerAware is summarized as :

    .. math::
        k(X, dX) = \\alpha * RBF(X, dX, \\ell) + \\nu \\delta

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

    length_scale: float or ndarray of shape (ndim_X,), default=1.0
        Length scale of the modified RBF kernel.

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

    noise_level_dX: float, None, or ndarray of shape (ndim_dX), default=1.0
        Parameter controlling the noise level of the derivative observations dX.

    noise_level_dX_bounds: "fixed" or pair of floats > 0, default=(1e-5, 1e5)
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
                 noise_level=1.0, noise_level_bounds=(1e-5, 1e5),
                 noise_level_dX=1.0, noise_level_dX_bounds=(1e-5, 1e5)):
        self.constant_value = np.asarray(constant_value)
        self.constant_value_bounds = constant_value_bounds
        self.length_scale = np.asarray(length_scale)
        self.length_scale_bounds = length_scale_bounds
        if noise_level is None:
            self.noisy_X = False
            self.noise_level = np.array(0)
            self.noise_level_bounds = "fixed"
        else:
            self.noisy_X = True
            self.noise_level = np.asarray(noise_level)
            self.noise_level_bounds = noise_level_bounds
        if noise_level_dX is None:
            self.noisy_dX = False
            self.noise_level_dX = np.array(0)
            self.noise_level_dX_bounds = "fixed"
        else:
            self.noisy_dX = True
            self.noise_level_dX = np.asarray(noise_level_dX)
            self.noise_level_dX_bounds = noise_level_dX_bounds

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
        return Hyperparameter("noise_level", "numeric", self.noise_level_bounds)

    @property
    def anisotropic_noise_level_dX(self):
        return np.iterable(self.noise_level_dX) and len(self.noise_level_dX) > 1

    @property
    def hyperparameter_noise_level_dX(self):
        if self.anisotropic_noise_level_dX:
            return Hyperparameter("noise_level_dX", "numeric",
                                  self.noise_level_dX_bounds,
                                  len(self.noise_level_dX))
        else:
            return Hyperparameter("noise_level_dX", "numeric",
                                   self.noise_level_dX_bounds)


    def __call__(self, X, dX=None, idX=None, eval_gradient=False):
        """Returns the kernel and optionally its gradient.

        Parameters
        ----------
        X: ndarray of shape (nsamples_X, ndim_X)
            Coordinates of the function observations.

        dX: ndarray of shape (nsamples_dX, ndim_dX),
            default=None
            Coordinates of the derivative observations.
            If None, dX is assumed to be equal to X.

        idX: ndarrat of shape (ndim_dX,)
            default=None
            Array indicating the index of the dimensions along which the
            derivative information was taken. If None, it is assumed that
            the derivative information is provided in all dimensions
            (ndim_dX = ndim_X).

        eval_gradient: bool, default=False
            If True, the gradient with respect to the log of the
            kernel hyperparameters is also returned.

        Returns
        -------
        K: array of shape (nsamples_X+nsamples_dX, nsamples_X+nsamples_dX)
            Kernel.

        K_grad: ndimarray of shape (nsamples_X+nsamples_dX, nsamples_X+nsamples_dX, 3)
            Gradient of the kernel wrt the log of the hyperparameters.
            Only returned if eval_gradient is True.
        """
        self._check_length_scale(X, self.length_scale)

        if dX is None:
            dX = X
        return self._kernel_hybrid(X, dX, idX, eval_gradient=eval_gradient)

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

    def _cov_yy(self, X, Y=None, noisy=True, eval_gradient=False):
        if Y is None:
            (nX, ndim) = X.shape
            K = self.constant_value * self._rbf(X)
            if noisy:
                K += self.noise_level * np.eye(nX)

            if eval_gradient:
                (K_grad_const, K_grad_lensc, K_grad_noise, K_grad_noise_dX) =\
                    self._initialize_gradients((nX, nX))
                # -- wrt the log of the constant value -- #
                if not self.hyperparameter_constant_value.fixed:
                    K_grad_const = self._rbf(X)[:, :, np.newaxis]
                    K_grad_const *= self.constant_value
                # -- wrt the log of the length scale -- #
                if not self.hyperparameter_length_scale.fixed:
                    if not self.anisotropic_length_scale:
                        dists2 = pdist(X / self.length_scale, metric='sqeuclidean')
                        dists2 = squareform(dists2)
                        grad = self.constant_value * dists2 * self._rbf(X)
                        K_grad_lensc = grad[:, :, np.newaxis]
                    else:
                        dists2 = (X[:, np.newaxis, :] - X[np.newaxis, :, :])**2
                        dists2 /= self.length_scale**2
                        grad = self.constant_value * dists2 * self._rbf(X)[..., np.newaxis]
                        K_grad_lensc = grad
                # -- wrt the log of the noise level -- #
                if noisy and not self.hyperparameter_noise_level.fixed:
                        K_grad_noise += self.noise_level * np.eye(nX)[:, :, np.newaxis]
                return K, np.concatenate((K_grad_const,
                                          K_grad_lensc,
                                          K_grad_noise,
                                          K_grad_noise_dX), axis=-1)
            return K
        else:
            if eval_gradient:
                raise ValueError("Grad can only be evaluated when Y is None.")
            K = self.constant_value * self._rbf(X, Y)
            return K

    def _cov_ww(self, dX, dY=None, noisy=True, eval_gradient=False):
        if dY is None:
            dY = dX
        else:
            if eval_gradient:
                raise ValueError("Grad can only be evaluated when dY is None.")
        (ndX, ndim) = dX.shape
        (ndY, _) = dY.shape
        if not self.anisotropic_length_scale:
            length_scale = np.array([self.length_scale] * ndim)
        else:
            length_scale = self.length_scale
        if noisy:
            if not self.anisotropic_noise_level_dX:
                noise_level_dX = np.array([self.noise_level_dX] * ndim)
            else:
                noise_level_dX = self.noise_level_dX

        K = np.zeros((ndX*ndim, ndY*ndim))
        (K_grad_const, K_grad_lensc, K_grad_noise, K_grad_noise_dX) =\
            self._initialize_gradients((ndX*ndim, ndY*ndim))
        for i in range(ndim):
            for j in range(ndim):
                dist_i = dX[:, i].reshape(-1, 1) - dY[:, i].reshape(-1, 1).T
                dist_j = dX[:, j].reshape(-1, 1) - dY[:, j].reshape(-1, 1).T
                dist_i_scl = dist_i * 1. / length_scale[i]**2  # scaled
                dist_j_scl = dist_j * 1. / length_scale[j]**2  # scaled
                dij = (i==j) / length_scale[i]**2
                coeff = dij - (dist_i_scl * dist_j_scl)
                Kij = self.constant_value * coeff * self._rbf(dX, dY)
                if noisy and i==j:
                    Kij += noise_level_dX[i] * np.eye(ndX, ndY)
                K[i*ndX:(i+1)*ndX, j*ndY:(j+1)*ndY] = Kij
                if eval_gradient:
                    # -- wrt the log of the constant value -- #
                    if not self.hyperparameter_constant_value.fixed:
                        K_grad_const[i*ndX:(i+1)*ndX, j*ndX:(j+1)*ndX] = Kij[:, :, np.newaxis]
                    # -- wrt the log of the length scale -- #
                    if not self.hyperparameter_length_scale.fixed:
                        if not self.anisotropic_length_scale:
                            dists2 = pdist(dX / self.length_scale, metric='sqeuclidean')
                            dists2 = squareform(dists2)
                            d1 = coeff * dists2 * self._rbf(dX)
                            d1 = d1[:, :, np.newaxis]
                            dcoeff = -2*(i==j) / self.length_scale**2
                            dcoeff += 4*(dist_i_scl * dist_j_scl)
                            d2 = dcoeff * self._rbf(dX)
                            d2 = d2[:, :, np.newaxis]
                        else:
                            dists2 = (dX[:, np.newaxis, :] - dX[np.newaxis, :, :])**2
                            dists2 /= self.length_scale**2
                            d1 = coeff[..., np.newaxis] * dists2 * self._rbf(dX)[..., np.newaxis]
                            dcoeff = 4*(dist_i_scl * dist_j_scl)
                            dcoeff = np.repeat(dcoeff[..., np.newaxis], len(self.length_scale), axis=2)
                            dcoeff -= 2*(i==j) / self.length_scale**2
                            d2 = dcoeff * self._rbf(dX)[..., np.newaxis]
                        K_grad_lensc[i*ndX:(i+1)*ndX, j*ndX:(j+1)*ndX]= self.constant_value * (d1 + d2)
                    # -- wrt the noise dX -- #
                    if noisy and not self.hyperparameter_noise_level_dX.fixed:
                        if i==j:
                            if not self.anisotropic_noise_level_dX:
                                noise_grad = self.noise_level_dX * np.eye(ndX, ndY)[:, :, np.newaxis]
                                K_grad_noise_dX[i*ndX:(i+1)*ndX, j*ndY:(j+1)*ndY] += noise_grad
                            else:
                                noise_grad = self.noise_level_dX[i] * np.eye(ndX, ndY)
                                K_grad_noise_dX[i*ndX:(i+1)*ndX, j*ndY:(j+1)*ndY, i] += noise_grad
        if eval_gradient:
            return K, np.concatenate((K_grad_const,
                                      K_grad_lensc,
                                      K_grad_noise,
                                      K_grad_noise_dX), axis=-1)
        else:
            return K

    def _cov_wy(self, X, dX, idX=None, eval_gradient=False):
        (nX, ndimX) = X.shape
        (ndX, ndimdX) = dX.shape
        if (ndimX == ndimdX):
            idX_ = np.arange(ndimdX)
        elif (ndimX != ndimdX):
            if idX is None:
                raise ValueError(
                    "Please specify the index of the dimensions along "
                    "which the derviatives are taken, e.g. "
                    "idX=[0, 2] if only the derivatives along the 0th and "
                    "2nd dimensions are available."
                    )
            else:
                idX_ = idX
        X_ = X[:, idX_]

        if not self.anisotropic_length_scale:
            length_scale = np.array([self.length_scale] * ndimdX)
        else:
            length_scale = self.length_scale

        K = np.zeros((ndX*ndimdX, nX))
        (K_grad_const, K_grad_lensc, K_grad_noise, K_grad_noise_dX) =\
            self._initialize_gradients((ndX*ndimdX, nX))
        for i in range(len(idX_)):
            dist = (dX[:, i].reshape(-1, 1) - X_[:, i].reshape(-1, 1).T)
            coeff = dist / length_scale[i]**2
            Kij = -self.constant_value * coeff * self._rbf(dX, X_)
            K[i*ndX:(i+1)*ndX, :] = Kij
            if eval_gradient:
                # -- wrt the log of the constant value -- #
                if not self.hyperparameter_constant_value.fixed:
                    K_grad_const[i*ndX:(i+1)*ndX, :] = Kij[:, :, np.newaxis]
                # -- wrt the log of the length scale -- #
                if not self.hyperparameter_length_scale.fixed:
                    if not self.anisotropic_length_scale:
                        dists2 = cdist(dX / self.length_scale,
                                       X_ / self.length_scale,
                                       metric='sqeuclidean')
                        d1 = -coeff * dists2 * self._rbf(dX, X_)
                        d1 = d1[:, :, np.newaxis]
                        dcoeff = 2*dist / self.length_scale**2
                        d2 = dcoeff * self._rbf(dX, X_)
                        d2 = d2[:, :, np.newaxis]
                    else:
                        dists2 = (dX[:, np.newaxis, :] - X_[np.newaxis, :, :])**2
                        dists2 /= self.length_scale**2
                        d1 = -coeff[..., np.newaxis] * dists2
                        d1 = d1 * self._rbf(dX, X_)[..., np.newaxis]
                        dcoeff = np.repeat(dist[..., np.newaxis], len(self.length_scale), axis=2)
                        dcoeff = 2 * dcoeff / self.length_scale**2
                        d2 = (dcoeff * self._rbf(dX, X_)[..., np.newaxis])
                    K_grad_lensc[i*ndX:(i+1)*ndX, :] = self.constant_value * (d1 + d2)
        if eval_gradient:
            return K, np.concatenate((K_grad_const,
                                      K_grad_lensc,
                                      K_grad_noise,
                                      K_grad_noise_dX), axis=-1)
        else:
            return K

    def _kernel_hybrid(self, X, dX, idX,
                       eval_gradient=False):
        if eval_gradient:
            Kww, Kww_grad = self._cov_ww(dX, noisy=self.noisy_dX, eval_gradient=True)
            Kwy, Kwy_grad = self._cov_wy(X, dX, idX, eval_gradient=True)
            Kyy, Kyy_grad = self._cov_yy(X, noisy=self.noisy_X, eval_gradient=True)
            K = np.block([[Kyy, Kwy.T], [Kwy, Kww]])
            K_grad = np.zeros((K.shape[0], K.shape[1], Kyy_grad.shape[2]))
            for i in range(Kyy_grad.shape[2]):
                K_grad[:, :, i] = np.block(
                    [[Kyy_grad[:, :, i], Kwy_grad[:, :, i].T],
                     [Kwy_grad[:, :, i], Kww_grad[:, :, i]]])
            return K, K_grad
        else:
            Kww = self._cov_ww(dX, noisy=self.noisy_dX)
            Kwy = self._cov_wy(X, dX, idX)
            Kyy = self._cov_yy(X, noisy=self.noisy_X)
            K = np.block([[Kyy, Kwy.T], [Kwy, Kww]])
            return K


    def __repr__(self):
        if not self.anisotropic_length_scale:
            desc = "Constant({0:.3g}**2) * RBF(length_scale={1:.3g})".format(
                np.sqrt(self.constant_value),
                np.ravel(self.length_scale)[0])
        else:
            desc = "Constant({0:.3g}**2) * RBF(length_scale=[{1}])".format(
                np.sqrt(self.constant_value),
                ", ".join(map("{0:.3g}".format, self.length_scale)))
        if self.noisy_X:
            desc += " + WhiteKernel_X(noise_level={0:.3g})".format(
                self.noise_level)
        if self.noisy_dX:
            if not self.anisotropic_noise_level_dX:
                desc += " + WhiteKernel_dX(noise_level={0:.3g})".format(
                    self.noise_level_dX)
            else:
                desc += " + WhiteKernel_dX(noise_level=[{0}])".format(
                    ", ".join(map("{0:.3g}".format, self.noise_level_dX)))
        return desc


    def _check_length_scale(self, X, length_scale):
        if np.ndim(length_scale) > 1:
            raise ValueError("length_scale cannot be of dimension"
                             " greater than 1.")


    def _initialize_gradients(self, kernel_dims):
        # -- wrt the log of the constant value -- #
        if self.hyperparameter_constant_value.fixed:
            K_grad_const = np.empty((kernel_dims[0], kernel_dims[1], 0))
        else:
            K_grad_const = np.zeros((kernel_dims[0], kernel_dims[1], 1))
        # -- wrt the log of the length scale -- #
        if self.hyperparameter_length_scale.fixed:
            K_grad_lensc = np.empty((kernel_dims[0], kernel_dims[1], 0))
        else:
            if not self.anisotropic_length_scale:
                K_grad_lensc = np.zeros((kernel_dims[0], kernel_dims[1], 1))
            else:
                K_grad_lensc = np.zeros((kernel_dims[0], kernel_dims[1], len(self.length_scale)))
        # -- wrt the log of the noise level of X -- #
        if self.hyperparameter_noise_level.fixed:
            K_grad_noise = np.empty((kernel_dims[0], kernel_dims[1], 0))
        else:
            K_grad_noise = np.zeros((kernel_dims[0], kernel_dims[1], 1))
        # -- wrt the log of the noise level of dX -- #
        if self.hyperparameter_noise_level_dX.fixed:
            K_grad_noise_dX = np.empty((kernel_dims[0], kernel_dims[1], 0))
        else:
            if not self.anisotropic_noise_level_dX:
                K_grad_noise_dX = np.zeros((kernel_dims[0], kernel_dims[1], 1))
            else:
                K_grad_noise_dX = np.zeros((kernel_dims[0], kernel_dims[1], len(self.noise_level_dX)))
        return (K_grad_const, K_grad_lensc, K_grad_noise, K_grad_noise_dX)
