import warnings

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.gaussian_process.kernels import (
    Hyperparameter,
    Kernel,
    NormalizedKernelMixin,
    StationaryKernelMixin,
)

__all__ = [
    "RegularKernel",
    "validate_scalar",
    "validate_scalar_or_array",
    "validate_bounds",
]


def validate_scalar(value, name):
    """Validate that the value is a scalar."""
    if not np.isscalar(value):
        raise ValueError(f"{name} must be a scalar, got {type(value)}")
    return value


def validate_scalar_or_array(value, name):
    """Validate that the value is a scalar or a 1D array."""
    if np.isscalar(value):
        return value
    if isinstance(value, (np.ndarray, np.generic)):
        if value.ndim == 1:
            return value
        else:
            raise ValueError(f"{name} must be a scalar or a 1D array, got {type(value)}")
    raise ValueError(f"{name} must be a scalar or a 1D array, got {type(value)}")


def validate_bounds(bounds, name):
    """Validate that the bounds are either 'fixed' or a pair of floats."""
    if bounds == "fixed":
        return bounds
    if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
        lower, upper = bounds
        if not (np.isscalar(lower) and np.isscalar(upper)):
            raise ValueError(f"{name} must be a pair of floats, got {bounds}")
        return bounds
    raise ValueError(f"{name} must be 'fixed' or a pair of floats, got {bounds}")


class RegularKernel(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    """Kernel for regular Gaussian Process Regression (GPR).

    RegularKernel can be summarized as :
    .. math::
        K(X, Y) = a^2 * RBF(X, Y, \\ell) + \\sigma^2 I,

    where :math:`a > 0` is the amplitude of the radial-basis function
    (RBF) with length scale :math:`\\ell > 0`. White noise with variance
    :math:`\\sigma^2` is added to account for statistical fluctuations.

    The implementation of the RBF kernel is based on SKlearn's RBF kernel.
    See [3].

    Parameters
    ----------
    amplitude: float, default=1.0
        Amplitude of the RBF kernel.

    amplitude_bounds: 'fixed' or pair of floats, default=(1e-5, 1e5)
        The lower and upper bounds of 'amplitude'.
        If 'fixed', the amplitude parameter is not changed during
        hyperparameter tuning.

    length_scale: float or ndarray of shape (ndims,), default=1.0
        Length scale of the RBF kernel.

    length_scale_bounds: 'fixed' or pair of floats > 0, default=(1e-5, 1e5)
        The lower and upper bounds of 'length_scale'.
        If 'fixed', the length_scale parameter is not changed during
        hyperparameter tuning.

    noise_level: float or None, default=0.01
        Square root if the variance of the added white noise.

    noise_level_bounds: 'fixed' or pair of floats > 0, default=(1e-5, 1e5)
        The lower and upper bounds of 'noise_level'.
        If 'fixed', the noise_level parameter is not changed during
        hyperparameter tuning.

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

    def __init__(
        self,
        amplitude=1.0,
        amplitude_bounds=(1e-5, 1e5),
        length_scale=1.0,
        length_scale_bounds=(1e-5, 1e5),
        noise_level=1e-2,
        noise_level_bounds=(1e-2, 1e4),
    ):
        self.amplitude = validate_scalar(amplitude, "amplitude")
        self.amplitude_bounds = validate_bounds(amplitude_bounds, "amplitude_bounds")
        self.length_scale = validate_scalar_or_array(length_scale, "length_scale")
        self.length_scale_bounds = validate_bounds(length_scale_bounds, "length_scale_bounds")
        noise_level = noise_level if noise_level is not None else 0.0
        self.noise_level = validate_scalar(noise_level, "noise_level")
        self.noise_level_bounds = validate_bounds(noise_level_bounds, "noise_level_bounds")

    @property
    def hyperparameter_amplitude(self):
        return Hyperparameter("amplitude", "numeric", self.amplitude_bounds)

    @property
    def anisotropic_length_scale(self):
        return np.iterable(self.length_scale) and len(self.length_scale) > 1

    @property
    def hyperparameter_length_scale(self):
        if self.anisotropic_length_scale:
            return Hyperparameter(
                "length_scale",
                "numeric",
                self.length_scale_bounds,
                len(self.length_scale),
            )
        else:
            return Hyperparameter("length_scale", "numeric", self.length_scale_bounds)

    @property
    def hyperparameter_noise_level(self):
        return Hyperparameter("noise_level", "numeric", self.noise_level_bounds)

    def __call__(self, X, Y=None, add_noise=True, eval_gradient=False):
        """Returns the kernel and optionally its gradients.

        Parameters
        ----------
        X: ndarray of shape (n_sampX, n_feat)

        Y: ndarray of shape (n_sampY, n_feat), default=None
            If None, k(X, X) is evaluated instead.

        add_noise: bool, default=True
            If True, the white noise is added to the diagonal of the kernel.

        eval_gradient: bool, default=False
            If True, the gradients with respect to the log of the
            hyperparameters are also returned.

        Returns
        -------
        K: ndarray of shape (n_sampX, n_sampY)
            Kernel K(X, Y)

        K_gradient: ndarray of shape (n_sampX, n_sampX, n_hyperparams)
            The gradient of the kernel k(X, X) with respect to the
            hyperparameters of the kernel. Only returned when eval_gradient
            is True.
        """
        self._check_length_scale(X, self.length_scale)

        return self._cov_yy(X, Y=Y, add_noise=add_noise, eval_gradient=eval_gradient)

    def _rbf(self, X, Y=None):
        ls = self.length_scale

        if Y is None:
            dists2 = pdist(X / ls, metric="sqeuclidean")
            K = np.exp(-0.5 * dists2)
            K = squareform(K)
            np.fill_diagonal(K, 1)
            return K
        else:
            dists2 = cdist(X / ls, Y / ls, metric="sqeuclidean")
            return np.exp(-0.5 * dists2)

    def _cov_yy(self, X, add_noise, Y=None, eval_gradient=False):
        amp = self.amplitude
        ls = self.length_scale
        noise = self.noise_level

        if Y is None:
            (n_samples, _) = X.shape

            rbf = self._rbf(X)
            K = amp**2 * rbf
            if noise > 0 and add_noise:
                K += noise**2 * np.eye(n_samples)

            if not eval_gradient:
                return K

            # gradient with respect to the log amplitude parameter
            if self.hyperparameter_amplitude.fixed:
                dK_damp = np.empty((n_samples, n_samples, 0))
            else:
                dK_damp = (2 * amp**2 * rbf)[:, :, np.newaxis]
            # with respect to the log length_scale parameter
            if self.hyperparameter_length_scale.fixed:
                dK_dls = np.empty((n_samples, n_samples, 0))
            else:
                if not self.anisotropic_length_scale:
                    dists2 = squareform(pdist(X / ls, metric="sqeuclidean"))
                    dK_dls = amp**2 * dists2 * rbf
                    dK_dls = dK_dls[:, :, np.newaxis]
                else:
                    dists2 = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2
                    dists2 /= ls**2
                    dK_dls = amp**2 * rbf[..., np.newaxis] * dists2
            # with respect to the log noise_level parameter
            if self.hyperparameter_noise_level.fixed:
                dK_dnoise = np.empty((n_samples, n_samples, 0))
            else:
                dK_dnoise = np.eye(n_samples)[:, :, np.newaxis]
                dK_dnoise *= 2 * noise**2
            return K, np.concatenate((dK_damp, dK_dls, dK_dnoise), axis=-1)
        else:
            if X.shape[1] != Y.shape[1]:
                raise ValueError("The number of features of X and Y must be equal.")
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            if add_noise:
                warnings.warn(
                    "Warning: noise is not added when Y is not None.",
                    UserWarning,
                )
            K = amp**2 * self._rbf(X, Y)
            return K

    def _check_length_scale(self, X, length_scale):
        """Check the length_scale parameter."""
        if np.ndim(length_scale) == 1 and X.shape[1] != length_scale.shape[0]:
            raise ValueError(
                f"Anisotropic kernels must have the same number of "
                f"dimensions as data ({length_scale.shape[0]}!={X.shape[1]})"
            )

    def __repr__(self):
        if self.anisotropic_length_scale:
            scales = ", ".join(f"{s:.3g}" for s in self.length_scale)
            description = f"{self.amplitude:.3g}**2 * RBF(length_scale=[{scales}])"
        else:
            ls0 = np.ravel(self.length_scale)[0]
            description = f"{self.amplitude:.3g}**2 * RBF(length_scale={ls0:.3g})"
        description += f" + WhiteKernel(noise_level={self.noise_level:.3g})"
        return description

    def _repr_latex_(self):
        return self.__repr__()
