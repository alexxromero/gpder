import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.gaussian_process.kernels import StationaryKernelMixin
from sklearn.gaussian_process.kernels import NormalizedKernelMixin
from sklearn.gaussian_process.kernels import Kernel
from sklearn.gaussian_process.kernels import Hyperparameter

__all__ = ["RegularKernel"]


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
        hyperparameter tunning.

    length_scale: float or ndarray of shape (ndims,), default=1.0
        Length scale of the RBF kernel.

    length_scale_bounds: 'fixed' or pair of floats > 0, default=(1e-5, 1e5)
        The lower and upper bounds of 'length_scale'.
        If 'fixed', the length_scale parameter is not changed during
        hyperparameter tunning.

    noise_level: float or None, default=1.0
        Square root if the variance of the added white noise.

    noise_level_bounds: 'fixed' or pair of floats > 0, default=(1e-5, 1e5)
        The lower and upper bounds of 'noise_level'.
        If 'fixed', the noise_level parameter is not changed during
        hyperparameter tunning.

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
        self.amplitude = amplitude
        self.amplitude_bounds = amplitude_bounds
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds
        if noise_level is None:
            self.noise_level = 0.0
            self.noise_level_bounds = "fixed"
        else:
            self.noise_level = noise_level
            self.noise_level_bounds = noise_level_bounds

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

    def __call__(self, X, Y=None, eval_gradient=False):
        """Returns the kernel and optionally its gradients.

        Parameters
        ----------
        X: ndarray of shape (n_samples_X, n_features)

        Y: ndarray of shape (n_samples_Y, n_features), default=None
            If None, k(X, X) is evaluated instead.

        eval_gradient: bool, default=False
            If True, the gradients with respect to the log of the
            hyperparameters are also returned.

        Returns
        -------
        K: ndarray of shape (n_samples_X, n_samples_Y)
            Kernel K(X, Y)

        K_gradient: ndarray of shape (n_samples_X, n_samples_X, n_hyperparams)
            The gradient of the kernel k(X, X) with respect to the
            hyperparameters of the kernel. Only returned when eval_gradient
            is True.
        """
        self._check_length_scale(X, self.length_scale)

        return self._cov_yy(X, Y, eval_gradient=eval_gradient)

    def _rbf(self, X, Y=None):
        if Y is None:
            dists2 = pdist(X / self.length_scale, metric="sqeuclidean")
            K = np.exp(-0.5 * dists2)
            K = squareform(K)
            np.fill_diagonal(K, 1)
            return K
        else:
            dists2 = cdist(
                X / self.length_scale, Y / self.length_scale, metric="sqeuclidean"
            )
            return np.exp(-0.5 * dists2)

    def _cov_yy(self, X, Y=None, add_noise=True, eval_gradient=False):
        if Y is None:
            (n_samples, _) = X.shape
            K = self.amplitude**2 * self._rbf(X)
            if add_noise and self.noise_level:
                K += self.noise_level**2 * np.eye(n_samples)

            if eval_gradient:
                # with respect to the amplitude parameter
                if self.hyperparameter_amplitude.fixed:
                    dK_damplitude = np.empty((n_samples, n_samples, 0))
                else:
                    dK_damplitude = self._rbf(X)[:, :, np.newaxis]
                    dK_damplitude *= 2 * self.amplitude
                # with respect to the length_scale parameter
                if self.hyperparameter_length_scale.fixed:
                    dK_dlength_scale = np.empty((n_samples, n_samples, 0))
                else:
                    if not self.anisotropic_length_scale:
                        dists2 = pdist(X / self.length_scale, metric="sqeuclidean")
                        dists2 = squareform(dists2)
                        dK_dlength_scale = self.amplitude**2 * dists2
                        dK_dlength_scale *= self._rbf(X)
                        dK_dlength_scale = dK_dlength_scale[:, :, np.newaxis]
                    else:
                        dists2 = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2
                        dists2 /= self.length_scale**2
                        dK_dlength_scale = self.amplitude**2 * dists2
                        dK_dlength_scale *= self._rbf(X)[..., np.newaxis]
                # with respect to the noise_level parameter
                if self.hyperparameter_noise_level.fixed:
                    dK_dnoise_level = np.empty((n_samples, n_samples, 0))
                else:
                    dK_dnoise_level = np.eye(n_samples)[:, :, np.newaxis]
                    dK_dnoise_level *= self.noise_level**2
                return K, np.concatenate(
                    (dK_damplitude, dK_dlength_scale, dK_dnoise_level), axis=-1
                )
            else:
                return K
        else:
            (n_samples, _) = X.shape
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y " "is None.")
            K = self.amplitude**2 * self._rbf(X, Y)
            if add_noise and self.noise_level:
                K += self.noise_level**2 * np.eye(n_samples)
            return K

    def _check_length_scale(self, X, length_scale):
        """Check the length_scale parameter."""
        if np.ndim(length_scale) == 1 and X.shape[1] != length_scale.shape[0]:
            raise ValueError(
                "Anisotropic kernels must have the same number of"
                "dimensions as data (%d!=%d)" % (length_scale.shape[0], X.shape[1])
            )

    def __repr__(self):
        if self.anisotropic_length_scale:
            description = "{0:.3g}**2 * RBF(length_scale=[{1}])".format(
                self.amplitude, ", ".join(map("{0:.3g}".format, self.length_scale))
            )
        else:
            description = "{0:.3g}**2 * RBF(length_scale={1:.3g})".format(
                self.amplitude, np.ravel(self.length_scale)[0]
            )
        description += " + WhiteKernel(noise_level={0:.3g})".format(self.noise_level)
        return description

    def _repr_latex_(self):
        return self.__repr__()