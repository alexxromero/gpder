import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.gaussian_process.kernels import StationaryKernelMixin
from sklearn.gaussian_process.kernels import NormalizedKernelMixin
from sklearn.gaussian_process.kernels import Kernel
from sklearn.gaussian_process.kernels import Hyperparameter
from sklearn.utils.validation import _num_samples

__all__ = ["DerivativeKernel"]


class DerivativeKernel(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    """Kernel for Gaussian Process Regression (GPR) with derivative observations.
    This kernel is a modification of the RegularKernel.

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
        Square root if the variance of white noise added to the kernel between function observations.

    noise_level_bounds: 'fixed' or pair of floats > 0, default=(1e-5, 1e5)
        The lower and upper bounds of 'noise_level'.
        If 'fixed', the noise_level parameter is not changed during
        hyperparameter tunning.

    noise_level_der: float, None, or ndarray of shape (ndim_dX), default=1.0
        Square root if the variance of the white noise added to the kernel between derivative observations.

    noise_level_der_bounds: "fixed" or pair of floats > 0, default=(1e-5, 1e5)
        The lower and upper bounds of 'noise_level_der'.
        If "fixed", the noise_level_der parameter is not changed during
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

    def __init__(
        self,
        amplitude=1.0,
        amplitude_bounds=(1e-5, 1e5),
        length_scale=1.0,
        length_scale_bounds=(1e-5, 1e5),
        noise_level=1e-2,
        noise_level_bounds=(1e-2, 1e4),
        noise_level_der=1e-2,
        noise_level_der_bounds=(1e-2, 1e4),
    ):
        self.amplitude = amplitude
        self.amplitude_bounds = amplitude_bounds
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds
        if noise_level is None:
            self.noise_level = 0
            self.noise_level_bounds = "fixed"
        else:
            self.noise_level = noise_level
            self.noise_level_bounds = noise_level_bounds
        if noise_level_der is None:
            self.noise_level_der = 0
            self.noise_level_der_bounds = "fixed"
        else:
            self.noise_level_der = noise_level_der
            self.noise_level_der_bounds = noise_level_der_bounds

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

    @property
    def anisotropic_noise_level_der(self):
        return np.iterable(self.noise_level_der) and len(self.noise_level_der) > 1

    @property
    def hyperparameter_noise_level_der(self):
        if self.anisotropic_noise_level_der:
            return Hyperparameter(
                "noise_level_der",
                "numeric",
                self.noise_level_der_bounds,
                len(self.noise_level_der),
            )
        else:
            return Hyperparameter(
                "noise_level_der", "numeric", self.noise_level_der_bounds
            )

    def __call__(self, X, dX=None, idx=None, eval_gradient=False):
        """Returns the kernel and optionally its gradients.

        Parameters
        ----------
        X: ndarray of shape (n_samples_X, n_features)
            Function input.

        dX: ndarray of shape (n_samples_dX, n_features_dX), default=None
            Derivative input. If None, then dX is assumed to be equal to X.

        idx: ndarray of shape (n_features_dX,)
            Indices of the dimensions of X along which the derivatives are evaluated.
            If None, then idx is assumed to be equal to the range (0, n_features_X).

        eval_gradient: bool, default=False
            If True, the gradients with respect to the log of the
            kernel hyperparameters are also returned.

        Returns
        -------
        K: ndarray of shape (n_samples_X + n_samples_dX * n_features_dX, n_samples_X + n_samples_dX * n_features_dX)
            Kernel.

        K_gradient: ndarray of shape (n_samples_X + n_samples_dX * n_features_dX, n_samples_X + n_samples_dX * n_features_dX, n_params)
            The gradient of the kernel with respect to the
            hyperparameters of the kernel. Only returned when eval_gradient
            is True.
        """
        self._check_length_scale(X, self.length_scale)

        if dX is None:
            dX = X
        return self._kernel_hybrid(X, dX, idx, eval_gradient=eval_gradient)

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
            K = np.exp(-0.5 * dists2)
            return K

    def _cov_yy(self, X, Y=None, add_noise=True, eval_gradient=False):
        """Covariance between function observations at inputs X and Y."""
        amplitude = self.amplitude

        length_scale = np.array(self.length_scale)

        if Y is None:
            (n_samples, _) = X.shape
            K = amplitude**2 * self._rbf(X)
            if add_noise and self.noise_level:
                K += self.noise_level**2 * np.eye(n_samples)

            if eval_gradient:
                (
                    dK_damplitude,
                    dK_dlength_scale,
                    dK_dnoise_level,
                    dK_dnoise_level_der,
                ) = self._initialize_gradients((n_samples, n_samples))
                # with respect to the amplitude parameter
                if not self.hyperparameter_amplitude.fixed:
                    dK_damplitude = self._rbf(X)[:, :, np.newaxis]
                    dK_damplitude *= 2 * amplitude
                # with respect to the length_scale parameter
                if not self.hyperparameter_length_scale.fixed:
                    if not self.anisotropic_length_scale:
                        dists2 = pdist(X / length_scale, metric="sqeuclidean")
                        dists2 = squareform(dists2)
                        grad = amplitude**2 * dists2 * self._rbf(X)
                        dK_dlength_scale = grad[:, :, np.newaxis]
                    else:
                        dists2 = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2
                        dists2 /= length_scale**2
                        grad = amplitude**2 * dists2 * self._rbf(X)[..., np.newaxis]
                        dK_dlength_scale = grad
                # with respect to the noise_level parameter
                if not self.hyperparameter_noise_level.fixed:
                    dK_dnoise_level += (
                        self.noise_level**2 * np.eye(n_samples)[:, :, np.newaxis]
                    )
                # no dependency on noise_level_der
                return K, np.concatenate(
                    (
                        dK_damplitude,
                        dK_dlength_scale,
                        dK_dnoise_level,
                        dK_dnoise_level_der,
                    ),
                    axis=-1,
                )
            else:
                return K
        else:
            if eval_gradient:
                raise ValueError("Grad can only be evaluated when Y is None.")
            if add_noise and self.noise_level:
                raise ValueError("Noise is only added when Y is None.")
            return amplitude**2 * self._rbf(X, Y)

    def _cov_ww(self, dX, dy=None, idx=None, add_noise=True, eval_gradient=False):
        """Covariance between derivative observations at inputs dX and dy."""
        if dy is None:
            dy = dX
        else:
            if dX.shape[1] != dy.shape[1]:
                raise ValueError("The number of features of dX and dy must be equal.")
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when dy is None.")

        (n_samples_dX, n_features_dX) = dX.shape
        (n_samples_dY, _) = dy.shape

        amplitude = self.amplitude

        if not self.anisotropic_length_scale:
            length_scale = np.repeat(self.length_scale, n_features_dX)
        else:
            length_scale = np.array(self.length_scale)

        if not self.anisotropic_noise_level_der:
            noise_level_der = np.repeat(self.noise_level_der, n_features_dX)
        else:
            noise_level_der = np.array(self.noise_level_der)

        if idx is None:
            grad_idx = np.arange(
                dX.shape[1]
            )  # Assuming all dimensions are accounted for
        else:
            grad_idx = idx

        K = np.zeros((n_samples_dX * len(grad_idx), n_samples_dY * len(grad_idx)))
        if eval_gradient:
            (dK_damplitude, dK_dlength_scale, dK_dnoise_level, dK_dnoise_level_der) = (
                self._initialize_gradients(
                    (n_samples_dX * len(grad_idx), n_samples_dY * len(grad_idx))
                )
            )

        for i, i_dim in enumerate(grad_idx):
            for j, j_dim in enumerate(grad_idx):
                dist_i = dX[:, i_dim].reshape(-1, 1) - dy[:, i_dim].reshape(-1, 1).T
                dist_i *= 1.0 / length_scale[i_dim] ** 2
                dist_j = dX[:, j_dim].reshape(-1, 1) - dy[:, j_dim].reshape(-1, 1).T
                dist_j *= 1.0 / length_scale[j_dim] ** 2
                dist_ii = (i_dim == j_dim) * (1.0 / length_scale[i_dim] ** 2)
                coeff = dist_ii - (dist_i * dist_j)
                K_ij = amplitude**2 * coeff * self._rbf(dX, dy)
                K[
                    i * n_samples_dX : (i + 1) * n_samples_dX,
                    j * n_samples_dY : (j + 1) * n_samples_dY,
                ] = K_ij
                if add_noise and i_dim == j_dim:
                    K[
                        i * n_samples_dX : (i + 1) * n_samples_dX,
                        j * n_samples_dY : (j + 1) * n_samples_dY,
                    ] += noise_level_der[i_dim]**2 * np.eye(n_samples_dX, n_samples_dY)
                if eval_gradient:
                    # with respect to the amplitude parameter
                    if not self.hyperparameter_amplitude.fixed:
                        dK_damplitude[
                            i * n_samples_dX : (i + 1) * n_samples_dX,
                            j * n_samples_dY : (j + 1) * n_samples_dY,
                        ] = (2 * amplitude * coeff * self._rbf(dX, dy))[:, :, np.newaxis]
                    # with respect to the length_scale parameter
                    if not self.hyperparameter_length_scale.fixed:
                        if not self.anisotropic_length_scale:
                            dists2 = pdist(dX / length_scale, metric="sqeuclidean")
                            dists2 = squareform(dists2)
                            d1 = coeff * dists2 * self._rbf(dX)
                            d1 = d1[:, :, np.newaxis]
                            dcoeff = (
                                -2 * (i_dim == j_dim) * (1.0 / length_scale[0] ** 2)
                            )
                            dcoeff += 4 * (dist_i * dist_j)
                            d2 = dcoeff * self._rbf(dX)
                            d2 = d2[:, :, np.newaxis]
                        else:
                            dist2 = (
                                dX[:, np.newaxis, :] - dX[np.newaxis, :, :]
                            ) ** 2 * (1.0 / length_scale**2)
                            d1 = (
                                coeff[..., np.newaxis]
                                * dist2
                                * self._rbf(dX)[..., np.newaxis]
                            )
                            dcoeff = 4 * (dist_i * dist_j)
                            dcoeff = np.repeat(
                                dcoeff[:, :, np.newaxis], n_features_dX, axis=2
                            )
                            dcoeff -= 2 * (i_dim == j_dim) * (1.0 / length_scale**2)
                            d2 = dcoeff * self._rbf(dX)[..., np.newaxis]
                        dK_dlength_scale[
                            i * n_samples_dX : (i + 1) * n_samples_dX,
                            j * n_samples_dY : (j + 1) * n_samples_dY,
                        ] = amplitude**2 * (d1 + d2)
                    # no dependence on noise_level
                    # with respect to the noise_level_der parameter
                    if add_noise and not self.hyperparameter_noise_level_der.fixed:
                        if i_dim == j_dim:
                            if not self.anisotropic_noise_level_der:
                                noise_grad = noise_level_der[i_dim]**2 * np.eye(
                                    n_samples_dX, n_samples_dY
                                )
                                dK_dnoise_level_der[
                                    i * n_samples_dX : (i + 1) * n_samples_dX,
                                    j * n_samples_dY : (j + 1) * n_samples_dY,
                                ] = noise_grad[..., np.newaxis]
                            else:
                                noise_grad = noise_level_der[i_dim]**2 * np.eye(
                                    n_samples_dX, n_samples_dY
                                )
                                dK_dnoise_level_der[
                                    i * n_samples_dX : (i + 1) * n_samples_dX,
                                    j * n_samples_dY : (j + 1) * n_samples_dY,
                                    i_dim,
                                ] = noise_grad
        if eval_gradient:
            return K, np.concatenate(
                (dK_damplitude, dK_dlength_scale, dK_dnoise_level, dK_dnoise_level_der),
                axis=-1,
            )
        else:
            return K

    def _cov_wy(self, dX, Y, idx=None, eval_gradient=False):
        """Covariance between derivative (dX) and function (Y) observations.
        Note that cov_wy = cov_yw.T"""
        (n_samples_dX, n_features_dX) = dX.shape
        (n_samples_Y, n_features_Y) = Y.shape

        amplitude = self.amplitude

        if not self.anisotropic_length_scale:
            length_scale = np.repeat(self.length_scale, n_features_dX)
        else:
            length_scale = np.array(self.length_scale)

        if idx is None:
            grad_idx = np.arange(n_features_Y)
        else:
            grad_idx = idx

        K = np.zeros((n_samples_dX * len(grad_idx), n_samples_Y))
        if eval_gradient:
            (dK_damplitude, dK_dlength_scale, dK_dnoise_level, dK_dnoise_level_der) = (
                self._initialize_gradients((n_samples_dX * len(grad_idx), n_samples_Y))
            )

        for i, i_dim in enumerate(grad_idx):
            dist_i = dX[:, i_dim].reshape(-1, 1) - Y[:, i_dim].reshape(-1, 1).T
            dist_i_scl = dist_i * (1.0 / length_scale[i_dim] ** 2)
            K_i = -1.0 * amplitude**2 * dist_i_scl * self._rbf(dX, Y)
            K[i * n_samples_dX : (i + 1) * n_samples_dX] = K_i
            if eval_gradient:
                # with respect to the amplitude parameter
                if not self.hyperparameter_amplitude.fixed:
                    dK_i_amp = -2.0 * amplitude * dist_i_scl * self._rbf(dX, Y)
                    dK_damplitude[i * n_samples_dX : (i + 1) * n_samples_dX, :] = dK_i_amp[
                        :, :, np.newaxis
                    ]
                # with respect to the length_scale parameter
                if not self.hyperparameter_length_scale.fixed:
                    if not self.anisotropic_length_scale:
                        dists2 = cdist(
                            dX / self.length_scale,
                            Y / self.length_scale,
                            metric="sqeuclidean",
                        )
                        d1 = -1.0 * dist_i_scl * dists2 * self._rbf(dX, Y)
                        d1 = d1[:, :, np.newaxis]
                        dcoeff = 2.0 * dist_i * (1.0 / self.length_scale**2)
                        d2 = dcoeff * self._rbf(dX, Y)
                        d2 = d2[:, :, np.newaxis]
                    else:
                        dist2 = (dX[:, np.newaxis, :] - Y[np.newaxis, :, :]) ** 2
                        dist2 /= length_scale**2
                        d1 = -1.0 * dist_i_scl[..., np.newaxis] * dist2
                        d1 *= self._rbf(dX, Y)[..., np.newaxis]
                        dcoeff = 2.0 * np.repeat(
                            dist_i[..., np.newaxis], n_features_Y, axis=2
                        )
                        dcoeff /= length_scale**2
                        d2 = dcoeff * self._rbf(dX, Y)[..., np.newaxis]
                    dK_dlength_scale[i * n_samples_dX : (i + 1) * n_samples_dX, :] = (
                        amplitude**2 * (d1 + d2)
                    )
                # no dependence on noise_level
                # no dependence on noise_level_der
        if eval_gradient:
            return K, np.concatenate(
                (dK_damplitude, dK_dlength_scale, dK_dnoise_level, dK_dnoise_level_der),
                axis=-1,
            )
        else:
            return K

    def _kernel_hybrid(self, X, dX, idx=None, eval_gradient=False):
        """Returns the composite covariance between function and derivative observations,
        and optionally its gradient."""
        if eval_gradient:
            (K_yy, dK_yy) = self._cov_yy(X=X, eval_gradient=True)
            (K_ww, dK_ww) = self._cov_ww(dX=dX, idx=idx, eval_gradient=True)
            (K_wy, dK_wy) = self._cov_wy(dX=dX, Y=X, idx=idx, eval_gradient=True)
            K = np.block([[K_yy, K_wy.T], [K_wy, K_ww]])
            dK = np.zeros((K.shape[0], K.shape[1], dK_yy.shape[2]))
            for i in range(dK_yy.shape[2]):
                dK[:, :, i] = np.block(
                    [
                        [dK_yy[:, :, i], dK_wy[:, :, i].T],
                        [dK_wy[:, :, i], dK_ww[:, :, i]],
                    ]
                )
            return K, dK
        else:
            K_yy = self._cov_yy(X)
            K_ww = self._cov_ww(dX, idx=idx)
            K_wy = self._cov_wy(dX, X, idx=idx)
            K = np.block([[K_yy, K_wy.T], [K_wy, K_ww]])
            return K

    def _check_length_scale(self, X, length_scale):
        """Check the length_scale parameter."""
        if np.ndim(length_scale) == 1 and X.shape[1] != len(length_scale):
            raise ValueError(
                "Anisotropic kernels must have the same number of"
                "dimensions as data (%d!=%d)" % (len(length_scale), X.shape[1])
            )

    def _initialize_gradients(self, dims):
        # with respect to the amplitude parameter
        if self.hyperparameter_amplitude.fixed:
            dK_damplitude = np.empty(dims + (0,))
        else:
            dK_damplitude = np.zeros(dims + (1,))
        # with respect to the length_scale parameter
        if self.hyperparameter_length_scale.fixed:
            dK_dlength_scale = np.empty(dims + (0,))
        else:
            if not self.anisotropic_length_scale:
                dK_dlength_scale = np.zeros(dims + (1,))
            else:
                dK_dlength_scale = np.zeros(dims + (len(self.length_scale),))
        # with respect to the noise_level parameter
        if self.hyperparameter_noise_level.fixed:
            dK_dnoise_level = np.empty(dims + (0,))
        else:
            dK_dnoise_level = np.zeros(dims + (1,))
        # with respect to the noise_level_der parameter
        if self.hyperparameter_noise_level_der.fixed:
            dK_dnoise_level_der = np.empty(dims + (0,))
        else:
            if not self.anisotropic_noise_level_der:
                dK_dnoise_level_der = np.zeros(dims + (1,))
            else:
                dK_dnoise_level_der = np.zeros(dims + (len(self.noise_level_der),))
        return (dK_damplitude, dK_dlength_scale, dK_dnoise_level, dK_dnoise_level_der)

    def __repr__(self):
        if not self.anisotropic_length_scale:
            desc = "{0:.3g}**2 * DerivativeRBF(length_scale={1:.3g})".format(
                self.amplitude, np.ravel(self.length_scale)[0]
            )
        else:
            desc = "{0:.3g}**2 * DerivativeRBF(length_scale=[{1}])".format(
                self.amplitude, ", ".join(map("{0:.3g}".format, self.length_scale))
            )
        desc += " + WhiteKernel(noise_level={0:.3g})".format(self.noise_level)
        if not self.anisotropic_noise_level_der:
            desc += " + WhiteKernel_der(noise_level={0:.3g})".format(
                self.noise_level_der
            )
        else:
            desc += " + WhiteKernel_der(noise_level=[{0}])".format(
                ", ".join(map("{0:.3g}".format, self.noise_level_der))
            )
        return desc
