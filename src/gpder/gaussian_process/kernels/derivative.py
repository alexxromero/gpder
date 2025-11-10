import warnings

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.gaussian_process.kernels import (
    Hyperparameter,
    Kernel,
    NormalizedKernelMixin,
    StationaryKernelMixin,
)

from gpder.gaussian_process.kernels.regular import (
    validate_bounds,
    validate_scalar,
    validate_scalar_or_array,
)

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
        hyperparameter tuning.

    length_scale: float or ndarray of shape (ndims,), default=1.0
        Length scale of the RBF kernel.

    length_scale_bounds: 'fixed' or pair of floats > 0, default=(1e-5, 1e5)
        The lower and upper bounds of 'length_scale'.
        If 'fixed', the length_scale parameter is not changed during
        hyperparameter tuning.

    noise_level: float or None, default=1.0
        Square root if the variance of white noise added to the kernel between function observations.

    noise_level_bounds: 'fixed' or pair of floats > 0, default=(1e-5, 1e5)
        The lower and upper bounds of 'noise_level'.
        If 'fixed', the noise_level parameter is not changed during
        hyperparameter tuning.

    noise_level_der: float, None, or ndarray of shape (ndim_dX), default=1.0
        Square root if the variance of the white noise added to the kernel between derivative observations.

    noise_level_der_bounds: "fixed" or pair of floats > 0, default=(1e-5, 1e5)
        The lower and upper bounds of 'noise_level_der'.
        If "fixed", the noise_level_der parameter is not changed during
        the hyperparameter tuning.

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
        self.amplitude = validate_scalar(amplitude, "amplitude")
        self.amplitude_bounds = validate_bounds(amplitude_bounds, "amplitude_bounds")
        self.length_scale = validate_scalar_or_array(length_scale, "length_scale")
        self.length_scale_bounds = validate_bounds(length_scale_bounds, "length_scale_bounds")
        noise_level = noise_level if noise_level is not None else 0.0
        self.noise_level = validate_scalar(noise_level, "noise_level")
        self.noise_level_bounds = validate_bounds(noise_level_bounds, "noise_level_bounds")
        noise_level_der = noise_level_der if noise_level_der is not None else 0.0
        self.noise_level_der = validate_scalar_or_array(noise_level_der, "noise_level_der")
        self.noise_level_der_bounds = validate_bounds(
            noise_level_der_bounds, "noise_level_der_bounds"
        )

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
            return Hyperparameter("noise_level_der", "numeric", self.noise_level_der_bounds)

    def __call__(self, X, dX=None, add_noise=True, idx=None, eval_gradient=False):
        """Returns the kernel and optionally its gradients.

        Parameters
        ----------
        X: ndarray of shape (n_sampX, n_feat)
            Function input.

        dX: ndarray of shape (n_sampdX, n_featdX), default=None
            Derivative input. If None, then dX is assumed to be equal to X.

        idx: ndarray of shape (n_featdX,)
            Indices of the dimensions of X along which the derivatives are evaluated.
            If None, then idx is assumed to be equal to the range (0, n_feat_X).

        add_noise: bool, default=True
            If True, the white noise is added to the diagonal of the kernel.

        eval_gradient: bool, default=False
            If True, the gradients with respect to the log of the
            kernel hyperparameters are also returned.

        Returns
        -------
        K: ndarray of shape
           (n_sampX + n_sampdX * n_featdX, n_sampX + n_sampdX * n_featdX)
            Kernel.

        K_gradient: ndarray of shape
            (n_sampX + n_sampdX * n_featdX, n_sampX + n_sampdX * n_featdX, n_params)
            The gradient of the kernel with respect to the
            hyperparameters of the kernel. Only returned when eval_gradient
            is True.
        """
        self._check_length_scale(X, self.length_scale)

        if dX is None:
            dX = X
        return self._kernel_hybrid(
            X, dX=dX, idx=idx, add_noise=add_noise, eval_gradient=eval_gradient
        )

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
            K = np.exp(-0.5 * dists2)
            return K

    def _cov_yy(self, X, add_noise, Y=None, eval_gradient=False):
        """Covariance between function observations at inputs X and Y."""
        amp = self.amplitude
        ls = self.length_scale
        noise = self.noise_level

        if Y is None:
            (n_samples, _) = X.shape

            rbf = self._rbf(X)
            K = amp**2 * rbf
            if add_noise and self.noise_level:
                K += self.noise_level**2 * np.eye(n_samples)

            if not eval_gradient:
                return K

            (
                dK_damp,
                dK_dls,
                dK_dnoise,
                dK_dnoise_der,
            ) = self._initialize_gradients((n_samples, n_samples))

            # with respect to the amplitude parameter
            if not self.hyperparameter_amplitude.fixed:
                dK_damp = (2 * amp**2 * rbf)[:, :, np.newaxis]

            # with respect to the length_scale parameter
            if not self.hyperparameter_length_scale.fixed:
                if not self.anisotropic_length_scale:
                    dists2 = squareform(pdist(X / ls, metric="sqeuclidean"))
                    grad = amp**2 * dists2 * rbf
                    dK_dls = grad[:, :, np.newaxis]
                else:
                    dists2 = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2
                    dists2 /= ls**2
                    grad = amp**2 * rbf[..., np.newaxis] * dists2
                    dK_dls = grad

            # with respect to the noise_level parameter
            if not self.hyperparameter_noise_level.fixed:
                dK_dnoise += 2 * noise**2 * np.eye(n_samples)[:, :, np.newaxis]

            # no dependency on noise_level_der
            return K, np.concatenate(
                (
                    dK_damp,
                    dK_dls,
                    dK_dnoise,
                    dK_dnoise_der,
                ),
                axis=-1,
            )

        else:
            if X.shape[1] != Y.shape[1]:
                raise ValueError("The number of features of X and Y must be equal.")
            if eval_gradient:
                raise ValueError("Grad can only be evaluated when Y is None.")
            if add_noise:
                warnings.warn(
                    "Warning: noise is not added when Y is not None.",
                    UserWarning,
                )
            return amp**2 * self._rbf(X, Y)

    def _cov_ww(self, dX, add_noise, dy=None, idx=None, eval_gradient=False):
        """Covariance between derivative observations at inputs dX and dy."""
        if dy is None:
            dy = dX
        else:
            if dX.shape[1] != dy.shape[1]:
                raise ValueError("The number of features of dX and dy must be equal.")
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when dy is None.")
            if add_noise:
                warnings.warn(
                    "Warning: noise is not added when dY is not None.",
                    UserWarning,
                )
                add_noise = False

        (n_sampdX, n_featdX) = dX.shape
        (n_sampdY, _) = dy.shape

        amp = self.amplitude

        ls = np.asarray(self.length_scale)
        if not self.anisotropic_length_scale:
            ls = np.repeat(ls, n_featdX)

        if add_noise and self.noise_level_der > 0:
            noise_der = np.array(self.noise_level_der)
            if not self.anisotropic_noise_level_der:
                noise_der = np.repeat(noise_der, n_featdX)

        grad_idx = np.arange(dX.shape[1]) if idx is None else idx

        rbf = self._rbf(dX, dy)

        K = np.zeros((n_sampdX * len(grad_idx), n_sampdY * len(grad_idx)))
        if eval_gradient:
            (dK_damp, dK_dls, dK_dnoise, dK_dnoise_der) = self._initialize_gradients(
                (n_sampdX * len(grad_idx), n_sampdY * len(grad_idx))
            )

        for i, i_dim in enumerate(grad_idx):
            for j, j_dim in enumerate(grad_idx):
                dist_i = dX[:, i_dim].reshape(-1, 1) - dy[:, i_dim].reshape(-1, 1).T
                dist_i *= 1.0 / ls[i_dim] ** 2
                dist_j = dX[:, j_dim].reshape(-1, 1) - dy[:, j_dim].reshape(-1, 1).T
                dist_j *= 1.0 / ls[j_dim] ** 2
                dist_ii = (i_dim == j_dim) * (1.0 / ls[i_dim] ** 2)
                coeff = dist_ii - (dist_i * dist_j)
                K_ij = amp**2 * coeff * rbf
                K[
                    i * n_sampdX : (i + 1) * n_sampdX,
                    j * n_sampdY : (j + 1) * n_sampdY,
                ] = K_ij

                if add_noise and self.noise_level_der > 0 and i_dim == j_dim:
                    K[
                        i * n_sampdX : (i + 1) * n_sampdX,
                        j * n_sampdY : (j + 1) * n_sampdY,
                    ] += noise_der[i_dim] ** 2 * np.eye(n_sampdX, n_sampdY)

                if eval_gradient:
                    # with respect to the log amplitude parameter
                    if not self.hyperparameter_amplitude.fixed:
                        dK_damp[
                            i * n_sampdX : (i + 1) * n_sampdX,
                            j * n_sampdY : (j + 1) * n_sampdY,
                        ] = (2 * amp**2 * coeff * self._rbf(dX, dy))[:, :, np.newaxis]

                    # with respect to the log length_scale parameter
                    if not self.hyperparameter_length_scale.fixed:
                        if not self.anisotropic_length_scale:
                            dists2 = squareform(pdist(dX / ls, metric="sqeuclidean"))
                            d1 = (coeff * dists2 * rbf)[:, :, np.newaxis]
                            dcoeff = -2 * float(i_dim == j_dim) * (1.0 / ls[0] ** 2)
                            dcoeff += 4 * (dist_i * dist_j)
                            d2 = dcoeff * rbf
                            d2 = d2[:, :, np.newaxis]
                        else:
                            dist2 = (dX[:, np.newaxis, :] - dX[np.newaxis, :, :]) ** 2
                            dist2 /= ls**2
                            d1 = coeff[..., np.newaxis] * dist2 * rbf[..., np.newaxis]
                            dcoeff = 4 * (dist_i * dist_j)
                            dcoeff = np.repeat(dcoeff[:, :, np.newaxis], n_featdX, axis=2)
                            dcoeff -= 2 * float(i_dim == j_dim) * (1.0 / ls**2)
                            d2 = dcoeff * rbf[..., np.newaxis]
                        dK_dls[
                            i * n_sampdX : (i + 1) * n_sampdX,
                            j * n_sampdY : (j + 1) * n_sampdY,
                        ] = amp**2 * (d1 + d2)

                    # no dependence on noise_level
                    # with respect to the log noise_level_der parameter
                    if (
                        add_noise
                        and self.noise_level_der > 0
                        and not self.hyperparameter_noise_level_der.fixed
                    ):
                        if i_dim == j_dim:
                            if not self.anisotropic_noise_level_der:
                                noise_grad = (
                                    2 * noise_der[i_dim] ** 2 * np.eye(n_sampdX, n_sampdY)
                                )[..., np.newaxis]
                                dK_dnoise_der[
                                    i * n_sampdX : (i + 1) * n_sampdX,
                                    j * n_sampdY : (j + 1) * n_sampdY,
                                ] = noise_grad
                            else:
                                noise_grad = 2 * noise_der[i_dim] ** 2 * np.eye(n_sampdX, n_sampdY)
                                dK_dnoise_der[
                                    i * n_sampdX : (i + 1) * n_sampdX,
                                    j * n_sampdY : (j + 1) * n_sampdY,
                                    i_dim,
                                ] = noise_grad
        if eval_gradient:
            return K, np.concatenate(
                (dK_damp, dK_dls, dK_dnoise, dK_dnoise_der),
                axis=-1,
            )
        else:
            return K

    def _cov_wy(self, dX, Y, idx=None, eval_gradient=False):
        """Covariance between derivative (dX) and function (Y) observations.
        Note that cov_wy = cov_yw.T"""
        (n_sampdX, n_featdX) = dX.shape
        (n_sampY, n_feat_Y) = Y.shape

        rbf = self._rbf(dX, Y)

        amp = self.amplitude

        ls = np.array(self.length_scale)
        if not self.anisotropic_length_scale:
            ls = np.repeat(ls, n_featdX)
            ls_scalar = ls[0]

        grad_idx = idx if idx is not None else np.arange(n_featdX)

        K = np.zeros((n_sampdX * len(grad_idx), n_sampY))
        if eval_gradient:
            (dK_damp, dK_dls, dK_dnoise, dK_dnoise_der) = self._initialize_gradients(
                (n_sampdX * len(grad_idx), n_sampY)
            )

        for i, i_dim in enumerate(grad_idx):
            dist_i = dX[:, i_dim].reshape(-1, 1) - Y[:, i_dim].reshape(-1, 1).T
            dist_i_scl = dist_i * (1.0 / ls[i_dim] ** 2)
            K_i = -1.0 * amp**2 * dist_i_scl * rbf
            K[i * n_sampdX : (i + 1) * n_sampdX] = K_i
            if eval_gradient:

                # with respect to the log amplitude parameter
                if not self.hyperparameter_amplitude.fixed:
                    dK_i_amp = -2.0 * amp**2 * dist_i_scl * rbf
                    dK_damp[i * n_sampdX : (i + 1) * n_sampdX, :] = dK_i_amp[:, :, np.newaxis]

                # with respect to the log length_scale parameter
                if not self.hyperparameter_length_scale.fixed:
                    if not self.anisotropic_length_scale:
                        dists2 = cdist(
                            dX / ls_scalar,
                            Y / ls_scalar,
                            metric="sqeuclidean",
                        )
                        d1 = (-1.0 * dist_i_scl * dists2 * rbf)[:, :, np.newaxis]
                        dcoeff = 2.0 * dist_i * (1.0 / ls_scalar**2)
                        d2 = (dcoeff * rbf)[:, :, np.newaxis]
                    else:
                        dist2 = (dX[:, np.newaxis, :] - Y[np.newaxis, :, :]) ** 2
                        dist2 /= ls**2
                        d1 = -1.0 * dist_i_scl[..., np.newaxis] * dist2
                        d1 *= rbf[..., np.newaxis]
                        dcoeff = 2.0 * np.repeat(dist_i[..., np.newaxis], n_feat_Y, axis=2)
                        dcoeff /= ls**2
                        d2 = dcoeff * rbf[..., np.newaxis]
                    dK_dls[i * n_sampdX : (i + 1) * n_sampdX, :] = amp**2 * (d1 + d2)

                # no dependence on noise_level
                # no dependence on noise_level_der
        if eval_gradient:
            return K, np.concatenate(
                (dK_damp, dK_dls, dK_dnoise, dK_dnoise_der),
                axis=-1,
            )
        else:
            return K

    def _kernel_hybrid(self, X, dX, add_noise, idx=None, eval_gradient=False):
        """Returns the composite covariance between function and derivative observations,
        and optionally its gradient."""
        if eval_gradient:
            (K_yy, dK_yy) = self._cov_yy(X=X, eval_gradient=True, add_noise=add_noise)
            (K_ww, dK_ww) = self._cov_ww(dX=dX, idx=idx, eval_gradient=True, add_noise=add_noise)
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
            K_yy = self._cov_yy(X, add_noise=add_noise)
            K_ww = self._cov_ww(dX, idx=idx, add_noise=add_noise)
            K_wy = self._cov_wy(dX, X, idx=idx)
            K = np.block([[K_yy, K_wy.T], [K_wy, K_ww]])
            return K

    def _check_length_scale(self, X, length_scale):
        """Check the length_scale parameter."""
        if np.ndim(length_scale) == 1 and X.shape[1] != len(length_scale):
            raise ValueError(
                "Anisotropic kernels must have the same number of "
                f"dimensions as data ({len(length_scale):d}!={X.shape[1]:d})"
            )

    def _initialize_gradients(self, dims):
        def alloc(fixed: bool, size: int):
            return np.empty(dims + (0,)) if fixed else np.zeros(dims + (size,))

        ls_size = 1 if not self.anisotropic_length_scale else np.atleast_1d(self.length_scale).size
        nld_size = (
            1 if not self.anisotropic_noise_level_der else np.atleast_1d(self.noise_level_der).size
        )

        dK_damp = alloc(self.hyperparameter_amplitude.fixed, 1)
        dK_dls = alloc(self.hyperparameter_length_scale.fixed, ls_size)
        dK_dnoise = alloc(self.hyperparameter_noise_level.fixed, 1)
        dK_dnoise_der = alloc(self.hyperparameter_noise_level_der.fixed, nld_size)

        return (
            dK_damp,
            dK_dls,
            dK_dnoise,
            dK_dnoise_der,
        )

    def __repr__(self):
        if not self.anisotropic_length_scale:
            ls0 = np.ravel(self.length_scale)[0]
            desc = f"{self.amplitude:.3g}**2 * " f"DerivativeRBF(length_scale={ls0:.3g})"
        else:
            scales = ", ".join(f"{s:.3g}" for s in self.length_scale)
            desc = f"{self.amplitude:.3g}**2 * " f"DerivativeRBF(length_scale=[{scales}])"

        desc += f" + WhiteKernel(noise_level={self.noise_level:.3g})"

        if not self.anisotropic_noise_level_der:
            desc += f" + WhiteKernel_der(noise_level={self.noise_level_der:.3g})"
        else:
            noise_scales = ", ".join(f"{s:.3g}" for s in self.noise_level_der)
            desc += f" + WhiteKernel_der(noise_level=[{noise_scales}])"

        return desc
