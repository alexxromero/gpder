"""Gaussian Process models with warped output spaces."""

import numpy as np
from sklearn import clone

from . import gaussian_process
from .gaussian_process import GaussianProcessRegressor

__all__ = ['GPRSquareRootWarp']


class GPRSquareRootWarp(GaussianProcessRegressor):
    """GP whose output space is to be warped by the squared root transformation.
    """
    def __init__(self, kernel=None, alpha=1e-10, optimizer="fmin_l_bfgs_b",
                 n_restarts_optimizer=0, mean=0, dmean=0, copy_data=True,
                 random_state=None, alpha_warp=1e-3):
        self._alpha_warp = alpha_warp
        super().__init__(kernel, alpha, optimizer, n_restarts_optimizer,
                         mean, dmean, copy_data, random_state)
    @property
    def warped(self):
        return True


    def _transform_obs(self, y, alpha):
        return np.sqrt(2 * (y - alpha ))

    def _transform_derivative_obs(self, y, dy, alpha):
        return 1 / self._transform_obs(y, alpha).reshape(-1, 1) * dy

    def fit(self, X, y, dX=None, dy=None, idX=None):
        y_warped = self._transform_obs(y, self._alpha_warp)
        if dy is not None:
            dy_warped = self._transform_derivative_obs(y, dy, self._alpha_warp)
        else:
            dy_warped = None
        super().fit(X, y_warped, dX, dy_warped)

    def predict_latent(self, X, return_cov=False, return_std=False):
        mu = super().predict(X)
        if return_cov:
            _, cov = super().predict(X, return_cov=True)
            return mu, cov
        elif return_std:
            _, std = super().predict(X, return_std=True)
            return mu, std
        else:
            return mu

    def predict(self, X, return_cov=False, return_std=False):
        if return_cov:
            mu, cov = self.predict_latent(X, return_cov=True)
            mu_invtrans = self._alpha_warp + mu**2 / 2
            return mu_invtrans, cov*mu**2
        elif return_std:
            mu, std = self.predict_latent(X, return_std=True)
            mu_invtrans = self._alpha_warp + mu**2 / 2
            return mu_invtrans, std*np.squeeze(mu, axis=1)*2
        else:
            mu = self.predict_latent(X)
            mu_invtrans = self._alpha_warp + mu**2 / 2
            return mu_invtrans

    def predict_der_latent(self, dX, return_cov=False, return_std=False):
        mu, cov = super().predict_der(dX, return_cov=True)
        if return_cov:
            _, cov = super().predict_der(dX, return_cov=True)
            return mu, cov
        elif return_std:
            _, std = super().predict_der(dX, return_std=True)
            return mu, std
        else:
            return mu

    def predict_der(self, dX, return_cov=False, return_std=False):
        if return_cov:
            mu, cov = self.predict_der_latent(dX, return_cov=True)
            return mu, cov*mu**2
        elif return_std:
            mu, std = self.predict_der_latent(dX, return_std=True)
            return mu, std*np.squeeze(mu, axis=1)**2
        else:
            mu = self.predict_der_latent(dX)
            return mu
