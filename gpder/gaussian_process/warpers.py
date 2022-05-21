"""Gaussian Process models with warped latent spaces."""

import numpy as np
from sklearn import clone

from . import gaussian_process
from .gaussian_process import GaussianProcessRegressor

__all__ = ['GPWarper']


class GPWarper(GaussianProcessRegressor):
    """GP whose latent space is to be transformed or 'warped'.
       Two warping options are supported: log-transformation and
       square-root-transformation.
    """
    def __init__(self, transform, kernel=None, alpha=1e-10, optimizer="fmin_l_bfgs_b",
                 n_restarts_optimizer=0, mean=0, dmean=0, copy_data=True,
                 random_state=None, alpha_warp=1e-3):
        self._alpha_warp = alpha_warp
        if transform in ["log", "squareroot"]:
            self._transform_type = transform
        else:
            raise ValueError('transform must be one of "log" or "squareroot".')

        super().__init__(kernel, alpha, optimizer, n_restarts_optimizer,
                         mean, dmean, copy_data, random_state)
    @property
    def warped(self):
        return True

    def fit(self, X, y, dX=None, dy=None, idX=None):
        y_warped = self._transform_obs(y)
        if dy is not None:
            dy_warped = self._transform_grad_obs(y, dy)
        else:
            dy_warped = None
        super().fit(X, y_warped, dX, dy_warped)

    def predict_latent(self, X, return_cov=False, return_std=False):
        if return_cov:
            return super().predict(X, return_cov=True)
        elif return_std:
            return super().predict(X, return_std=True)
        else:
            return super().predict(X)

    def predict(self, X, return_cov=False, return_std=False):
        mu, std = self.predict_latent(X, return_std=True)
        mu, std = mu.ravel(), std.ravel()
        mu_inv = self._transform_obs_back(mu, std)
        if return_cov:
            _, cov = self.predict_latent(X, return_cov=True)
            cov_inv = self._transform_cov_back(cov, mu)
            return mu_inv, cov_inv
        elif return_std:
            _, std = self.predict_latent(X, return_std=True)
            std_inv = self._transform_std_back(std, mu)
            return mu_inv, std_inv
        else:
            return mu_inv

    def predict_der_latent(self, dX, return_cov=False, return_std=False,
                           flatten=True):
        if return_cov:
            return super().predict_der(dX, return_cov=True, flatten=flatten)
        elif return_std:
            return super().predict_der(dX, return_std=True, flatten=flatten)
        else:
            return super().predict_der(dX, flatten=flatten)

    def predict_der(self, dX, return_cov=False, return_std=False):
        y_mu, y_std = self.predict_latent(dX, return_std=True)
        y_mu, y_std = y_mu.ravel(), y_std.ravel()
        dy_mu, dy_std = self.predict_der_latent(dX, return_std=True, flatten=False)
        dy_mu_inv = self._transform_grad_obs_back(y_mu, dy_mu, y_std)
        if return_cov:
            _, dy_cov = self.predict_der_latent(dX, return_cov=True)
            cov_inv = self._transform_cov_back(dy_cov, np.tile(y_mu, dy_mu.shape[1]))
            return dy_mu_inv, cov_inv
        elif return_std:
            std_inv = self._transform_std_back(dy_std, y_mu)
            return dy_mu_inv, std_inv
        else:
            return dy_mu_inv

    def _transform_obs(self, y):
        if self._transform_type == "log":
            return np.log(y)
        elif self._transform_type == "squareroot":
            return np.sqrt(2 * (y - self._alpha_warp))

    def _transform_obs_back(self, y, std=1):
        if self._transform_type == "log":
            return np.exp(y + 0.5 * std**2)
        elif self._transform_type == "squareroot":
            return self._alpha_warp + 0.5 * y**2

    def _transform_grad_obs(self, y, dy):
        if self._transform_type == "log":
            return 1. / y.reshape(-1, 1) * dy
        elif self._transform_type == "squareroot":
            return 1. / self._transform_obs(y).reshape(-1, 1) * dy

    def _transform_grad_obs_back(self, y, dy, std=1):
        if self._transform_type == "log":
            return np.exp(y + 0.5 * std**2).reshape(-1, 1) * dy
        elif self._transform_type == "squareroot":
            return y.reshape(-1, 1) * dy

    def _transform_cov_back(self, cov, y):
        if self._transform_type == "log":
            return np.exp(y + 0.5 * cov**2)**2 * (np.exp(cov**2) - 1.)
        elif self._transform_type == "squareroot":
            return cov * y**2

    def _transform_std_back(self, std, y):
        if std.ndim > 1:
            y = y.reshape(-1, 1)
        if self._transform_type == "log":
            return np.exp(y + 0.5 * std**2)**2 * (np.exp(std**2) - 1.)
        elif self._transform_type == "squareroot":
            return std * y**2
