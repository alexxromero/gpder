"""Gaussian process regressor (GPR) with derivative observations."""

import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
from .kernel import kernel_XX, kernel_dXX, kernel_deraware

__all__ = ['GaussianProcessRegressor']

class GaussianProcessRegressor:
    """Implements gaussian process regression (GPR) w/derivative observations.

    The GPR with derivative observations is base don XXX.

    Attributes:
        length_scale:
        length_scale_bounds:
    """

    def __init__(self, length_scale, length_scale_bounds=(1e-5, 1e5),
                 copy_data=True, random_state=None):
        self.length_scale = self._verify_length_scale(length_scale)
        self.length_scale_bounds = length_scale_bounds
        self.copy_data = copy_data
        self.random_state = random_state


    def fit(self, X, y, dX=None, dy=None, alpha=1e-10):
        """Fit GPR model.

        Args:
            X:
            y:
            alpha:

        Returns:
        """
        self.X_train, self.y_train = self._verify_X_y(X, y)
        self.has_derinfo = False
        if dX and dy:
            self.has_derinfo = True
            self.dX_train, self.dy_train = self._verify_dX_dy(dX, dy)
        self.alpha = alpha

        if self.has_derinfo:
            self.kernel_train = kernel_deraware(self.X_train, self.dX_train,
                                                self.length_scale)
            self.y_chol_ = np.concatenate((self.y_train, self.dy_train))
        else:
            self.kernel_train = kernel_XX(self.X_train, self.X_train,
                                          self.length_scale)
            self.y_chol_ = self.y_train

        self.kernel_train += self.alpha*np.eye((self.kernel_train.shape[0]))
        self.lower = True
        try:
            self.L_chol = cholesky(self.kernel_train, lower=self.lower)
        except np.linalg.linAlgError as exc:
            print("The kernel is not positive definite. Try increasing alpha.")
            raise
        self.alpha_chol = cho_solve((self.L_chol, self.lower), self.y_chol_)
        return self


    def predict(self, X_star):
        """
        """
        self.X_star = self._verify_X_star(X_star)
        self.kernel_post, kernel_X_star = self.calc_kernel_post_and_X_star()
        self.mu = self.kernel_post.dot(self.alpha_chol)
        v = cho_solve((self.L_chol, self.lower), self.kernel_post.T)
        L_inv = solve_triangular(self.L_chol.T, np.eye(self.L_chol.shape[0]))
        K_inv = L_inv.dor(L_inv.T)
        std2 = np.diag(kernel_X_star)-np.einsum("ij,ij->i",
                                               np.dot(self.kernel_post, K_inv),
                                               self.kernel_post)
        self.std = np.sqrt(std2)


    def calc_kernel_post_and_star(self):
        kernel_star = kernel_XX(self.X_star, self.X_star,
                                self.length_scale)
        kernel_star_train = kernel_XX(self.X_star, self.X_train,
                                      self.length_scale)
        if self.has_derinfo:
            kernel_der_star = kernel_dXX(self.dX_train, self.X_star,
                                         self.length_scale)
            kernel_post = np.block([kernel_star_train, kernel_der_star.T])
        else:
            kernel_post = kernel_star_train
        return kernel_post, kernel_star


    def _verify_X_y(self, X, y):
        X_ = np.array(X, copy=True) if self.copy_data else X
        y_ = np.array(y, copy=True) if self.copy_data else y
        if X_.shape[0] != y_.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        if X_.ndim != 2:
            raise ValueError("X must be a 2D array.")
        return X_, y_


    def _verify_dX_dy(self, dX, dy):
        dX_ = np.array(dX, copy=True) if self.copy_data else dX
        dy_ = np.array(dy, copy=True) if self.copy_data else dy
        if dX_.shape[0] != dy_.shape[0]:
            raise ValueError("dX and dy must have the same number of samples.")
        if dX_.ndim != 2:
            raise ValueError("dX must be a 2D array.")
        return dX_, dy_


    def _verify_X_star(self, X_star):
        X_ = np.array(X_star, copy=True) if self.copy_data else X_star
        if X_.ndim != 2:
            raise ValueError("X_star must be a 2D array.")
        if X_.shape[1] != self.X_train.shape[1]:
            raise ValueError("X_star must have the same number of features "
                             "as X.")
        return X_
