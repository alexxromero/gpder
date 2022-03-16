import numpy as np
import scipy
import sklearn
import copy
from copy import deepcopy

class AcquisitionFunction():
    """Acquisition function for the BayesUncertaintyOptimization class.
    """

    def __init__(self, kind):
        accepted_kinds = ['det', 'trace']
        if kind not in accepted_kinds:
            raise ValueError("kind has to be one of {}".format(accepted_kinds))
        else:
            self.kind = kind

    def utility(self, X, Xpred, gp):
        if self.kind=='trace':
            return self.expected_cov_trace(X, Xpred, gp)
        elif self.kind=='det':
            return self.expected_cov_det(X, Xpred, gp)

    def _expected_cov(X, Xpred, gp):
        gp_temp = deepcopy(gp)
        gp_temp.optimizer = None
        X = np.reshape(X, (-1, gp_temp.X_train_.shape[1]))
        y = gp_temp.predict(X=X).reshape(-1, gp_temp.y_train_.shape[1])
        X_temp = np.vstack((gp_temp.X_train_, X))
        y_temp = np.vstack((gp_temp.y_train_, y))
        if gp_temp._has_derinfo:
            dy = gp_temp.predict_der(dX=X).reshape(-1, gp_temp.dy_train_.shape[1])
            dX_temp = np.vstack((gp_temp.dX_train_, X))
            dy_temp = np.vstack((gp_temp.dy_train_, dy))
            gp_temp.fit(X=X_temp, y=y_temp, dX=dX_temp, dy=dy_temp)
        else:
            gp_temp.fit(X=X_temp, y=y_temp)
        if hasattr(gp, 'warped'):
            _, cov = gp_temp.predict_latent(Xpred, return_cov=True)
        else:
            _, cov = gp_temp.predict(Xpred, return_cov=True)
        return cov
    
    def _current_cov(Xpred, gp):
        gp_current = deepcopy(gp)
        gp_current.optimizer = None
        if hasattr(gp, 'warped'):
            _, cov = gp_current.predict_latent(Xpred, return_cov=True)
        else:
            _, cov = gp_current.predict(Xpred, return_cov=True)
        return cov

    @staticmethod
    def expected_cov_det(X, Xpred, gp):
        exp_det = np.linalg.det(AcquisitionFunction._expected_cov(X, Xpred, gp))
        current_det = np.linalg.det(AcquisitionFunction._current_cov(Xpred, gp))
        return 1 - exp_det / current_det
    
    @staticmethod
    def expected_cov_trace(X, Xpred, gp):
        exp_trace = np.trace(AcquisitionFunction._expected_cov(X, Xpred, gp))
        current_trace = np.trace(AcquisitionFunction._current_cov(Xpred, gp))
        return 1 - exp_trace / current_trace
    


