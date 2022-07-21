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

    def utility(self, X, X_query, gp, freeze_params=True, batch_size=None):
        if self.kind=='trace':
            return self.expected_cov_trace(X, X_query, gp,
                                           freeze_params, batch_size)
        # elif self.kind=='det':
        #     return self.expected_cov_det(X, X_query, gp)

    def _expected_cov(X, X_query, gp, freeze_params):
        gp_temp = deepcopy(gp)
        if freeze_params:
            gp_temp.optimizer=None
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
            _, cov = gp_temp.predict_latent(X_query, return_cov=True)
        else:
            _, cov = gp_temp.predict(X_query, return_cov=True)
        return cov

    def _current_cov(X_query, gp, freeze_params):
        gp_current = deepcopy(gp)
        if freeze_params:
            gp_current.optimizer=None
        if hasattr(gp, 'warped'):
            _, cov = gp_current.predict_latent(X_query, return_cov=True)
        else:
            _, cov = gp_current.predict(X_query, return_cov=True)
        return cov

    # @staticmethod
    # def expected_cov_det(X, X_query, gp):
    #     exp_det = np.linalg.det(AcquisitionFunction._expected_cov(X, X_query, gp))
    #     current_det = np.linalg.det(AcquisitionFunction._current_cov(X_query, gp))
    #     return 1 - exp_det / current_det

    @staticmethod
    def expected_cov_trace(X, X_query, gp, freeze_params=True, batch_size=None):
        if batch_size is not None:
            nbatches = int(X_query.shape[0] / batch_size)
            if X_query.shape[0] % batch_size > 0:
                nbatches +=1
            exp_trace, current_trace = 0, 0
            for i in range(nbatches):
                X_query_batch = X_query[i*batch_size : (i+1)*batch_size]
                exp_trace += np.trace(
                    AcquisitionFunction._expected_cov(X, X_query_batch, gp, freeze_params)
                    )
                current_trace += np.trace(
                    AcquisitionFunction._current_cov(X_query_batch, gp, freeze_params)
                    )
        else:
            exp_trace = np.trace(
                AcquisitionFunction._expected_cov(X, X_query, gp, freeze_params)
                )
            current_trace = np.trace(
                AcquisitionFunction._current_cov(X_query, gp, freeze_params)
                )
        return 1 - exp_trace / current_trace
