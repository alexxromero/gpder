import numpy as np
import scipy
import sklearn
import copy
from copy import deepcopy

class AcquisitionFunction():
    """Acquisition function for the BayesUncertaintyOptimization class.
    """

    def __init__(self):

    def utility(self, X, X_query, gp, trace_sigma_query, batch_size=None):
        return self.expected_cov_trace(
            X, X_query, trace_sigma_query, gp, batch_size
            )

    def _expected_cov(X, X_query, gp):
        gp_temp = deepcopy(gp)
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

    @staticmethod
    def expected_cov_trace(X, X_query, gp, current_trace, batch_size=None):
        if batch_size is not None:
            nbatches = int(X_query.shape[0] / batch_size)
            if X_query.shape[0] % batch_size > 0:
                nbatches +=1
            exp_trace = 0
            for i in range(nbatches):
                X_query_batch = X_query[i*batch_size : (i+1)*batch_size]
                exp_trace += np.trace(
                    AcquisitionFunction._expected_cov(X, X_query_batch, gp, freeze_params)
                    )
        else:
            exp_trace = np.trace(
                AcquisitionFunction._expected_cov(X, X_query, gp, freeze_params)
                )
        return 1 - exp_trace / current_trace
