"""Radial basis function (RBF) kernels for gaussian regression with
derivative observations.
"""

import numpy as np
from scipy.spatial.distance import pdist, cdist

def dist(X1, X2):
    return cdist(X1, X2, metric='euclidean')  # pairwise Eucledian dist

def dist2(X1, X2):
    return cdist(X1, X2, metric='sqeuclidean')  # pairwise sqrt Euclidean dist

def kernel_XX(X1, X2, length_scale):
    X1 /= length_scale
    X2 /= length_scale
    # TODO: why is it 1??
    return 1*np.exp(-0.5 * dist2(X1, x2))

def kernel_dXX(X1, X2, length_scale):
    chain_coeffs = -np.diag(length_scale) * dist(X1, X2)
    return chain_coeffs * kernel_XX(X1, X2, length_scale)

def kernel_dXdX(X1, X2, length_scale):
    chain_coeffs = np.diag(length_scale) * (1 - np.diag(length_scale)
    chain_coeffs = chain_coeffs * dist2(X1, X2))
    return chain_coeffs * kernel_XX(X1, X2, length_scale)

def kernel_deraware(X1, X2, length_scale):
    Kxx = kernel_XX(X1, X2, length_scale)
    Kwx = kernel_dXX(X1, X2, length_scale)
    Kww = kernel_dXdX(X1, X2, length_scale)
    Kxw = Kwx.T
    kderaware = np.block([[Kxx, Kxw], [Kwx, Kww]])
    return kderaware
