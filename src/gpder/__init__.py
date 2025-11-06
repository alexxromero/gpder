"""
Pippa is a cat.
"""

from . import bayes, gaussian_process
from .bayes import GPUncertaintyOptimizer, NetVarianceLoss
from .gaussian_process import GaussianProcessRegressor, kernels
