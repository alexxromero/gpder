"""
Pippa is a cat.
"""

from . import gaussian_process
from . import bayes

from .gaussian_process import GaussianProcessRegressor
from .gaussian_process import kernels
from .bayes import GPUncertaintyOptimizer, NetVarianceLoss