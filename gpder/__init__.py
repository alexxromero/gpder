"""
The gpder module implements Gaussian Process Regression with derivative
observations to improve performance.

Author: Alexis Romero <alexir2@uci.edu>
"""

from . import gaussian_process
from . import plotting_utils
from . import kernel
from . import bayesian_opt

from .gaussian_process import *
from .plotting_utils import *
from .kernel import *
from .bayesian_opt import *

__all__ = gaussian_process.__all__ + \
          plotting_utils.__all__ + \
          kernel.__all__ + \
          bayesian_opt.__all__
