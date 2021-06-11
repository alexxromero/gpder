"""
The gpder module implements Gaussian Process Regression with derivative
observations to improve performance.

Author: Alexis Romero <alexir2@uci.edu>
"""

from . import gaussian_process
from . import bayesian_opt
from . import bayesian_opt_uncertainty
from . import bayesian_opt_utils

from .gaussian_process import *
from .bayesian_opt import *
from .bayesian_opt_uncertainty import *
from .bayesian_opt_utils import *

__all__ = gaussian_process.__all__ + \
          bayesian_opt.__all__ + \
          bayesian_opt_uncertainty.__all__ + \
          bayesian_opt_utils.__all__
