"""
The gpder module implements Gaussian Process Regression with derivative
observations.

Author: Alexis Romero <alexir2@uci.edu>
"""

from . import gaussian_process
from . import bayes

from .gaussian_process import *
from .bayes import *

__all__ = gaussian_process.__all__ + \
          bayes.__all__
