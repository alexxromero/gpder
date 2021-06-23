"""Custom kernels for Gaussian Process Regression.

GPKernelDerAware is based on
https://papers.nips.cc/paper/2002/file/5b8e4fd39d9786228649a8a8bec4e008-Paper.pdf
"""

from . import deraware_kernel
from . import kernel

from .deraware_kernel import GPKernelDerAware
from .kernel import GPKernel
