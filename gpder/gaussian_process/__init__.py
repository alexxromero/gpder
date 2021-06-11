"""The mod gaussian_process implements Gaussian Process regression with
the custom kernel
    kernel = ConstantKernel * RBF + WhiteKernel

Derivative observations can be used to aid the regression, as implemented in
https://papers.nips.cc/paper/2002/file/5b8e4fd39d9786228649a8a8bec4e008-Paper.pdf
"""

from . import gaussian_process
from . import deraware_kernel
from . import kernel

from .gaussian_process import GaussianProcessRegressor
from .deraware_kernel import GPKernelDerAware
from .kernel import GPKernel
