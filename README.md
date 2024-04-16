# GPder 

This package offers an implementation of the Gaussian Process (GP) Regression 
algorithm with and without derivative information. 

## Description 
The following kernels can be used:
- RegularKernel: Kernel for regular GP regression

    $k(x_i, x_j) = \alpha^2 \mathrm{exp} \left( -\frac{\mid \mid x_i - x_j \mid \mid^2 }{2 \ell^2} \right) + \sigma^2 I$

- DerivativeKernel: Kernel for GP regression with derivative observations. Has the same form as the regular kernel but the covariance term is expanded to include derivative observations. The added noise is also expanded with the derivative noise parameter $\sigma^2_{\nabla}$.

    $k(\bm{x}_i, \bm{x}_j) = \alpha^2 \mathrm{exp} \left( -\frac{\mid \mid \bm{x}_i - \bm{x}_j \mid \mid^2 }{2\bm{\ell}^2} \right)_{\mathrm{expanded}} + \sigma^2_{\mathrm{expanded}} I$


See PAPER.

### Install

```
pip install gpder
```

### References

TITLE OF PAPER