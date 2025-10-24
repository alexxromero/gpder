# GPder 

This package offers an implementation of the Gaussian Process (GP) Regression 
algorithm with and without derivative information. 

## Description 
The following kernels can be used:
- RegularKernel: Kernel for regular GP regression

    $k(x_i, x_j) = \alpha^2 \mathrm{exp} \left( -\frac{\mid \mid x_i - x_j \mid \mid^2 }{2 \ell^2} \right) + \sigma^2 I$

- DerivativeKernel: Kernel for GP regression with derivative observations. Has the same form as the regular kernel but the covariance term is expanded to include derivative observations. The added noise is also expanded with the derivative noise parameter $\sigma^2_{\nabla}$.

    $k({x}_i, {x}_j) = \alpha^2 \mathrm{exp} \left( -\frac{\mid \mid {x}_i - {x}_j \mid \mid^2 }{2{\ell}^2} \right) _{\mathrm{expanded}} + \sigma^2 _{\mathrm{expanded}} I$


See "Efficient Estimation of Unfactorizable Systematic Uncertainties".

### Install

```
pip install gpder
```

### References

Efficient Estimation of Unfactorizable Systematic Uncertainties
