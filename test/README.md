The purpose of this program is to test whether Gaussian Process Regression (GPR) could be used in mimicing the phase space distribution generated from MQCLE or SE.

It could be used to depict each element of the 2-by-2 partial Wigner transformed density matrix (PWTDM).

The GPR is implemented by myself, with linear algebra in [Eigen](http://eigen.tuxfamily.org) and the kernel from [Shogun](https://www.shogun-toolbox.org/), and the optimization of hyperparameters uses the Simplex algorithm from [NLOPT](https://nlopt.readthedocs.io).

To build, simply run "make" with Intel(R) compilers, or to modify makefile for other compilers (with linkage to [Intel(R) MKL](https://software.intel.com/content/www/us/en/develop/tools/math-kernel-library.html) required by [Eigen](http://eigen.tuxfamily.org)).

References:
> 1. Steven G. Johnson, The NLopt nonlinear-optimization package, http://github.com/stevengj/nlopt
> 2. Gael Guennebaud and Benoit Jacob and others, Eigen v3, http://eigen.tuxfamily.org
> 3. Soeren Sonnenburg, Heiko Strathmann, Sergey Lisitsyn, Viktor Gal, Fernando J. Iglesias García, Wu Lin, … Björn Esser. (2019, July 5). shogun-toolbox/shogun: Shogun 6.1.4 (Version shogun_6.1.4). Zenodo. http://doi.org/10.5281/zenodo.591641