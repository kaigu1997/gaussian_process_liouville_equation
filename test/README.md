The purpose of this program is to test whether Gaussian Process Regression (GPR) could be used in mimicing the phase space distribution generated from MQCLE or SE.

It could be used to depict each element of the 2-by-2 partial Wigner transformed density matrix (PWTDM).

The GPR is used from the shogun library, and the optimization of hyperparameters uses the Simplex algorithm from GNU Scientific Library.

To build, simply run "make" with Intel(R) compilers, or to modify makefile for other compilers.

References:
> Soeren Sonnenburg, Heiko Strathmann, Sergey Lisitsyn, Viktor Gal, Fernando J. Iglesias García, Wu Lin, … Björn Esser. (2019, July 5). shogun-toolbox/shogun: Shogun 6.1.4 (Version shogun_6.1.4). Zenodo. http://doi.org/10.5281/zenodo.591641
> M. Galassi et al, GNU Scientific Library Reference Manual (3rd Ed.), ISBN 0954612078.