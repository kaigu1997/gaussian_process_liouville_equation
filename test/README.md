The purpose of this program is to test whether Gaussian Process Regression (GPR) could be used in mimicing the phase space distribution generated from MQCLE or SE.

It could be used to depict each element of the 2-by-2 partial Wigner transformed density matrix (PWTDM).

The GPR is implemented by myself, and the optimization of hyperparameters uses the Simplex algorithm from GNU Scientific Library.

To build, simply run "make" with Intel(R) compilers, or to modify makefile for other compilers (with linkage to Intel(R) MKL library required by Eigen).

References:
> M. Galassi et al, GNU Scientific Library Reference Manual (3rd Ed.), ISBN 0954612078.