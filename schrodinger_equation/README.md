The purpose of this program is to give an exact solution of quantum mechanic problem using Discrete Variable Representation (DVR)[1] with Absorbing Boundary Condition[2-3].

This program could be used to solve exact solution under diabatic basis ONLY.

It requires C++17 or newer C++ standards when compiling and needs connection to Intel(R) Math Kernel Library (MKL).

To build, simply run "make" with Intel(R) compilers, or to modify makefile for other compilers. Then running bat.sh or the executable mqcl with proper input. The input file format is in bat.sh.

This is going to merge with schrodinger equation into one folder.

Reference:
> 1. Colbert D T, Miller W H. A novel discrete variable representation for quantum mechanical reactive scattering via the S-matrix Kohn method. J. Chem. Phys., 1992, 96(3): 1982-1991.
> 2. Manolopoulos D E. Derivation and reflection properties of a transmission-free absorbing potential. J. Chem. Phys., 2002, 117(21): 9552-9559.
> 3. Gonzalez-Lezana T, Rackham E J, Manolopoulos D E. Quantum reactive scattering with a transmission-free absorbing potential. J. Chem. Phys., 2004, 120(5): 2247-2254.