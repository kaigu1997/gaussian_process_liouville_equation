# GPR for MQCLE
This project uses Gaussian Process Regression (GPR) for the evolution of Mixed Quantum-Classical Liouville Equation (MQCLE).
## Overview
We includes
- ***the documents, mainly the reading notes of literatures, and a notes of calculating derivatives from fft***, and
- exact solutions, i.e., grided solution of Schrodinger Equation (SE) and MQCLE and their plotting codes, and
- ***test codes, which test whether the GPR works well for depicting the phase space distribution at a moment (TBD)***, and
- ***the main program, which uses GPR to depict the phase space distribution at each moment, and sampling from the distribution, where the chosen points are used for evolution to the next time step***.

Those in ***boldslash style*** are works that have not yet finished.
## Things to Be Done
- altogether gitignore, gitattributes, makefile
- combined exact solutions (SE and MQCLE) into one subfolder, with common interface
- notes, in 简体中文 (for native English speakers, the original literatures are enough)
- test code and main program, commented in Doxygen style
## Reference
> 1. Colbert D T, Miller W H. A novel discrete variable representation for quantum mechanical reactive scattering via the S-matrix Kohn method. J. Chem. Phys., 1992, 96(3): 1982-1991.
> 2. Manolopoulos D E. Derivation and reflection properties of a transmission-free absorbing potential. J. Chem. Phys., 2002, 117(21): 9552-9559.
> 3. Gonzalez-Lezana T, Rackham E J, Manolopoulos D E. Quantum reactive scattering with a transmission-free absorbing potential. J. Chem. Phys., 2004, 120(5): 2247-2254.
> 4. Kapral R. Quantum dynamics in open quantum-classical systems. J. Phys.: Condens. Matter, 2015, 27(7): 073201.
> 5. Williams C K I, Rasmussen C E. Gaussian processes for machine learning. Cambridge, MA: MIT press, 2006.