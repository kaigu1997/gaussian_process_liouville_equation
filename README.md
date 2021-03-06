# GPR for MQCLE
This project uses Gaussian Process Regression (GPR) for the evolution of Mixed Quantum-Classical Liouville Equation (MQCLE).
## Overview
We includes
- the documents, i.e., the reading notes of literatures, and
- exact solutions, i.e., grided solution of Schrodinger Equation (SE) and MQCLE and their plotting codes, and
- test codes, which test whether the GPR works well for depicting the phase space distribution (of Liouville equation solution) at a moment, and
- ***the main program, which uses GPR to depict the phase space distribution at each moment, and sampling from the distribution, where the chosen points are used for evolution to the next time step***.

Those in ***boldslash style*** are works that have not yet finished.
## Things to Be Done
- altogether makefile
- combined exact solutions (SE and MQCLE) into one subfolder, with common interface
- main program, commented in Doxygen style
## Reference
> 1. Colbert D T, Miller W H. A novel discrete variable representation for quantum mechanical reactive scattering via the S-matrix Kohn method. J. Chem. Phys., 1992, 96(3): 1982-1991.
> 2. Manolopoulos D E. Derivation and reflection properties of a transmission-free absorbing potential. J. Chem. Phys., 2002, 117(21): 9552-9559.
> 3. Gonzalez-Lezana T, Rackham E J, Manolopoulos D E. Quantum reactive scattering with a transmission-free absorbing potential. J. Chem. Phys., 2004, 120(5): 2247-2254.
> 4. Kapral R. Quantum dynamics in open quantum-classical systems. J. Phys.: Condens. Matter, 2015, 27(7): 073201.
> 5. Williams C K I, Rasmussen C E. Gaussian processes for machine learning. Cambridge, MA: MIT press, 2006.
> 6. Soeren Sonnenburg, Heiko Strathmann, Sergey Lisitsyn, Viktor Gal, Fernando J. Iglesias García, Wu Lin, … Björn Esser. (2019, July 5). shogun-toolbox/shogun: Shogun 6.1.4 (Version shogun_6.1.4). Zenodo. http://doi.org/10.5281/zenodo.591641