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
- notes, in both English and 简体中文
- test code and main program, commented in Doxygen style