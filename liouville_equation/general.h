/// @file general.h
/// @brief Declarations of variables and functions used in main driver: I/O, evolution, etc 
///
/// This header file contains some functions that will be used 
/// in main function, but they are often call once or twice.
/// However, putting the function codes directly in main make the
/// main function and main file too long. In that case, these 
/// functions are moved to an individual file (general.cpp)
/// and the interfaces are declared here.

#ifndef GENERAL_H
#define GENERAL_H

#include <iostream>
#include <memory>
#include <tuple>
#include "pes.h"
#include "matrix.h"

const double pi = acos(-1.0); ///< mathematical constant, pi
const double hbar = 1.0; ///< physical constant, reduced Planck constant
const double PlanckH = 2 * pi * hbar; ///< physical constant, Planck constant
// Alpha and Beta are for cblas complex matrix multiplication functions
// as they behave as A=alpha*B*C+beta*A
const Complex Alpha(1.0, 0.0); ///< constant for cblas, A=alpha*B*C+beta*A
const Complex Beta(0.0, 0.0); ///< constant for cblas, A=alpha*B*C+beta*A
const double ChangeLim = 1e-5; ///< if the population change on each PES is smaller than ChangeLim, then it is stable and could stop simulation; not using now

/// real array constructed by unique_ptr, no need to free memory manually
typedef unique_ptr<double[]> RealVector;
/// complex array constructed by unique_ptr, no need to free memory manually
typedef unique_ptr<Complex[]> ComplexVector; 


// utility functions

/// @brief cut off
/// @param val the input value to do the cutoff
/// @return the value after cutoff
double cutoff(const double val);

/// sign function; return -1 for negative, 1 for positive, and 0 for 0
/// @param val a value of any type that have '<' and '>' and could construct 0.0
/// @return the sign of the value
template <typename valtype>
inline int sgn(const valtype& val)
{
    return (val > valtype(0.0)) - (val < valtype(0.0));
}

/// @brief returns (-1)^n
/// @param n an integer
/// @return -1 if n is odd, or 1 if n is even
int pow_minus_one(const int n);


// I/O functions

/// @brief read a double: mass, x0, etc
/// @param is an istream object (could be ifstream/isstream)
/// @return the real value read from the stream
double read_double(istream& is);

/// @brief to print current time
/// @param os an ostream object (could be ifstream/isstream)
/// @return the ostream object of the parameter after output the time
ostream& show_time(ostream& os);


// evolution related functions

/// @brief initialize the PWTDM (Partial Wigner-Transformed Density Matrix), and normalize it
/// @param NGrids the number of grids in density matrix, overall NGrids^2 sub-density matrices
/// @param GridPosition the position coordinate of each grid, i.e., x_i
/// @param GridMomentum the momentum coordinate of each grid, i.e., p_j
/// @param dx the grid spacing of position coordinate, for normalization
/// @param dp the grid spacing of momentum coordinate, for normalization
/// @param x0 the initial average position
/// @param p0 the initial average momentum
/// @param SigmaX the initial standard deviation of position
/// @param SigmaP the initial standard deviation of momentum
/// @param rho_adia the density matrix that will be initialized in adiabatic representation
void density_matrix_initialization
(
    const int NGrids,
    const double* const GridPosition,
    const double* const GridMomentum,
    const double dx,
    const double dp,
    const double x0,
    const double p0,
    const double SigmaX,
    const double SigmaP,
    ComplexMatrixMatrix& rho_adia
);

/// @brief calculate the population on each PES
/// @param NGrids the number of grids in density matrix, overall NGrids^2 sub-density matrices
/// @param dx the grid spacing of position coordinate, for normalization
/// @param dp the grid spacing of momentum coordinate, for normalization
/// @param rho_adia the density matrix that should be in adiabatic representation for calculating population
/// @param Population the array to store the calculated population on each PES
void calculate_population
(
    const int NGrids,
    const double dx,
    const double dp,
    const ComplexMatrixMatrix& rho_adia,
    double* const Population
);

/// @brief calculate average energy, x, and p
/// @param rho the density matrix for calculating the averages
/// @param NGrids the number of grids in density matrix, overall NGrids^2 sub-density matrices
/// @param Potential the 2D array containing V of all basis on each grid
/// @param GridPosition the position coordinate of each grid, i.e., x_i
/// @param GridMomentum the momentum coordinate of each grid, i.e., p_j
/// @param mass the mass of the bath
/// @param dx the grid spacing of position coordinate, for normalization
/// @param dp the grid spacing of momentum coordinate, for normalization
/// @param BasisOfRho the basis of density matrix for choosing the potential function
/// @return binded average, in the order of <E>, <x>, then <p>
tuple<double, double, double> calculate_average
(
    const ComplexMatrixMatrix& rho,
    const int NGrids,
    const RealMatrix* const* const Potential,
    const double* const GridPosition,
    const double* const GridMomentum,
    const double mass,
    const double dx,
    const double dp,
    const Representation BasisOfRho
);

/// @brief evolve the quantum liouville
/// @param rho the density matrix to evolve
/// @param NGrids the number of grids in density matrix, overall NGrids^2 sub-density matrices
/// @param Potential the 2D array containing V of all basis on each grid
/// @param Coupling the 2D array containing D of all basis on each grid
/// @param GridPosition the position coordinate of each grid, i.e., x_i
/// @param GridMomentum the momentum coordinate of each grid, i.e., p_j
/// @param mass the mass of the bath
/// @param dt the time step for this evolve
/// @param BasisOfRho the basis of density matrix for deciding the way of evolution and for choosing the potential function
/// @see classical_position_liouville_propagator(), classical_momentum_liouville_propagator()
void quantum_liouville_propagation
(
    ComplexMatrixMatrix& rho,
    const int NGrids,
    const RealMatrix* const* const Potential,
    const RealMatrix* const* const Coupling,
    const double* const GridPosition,
    const double* const GridMomentum,
    const double mass,
    const double dt,
    const Representation BasisOfRho
);

/// @brief evolve the classical position liouville
/// @param rho the density matrix to evolve
/// @param NGrids the number of grids in density matrix, overall NGrids^2 sub-density matrices
/// @param GridMomentum the momentum coordinate of each grid, i.e., p_j
/// @param mass the mass of the bath
/// @param TotalPositionLength equals to xmax-xmin, or (N-1)*dx
/// @param dx the grid spacing of position
/// @param dt the time step for this evolve
/// @see quantum_liouville_propagator(), classical_momentum_liouville_propagator()
void classical_position_liouville_propagator
(
    ComplexMatrixMatrix& rho,
    const int NGrids,
    const double* const GridMomentum,
    const double mass,
    const double TotalPositionLength,
    const double dx,
    const double dt
);

/// @brief evolve the quantum liouville
/// @param rho the density matrix to evolve
/// @param NGrids the number of grids in density matrix, overall NGrids^2 sub-density matrices
/// @param Force the 2D array containing F of all basis on each grid
/// @param GridPosition the position coordinate of each grid, i.e., x_i
/// @param TotalMomentumLength equals to pmax-pmin, or (N-1)*dp
/// @param dp the grid spacing of momentum
/// @param dt the time step for this evolve
/// @param BasisOfRho the basis of density matrix for deciding the way of evolution and for choosing the potential function
/// @see quantum_liouville_propagator(), classical_position_liouville_propagator()
void classical_momentum_liouville_propagator
(
    ComplexMatrixMatrix& rho,
    const int NGrids,
    const RealMatrix* const* const Force,
    const double* const GridPosition,
    const double TotalMomentumLength,
    const double dp,
    const double dt,
    const Representation BasisOfRho
);

#endif // !GENERAL_H
