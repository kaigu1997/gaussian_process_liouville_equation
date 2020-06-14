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

#include <complex>
#include <iostream>
#include <memory>
#include <tuple>
#include <valarray>

#include <mkl.h>
#ifndef EIGEN_USE_MKL_ALL
#define EIGEN_USE_MKL_ALL
#endif // ! EIGEN_USE_MKL_ALL
#include <Eigen/Eigen>

using namespace std;
using namespace Eigen;

/// a shorter name for double-precision complex number
using Complex = complex<double>;
/// array of complex that could do numerical calculations
using ComplexArray = valarray<Complex>;

const double pi = acos(-1.0); ///< mathematical constant, pi
const double hbar = 1.0; ///< physical constant, reduced Planck constant
const double PlanckH = 2 * pi * hbar; ///< physical constant, Planck constant
// Alpha and Beta are for cblas complex matrix multiplication functions
// as they behave as A=alpha*B*C+beta*A
const Complex Alpha = 1.0; ///< constant for cblas, A=alpha*B*C+beta*A
const Complex Beta = 0.0; ///< constant for cblas, A=alpha*B*C+beta*A
// for RK4, things are different
const ComplexArray RK4Parameter = { 1.0, 2.0, 2.0, 1.0 }; ///< coefficient in RK4. ki=f(x+dt/para,y+dt/para*k[i-1]), y+=ki*dt/6*para
const Complex RK4kAlpha = 1.0 / 1.0i / hbar; ///< constant for RK4-calling cblas, k_{n+1}=H/ihbar*(psi+dt/1~2*kn)

const double PplLim = 1e-4; ///< if the overall population on all PES is smaller than PplLim, it is stable and could stop simulation. Used only with ABC
const double ChangeLim = 1e-5; ///< if the population change on each PES is smaller than ChangeLim, then it is stable and could stop simulation


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
/// @param is an istream object (could be ifstream/isstream) to input
/// @return the real value read from the stream
double read_double(istream& is);

/// @brief to print current time
/// @param os an ostream object (could be ifstream/isstream) to output
/// @return the ostream object of the parameter after output the time
ostream& show_time(ostream& os);


// evolution related functions

/// @enum the boundary condition
/// @sa Hamiltonian_construction(), Evolve, output_phase_space_distribution()
enum BoundaryCondition
{
	Absorbing, ///< absorbing potential, have imaginary part on diagonal elements
	Periodic, ///< periodic boundary, using finite difference
	Reflective ///< reflective boundary, also using finite difference
};

#ifndef TestBoundaryCondition
const BoundaryCondition TestBoundaryCondition = Periodic; ///< the boundary condition used
#endif // !TestBoundaryCondition

/// @brief initialize the gaussian wavepacket, and normalize it
/// @param XCoordinate the coordinate of each grid, i.e., x_i
/// @param dx the grid spacing
/// @param x0 the initial average position
/// @param p0 the initial average momentum
/// @param SigmaX the initial standard deviation of position. SigmaP is not needed due to the minimum uncertainty principle
/// @return the wavefunction to be initialized in adiabatic representation
VectorXcd wavefunction_initialization
(
	const VectorXd& XCoordinate,
	const double x0,
	const double p0,
	const double SigmaX
);

/// @brief construct the diabatic Hamiltonian
/// @param NGrids the number of grids in wavefunction
/// @param XCoordinate the coordinate of each grid, i.e., x_i
/// @param dx the grid spacing
/// @param mass the mass of the bath (the nucleus mass)
/// @param xmin the left boundary, only used with ABC
/// @param xmax the right boundary, only used with ABC
/// @param AbsorbingRegionLength the extended region for absoptiong, only used with ABC
/// @return the Hamiltonian, a complex matrix, hermitian if no ABC
MatrixXcd Hamiltonian_construction
(
	const VectorXd& XCoordinate,
	const double dx,
	const double mass,
	const double xmin = 0.0,
	const double xmax = 0.0,
	const double AbsorbingRegionLength = 0.0
);

/// the class to do the evolution, inspired by FFT_HANDLE
class Evolution
{
private:
	// general variables
	const int dim; ///< the number of grids / the size of Hamiltonian
	const MatrixXcd Hamiltonian; ///< the Hamiltonain Matrix
	VectorXcd Intermediate1; ///< No ABC, psi_diag(0); with ABC, (psi(t)+k[i-1]*dt/n)/ihbar
	VectorXcd Intermediate2; ///< No ABC, psi_diag(t); with ABC, H*(psi(t)+k[i-1]*dt/n)/ihbar
	VectorXcd PsiAtTimeT; /// wavefunction at time t; the evolved result
	// used when Absorbed == false
	SelfAdjointEigenSolver<MatrixXcd> solver;
	const MatrixXcd EigVec; ///< the eigenvectors of Hamiltonian, the basis transformation matrix
	const MatrixXd EigVal; ///< the eigenvalues of Hamiltonian, the diagonalized Hamiltonian
	// used when Absorbed == true
	const double dt; ///< time step
	const ComplexArray RK4kBeta; ///< used in RK4, = dt/RK4Parameter[i]/i/hbar
	const ComplexArray RK4PsiAlpha; ///< used in RK4, = dt/6.0*RK4Parameter[i]
public:
	/// @brief the constructor
	/// @param IsAbsorb whether to use absorbing potential or not
	/// @param DiaH the diabatic Hamiltonian, should be hermitian
	/// @param Psi0 the initial diabatic wavefunction
	/// @param TimeStep the time interval, or dt
	Evolution
	(
		const MatrixXcd& DiaH,
		const VectorXcd& Psi0,
		const double TimeStep
	);
	/// @brief destructor
	~Evolution(void);
	/// @brief evolve for a time step
	/// @param Psi the diabatic wavefunction at time t-dt
	/// @param Time the time when the output psi would be
	void evolve(VectorXcd& Psi, const double Time);
};

/// @brief output the population on each grid and PES, i.e. |psi_i(x_j)|^2
/// @param os an ostream object (could be ofstream/osstream) to output
/// @param Psi the (adiabatic) wavefunction
void output_grided_population(ostream& os, const VectorXcd Psi);

/// @brief calculate the phase space distribution, and output it
/// @param os an ostream object (could be ofstream/osstream) to output
/// @param PGridCoordinate the coordinate of each grid on p direction, i.e., p_i
/// @param dx the grid spacing
/// @param p0 the initial average momentum
/// @param Psi the (adiabatic) wavefunction
void output_phase_space_distribution
(
	ostream& os,
	const VectorXd& PGridCoordinate,
	const double dx,
	const double p0,
	const VectorXcd& Psi
);

/// @brief calculate average energy, x, and p
/// @param XCoordinate the coordinate of each grid, i.e., x_i
/// @param dx the grid spacing
/// @param mass the mass of the bath (the nucleus mass)
/// @param Psi the (adiabatic) wavefunction
/// @return average energy, x, and p
tuple<double, double, double> calculate_average
(
	const VectorXd& XCoordinate,
	const double dx,
	const double mass,
	const VectorXcd& Psi
);

/// @brief calculate the population on each PES
/// @param NGrids the number of grids in wavefunction
/// @param dx the grid spacing
/// @param Psi the wavefunction
/// @param Population the array to save the population on each PES
/// @param Psi the (adiabatic) wavefunction
void calculate_population
(
	const int NGrids,
	const double dx,
	const VectorXcd& Psi,
	double* const Population
);

#endif // !GENERAL_H
