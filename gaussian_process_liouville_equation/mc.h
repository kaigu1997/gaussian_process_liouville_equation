/// @file mc.h
/// @brief Interface of functions related to Monte Carlo (MC) procedure, including MC, weight function, and density matrix generation

#ifndef MC_H
#define MC_H

#include "io.h"

/// The type for saving all autocorrelations
using AutoCorrelations = std::array<Eigen::VectorXd, NumElements>;

/// @brief The weight function for sorting / MC selection
/// @param x The input variable
/// @return The weight of the input, which is the square of it
inline double weight_function(const double x)
{
	return x * x;
}

/// @brief To get the element of density matrix
/// @param[in] DensityMatrix The density matrix
/// @param[in] ElementIndex The index of the element in density matrix
/// @return The element of density matrix
/// @details If the index corresponds to upper triangular elements, gives real part.
///
/// If the index corresponds to lower triangular elements, gives imaginary part.
///
/// If the index corresponds to diagonal elements, gives the original value.
inline double get_density_matrix_element(const QuantumComplexMatrix& DensityMatrix, const int ElementIndex)
{
	const int iPES = ElementIndex / NumPES, jPES = ElementIndex % NumPES;
	if (iPES <= jPES)
	{
		return DensityMatrix(iPES, jPES).real();
	}
	else
	{
		return DensityMatrix(iPES, jPES).imag();
	}
}

/// @brief To check if all elements are very small or not
/// @param[in] density The density matrices at known points
/// @return A matrix of boolean type and same size as density matrix, containing each element of density matrix is small or not
QuantumBoolMatrix is_very_small(const EvolvingDensity& density);

/// @brief To generate initial adiabatic PWTDM at the given place
/// @param[in] Params Parameters objects containing all the required information (r0, sigma0, mass)
/// @param[in] x Position of classical degree of freedom
/// @param[in] p Momentum of classical degree of freedom
/// @return The initial density matrix at the give phase point under adiabatic basis
/// @see learnt_distribution(), monte_carlo_selection()
QuantumComplexMatrix initial_distribution(const Parameters& Params, const ClassicalDoubleVector& x, const ClassicalDoubleVector& p);

/// To store the parameters used in MC process
class MCParameters
{
	int NumPoints, NOMC;
	double displacement;

public:
	static const double AboveMinFactor;
	/// @brief Constructor. Initialization of parameters.
	/// @param[in] Params Parameters object containing position, momentum, etc
	/// @param[in] NOMC_ The initial number of monte carlo steps
	/// @param[in] NGrids_ The initial number of grids to set the length of displacement
	MCParameters(const Parameters& Params, const int NOMC_ = 200);

	/// @brief To set the number of selected points
	/// @param[in] NumPoints_ The number of points to be selected
	void set_num_points(const int NumPoints_)
	{
		NumPoints = NumPoints_;
	}

	/// @brief To set the number of monte carlo steps
	/// @param[in] NOMC_ The number of monte carlo steps
	void set_num_MC_steps(const int NOMC_)
	{
		NOMC = NOMC_;
	}

	/// @brief To set the maximum possible displacement
	/// @param[in] displacement_ The maximum possible displacement of one monte carlo step on one direction
	void set_displacement(const double displacement_)
	{
		displacement = displacement_;
	}

	/// @brief To get the number of points selected
	/// @return The number of points selected
	int get_num_points(void) const
	{
		return NumPoints;
	}

	/// @brief To get the number of monte carlo steps
	/// @return The number of monte carlo steps
	int get_num_MC_steps(void) const
	{
		return NOMC;
	}

	/// @brief To get the number of monte carlo steps
	/// @return The number of monte carlo steps
	double get_max_dispalcement(void) const
	{
		return displacement;
	}
};

/// @brief Using Monte Carlo to select points
/// @param[in] Params The input parameters (mass, dt, etc)
/// @param[in] MCParams Monte carlo parameters (number of steps, maximum displacement, etc)
/// @param[in] distribution The function object to generate the distribution
/// @param[in] IsSmall The matrix that saves whether each element is small or not
/// @param[out] density The selected density matrices
/// @param[in] IsToBeEvolved Whether the points are at evolved places or original places
void monte_carlo_selection(
	const Parameters& Params,
	const MCParameters& MCParams,
	const DistributionFunction& distribution,
	const QuantumBoolMatrix& IsSmall,
	EvolvingDensity& density,
	const bool IsToBeEvolved = true);

/// @brief To optimize the number of the monte carlo steps by autocorrelation
/// @param[inout] MCParams Monte carlo parameters (number of steps, maximum displacement, etc)
/// @param[in] distribution The function object to generate the distribution
/// @param[in] IsSmall The matrix that saves whether each element is small or not
/// @param[in] density The selected density matrices
/// @return The autocorrelation of all tested number of steps if the element is not small
AutoCorrelations autocorrelation_optimize_steps(
	MCParameters& MCParams,
	const DistributionFunction& distribution,
	const QuantumBoolMatrix& IsSmall,
	const EvolvingDensity& density);

/// @brief To optimize the number of the monte carlo steps by autocorrelation
/// @param[inout] MCParams Monte carlo parameters (number of steps, maximum displacement, etc)
/// @param[in] distribution The function object to generate the distribution
/// @param[in] IsSmall The matrix that saves whether each element is small or not
/// @param[in] density The selected density matrices
/// @return The autocorrelation of all tested number of steps if the element is not small
AutoCorrelations autocorrelation_optimize_displacement(
	MCParameters& MCParams,
	const DistributionFunction& distribution,
	const QuantumBoolMatrix& IsSmall,
	const EvolvingDensity& density);

/// @brief To select points for new elements of density matrix
/// @param density The selected density matrices, new ones will also be put here
/// @param IsNew The matrix that saves whether the element is newly populated or not
/// @param NumPoints The number of points for each element
void new_element_point_selection(EvolvingDensity& density, const QuantumBoolMatrix& IsNew, const int NumPoints);

#endif // !MC_H
