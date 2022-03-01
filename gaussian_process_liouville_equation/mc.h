/// @file mc.h
/// @brief Interface of functions related to Metropolist Monte Carlo (MC) procedure,
/// including MC, weight function, and densitymatrix generation

#ifndef MC_H
#define MC_H

#include "stdafx.h"

#include "input.h"

/// The type for saving all autocorrelations
using AutoCorrelations = QuantumArray<Eigen::VectorXd>;

/// @brief To check if all elements are very small or not
/// @param[in] density The selected points in phase space for each element of density matrices
/// @return A matrix of boolean type and same size as density matrix,
/// containing each element of density matrix is smallor not
QuantumMatrix<bool> is_very_small(const AllPoints& density);

/// @brief To generate initial adiabatic PWTDM at the given place
/// @param[in] r0 Initial center of phase space
/// @param[in] SigmaR0 Initial standard deviation of phase space
/// @param[in] InitialPopulation Initial population on each of the PES
/// @param[in] r Given phase space coordinates
/// @return The initial density matrix at the give phase point under adiabatic basis
QuantumMatrix<std::complex<double>> initial_distribution(
	const ClassicalPhaseVector& r0,
	const ClassicalPhaseVector& SigmaR0,
	const std::array<double, NumPES>& InitialPopulation,
	const ClassicalPhaseVector& r);

/// @brief To store the parameters used in MC process
class MCParameters final
{
	std::size_t NumPoints, NOMC;
	double displacement;

public:
	static const double AboveMinFactor;
	/// @brief Constructor. Initialization of parameters.
	/// @param[in] InitParams InitialParameters object containing position, momentum, etc
	/// @param[in] NOMC_ The initial number of monte carlo steps
	/// @param[in] NGrids_ The initial number of grids to set the length of displacement
	MCParameters(const InitialParameters& InitParams, const std::size_t NOMC_ = 200);

	/// @brief To set the number of selected points
	/// @param[in] NumPoints_ The number of points to be selected
	void set_num_points(const std::size_t NumPoints_)
	{
		NumPoints = NumPoints_;
	}

	/// @brief To set the number of monte carlo steps
	/// @param[in] NOMC_ The number of monte carlo steps
	void set_num_MC_steps(const std::size_t NOMC_)
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
	std::size_t get_num_points(void) const
	{
		return NumPoints;
	}

	/// @brief To get the number of monte carlo steps
	/// @return The number of monte carlo steps
	std::size_t get_num_MC_steps(void) const
	{
		return NOMC;
	}

	/// @brief To get the number of monte carlo steps
	/// @return The number of monte carlo steps
	double get_max_displacement(void) const
	{
		return displacement;
	}
};

/// @brief Using Metropolis Monte Carlo to select points
/// @param[in] InitParams The input parameters (mass, dt, etc)
/// @param[in] MCParams Monte carlo parameters (number of steps, maximum displacement, etc)
/// @param[in] distribution The function object to generate the distribution
/// @param[inout] density The selected points in phase space for each element of density matrices
/// @param[in] IsToBeEvolved Whether the points are at evolved places or original places
void monte_carlo_selection(
	const MCParameters& MCParams,
	const DistributionFunction& distribution,
	AllPoints& density);

/// @brief To optimize the number of the monte carlo steps by autocorrelation
/// @param[inout] MCParams Monte carlo parameters (number of steps, maximum displacement, etc)
/// @param[in] distribution The function object to generate the distribution
/// @param[in] density The selected points in phase space for each element of density matrices
/// @return The autocorrelation of all tested number of steps if the element is not small
AutoCorrelations autocorrelation_optimize_steps(
	MCParameters& MCParams,
	const DistributionFunction& distribution,
	const AllPoints& density);

/// @brief To optimize the number of the monte carlo steps by autocorrelation
/// @param[inout] MCParams Monte carlo parameters (number of steps, maximum displacement, etc)
/// @param[in] distribution The function object to generate the distribution
/// @param[in] density The selected points in phase space for each element of density matrices
/// @return The autocorrelation of all tested number of steps if the element is not small
AutoCorrelations autocorrelation_optimize_displacement(
	MCParameters& MCParams,
	const DistributionFunction& distribution,
	const AllPoints& density);

/// @brief To get the number of points for each elements
/// @param[in] Points The selected points in phase space for each element of density matrices
/// @return The number of points for non-zero element
std::size_t get_num_points(const AllPoints& Points);

/// @brief To select points for new elements of density matrix
/// @param[inout] density The selected points in phase space for each element of density matrices
/// @param[in] IsNew The matrix that saves whether the element is newly populated or not
void new_element_point_selection(AllPoints& density, const QuantumMatrix<bool>& IsNew);

#endif // !MC_H
