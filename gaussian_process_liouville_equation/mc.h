/// @file mc.h
/// @brief Interface of functions related to Metropolist Monte Carlo (MC) procedure,
/// including MC, weight function, and densitymatrix generation

#ifndef MC_H
#define MC_H

#include "stdafx.h"

#include "storage.h"

/// @brief The type for saving all autocorrelations
using AutoCorrelations = QuantumStorage<Eigen::VectorXd>;

/// @brief To generate initial adiabatic partial Wigner-transformed density matrics at the given place
/// @param[in] r0 Initial center of phase space
/// @param[in] SigmaR0 Initial standard deviation of phase space
/// @param[in] r Given phase space coordinates
/// @param[in] RowIndex RowIndex Index of row of the element in density matrix. Must be a valid index (< @p NumPES )
/// @param[in] ColIndex RowIndex Index of row of the element in density matrix. Must be a valid index (< @p NumPES )
/// @param[in] InitialPopulation Initial population of wavefunction on each of the PES, whose squared sum will be normalized
/// @param[in] InitialPhaseFactor Initial phase factor of wavefunction for each of the PES
/// @return The initial density matrix at the give phase point under adiabatic basis
std::complex<double> initial_distribution(
	const ClassicalPhaseVector& r0,
	const ClassicalPhaseVector& SigmaR0,
	const ClassicalPhaseVector& r,
	const std::size_t RowIndex,
	const std::size_t ColIndex,
	const std::array<double, NumPES>& InitialPopulation = {1.0},
	const std::array<double, NumPES>& InitialPhaseFactor = {0}
);

/// @brief To generate the extra points for the density fitting
/// @param[in] density The selected points in phase space for each element of density matrices
/// @param[in] NumPoints The number of points for each elements
/// @param[in] mass Mass of classical degree of freedom
/// @param[in] dt Time interval
/// @param[in] NumTicks Number of @p dt since T=0; i.e., now is @p NumTicks * @p dt
/// @param[in] distribution The current distribution
/// @return The newly selected points with its corresponding distribution
AllPoints generate_extra_points(
	const AllPoints& density,
	const std::size_t NumPoints,
	const ClassicalVector<double>& mass,
	const double dt,
	const std::size_t NumTicks,
	const DistributionFunction& distribution
);

/// @brief To store the parameters used in MC process
class MCParameters final
{
	/// @brief The number of MCMC steps
	std::size_t NOMC;
	/// @brief The maximum displacement of one MCMC step
	double displacement;

public:
	/// @brief choose the minimal step or max displacement whose autocor < factor*min{autocor}
	static constexpr double AboveMinFactor = 1.1;

	/// @brief Constructor. Initialization of parameters.
	/// @param[in] InitialDisplacement The initial guess of displacement
	/// @param[in] InitialSteps The initial guess of the number of monte carlo steps
	MCParameters(const double InitialDisplacement, const std::size_t InitialSteps = 200):
		NOMC(InitialSteps), displacement(InitialDisplacement)
	{
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
/// @param[in] MCParams Monte carlo parameters (number of steps, maximum displacement, etc)
/// @param[in] mass Mass of classical degree of freedom
/// @param[in] dt Time interval
/// @param[in] NumTicks Number of @p dt since T=0; i.e., now is @p NumTicks * @p dt
/// @param[in] distribution The function object to generate the distribution
/// @param[inout] density The selected points in phase space for each element of density matrices
void monte_carlo_selection(
	const MCParameters& MCParams,
	const ClassicalVector<double>& mass,
	const double dt,
	const std::size_t NumTicks,
	const DistributionFunction& distribution,
	AllPoints& density
);

/// @brief To optimize the number of the monte carlo steps by autocorrelation
/// @param[inout] MCParams Monte carlo parameters (number of steps, maximum displacement, etc)
/// @param[in] distribution The function object to generate the distribution
/// @param[in] density The selected points in phase space for each element of density matrices
/// @return The autocorrelation of all tested number of steps
AutoCorrelations autocorrelation_optimize_steps(
	MCParameters& MCParams,
	const DistributionFunction& distribution,
	const AllPoints& density
);

/// @brief To optimize the number of the monte carlo steps by autocorrelation
/// @param[inout] MCParams Monte carlo parameters (number of steps, maximum displacement, etc)
/// @param[in] distribution The function object to generate the distribution
/// @param[in] density The selected points in phase space for each element of density matrices
/// @return The autocorrelation of all tested displacements
AutoCorrelations autocorrelation_optimize_displacement(
	MCParameters& MCParams,
	const DistributionFunction& distribution,
	const AllPoints& density
);

/// @brief To select points for new elements of density matrix
/// @param[inout] density The selected points in phase space for each element of density matrices
/// @param[in] extra_points The extra selected points
/// @param[in] IsNew The matrix that saves whether the element is newly populated or not
/// @param[in] mass Mass of classical degree of freedom
/// @param[in] dt Time interval
/// @param[in] NumTicks Number of @p dt since T=0; i.e., now is @p NumTicks * @p dt
/// @param[in] distribution The function object to generate the distribution
void new_element_point_selection(
	AllPoints& density,
	const AllPoints& extra_points,
	const QuantumMatrix<bool>& IsNew,
	const ClassicalVector<double>& mass,
	const double dt,
	const std::size_t NumTicks,
	const DistributionFunction& distribution
);

#endif // !MC_H
