/// @file opt.h
/// @brief Interface to functions of parameter optimization

#ifndef OPT_H
#define OPT_H

#include "stdafx.h"

#include "input.h"
#include "predict.h"

/// @brief The relative tolerance for averages
static constexpr double AverageTolerance = 0.05;

/// @brief To store parameters, kernels and optimization algorithms to use. @b
/// Also, to optimize parameters, and then predict density matrix and given point.
class Optimization final
{
public:
	/// @brief The type of optimization that decides the parameters
	enum OptimizationType
	{
		/// @brief Default, in case a type is needed but not the following
		Default,
		/// @brief Local optimization with previous parameters
		LocalPrevious,
		/// @brief Local optimization with initial parameters
		LocalInitial,
		/// @brief Global optimization
		Global
	};

	/// @brief The return type of optimization routine. @n
	/// First is the total error (MSE, log likelihood, etc) of all elements of density matrix. @n
	/// Second is a vector of std::size_t, containing the optimization steps of each element
	using Result = std::tuple<double, std::vector<std::size_t>, OptimizationType>;

	/// @brief Constructor. Initial parameters, kernels and optimization algorithms needed
	/// @param[in] InitParams InitialParameters object containing position, momentum, etc
	/// @param[in] InitialTotalEnergy The total energy of the system used for energy conservation
	/// @param[in] InitialPurity The purity of the partial Wigner transformed density matrix, used for purity conservation
	/// @param[in] LocalDiagonalAlgorithm The optimization algorithm for local optimization of diagonal elements
	/// @param[in] LocalOffDiagonalAlgorithm The optimization algorithm for local optimization of off-diagonal elements
	/// @param[in] ConstraintAlgorithm The optimization algorithm for local optimization with constraints (population, energy, purity, etc)
	/// @param[in] GlobalAlgorithm The optimization algorithm for global optimization
	/// @param[in] GlobalSubAlgorithm The subsidiary local optimization algorithm used in global optimization if needed
	Optimization(
		const InitialParameters& InitParams,
		const double InitialTotalEnergy,
		const double InitialPurity,
		const nlopt::algorithm LocalDiagonalAlgorithm = nlopt::algorithm::LN_NELDERMEAD,
		const nlopt::algorithm LocalOffDiagonalAlgorithm = nlopt::algorithm::LN_NELDERMEAD,
		const nlopt::algorithm ConstraintAlgorithm = nlopt::algorithm::LD_SLSQP,
		const nlopt::algorithm GlobalAlgorithm = nlopt::algorithm::GN_DIRECT_L,
		const nlopt::algorithm GlobalSubAlgorithm = nlopt::algorithm::LD_MMA
	);

	/// @brief To optimize parameters based on the given density
	/// @param[in] density The selected points in phase space for each element of density matrices
	/// @param[in] extra_points The extra selected points to reduce overfitting
	/// @return The total error (MSE, log likelihood, etc) of all elements, and the optimization steps
	Result optimize(
		const AllPoints& density,
		const AllPoints& extra_points
	);

	/// @brief To get the parameter of the elements
	/// @return The parameters of all the elements element of density matrix
	const QuantumStorage<ParameterVector>& get_parameters(void) const
	{
		return ParameterVectors;
	}

	/// @brief To get the lower bounds of the elements
	/// @return The lower bounds of all the elements of density matrix
	QuantumStorage<ParameterVector> get_lower_bounds(void) const;

	/// @brief To get the lower bounds of the elements
	/// @return The lower bounds of all the elements of density matrix
	QuantumStorage<ParameterVector> get_upper_bounds(void) const;


private:
	// local variables
	/// @brief The initial energy of the density matrix, and should be kept to the end
	const double TotalEnergy;
	/// @brief The initial purity of the density matrix, and should be kept to the end
	const double Purity;
	/// @brief The mass of classical degree of freedom
	const ClassicalVector<double> mass;
	/// @brief The initial parameter for each of the element and for reopt
	const ParameterVector InitialKernelParameter;
	/// @brief The initial parameter for complex kernels
	const ParameterVector InitialComplexKernelParameter;
	/// @brief Local NLOPT optimizers for each element
	QuantumStorage<nlopt::opt> LocalMinimizers;
	/// @brief Local NLOPT optimizer with diagonal constraints
	nlopt::opt DiagonalMinimizer;
	/// @brief Local NLOPT optimizer with constraints of all elements
	nlopt::opt FullMinimizer;
	/// @brief Global NLOPT optimizers for each element
	QuantumStorage<nlopt::opt> GlobalMinimizers;
	/// @brief The parameters for all elements of density matrix
	QuantumStorage<ParameterVector> ParameterVectors;
};

#endif // !OPT_H
