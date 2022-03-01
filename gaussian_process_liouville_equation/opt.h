/// @file opt.h
/// @brief Interface to functions of parameter optimization

#ifndef OPT_H
#define OPT_H

#include "stdafx.h"

#include "input.h"
#include "predict.h"

/// The training set for parameter optimization of one element of density matrix
/// passed as parameter to the optimization function.
/// First is feature, second is label
using ElementTrainingSet = std::tuple<Eigen::MatrixXd, Eigen::VectorXd>;

/// @brief To update the kernels
/// @param[out] kernels Array of constructed kernels
/// @param[in] ParameterVectors The parameters for all elements of density matrix
/// @param[in] density The selected points in phase space for each element of density matrices
/// @param[in] extra_points The extra selected points to reduce overfitting
void construct_predictors(
	OptionalKernels& kernels,
	const QuantumArray<ParameterVector>& ParameterVectors,
	const AllPoints& density,
	const AllPoints& extra_points);

/// @brief To store parameters, kernels and optimization algorithms to use.
/// And, to optimize parameters, and then predict density matrix and given point.
class Optimization final
{
public:
	/// The return type of optimization routine.
	/// First is the total error (MSE, log likelihood, etc) of all elements of density matrix.
	/// Second is a vector of std::size_t, containing the optimization steps of each element
	using Result = std::tuple<double, std::vector<std::size_t>>;

	/// @brief Constructor. Initial parameters, kernels and optimization algorithms needed
	/// @param[in] InitParams InitialParameters object containing position, momentum, etc
	/// @param[in] InitialTotalEnergy The total energy of the system used for energy conservation
	/// @param[in] InitialPurity The purity of the partial Wigner transformed density matrix, used for purity conservation
	/// @param[in] LocalAlgorithm The optimization algorithm for local optimization
	/// @param[in] ConstraintAlgorithm The optimization algorithm for local optimization with constraints (population, energy, purity, etc)
	/// @param[in] GlobalAlgorithm The optimization algorithm for global optimization
	/// @param[in] GlobalSubAlgorithm The subsidiary local optimization algorithm used in global optimization if needed
	Optimization(
		const InitialParameters& InitParams,
		const double InitialTotalEnergy,
		const double InitialPurity,
		const nlopt::algorithm LocalAlgorithm = nlopt::algorithm::LD_SLSQP,
		const nlopt::algorithm ConstraintAlgorithm = nlopt::algorithm::LD_SLSQP,
		const nlopt::algorithm GlobalAlgorithm = nlopt::algorithm::GN_DIRECT_L,
		const nlopt::algorithm GlobalSubAlgorithm = nlopt::algorithm::LD_MMA);

	/// @brief To optimize parameters based on the given density
	/// @param[in] density The selected points in phase space for each element of density matrices
	/// @param[in] extra_points The extra selected points to reduce overfitting
	/// @return The total error (MSE, log likelihood, etc) of all elements, and the optimization steps
	Result optimize(
		const AllPoints& density,
		const AllPoints& extra_points);

	/// @brief To get the parameter of the elements
	/// @return The parameters of all the elements element of density matrix
	const QuantumArray<ParameterVector>& get_parameters(void) const
	{
		return ParameterVectors;
	}

	/// @brief To get the lower bounds of the elements
	/// @return The lower bounds of all the elements of density matrix
	QuantumArray<ParameterVector> get_lower_bounds(void) const;

	/// @brief To get the lower bounds of the elements
	/// @return The lower bounds of all the elements of density matrix
	QuantumArray<ParameterVector> get_upper_bounds(void) const;


private:
	// local variables
	const double TotalEnergy;								  ///< The initial energy of the density matrix, and should be kept to the end
	const double Purity;									  ///< The initial purity of the density matrix, and should be kept to the end
	const ClassicalVector<double> mass;						  ///< The mass of classical degree of freedom
	const ParameterVector InitialParameter;					  ///< The initial parameter for each of the element and for reopt
	const ParameterVector InitialParameterForGlobalOptimizer; ///< The initial parameter, but modified for global optimizers
	std::vector<nlopt::opt> NLOptLocalMinimizers;			  ///< The vector containing all local NLOPT optimizers, one non-grad for each element
	std::vector<nlopt::opt> NLOptGlobalMinimizers;			  ///< The vector containing all global NLOPT optimizers
	QuantumArray<ParameterVector> ParameterVectors;			  ///< The parameters for all elements of density matrix
};

#endif // !OPT_H
