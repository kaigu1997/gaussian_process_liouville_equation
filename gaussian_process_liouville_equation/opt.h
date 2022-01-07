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
void construct_predictors(
	OptionalKernels& kernels,
	const QuantumArray<ParameterVector>& ParameterVectors,
	const AllPoints& density);

/// @brief To store parameters, kernels and optimization algorithms to use.
/// And, to optimize parameters, and then predict density matrix and given point.
class Optimization final
{
public:
	/// The return type of optimization routine.
	/// First is the total error (MSE, log likelihood, etc) of all elements of density matrix.
	/// Second is a vector of int, containing the optimization steps of each element
	using Result = std::tuple<double, std::vector<int>>;

	/// @brief Constructor. Initial parameters, kernels and optimization algorithms needed
	/// @param[in] Params Parameters object containing position, momentum, etc
	/// @param[in] InitialTotalEnergy The total energy of the system used for energy conservation
	/// @param[in] InitialPurity The purity of the partial Wigner transformed density matrix, used for purity conservation
	/// @param[in] LocalAlgorithm The optimization algorithm for local optimization
	/// @param[in] ConstraintAlgorithm The optimization algorithm for local optimization with constraints (population, energy, purity, etc)
	/// @param[in] GlobalAlgorithm The optimization algorithm for global optimization
	/// @param[in] GlobalSubAlgorithm The subsidiary local optimization algorithm used in global optimization if needed
	Optimization(
		const Parameters& Params,
		const double InitialTotalEnergy,
		const double InitialPurity,
		const nlopt::algorithm LocalAlgorithm = nlopt::algorithm::LD_SLSQP,
		const nlopt::algorithm ConstraintAlgorithm = nlopt::algorithm::LD_SLSQP,
		const nlopt::algorithm GlobalAlgorithm = nlopt::algorithm::GN_ISRES,
		const nlopt::algorithm GlobalSubAlgorithm = nlopt::algorithm::LD_MMA);

	/// @brief To optimize parameters based on the given density
	/// @param[in] density The selected points in phase space for each element of density matrices
	/// @param[in] mc_points The selected points for calculating mc integration
	/// @return The total error (MSE, log likelihood, etc) of all elements, and the optimization steps
	Result optimize(
		const AllPoints& density,
		const AllPoints& mc_points);

	/// @brief To get the parameter of the corresponding element
	/// @param[in] iPES The index of the row of the density matrix element
	/// @param[in] jPES The index of the column of the density matrix element
	/// @return The parameters of the corresponding element of density matrix
	const QuantumArray<ParameterVector>& get_parameters(void) const
	{
		return ParameterVectors;
	}

private:
	// local variables
	const double TotalEnergy;						///< The initial energy of the density matrix, and should be kept to the end
	const double Purity;							///< The initial purity of the density matrix, and should be kept to the end
	const ClassicalVector<double> mass;				///< The mass of classical degree of freedom
	const ParameterVector InitialParameter;			///< The initial parameter for each of the element and for reopt
	std::vector<nlopt::opt> NLOptLocalMinimizers;	///< The vector containing all local NLOPT optimizers, one non-grad for each element
	std::vector<nlopt::opt> NLOptGlobalMinimizers;	///< The vector containing all global NLOPT optimizers
	QuantumArray<ParameterVector> ParameterVectors; ///< The parameters for all elements of density matrix
};

#endif // !OPT_H
