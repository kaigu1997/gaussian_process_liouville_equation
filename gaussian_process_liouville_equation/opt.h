/// @file opt.h
/// @brief Interface to functions of parameter optimization

#ifndef OPT_H
#define OPT_H

#include "stdafx.h"

#include "input.h"
#include "kernel.h"

/// The training set for parameter optimization of one element of density matrix
/// passed as parameter to the optimization function.
/// First is feature, second is label
using ElementTrainingSet = std::tuple<Eigen::MatrixXd, Eigen::VectorXd>;

/// @brief To construct the training set for single element
/// @param[in] density The vector containing all known density matrix
/// @param[in] ElementIndex The index of the element in density matrix
/// @return The training set of the corresponding element in density matrix
ElementTrainingSet construct_element_training_set(const EigenVector<PhaseSpacePoint>& density, const int ElementIndex);

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
	/// @param[in] KernelTypes The vector containing all the kernel type used in optimization
	/// @param[in] LocalAlgorithm The optimization algorithm for local optimization
	/// @param[in] GlobalAlgorithm The optimization algorithm for global optimization
	Optimization(
		const Parameters& Params,
		const nlopt::algorithm LocalAlgorithm = nlopt::algorithm::LD_SLSQP,
		const nlopt::algorithm ConstraintAlgorithm = nlopt::algorithm::LD_SLSQP,
		const nlopt::algorithm GlobalAlgorithm = nlopt::algorithm::G_MLSL_LDS,
		const nlopt::algorithm GlobalSubAlgorithm = nlopt::algorithm::LD_MMA);

	/// @brief To optimize parameters based on the given density
	/// @param[in] density The vector containing all known density matrix
	/// @param[in] mass Mass of classical degree of freedom
	/// @param[in] TotalEnergy The total energy of the system used for energy conservation
	/// @param[in] Purity The purity of the partial Wigner transformed density matrix
	/// @param[in] IsSmall The matrix that saves whether each element is small or not
	/// @return The total error (MSE, log likelihood, etc) of all elements of density matrix
	Result optimize(
		const EigenVector<PhaseSpacePoint>& density,
		const ClassicalVector<double>& mass,
		const double TotalEnergy,
		const double Purity,
		const QuantumMatrix<bool>& IsSmall);

	/// @brief To get the parameter of the corresponding element
	/// @param[in] ElementIndex The index of the density matrix element
	/// @return The parameters of the corresponding element of density matrix
	const ParameterVector& get_parameter(const int ElementIndex) const
	{
		return ParameterVectors[ElementIndex];
	}

private:
	// local variables
	const int NumPoints;									   ///< The number of points selected for parameter optimization
	const ParameterVector InitialParameter;					   ///< The initial parameter for each of the element and for reopt
	std::vector<nlopt::opt> NLOptLocalMinimizers;			   ///< The vector containing all local NLOPT optimizers, one non-grad for each element
	std::vector<nlopt::opt> NLOptGlobalMinimizers;			   ///< The vector containing all global NLOPT optimizers
	std::array<ParameterVector, NumElements> ParameterVectors; ///< The parameters for all elements of density matrix
};

#endif // !OPT_H
