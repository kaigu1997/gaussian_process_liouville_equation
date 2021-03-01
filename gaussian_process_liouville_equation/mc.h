/// @file mc.h
/// @brief Interface of functions related to Monte Carlo (MC) procedure, including MC, weight function, and density matrix generation

#ifndef MC_H
#define MC_H

#include "io.h"

/// A number of optimizers to study the hyperparameters of gaussian process
using Optimizer = std::vector<nlopt::opt>;
/// Vector containing the type of all kernels
using KernelTypeList = std::vector<shogun::EKernelType>;
/// A vector containing weights and pointers to kernels, which should be the same length as KernelTypeList
using KernelList = std::vector<std::pair<double, std::shared_ptr<shogun::Kernel>>>;
/// The vector containing hyperparameters (or similarly: bounds, gradient, etc)
using ParameterVector = std::vector<double>;
/// The averages to calculate: <1>, <x>, <p>, <V>, <T>
using Averages = std::tuple<double, ClassicalDoubleVector, ClassicalDoubleVector, double, double>;

/// @brief To check if all elements are very small or not
/// @param[in] density The density matrices at known points
/// @return A matrix of boolean type and same size as density matrix, containing each element of density matrix is small or not
QuantumBoolMatrix is_very_small(const EvolvingDensity& density);

/// Store hyperparameters, kernels and optimization algorithms to use.
/// Optimize hyperparameters, and then predict density matrix and given point.
class Optimization final
{
private:
	// static constant variables
	static const double DiagMagMin, DiagMagMax, GaussMagMin, GaussMagMax, AbsoluteTolerance, InitialStepSize;
	// local variables
	const KernelTypeList TypesOfKernels;					  ///< The types of the kernels to use
	const int NumHyperparameters;							  ///< The number of hyperparameters to use, derived from the kernel types
	const int NumPoint;										  ///< The number of points selected for hyperparameter optimization
	Optimizer NLOptMinimizers;								  ///< The vector containing all NLOPT optimizers, one gradient and one non-grad for each element
	std::array<ParameterVector, NumElement> Hyperparameters;  ///< The hyperparameters for all elements of density matrix
	mutable std::array<KernelList, NumElement> Kernels;		  ///< The kernels with hyperparameters set for all elements of density matrix
	mutable std::array<Eigen::VectorXd, NumElement> KInvLbls; ///< The inverse of kernel matrix of training features times the training labels

public:
	/// @brief Constructor. Initial hyperparameters, kernels and optimization algorithms needed
	/// @param[in] params Parameters object containing position, momentum, etc
	/// @param[in] KernelTypes The vector containing all the kernel type used in optimization
	/// @param[in] NonGradAlgo The non-gradient optimization algorithm
	/// @param[in] GradAlgo The Gradient optimization algorithm
	Optimization(
		const Parameters& params,
		const KernelTypeList& KernelTypes = { shogun::EKernelType::K_DIAG, shogun::EKernelType::K_GAUSSIANARD },
		const nlopt::algorithm NonGradAlgo = nlopt::algorithm::LN_NELDERMEAD,
		const nlopt::algorithm GradAlgo = nlopt::algorithm::LD_TNEWTON_PRECOND_RESTART);
	/// @brief To optimize hyperparameters based on the given density
	/// @param[in] density The vector containing all known density matrix
	/// @param[in] IsSmall The matrix that saves whether each element is small or not
	/// @return The total negative log marginal likelihood of all elements of density matrix
	double optimize(const EvolvingDensity& density, const QuantumBoolMatrix& IsSmall);
	/// @brief To predict one of the element of the density matrix at the given phase space point
	/// @param[in] IsSmall The matrix that saves whether each element is small or not
	/// @param[in] x Position of classical degree of freedom
	/// @param[in] p Momentum of classical degree of freedom
	/// @return The element of the density matrix at the given point
	QuantumComplexMatrix predict(
		const QuantumBoolMatrix& IsSmall,
		const ClassicalDoubleVector& x,
		const ClassicalDoubleVector& p) const;
	/// @brief To print the grids of a certain element of density matrix
	/// @param[in] PhaseGrids All the grids required to calculate in phase space
	/// @param[in] ElementIndex The index of the density matrix element
	/// @return A 1-by-N matrix, N is the number of required grids
	Eigen::MatrixXd print_element(const Eigen::MatrixXd& PhaseGrids, const int ElementIndex) const;
	/// @brief To calculate the averages (population, <x>, <p>, <V>, <T>) from hyperparameters
	/// @param[in] IsSmall The matrix that saves whether each element is small or not
	/// @param[in] PESIndex The index of the potential energy surface, corresponding to (PESIndex, PESIndex) in density matrix
	/// @return Tuple of the 5 averages
	Averages calculate_average(const ClassicalDoubleVector& mass, const QuantumBoolMatrix& IsSmall, const int PESIndex) const;
	/// @brief To get the hyperparameter of the corresponding element
	/// @param[in] ElementIndex The index of the density matrix element
	/// @return The hyperparameters of the corresponding element of density matrix
	ParameterVector get_hyperparameter(const int ElementIndex) const
	{
		return Hyperparameters[ElementIndex];
	}
};

/// @brief Generate initial adiabatic PWTDM at the given place
/// @param[in] params Parameters objects containing all the required information (r0, sigma0, mass)
/// @param[in] x Position of classical degree of freedom
/// @param[in] p Momentum of classical degree of freedom
/// @return The initial density matrix at the give phase point under adiabatic basis
/// @see learnt_distribution(), monte_carlo_selection()
QuantumComplexMatrix initial_distribution(const Parameters& params, const ClassicalDoubleVector& x, const ClassicalDoubleVector& p);

/// @brief Using Monte Carlo to select points
/// @param[in] params Parameters objects containing all the required information (min, max, dmax)
/// @param[in] distribution The function object to generate the distribution
/// @param[in] IsSmall The matrix that saves whether each element is small or not
/// @param[out] density The selected density matrices
void monte_carlo_selection(
	const Parameters& params,
	const DistributionFunction& distribution,
	const QuantumBoolMatrix& IsSmall,
	EvolvingDensity& density);

#endif // !MC_H
