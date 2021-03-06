/// @file opt.h
/// @brief Interface to functions of hyperparameter optimization

#ifndef OPT_H
#define OPT_H

#include "mc.h"

/// Vector containing the type of all kernels
using KernelTypeList = std::vector<shogun::EKernelType>;
/// A vector containing weights and pointers to kernels, which should be the same length as KernelTypeList
using KernelList = std::vector<std::pair<double, std::shared_ptr<shogun::Kernel>>>;
/// The vector containing hyperparameters (or similarly: bounds, gradient, etc)
using ParameterVector = std::vector<double>;
/// The averages to calculate: <1>, <x>, <p>, <V>, <T>
using Averages = std::tuple<double, ClassicalDoubleVector, ClassicalDoubleVector, double, double>;

/// Store hyperparameters, kernels and optimization algorithms to use.
/// Optimize hyperparameters, and then predict density matrix and given point.
class Optimization final
{
private:
	// static constant variables
	static const double DiagMagMin, DiagMagMax, AbsoluteTolerance, InitialStepSize;
	// local variables
	const KernelTypeList TypesOfKernels;					   ///< The types of the kernels to use
	const int NumHyperparameters;							   ///< The number of hyperparameters to use, derived from the kernel types
	const int NumPoints;									   ///< The number of points selected for hyperparameter optimization
	std::vector<nlopt::opt> NLOptMinimizers;				   ///< The vector containing all NLOPT optimizers, one non-grad for each element
	std::array<ParameterVector, NumElements> Hyperparameters;  ///< The hyperparameters for all elements of density matrix
	std::array<Eigen::MatrixXd, NumElements> TrainingFeatures; ///< The training features (phase space coordinates) of each density matrix element
	std::array<KernelList, NumElements> Kernels;			   ///< The kernels with hyperparameters set for all elements of density matrix
	std::array<Eigen::VectorXd, NumElements> KInvLbls;		   ///< The inverse of kernel matrix of training features times the training labels

public:
	/// @brief Constructor. Initial hyperparameters, kernels and optimization algorithms needed
	/// @param[in] Params Parameters object containing position, momentum, etc
	/// @param[in] KernelTypes The vector containing all the kernel type used in optimization
	/// @param[in] Algorithm The non-gradient optimization algorithm
	Optimization(
		const Parameters& Params,
		const KernelTypeList& KernelTypes = { shogun::EKernelType::K_DIAG, shogun::EKernelType::K_GAUSSIANARD },
		const nlopt::algorithm Algorithm = nlopt::algorithm::LN_NELDERMEAD);

	/// @brief To optimize hyperparameters based on the given density
	/// @param[inout] density The vector containing all known density matrix, and normalized after optimization
	/// @param[in] IsSmall The matrix that saves whether each element is small or not
	/// @return The total error (MSE, log likelihood, etc) of all elements of density matrix
	double optimize(const EvolvingDensity& density, const QuantumBoolMatrix& IsSmall);

	/// @brief To normalize the training set
	/// @param[inout] density The vector containing all known density matrix, and normalized after optimization
	/// @param[in] IsSmall The matrix that saves whether each element is small or not
	/// @return The total population before normalization
	double normalize(EvolvingDensity& density, const QuantumBoolMatrix& IsSmall) const;

	/// @brief To update the TrainingFeatures and KInvLbls up-to-date
	/// @param[in] density The vector containing all known density matrix, and normalized after optimization
	/// @param[in] IsSmall The matrix that saves whether each element is small or not
	void update_training_set(const EvolvingDensity& density, const QuantumBoolMatrix& IsSmall);

	/// @brief To predict one of the elements of the density matrix at the given phase space point
	/// @param[in] IsSmall The matrix that saves whether each element is small or not
	/// @param[in] x Position of classical degree of freedom
	/// @param[in] p Momentum of classical degree of freedom
	/// @param[in] ElementIndex The index of the density matrix element
	/// @return The element of the density matrix at the given point
	double predict_element(
		const QuantumBoolMatrix& IsSmall,
		const ClassicalDoubleVector& x,
		const ClassicalDoubleVector& p,
		const int ElementIndex) const;

	/// @brief To predict the density matrix at the given phase space point
	/// @param[in] IsSmall The matrix that saves whether each element is small or not
	/// @param[in] x Position of classical degree of freedom
	/// @param[in] p Momentum of classical degree of freedom
	/// @return The density matrix at the given point
	QuantumComplexMatrix predict_matrix(
		const QuantumBoolMatrix& IsSmall,
		const ClassicalDoubleVector& x,
		const ClassicalDoubleVector& p) const;

	/// @brief To predict one of the elements of the density matrix at the given phase space points
	/// @param[in] IsSmall The matrix that saves whether each element is small or not
	/// @param[in] x Positions of classical degree of freedom
	/// @param[in] p Momenta of classical degree of freedom
	/// @param[in] ElementIndex The index of the density matrix element
	/// @return The element of the density matrix at the given points
	Eigen::VectorXd predict_elements(
		const QuantumBoolMatrix& IsSmall,
		const ClassicalVectors& x,
		const ClassicalVectors& p,
		const int ElementIndex) const;

	/// @brief To print the grids of a certain element of density matrix
	/// @param[in] PhaseGrids All the grids required to calculate in phase space
	/// @param[in] ElementIndex The index of the density matrix element
	/// @return A 1-by-N matrix, N is the number of required grids
	Eigen::VectorXd print_element(const QuantumBoolMatrix& IsSmall, const Eigen::MatrixXd& PhaseGrids, const int ElementIndex) const;

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

#endif // !OPT_H