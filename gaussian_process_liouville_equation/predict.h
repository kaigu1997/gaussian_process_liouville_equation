/// @file predict.h
/// @brief Interface of functions to predict averages using parameters

#ifndef PREDICT_H
#define PREDICT_H

#include "stdafx.h"

#include "kernel.h"
#include "opt.h"

/// @brief To construct the training set as a matrix from std::vector of x and p
/// @param[in] x Position of classical degree of freedom
/// @param[in] p Momentum of classical degree of freedom
/// @return The combination of the input
Eigen::VectorXd construct_training_feature(const EigenVector<ClassicalVector<double>>& x, const EigenVector<ClassicalVector<double>>& p);

/// The class for prediction of averages of a single element
class Prediction final
{
public:
	/// @brief The constructor
	/// @param[in] Parameters The parameters of the element
	/// @param[in] Features The training features (coordinates)
	/// @param[in] Labels The training labels (density matrix element)
	/// @param[in] IsCalculateAverage Whether to calculate averages or not
	Prediction(
		const ParameterVector& Params,
		const Eigen::MatrixXd& Features,
		const Eigen::VectorXd& Labels) :
		Prms(Params),
		TrainingFeatures(Features),
		Krn(Params, Features, Features, Labels, true)
	{
	}

	/// @brief To print the grids of a certain element of density matrix
	/// @param[in] PhaseGrids All the grids required to calculate in phase space
	/// @return A N-by-1 matrix, N is the number of required grids
	Eigen::VectorXd predict_elements(const Eigen::MatrixXd& PhaseGrids) const;

	/// @brief To calculate the population of the given surface
	/// @param[in] IsSmall The matrix that saves whether each element is small or not
	/// @param[in] PESIndex The index of the potential energy surface, corresponding to (PESIndex, PESIndex) in density matrix
	/// @return The population of the given surface
	double calculate_population(void) const
	{
		return Krn.get_population();
	}

	/// @brief To calculate the average position and momentum of the given surface
	/// @param[in] IsSmall The matrix that saves whether each element is small or not
	/// @param[in] PESIndex The index of the potential energy surface, corresponding to (PESIndex, PESIndex) in density matrix
	/// @return <x> and <p> of the given surface
	ClassicalPhaseVector calculate_1st_order_average(void) const
	{
		return Krn.get_1st_order_average() / Krn.get_population();
	}

	/// @brief To calculate the purity of the given element
	/// @return The purity of the given element
	double calculate_purity(void) const
	{
		return Krn.get_purity();
	}

private:
	const ParameterVector Prms;				///< The parameters for all elements of density matrix
	const Eigen::MatrixXd TrainingFeatures; ///< The training features (phase space coordinates) of each density matrix element
	const Kernel Krn;						///< The kernel
};

/// Array of predictors, whether have or not depending on IsSmall
using Predictions = QuantumArray<std::optional<Prediction>>;

/// @brief To update the predictors
/// @param[out] Predictors An array of predictors for prediction, whose size is NumElements
/// @param[in] Optimizer The optimization result of parameters
/// @param[in] IsSmall The matrix that saves whether each element is small or not
/// @param[in] density The selected density matrices
/// @return Array of predictors
void construct_predictors(
	Predictions& Predictors,
	const Optimization& Optimizer,
	const QuantumMatrix<bool>& IsSmall,
	const EigenVector<PhaseSpacePoint>& density);

/// @brief To predict the density matrix at the given phase space point
/// @param[in] Predictors An array of predictors for prediction, whose size is NumElements
/// @param[in] x Position of classical degree of freedom
/// @param[in] p Momentum of classical degree of freedom
/// @return The density matrix at the given point
QuantumMatrix<std::complex<double>> predict_matrix(
	const Predictions& Predictors,
	const ClassicalVector<double>& x,
	const ClassicalVector<double>& p);

/// @brief To calculate the population of the density matrix by parameters
/// @param[in] Predictors An array of predictors for prediction, whose size is NumElements
/// @return The population of the overall partial Wigner transformed density matrix
double calculate_population(const Predictions& Predictors);

/// @brief To calculate the average <x> and <p> of the density matrix by parameters
/// @param[in] Predictors An array of predictors for prediction, whose size is NumElements
/// @return The average <x> and <p> of the overall partial Wigner transformed density matrix
ClassicalPhaseVector calculate_1st_order_average(const Predictions& Predictors);

/// @brief To calculate the total energy of the density matrix by parameters
/// @param[in] Predictors An array of predictors for prediction, whose size is NumElements
/// @param[in] density The selected density matrices
/// @param[in] mass Mass of classical degree of freedom
/// @return The total energy of the overall partial Wigner transformed density matrix
double calculate_total_energy_average(
	const Predictions& Predictors,
	const EigenVector<PhaseSpacePoint>& density,
	const ClassicalVector<double>& mass);

/// @brief To calculate the purity of the density matrix by parameters
/// @param[in] Predictors An array of predictors for prediction, whose size is NumElements
/// @return The purity of the overall partial Wigner transformed density matrix
double calculate_purity(const Predictions& Predictors);

#endif // !PREDICT_H
