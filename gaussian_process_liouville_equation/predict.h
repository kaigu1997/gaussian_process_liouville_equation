/// @file predict.h
/// @brief Interface of functions to predict averages using parameters

#ifndef PREDICT_H
#define PREDICT_H

#include "stdafx.h"

#include "kernel.h"

/// @brief To calculate the average position and momentum of one element by averaging directly
/// @param[in] density The selected points in phase space
/// @return Average position and momentum
ClassicalPhaseVector calculate_1st_order_average(const EigenVector<PhaseSpacePoint>& density);

/// @brief To calculate the variance of position and momentum of one element
/// @param[in] density The selected points in phase space
/// @return Standard deviation of position and momentum
ClassicalPhaseVector calculate_standard_deviation(const EigenVector<PhaseSpacePoint>& density);

/// @brief To calculate the total energy
/// @param[in] density The selected points in phase space, must corresponding to a diagonal element in density matrix
/// @param[in] PESIndex The row and column index of the @p density in density matrix
/// @param[in] mass Mass of classical degree of freedom
/// @return Averaged total energy
double calculate_total_energy_average(
	const EigenVector<PhaseSpacePoint>& density,
	const int PESIndex,
	const ClassicalVector<double>& mass);

/// @brief To construct the training set as a matrix from std::vector of x and p
/// @param[in] x Position of classical degree of freedom
/// @param[in] p Momentum of classical degree of freedom
/// @return The combination of the input
Eigen::MatrixXd construct_training_feature(const EigenVector<ClassicalVector<double>>& x, const EigenVector<ClassicalVector<double>>& p);

/// @brief To print the grids of a certain element of density matrix
/// @param[in] kernel The kernel of the training set
/// @param[in] PhaseGrids All the grids required to calculate in phase space
/// @return A N-by-1 matrix, N is the number of required grids
Eigen::VectorXd predict_elements(const Kernel& kernel, const Eigen::MatrixXd& PhaseGrids);

/// @brief To calculate the derivative of the prediction over parameters
/// @param[in] kernel The kernel of the training set
/// @param[in] x Position of classical degree of freedom
/// @param[in] p Momentum of classical degree of freedom
/// @return A vector of the derivative of the kernel matrix over each of the parameter
ParameterVector prediction_derivative(const Kernel& kernel, const ClassicalVector<double>& x, const ClassicalVector<double>& p);

/// @brief To print the grids of a certain element of density matrix
/// @param[in] kernel The kernel of the training set
/// @param[in] PhaseGrids All the grids required to calculate in phase space
/// @return A N-by-1 matrix, N is the number of required grids
Eigen::VectorXd predict_variances(const Kernel& kernel, const Eigen::MatrixXd& PhaseGrids);

/// Array of predictors, whether have or not depending on IsSmall
using OptionalKernels = QuantumArray<std::optional<Kernel>>;

/// @brief To predict the density matrix at the given phase space point
/// @param[in] Kernels An array of predictors for prediction, whose size is NumElements
/// @param[in] x Position of classical degree of freedom
/// @param[in] p Momentum of classical degree of freedom
/// @return The density matrix at the given point
QuantumMatrix<std::complex<double>> predict_matrix(
	const OptionalKernels& Kernels,
	const ClassicalVector<double>& x,
	const ClassicalVector<double>& p);

/// @brief To calculate the purity of the density matrix by parameters
/// @param[in] Kernels An array of predictors for prediction, whose size is NumElements
/// @return The purity of the overall partial Wigner transformed density matrix
double calculate_purity(const OptionalKernels& Kernels);

#endif // !PREDICT_H
