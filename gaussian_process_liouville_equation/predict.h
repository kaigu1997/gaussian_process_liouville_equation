/// @file predict.h
/// @brief Interface of functions to predict averages using parameters

#ifndef PREDICT_H
#define PREDICT_H

#include "stdafx.h"

#include "kernel.h"

/// @brief To calculate the average position and momentum of one element by averaging directly
/// @param[in] density The selected points in phase space
/// @return Average position and momentum
ClassicalPhaseVector calculate_1st_order_average_one_surface(const EigenVector<PhaseSpacePoint>& density);

/// @brief To calculate the variance of position and momentum of one element
/// @param[in] density The selected points in phase space
/// @return Standard deviation of position and momentum
ClassicalPhaseVector calculate_standard_deviation_one_surface(const EigenVector<PhaseSpacePoint>& density);

/// @brief To calculate the total energy of one element by averaging directly
/// @param[in] density The selected points in phase space, must corresponding to a diagonal element in density matrix
/// @param[in] mass Mass of classical degree of freedom
/// @param[in] PESIndex The row and column index of the @p density in density matrix
/// @return Averaged total energy
double calculate_total_energy_average_one_surface(
	const EigenVector<PhaseSpacePoint>& density,
	const ClassicalVector<double>& mass,
	const int PESIndex);

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

/// @brief To calculate the population on one surface by analytica integration of parameters
/// @param[in] kernel The kernel for prediction
/// @return The population of that surface calculated by points
inline double calculate_population_one_surface(const Kernel& kernel)
{
	return kernel.get_population();
}

/// @brief To calculate the average position and momentum of one element by analytical integral of parameters
/// @param[in] kernel The kernel of the training set
/// @return Average position and momentum
inline ClassicalPhaseVector calculate_1st_order_average_one_surface(const Kernel& kernel)
{
	return kernel.get_1st_order_average() / kernel.get_population();
}

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

/// @brief To calculate the total population by analytical integration of parameters
/// @param[in] Kernels An array of kernels for prediction, whose size is NumElements
/// @return The overall population calculated by points
double calculate_population(const OptionalKernels& Kernels);

/// @brief To calculate the <x> and <p> by analytical integration of parameters
/// @param[in] Kernels An array of kernels for prediction, whose size is NumElements
/// @param[in] mc_points The selected points for calculating mc integration
/// @return The overall population calculated by points
ClassicalPhaseVector calculate_1st_order_average(const OptionalKernels& Kernels);

/// @brief To calculate the total energy by analytical integration
/// @param[in] Kernels An array of kernels for prediction, whose size is NumElements
/// @param[in] Energies The energy on each surfaces
/// @return The total energy calculated by parameters
double calculate_total_energy_average(const OptionalKernels& Kernels, const QuantumVector<double>& Energies);

/// @brief To calculate the purity of the density matrix by analytical integration of parameters
/// @param[in] Kernels An array of predictors for prediction, whose size is NumElements
/// @return The purity of the overall partial Wigner transformed density matrix
double calculate_purity(const OptionalKernels& Kernels);

/// @brief To calculate the derivative of analytically integrated population over parameters
/// @param[in] Kernels An array of predictors for prediction, whose size is NumElements
/// @return The derivative of overall population calculated by parameters over parameters
ParameterVector population_derivative(const OptionalKernels& Kernels);

/// @brief To calculate the derivative of analytically integrated energy over parameters
/// @param[in] Kernels An array of predictors for prediction, whose size is NumElements
/// @param[in] Energies The energy on each surfaces
/// @return The derivative of overall energy calculated by parameters over parameters
ParameterVector energy_derivative(const OptionalKernels& Kernels, const QuantumVector<double>& Energies);

/// @brief To calculate the derivative of analytically integrated purity over parameters
/// @param[in] Kernels An array of kernels for prediction, whose size is NumElements
/// @return The derivative of overall purity calculated by parameters over parameters
ParameterVector purity_derivative(const OptionalKernels& Kernels);

#endif // !PREDICT_H
