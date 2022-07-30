/// @file predict.h
/// @brief Interface of functions to predict averages using parameters

#ifndef PREDICT_H
#define PREDICT_H

#include "stdafx.h"

#include "complex_kernel.h"
#include "kernel.h"

/// @brief To calculate the average position and momentum of one element by averaging directly
/// @param[in] density The selected points in phase space
/// @return Average position and momentum
ClassicalPhaseVector calculate_1st_order_average_one_surface(const ElementPoints& density);

/// @brief To calculate the variance of position and momentum of one element
/// @param[in] density The selected points in phase space
/// @return Standard deviation of position and momentum
ClassicalPhaseVector calculate_standard_deviation_one_surface(const ElementPoints& density);

/// @brief To calculate the total energy of one element by averaging directly
/// @param[in] density The selected points in phase space, must corresponding to a diagonal element in density matrix
/// @param[in] mass Mass of classical degree of freedom
/// @param[in] PESIndex The row and column index of the @p density in density matrix
/// @return Averaged total energy
double calculate_total_energy_average_one_surface(
	const ElementPoints& density,
	const ClassicalVector<double>& mass,
	const std::size_t PESIndex);

/// @brief To calculate the total energy of each element by averaging directly
/// @param[in] density The selected points in phase space, must corresponding to a diagonal element in density matrix
/// @param[in] mass Mass of classical degree of freedom
/// @return Averaged total energy on each surfaces
QuantumVector<double> calculate_total_energy_average_each_surface(const AllPoints& density, const ClassicalVector<double>& mass);

/// @brief To calculate the population on one surface by analytica integration of parameters
/// @param[in] kernel The kernel for prediction
/// @return The population of that surface calculated by points
inline double calculate_population_one_surface(const std::unique_ptr<KernelBase>& kernel)
{
	const Kernel& KernelRef = dynamic_cast<const Kernel&>(*kernel.get());
	return KernelRef.get_population();
}

/// @brief To calculate the average position and momentum of one element by analytical integral of parameters
/// @param[in] kernel The kernel of the training set
/// @return Average position and momentum
inline ClassicalPhaseVector calculate_1st_order_average_one_surface(const std::unique_ptr<KernelBase>& kernel)
{
	const Kernel& KernelRef = dynamic_cast<const Kernel&>(*kernel.get());
	return KernelRef.get_1st_order_average() / KernelRef.get_population();
}

/// Array of predictors, whether have or not depending on IsSmall
using OptionalKernels = QuantumArray<std::unique_ptr<KernelBase>>;

/// @brief To predict the density matrix at the given phase space point
/// @param[in] Kernels An array of predictors for prediction, whose size is NumElements
/// @param[in] r Phase space coordinates of classical degree of freedom
/// @return The density matrix at the given point
QuantumMatrix<std::complex<double>> predict_matrix(const OptionalKernels& Kernels, const ClassicalPhaseVector& r);

/// @brief To predict the density matrix at the given phase space point, after comparing with variance
/// @param[in] Kernels An array of predictors for prediction, whose size is NumElements
/// @param[in] r Phase space coordinates of classical degree of freedom
/// @return The density matrix at the given point
QuantumMatrix<std::complex<double>> predict_matrix_with_variance_comparison(const OptionalKernels& Kernels, const ClassicalPhaseVector& r);

/// @brief To calculate the total population by analytical integration of parameters
/// @param[in] Kernels An array of kernels for prediction, whose size is NumElements
/// @return The overall population calculated by points
double calculate_population(const OptionalKernels& Kernels);

/// @brief To calculate the @<x@> and @<p@> by analytical integration of parameters
/// @param[in] Kernels An array of kernels for prediction, whose size is NumElements
/// @return The overall @<x@> and @<p@> calculated by points
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
ParameterVector total_energy_derivative(const OptionalKernels& Kernels, const QuantumVector<double>& Energies);

/// @brief To calculate the derivative of analytically integrated purity over parameters
/// @param[in] Kernels An array of kernels for prediction, whose size is NumElements
/// @return The derivative of overall purity calculated by parameters over parameters
ParameterVector purity_derivative(const OptionalKernels& Kernels);

/// @brief To generate the extra points for the density fitting
/// @param[in] density The selected points in phase space for each element of density matrices
/// @param[in] NumPoints The number of points for each elements
/// @param[in] distribution The current distribution
/// @return The newly selected points with its corresponding distribution
AllPoints generate_extra_points(const AllPoints& density, const std::size_t NumPoints, const DistributionFunction& distribution);

#endif // !PREDICT_H
