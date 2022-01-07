/// @file mc_predict.h
/// @brief Interface to the monte carlo integration

#ifndef MC_PREDICT_H
#define MC_PREDICT_H

#include "stdafx.h"

#include "input.h"
#include "predict.h"

/// @brief To generate the points for the monte carlo integration
/// @param[in] Params Parameters objects containing all the required information (r0, sigma0, mass)
/// @param[in] NumPoints The number of points for each elements
/// @return The selected points, the real part of whose first element is the weight
AllPoints mc_points_generator(const AllPoints& density, const int NumPoints);

/// @brief To calculate the population on one surface by monte carlo integration
/// @param[in] kernel The kernel for prediction
/// @param[in] mc_points The selected points for calculating mc integration
/// @return The population of that surface calculated by points
double calculate_population_one_surface(const Kernel& kernel, const AllPoints& mc_points);

/// @brief To calculate the <x> and <p> on one surface by monte carlo integration
/// @param[in] kernel The kernel for prediction
/// @param[in] mc_points The selected points for calculating mc integration
/// @return The <x> and <p> of that surface calculated by points
ClassicalPhaseVector calculate_1st_order_average_one_surface(const Kernel& kernel, const AllPoints& mc_points);

/// @brief To calculate the energy on one surface by monte carlo integration
/// @param[in] kernel The kernel for prediction
/// @param[in] mc_points The selected points for calculating mc integration
/// @param[in] mass Mass of classical degree of freedom
/// @param[in] PESIndex The row and column index of the element in density matrix
/// @return The energy of that surface calculated by points
double calculate_total_energy_average_one_surface(
	const Kernel& kernel,
	const AllPoints& mc_points,
	const ClassicalVector<double>& mass,
	const int PESIndex);

/// @brief To calculate the population by monte carlo integration
/// @param[in] Kernels An array of kernels for prediction, whose size is NumElements
/// @param[in] mc_points The selected points for calculating mc integration
/// @return The overall population calculated by points
double calculate_population(const OptionalKernels& Kernels, const AllPoints& mc_points);

/// @brief To calculate the <x> and <p> by monte carlo integration
/// @param[in] Kernels An array of kernels for prediction, whose size is NumElements
/// @param[in] mc_points The selected points for calculating mc integration
/// @return The overall <x> and <p> calculated by points
ClassicalPhaseVector calculate_1st_order_average(const OptionalKernels& Kernels, const AllPoints& mc_points);

/// @brief To calculate the total energy by monte carlo integration
/// @param[in] Kernels An array of kernels for prediction, whose size is NumElements
/// @param[in] mc_points The selected points for calculating mc integration
/// @param[in] mass Mass of classical degree of freedom
/// @return The total energy calculated by points
double calculate_total_energy_average(
	const OptionalKernels& Kernels,
	const AllPoints& mc_points,
	const ClassicalVector<double>& mass);

/// @brief To calculate the purity by monte carlo integration
/// @param[in] Kernels An array of kernels for prediction, whose size is NumElements
/// @param[in] mc_points The selected points for calculating mc integration
/// @return The overall purity calculated by points selected from (iPES, jPES) in density matrix
double calculate_purity(const OptionalKernels& Kernels, const AllPoints& mc_points);

/// @brief To calculate the derivative of monte carlo integrated population over parameters
/// @param[in] Kernels An array of predictors for prediction, whose size is NumElements
/// @param[in] mc_points The selected points for calculating mc integration
/// @return The derivative of overall population calculated by points over parameters
ParameterVector population_derivative(const OptionalKernels& Kernels, const AllPoints& mc_points);

/// @brief To calculate the derivative of monte carlo integrated total energy over parameters
/// @param[in] Kernels An array of kernels for prediction, whose size is NumElements
/// @param[in] mc_points The selected points for calculating mc integration
/// @param[in] mass Mass of classical degree of freedom
/// @return The derivative of total energy calculated by points over parameters
ParameterVector total_energy_derivative(
	const OptionalKernels& Kernels,
	const AllPoints& mc_points,
	const ClassicalVector<double>& mass);

/// @brief To calculate the derivative of monte carlo integrated total energy over parameters
/// @param[in] Kernels An array of kernels for prediction, whose size is NumElements
/// @param[in] mc_points The selected points for calculating mc integration
/// @return The derivative of overall purity calculated by points over parameters
ParameterVector purity_derivative(const OptionalKernels& Kernels, const AllPoints& mc_points);

#endif // !MC_PREDICT_H