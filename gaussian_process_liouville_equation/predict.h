/// @file predict.h
/// @brief Interface of functions to predict averages using parameters

#ifndef PREDICT_H
#define PREDICT_H

#include "stdafx.h"

#include "complex_kernel.h"
#include "kernel.h"
#include "storage.h"

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
	const std::size_t PESIndex
);

/// @brief To calculate the total energy of each element by averaging directly
/// @param[in] density The selected points in phase space, must corresponding to a diagonal element in density matrix
/// @param[in] mass Mass of classical degree of freedom
/// @return Averaged total energy on each surfaces
QuantumVector<double> calculate_total_energy_average_each_surface(const AllPoints& density, const ClassicalVector<double>& mass);

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

/// @brief The training sets of all elements
using AllTrainingSets = QuantumStorage<ElementTrainingSet>;

/// @brief To construct the training set for all elements
/// @param[in] density The selected points in phase space for each element of density matrices
/// @return The training set of all elements in density matrix
AllTrainingSets construct_training_sets(const AllPoints& density);

/// @brief Class of kernels for all elements
class Kernels final: public QuantumStorage<Kernel, ComplexKernel>
{
public:
	/// @brief The type of the base class
	using BaseType = QuantumStorage<Kernel, ComplexKernel>;

	/// @brief The constructor using the training sets
	/// @param[in] ParameterVectors The parameters for all elements of density matrix
	/// @param[in] TrainingSets The training sets of all elements of density matrix
	/// @param[in] IsToCalculateError Whether to calculate LOOCV squared error or not
	/// @param[in] IsToCalculateAverage Whether to calculate averages (@<1@>, @<r@>, and purity) or not
	/// @param[in] IsToCalculateDerivative Whether to calculate derivative of each kernel or not
	Kernels(
		const QuantumStorage<ParameterVector>& ParameterVectors,
		const AllTrainingSets& TrainingSets,
		const bool IsToCalculateError,
		const bool IsToCalculateAverage,
		const bool IsToCalculateDerivative
	);

	/// @brief The default constructor
	/// @param[in] ParameterVectors The parameters for all elements of density matrix
	/// @param[in] density The selected points in phase space for each element of density matrices
	Kernels(const QuantumStorage<ParameterVector>& ParameterVectors, const AllPoints& density);

	Kernels() = delete;

	/// @brief To calculate the total population by analytical integration of parameters
	/// @return The overall population calculated by points
	double calculate_population(void) const;

	/// @brief To calculate the @<x@> and @<p@> by analytical integration of parameters
	/// @return The overall @<x@> and @<p@> calculated by points
	ClassicalPhaseVector calculate_1st_order_average(void) const;

	/// @brief To calculate the total energy by analytical integration
	/// @param[in] Energies The energy on each surfaces
	/// @return The total energy calculated by parameters
	double calculate_total_energy_average(const QuantumVector<double>& Energies) const;

	/// @brief To calculate the purity of the density matrix by analytical integration of parameters
	/// @return The purity of the overall partial Wigner transformed density matrix
	double calculate_purity(void) const;

	/// @brief To calculate the derivative of analytically integrated population over parameters
	/// @return The derivative of overall population calculated by parameters over parameters
	ParameterVector population_derivative(void) const;

	/// @brief To calculate the derivative of analytically integrated energy over parameters
	/// @param[in] Energies The energy on each surfaces
	/// @return The derivative of overall energy calculated by parameters over parameters
	ParameterVector total_energy_derivative(const QuantumVector<double>& Energies) const;

	/// @brief To calculate the derivative of analytically integrated purity over parameters
	/// @return The derivative of overall purity calculated by parameters over parameters
	ParameterVector purity_derivative(void) const;
};

#endif // !PREDICT_H
