/// @file gpr.h
/// @brief The header file containing gaussian process regression interfaces

#pragma once
#ifndef GPR_H
#define GPR_H

#include "stdafx.h"

/// @brief Calculate the population from phase space distribution
/// @param[in] PhaseSpaceDistribution The phase space distribution
/// @param[in] dx The grid size of x direction
/// @param[in] dp The grid size of p direction
/// @return \f$ <\rho_{ii}> \f$
QuantumVectorDouble calculate_population_from_grid(const SuperMatrix& PhaseSpaceDistribution, const double dx, const double dp);

/// @brief Calculate the potential energy of the distribution from grids
/// @param[in] AdiabaticDistribution The gridded whole PWTDM
/// @param[in] x The coordinates of x grids
/// @param[in] p The coordinates of p grids
/// @return The potential energy on each PES
QuantumVectorDouble calculate_potential_energy_from_grid(const SuperMatrix& AdiabaticDistribution, const Eigen::VectorXd& x, const Eigen::VectorXd& p);

/// @brief Calculate the kinetic energy of the distribution from grids
/// @param[in] AdiabaticDistribution The gridded whole PWTDM
/// @param[in] x The coordinates of x grids
/// @param[in] p The coordinates of p grids
/// @return The kinetic energy on each PES
QuantumVectorDouble calculate_kinetic_energy_from_grid(const SuperMatrix& AdiabaticDistribution, const Eigen::VectorXd& x, const Eigen::VectorXd& p);

/// @brief Set the initial value of hyperparameters and its upper/lower bounds
/// @param[in] TypesOfKernels The vector containing all the kernel type used in optimization
/// @param[in] data The gridded whole PWTDM
/// @param[in] x The coordinates of x grids
/// @param[in] p The coordinates of p grids
/// @return A vector of vector, {lower_bound, upper_bound, hyperparameter}
std::vector<ParameterVector> set_initial_value(
	const KernelTypeList& TypesOfKernels,
	const SuperMatrix& data,
	const Eigen::VectorXd& x,
	const Eigen::VectorXd& p);

/// @brief Judge if all points has very small weight
/// @param[in] data The gridded phase space distribution
/// @return If abs of all value are below a certain limit, return true, else return false
QuantumMatrixBool is_very_small_everywhere(const SuperMatrix& data);

/// @brief Generate the training set from the chosen point set
/// @param[in] data The gridded whole PWTDM
/// @param[in] IsSmall The judgement of whether each of the elements are small or not
/// @param[in] x The gridded position coordinates
/// @param[in] p The gridded momentum coordinates
/// @return The training set, labels from the density matrix, features from the coordinates
FullTrainingSet generate_training_set(
	const SuperMatrix& data,
	const QuantumMatrixBool& IsSmall,
	const Eigen::VectorXd& x,
	const Eigen::VectorXd& p);

/// @brief To opmitize the hyperparameters
/// @param[in] TrainingFeatures The features of the training set
/// @param[in] TrainingLabels The labels of the training set
/// @param[in] TypesOfKernels The types of all kernels used in Gaussian Process
/// @param[in] IsSmall The judgement of whether each of the elements are small or not
/// @param[in] ALGO The optimization algorithm, which is the nlopt::algorithm enum type
/// @param[in] LowerBound The lower bound of all the hyperparameters
/// @param[in] UpperBound The upper bound of all the hyperparameters
/// @param[in] InitialHyperparameters The initial guess of all the hyperparameters
/// @return The vector containing all hyperparameters, and the last element is the final function value
ParameterVector optimize(
	const SuperMatrix& TrainingFeatures,
	const VectorMatrix& TrainingLabels,
	const KernelTypeList& TypesOfKernels,
	const QuantumMatrixBool& IsSmall,
	const nlopt::algorithm& ALGO,
	const ParameterVector& LowerBound,
	const ParameterVector& UpperBound,
	const ParameterVector& InitialHyperparameters);

/// @brief Predict labels of test set
/// @param[inout] PredictedDistribution The phase space distribution waiting to be predicted
/// @param[in] TrainingFeatures The features of the training set
/// @param[in] TrainingLabels The labels of the training set
/// @param[in] TypesOfKernels The types of all kernels used in Gaussian Process
/// @param[in] IsSmall The judgement of whether each of the elements are small or not
/// @param[in] x The coordinates of x grids
/// @param[in] p The coordinates of p grids
/// @param[in] Hyperparameters Optimized hyperparameters used in kernel
void predict_phase(
	SuperMatrix& PredictedDistribution,
	const SuperMatrix& TrainingFeatures,
	const VectorMatrix& TrainingLabels,
	const KernelTypeList& TypesOfKernels,
	const QuantumMatrixBool& IsSmall,
	const Eigen::VectorXd& x,
	const Eigen::VectorXd& p,
	const ParameterVector& Hyperparameters);

/// @brief Calculate the normalization factor, or \f$ <\rho_{ii}> \f$
/// @param[in] TrainingFeatures The features of the training set
/// @param[in] TrainingLabels The labels of the training set
/// @param[in] TypesOfKernels The types of all kernels used in Gaussian Process
/// @param[in] IsSmall The judgement of whether each of the elements are small or not
/// @param[in] Hyperparameters Optimized hyperparameters used in kernel
/// @return \f$ <\rho_{ii}> \f$ for each i
QuantumVectorDouble calculate_population_from_gpr(
	const SuperMatrix& TrainingFeatures,
	const VectorMatrix& TrainingLabels,
	const KernelTypeList& TypesOfKernels,
	const QuantumMatrixBool& IsSmall,
	const ParameterVector& Hyperparameters);

/// @brief Calculate potential energy from the hyperparameters of Gaussian process
/// @param[in] TrainingFeatures The features of the training set
/// @param[in] TrainingLabels The labels of the training set
/// @param[in] TypesOfKernels The types of all kernels used in Gaussian Process
/// @param[in] IsSmall The judgement of whether each of the elements are small or not
/// @param[in] Hyperparameters Optimized hyperparameters of this element used in kernel
/// @return The potential energy of all the diagonal elements, potential energy from grids, and kinetic energy from analytical integration
QuantumVectorDouble calculate_potential_energy_from_gpr(
	const SuperMatrix& TrainingFeatures,
	const VectorMatrix& TrainingLabels,
	const KernelTypeList& TypesOfKernels,
	const QuantumMatrixBool& IsSmall,
	const ParameterVector& Hyperparameters);

/// @brief Calculate potential energy of the Gaussian process predicted phase space distribution
/// @param[in] TrainingFeatures The features of the training set
/// @param[in] TrainingLabels The labels of the training set
/// @param[in] TypesOfKernels The types of all kernels used in Gaussian Process
/// @param[in] IsSmall The judgement of whether each of the elements are small or not
/// @param[in] Hyperparameters Optimized hyperparameters of this element used in kernel
/// @return The total energy of all the diagonal elements, potential energy from grids, and kinetic energy from analytical integration
QuantumVectorDouble calculate_kinetic_energy_from_gpr(
	const SuperMatrix& TrainingFeatures,
	const VectorMatrix& TrainingLabels,
	const KernelTypeList& TypesOfKernels,
	const QuantumMatrixBool& IsSmall,
	const ParameterVector& Hyperparameters);

/// @brief To make the distribution following population/energy conservation
/// @param[inout] PredictedDistribution The phase space distribution which needs adding constraints
/// @param[in] TrainingFeatures The features of the training set
/// @param[in] TrainingLabels The labels of the training set
/// @param[in] TypesOfKernels The types of all kernels used in Gaussian Process
/// @param[in] IsSmall The judgement of whether each of the elements are small or not
/// @param[in] InitialTotalEnergy The total energy for energy conservation calculation
/// @param[in] x The coordinates of x grids
/// @param[in] p The coordinates of p grids
/// @param[in] Hyperparameters Optimized hyperparameters used in kernel
void obey_conservation(
	SuperMatrix& PredictedDistribution,
	const SuperMatrix& TrainingFeatures,
	VectorMatrix& TrainingLabels,
	const KernelTypeList& TypesOfKernels,
	const QuantumMatrixBool& IsSmall,
	const double InitialTotalEnergy,
	const Eigen::VectorXd& x,
	const Eigen::VectorXd& p,
	const ParameterVector& Hyperparameters);

/// @brief Calculate the mean squared error
/// @param[in] lhs The left-hand side PWTDM
/// @param[in] rhs The right-hand side PWTDM
/// @return The MSE between lhs and rhs on each element of PWTDM
QuantumMatrixDouble mean_squared_error(const SuperMatrix& lhs, const SuperMatrix& rhs);

#endif // !GPR_H
