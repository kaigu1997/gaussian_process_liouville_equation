/// @file kernel.h
/// @brief Interface to the kernel and the related

#ifndef KERNEL_H
#define KERNEL_H

#include "stdafx.h"

/// The vector containing parameters (or similarly: bounds, gradient, etc)
using ParameterVector = std::vector<double>;

/// @brief The class for kernel, including parameters, kernel matrix, its inverse, expected average, and derivatives
class Kernel final
{
public:
	/// The unpacked parameters for the kernels.
	/// First is the noise, then the Gaussian kernel parameters: magnitude and characteristic lengths
	using KernelParameter = std::tuple<double, std::tuple<double, ClassicalPhaseVector>>;

	static const std::size_t NumTotalParameters; ///< The overall number of parameters, include 1 for noise, 1 for magnitude of Gaussian and phasedim for characteristic length of Gaussian

	/// @brief To set up the values for the initial parameter
	/// @param[in] xSigma The variance of position
	/// @param[in] pSigma The variance of momentum
	/// @return Vector containing all parameters for one element
	static ParameterVector set_initial_parameters(const ClassicalVector<double>& xSigma, const ClassicalVector<double>& pSigma);

	/// @brief To set up the bounds for the parameter
	/// @param[in] xSize The size of the box on x direction
	/// @param[in] pSize The size of the box on p direction
	/// @return Lower and upper bounds
	static std::array<ParameterVector, 2> set_parameter_bounds(const ClassicalVector<double>& xSize, const ClassicalVector<double>& pSize);

	/// @brief The constructor
	/// @param[in] Parameter Parameters used in the kernel
	/// @param[in] left_feature The left feature
	/// @param[in] right_feature The right feature
	/// @param[in] label The label of kernels if used
	/// @param[in] IsCalculateAverage Whether to calculate averages (<rho>, <x> and <p>, and purity) or not
	/// @param[in] IsCalculateDerivative Whether to calculate derivative of each kernel or not
	Kernel(
		const ParameterVector& Parameter,
		const Eigen::MatrixXd& left_feature,
		const Eigen::MatrixXd& right_feature,
		const std::optional<Eigen::VectorXd>& label = std::nullopt,
		const bool IsCalculateAverage = false,
		const bool IsCalculateDerivative = false);

	/// @brief To get the kernel matrix
	/// @return The kernel matrix
	const Eigen::MatrixXd& get_kernel(void) const
	{
		return KernelMatrix;
	}

	/// @brief To get the labels
	/// @return The labels
	const Eigen::VectorXd& get_label(void) const
	{
		assert(Label.has_value() == true);
		return Label.value();
	}

	/// @brief To get the natural log of the determinant of the kernel matrix
	/// @return Log determinant of the kernel matrix
	double get_log_determinant(void) const
	{
		assert(LogDeterminant.has_value() == true);
		return LogDeterminant.value();
	}

	/// @brief To get the inverse of the kernel matrix
	/// @return The inverse matrix
	const Eigen::MatrixXd& get_inverse(void) const
	{
		assert(Inverse.has_value() == true);
		return Inverse.value();
	}

	/// @brief To get the inverse of kernel times label
	/// @return The inverse of kernel matrix times label
	const Eigen::VectorXd& get_inverse_label(void) const
	{
		assert(InvLbl.has_value() != 0);
		return InvLbl.value();
	}

	/// @brief To get the population integration
	/// @return The population
	double get_population(void) const
	{
		assert(Population.has_value() == true);
		return Population.value();
	}

	/// @brief To get the <x> and <p> integration
	/// @return The <x> and <p>
	const ClassicalPhaseVector& get_first_order_average(void) const
	{
		assert(FirstOrderAverage.has_value() == true);
		return FirstOrderAverage.value();
	}

	/// @brief To get the purity integration
	/// @return The purity
	double get_purity(void) const
	{
		assert(Purity.has_value() == true);
		return Purity.value();
	}

	/// @brief To get the derivatives over each of the parameter
	/// @return The derivative of the kernel matrix over each of the parameter
	const EigenVector<Eigen::MatrixXd>& get_derivatives(void) const
	{
		assert(Derivatives.has_value() == true);
		return Derivatives.value();
	}

	/// @brief To get the kernel inverse times the derivatives
	/// @return The product of inverse of kernel matrix times each of the derivatives
	const EigenVector<Eigen::MatrixXd>& get_inv_deriv(void) const
	{
		assert(InvDeriv.has_value() == true);
		return InvDeriv.value();
	}

	/// @brief To get the derivative of kernel inverse over parameters
	/// @return The derivative of inverse over parameters, which is the negative inverse times derivative times inverse
	const EigenVector<Eigen::MatrixXd>& get_negative_inv_deriv_inv(void) const
	{
		assert(NegInvDerivInv.has_value() == true);
		return NegInvDerivInv.value();
	}

	/// @brief To get the derivative of InvLbl over parameters
	/// @return The derivative of InvLbl over parameters, which is the negative inverse times derivative times inverse times label
	const EigenVector<Eigen::VectorXd>& get_negative_inv_deriv_inv_lbl(void) const
	{
		assert(NegInvDerivInvLbl.has_value());
		return NegInvDerivInvLbl.value();
	}

	/// @brief To get the derivative of population over parameters
	/// @return The derivative of population over parameters
	const ParameterVector& get_population_derivative(void) const
	{
		assert(PopulationDeriv.has_value());
		return PopulationDeriv.value();
	}

	/// @brief To get the derivative of purity over parameters
	/// @return The derivative of purity over parameters
	const ParameterVector& get_purity_derivative(void) const
	{
		assert(PurityDeriv.has_value());
		return PurityDeriv.value();
	}

private:
	const KernelParameter KernelParams;									 ///< All parameters used in the kernel
	const Eigen::MatrixXd LeftFeature;									 ///< The left feature
	const Eigen::MatrixXd RightFeature;									 ///< The right feature
	const Eigen::MatrixXd KernelMatrix;									 ///< The kernel matrix
	const bool IsTrainingSet;											 ///< The flag whether it is training set, for construction only
	const std::optional<Eigen::VectorXd> Label;							 ///< The labels corresponding to feature, if the left and right features are the same
	const std::optional<Eigen::LLT<Eigen::MatrixXd>> DecompOfKernel;	 ///< The Cholesky decomposition of the kernel matrix (if necessary)
	const std::optional<double> LogDeterminant;							 ///< The natural log of determinant of kernel matrix
	const std::optional<Eigen::MatrixXd> Inverse;						 ///< The inverse of the kernel
	const std::optional<Eigen::VectorXd> InvLbl;						 ///< The inverse times label
	const bool IsAverageCalculated;										 ///< Whether the averages are calculated, for construction only
	const std::optional<double> Population;								 ///< The population calculated by parameters
	const std::optional<ClassicalPhaseVector> FirstOrderAverage;		 ///< The <x> and <p> calculated by parameters
	const std::optional<double> Purity;									 ///< The purity calculated by parameters
	const bool IsDerivativeCalculated;									 ///< whether the derivatives over parameters are calculated, for construction only
	const std::optional<EigenVector<Eigen::MatrixXd>> Derivatives;		 ///< Derivatives of kernel matrices over parameters
	const std::optional<EigenVector<Eigen::MatrixXd>> InvDeriv;			 ///< The product of inverse times derivatives
	const std::optional<EigenVector<Eigen::MatrixXd>> NegInvDerivInv;	 ///< The derivative of inverse matrix over parameters
	const std::optional<EigenVector<Eigen::VectorXd>> NegInvDerivInvLbl; ///< The derivative of InvLbl over parameters
	const bool IsAverageDerivativeCalculated;							 ///< Whether the derivatives of averages are calculated, for construction only
	const std::optional<ParameterVector> PopulationDeriv;				 ///< The derivative of population over parameters
	const std::optional<ParameterVector> PurityDeriv;					 ///< The derivative of purity over parameters
};

#endif // !KERNEL_H
