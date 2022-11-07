/// @file complex_kernel.h
/// @brief The kernel for Complex Gaussian Progress for Regression (CGPR)

#ifndef COMPLEX_KERNEL_H
#define COMPLEX_KERNEL_H

#include "stdafx.h"

#include "kernel.h"

/// @brief The kernel for CGPR
/// @details The CGPR consists of two parts: kernel @f$ k(x,x')=k_R(x,x')+k_I(x,x') @f$,
/// and pseudo-kernel @f$ \tilde{k}(x,x')=k_R(x,x')-k_I(x,x')+2\mathrm{i}k_C(x,x') @f$.
class ComplexKernelBase
{
public:
	/// @brief Real and Imaginary kernels
	static constexpr std::size_t NumKernels = 2;
	/// @brief The overall number of parameters, including 1 for magnitude Gaussian
	/// and @p PhaseDim for characteristic length of real, imaginary and correlation Gaussian,
	/// and 1 for noise
	static constexpr std::size_t NumTotalParameters = 1 + NumKernels * (KernelBase::NumTotalParameters - 1) + 1;
	/// @brief Alias name to the base kernel
	static constexpr double RescaleMaximum = KernelBase::RescaleMaximum;

	/// @brief The unpacked parameters for the complex kernels. @n
	/// First the global magnitude, then the magnitude and the characteristic lengths of real and imaginary kernel, third the noise
	using KernelParameter = std::tuple<double, std::array<std::tuple<double, ClassicalPhaseVector>, 2>, double>;
	/// @brief The serialized type for parameters of kernel. Generally for derivative
	/// @tparam T The type of the item whose derivative is calculated
	template <typename T>
	using ParameterArray = std::array<T, NumTotalParameters>;

	/// @brief The constructor for same features without label, generally used for variance
	/// @param[in] Parameter Parameters used in the kernel
	/// @param[in] left_feature The left feature
	/// @param[in] right_feature The right feature
	/// @param[in] IsToCalculateDerivative Whether to calculate derivative of each kernel or not
	ComplexKernelBase(
		const KernelParameter& Parameter,
		const PhasePoints& left_feature,
		const PhasePoints& right_feature,
		const bool IsToCalculateDerivative
	);

	/// @brief To get the parameters
	/// @return The parameters in tuple form
	const KernelParameter& get_formatted_parameters(void) const
	{
		return KernelParams;
	}

	/// @brief To get the parameters
	/// @return The parameters for the real kernel in tuple form
	const KernelBase::KernelParameter& get_real_kernel_parameters(void) const
	{
		return RealParams;
	}

	/// @brief To get the parameters
	/// @return The parameters for the imaginary kernel in tuple form
	const KernelBase::KernelParameter& get_imaginary_kernel_parameters(void) const
	{
		return ImagParams;
	}

	/// @brief To get the parameters
	/// @return The parameters for the kernel of real-imaginary correlation in tuple form
	const KernelBase::KernelParameter& get_correlation_kernel_parameters(void) const
	{
		return CorrParams;
	}

	/// @brief To get the left feature
	/// @return The left feature
	const PhasePoints& get_left_feature(void) const
	{
		return LeftFeature;
	}

	/// @brief To get the right feature
	/// @return The right feature
	const PhasePoints& get_right_feature(void) const
	{
		return RightFeature;
	}

	/// @brief To get the kernel matrix
	/// @return The kernel matrix
	const Eigen::MatrixXd& get_kernel(void) const
	{
		return KernelMatrix;
	}

	/// @brief To get the pseudo kernel matrix
	/// @return The pseudo kernel matrix
	const Eigen::MatrixXcd& get_pseudo_kernel(void) const
	{
		return PseudoKernelMatrix;
	}

	/// @brief To get the derivative over each of the parameter
	/// @return The derivative of the kernel matrix over each of the parameter
	const ParameterArray<Eigen::MatrixXd>& get_derivative(void) const
	{
		assert(Derivatives.has_value());
		return Derivatives.value();
	}

	/// @brief To get the derivative of pseudo kernel matrix over each of the parameter
	/// @return The derivative of the pseudo kernel matrix over each of the parameter
	const ParameterArray<Eigen::MatrixXcd>& get_pseudo_derivative(void) const
	{
		assert(PseudoDerivatives.has_value());
		return PseudoDerivatives.value();
	}

private:
	/// @brief All parameters used in kernel
	const KernelParameter KernelParams;
	/// @brief Parameters for the real kernel
	const KernelBase::KernelParameter RealParams;
	/// @brief Parameters for the imaginary kernel
	const KernelBase::KernelParameter ImagParams;
	/// @brief Parameters for the correlation kernel
	const KernelBase::KernelParameter CorrParams;
	/// @brief The left feature
	const PhasePoints LeftFeature;
	/// @brief The right feature
	const PhasePoints RightFeature;
	/// @brief The real kernel @f$ k_R(x,x') @f$
	const KernelBase RealKernel;
	/// @brief The imaginary kernel @f$ k_I(x,x') @f$
	const KernelBase ImagKernel;
	/// @brief The correlation kernel @f$ k_C(x,x') @f$
	const KernelBase CorrKernel;
	/// @brief The covariance matrix @f$ K @f$
	const Eigen::MatrixXd KernelMatrix;
	/// @brief The pseudo-covariance matrix @f$ \tilde{K} @f$
	const Eigen::MatrixXcd PseudoKernelMatrix;
	/// @brief Derivatives of kernel matrix over parameters
	const std::optional<ParameterArray<Eigen::MatrixXd>> Derivatives;
	/// @brief Derivatives of pseudo-kernel matrix over parameters
	const std::optional<ParameterArray<Eigen::MatrixXcd>> PseudoDerivatives;
};

/// @brief The class for complex kernel of training set, including its inverse, label, expected average, and their derivatives. @n
/// Label, its product with inverse, LOOCV error and their derivatives are rescaled. @n
/// Purity and its derivatives are at original scale.
class TrainingComplexKernel final: public ComplexKernelBase
{
public:
	/// @brief Alias name from base class
	static constexpr std::size_t NumTotalParameters = ComplexKernelBase::NumTotalParameters;

	/// @brief Alias name from base class
	/// @tparam T The type of the item whose derivative is calculated
	template <typename T>
	using ParameterArray = ComplexKernelBase::ParameterArray<T>;

	/// @brief The constructor for feature and label, and average is generally calculated
	/// @param[in] Parameter Parameters used in the kernel
	/// @param[in] TrainingSet The feature and label for training
	/// @param[in] IsToCalculateError Whether to calculate LOOCV squared error or not
	/// @param[in] IsToCalculateAverage Whether to calculate averages (@<1@>, @<r@>, and purity) or not
	/// @param[in] IsToCalculateDerivative Whether to calculate derivative of each kernel or not
	TrainingComplexKernel(
		const ParameterVector& Parameter,
		const ElementTrainingSet& TrainingSet,
		const bool IsToCalculateError,
		const bool IsToCalculateAverage,
		const bool IsToCalculateDerivative
	);

	/// @brief To get the parameters
	/// @return The parameters in std::vector
	const ParameterVector& get_parameters(void) const
	{
		assert(Params.size() == NumTotalParameters);
		return Params;
	}

	/// @brief To get the rescale factor
	/// @return The rescale factor
	double get_rescale_factor(void) const
	{
		return RescaleFactor;
	}

	/// @brief To get the proper magnitude
	/// @return The magnitude maximizing likelihood
	double get_magnitude(void) const
	{
		const double within_sqrt = Label.dot(UpperPartOfAugmentedInverseLabel).real() / Label.size();
		if (within_sqrt < 0)
		{
			spdlog::warn("{}Magnitude square of complex kernel is negative, {}. Using its absval instead.", indents<3>::apply(), within_sqrt);
			return std::sqrt(-within_sqrt);
		}
		else
		{
			return std::sqrt(within_sqrt);
		}
	}

	/// @brief To get the upper left block of the inverse of augmented kernel
	/// @return The upper left block of the inverse of augmented kernel
	const Eigen::MatrixXcd& get_upper_left_block_of_augmented_inverse(void) const
	{
		return UpperLeftBlockOfAugmentedKernelInverse;
	}

	/// @brief To get the lower left block of the inverse of augmented kernel
	/// @return The lower left block of the inverse of augmented kernel
	const Eigen::MatrixXcd& get_lower_left_block_of_augmented_inverse(void) const
	{
		return LowerLeftBlockOfAugmentedKernelInverse;
	}

	/// @brief To get the upper part of augmented inverse times augmented label
	/// @return The upper part of augmented inverse times augmented label
	const Eigen::VectorXcd& get_upper_part_of_augmented_inverse_times_label(void) const
	{
		return UpperPartOfAugmentedInverseLabel;
	}

	/// @brief To get the squared error (LOOCV or comparsion with label)
	/// @return The error
	double get_error(void) const
	{
		assert(Error.has_value());
		return Error.value();
	}

	/// @brief To get the purity integration
	/// @return The purity
	double get_purity(void) const
	{
		assert(Purity.has_value());
		return Purity.value();
	}

	/// @brief To get the derivative of upper part of augmented inverse times augmented label over parameters
	/// @return The derivative of upper part of augmented inverse times augmented label over parameters
	const ParameterArray<Eigen::VectorXcd>& get_upper_part_of_augmented_inverse_times_label_derivative(void) const
	{
		assert(UpperAugmentedInvLblDerivatives.has_value());
		return UpperAugmentedInvLblDerivatives.value();
	}

	/// @brief To get the derivative of error over parameters
	/// @return The derivative of error over parameters
	const ParameterArray<double>& get_error_derivative(void) const
	{
		assert(ErrorDerivatives.has_value());
		return ErrorDerivatives.value();
	}

	/// @brief To get the derivative of purity over parameters
	/// @return The derivative of purity over parameters
	const ParameterArray<double>& get_purity_derivative(void) const
	{
		assert(PurityDerivatives.has_value());
		return PurityDerivatives.value();
	}

private:
	/// @brief Parameters in vector
	const ParameterVector Params;
	/// @brief The factor for rescaling
	const double RescaleFactor;
	/// @brief The label used for training set
	const Eigen::VectorXcd Label;
	/// @brief The Cholesky decomposition of the kernel matrix
	const Eigen::LDLT<Eigen::MatrixXcd> DecompositionOfKernel;
	/// @brief @f$ K^{-1}\tilde{K}^* @f$
	const Eigen::MatrixXcd KernelInversePseudoConjugate;
	/// @brief The upper left block of the inverse of augmented kernel
	const Eigen::MatrixXcd UpperLeftBlockOfAugmentedKernelInverse;
	/// @brief The lower left block of the inverse of augmented kernel
	const Eigen::MatrixXcd LowerLeftBlockOfAugmentedKernelInverse;
	/// @brief The upper part of augmented inverse times augmented label
	const Eigen::VectorXcd UpperPartOfAugmentedInverseLabel;
	/// @brief Training set LOOCV squared error or squared error for test set
	const std::optional<double> Error;
	/// @brief Parameters for the auxiliary real kernel
	const std::optional<KernelBase::KernelParameter> PurityAuxiliaryRealParams;
	/// @brief Parameters for the auxiliary imaginary kernel
	const std::optional<KernelBase::KernelParameter> PurityAuxiliaryImagParams;
	/// @brief Parameters for the auxiliary correlation kernel
	const std::optional<KernelBase::KernelParameter> PurityAuxiliaryCorrParams;
	/// @brief Parameters for the auxiliary real-correlation kernel
	const std::optional<KernelBase::KernelParameter> PurityAuxiliaryRealCorrParams;
	/// @brief Parameters for the auxiliary imaginary-correlation kernel
	const std::optional<KernelBase::KernelParameter> PurityAuxiliaryImagCorrParams;
	/// @brief The real kernel with sqrt(2) times characteristic length
	const std::optional<KernelBase> PurityAuxiliaryRealKernel;
	/// @brief The imag kernel with sqrt(2) times characteristic length
	const std::optional<KernelBase> PurityAuxiliaryImagKernel;
	/// @brief The cor kernel with sqrt(2) times characteristic length
	const std::optional<KernelBase> PurityAuxiliaryCorrKernel;
	/// @brief The real-corr kernel with root mean square characteristic length
	const std::optional<KernelBase> PurityAuxiliaryRealCorrKernel;
	/// @brief The imag-corr kernel with root mean square characteristic length
	const std::optional<KernelBase> PurityAuxiliaryImagCorrKernel;
	/// @brief The purity calculated by parameters
	const std::optional<double> Purity;
	/// @brief Derivatives of @p UpperLeftBlockOfAugmentedKernelInverse
	const std::optional<ParameterArray<Eigen::MatrixXcd>> UpperLeftAugmentedInverseDerivatives;
	/// @brief Derivatives of @p LowerLeftBlockOfAugmentedKernelInverse
	const std::optional<ParameterArray<Eigen::MatrixXcd>> LowerLeftAugmentedInverseDerivatives;
	/// @brief Derivatives of @p UpperPartOfAugmentedInverseLabel
	const std::optional<ParameterArray<Eigen::VectorXcd>> UpperAugmentedInvLblDerivatives;
	/// @brief Derivative of the squared error over parameters in array
	const std::optional<ParameterArray<double>> ErrorDerivatives;
	/// @brief Derivatives of purity over parameters in array
	const std::optional<ParameterArray<double>> PurityDerivatives;
};


/// @brief The class of kernels doing predictions. The left and right feature is generally different. @n
/// Cutoff prediction are exact. Error and its derivatives are rescaled.
class PredictiveComplexKernel final: public ComplexKernelBase
{
public:
	/// @brief Alias name from base class
	static constexpr std::size_t NumTotalParameters = ComplexKernelBase::NumTotalParameters;

	/// @brief Alias name from base class
	/// @tparam T The type of the item whose derivative is calculated
	template <typename T>
	using ParameterArray = ComplexKernelBase::ParameterArray<T>;

	/// @brief The constructor for different features
	/// @param[in] TestFeature The feature for test set
	/// @param[in] kernel The kernel of training set
	/// @param[in] IsToCalculateDerivative Whether to calculate derivative or not
	/// @param[in] TestLabel The label corresponding to the feature. Once given, the squared error and derivative (if needed) is calculated
	PredictiveComplexKernel(
		const PhasePoints& TestFeature,
		const TrainingComplexKernel& kernel,
		const bool IsToCalculateDerivative,
		const std::optional<Eigen::VectorXcd> TestLabel = std::nullopt
	);

	/// @brief To get the variance of the test input
	/// @return The variance of each input, or the diagonal elements of the covariance matrix
	const Eigen::VectorXd& get_variance(void) const
	{
		return ElementwiseVariance;
	}

	/// @brief To get the cutoff prediction of the test input by compared with variance
	/// @return The cutoff prediction
	const Eigen::VectorXcd& get_cutoff_prediction(void) const
	{
		return CutoffPrediction;
	}

	/// @brief To get the squared error (LOOCV or comparsion with label)
	/// @return The error
	double get_error(void) const
	{
		assert(Error.has_value());
		return Error.value();
	}

	/// @brief To get the derivative of error over parameters
	/// @return The derivative of error over parameters
	const ParameterArray<double>& get_error_derivative(void) const
	{
		assert(ErrorDerivatives.has_value());
		return ErrorDerivatives.value();
	}

private:
	/// @brief The factor for rescaling
	const double RescaleFactor;
	/// @brief Prediction from regression
	const Eigen::VectorXcd Prediction;
	/// @brief Variance for each input feature
	const Eigen::VectorXd ElementwiseVariance;
	/// @brief A cutoff version of prediction
	const Eigen::VectorXcd CutoffPrediction;
	/// @brief The labels corresponding to feature when the features are the same
	const std::optional<Eigen::VectorXcd> Label;
	/// @brief Squared error for test set
	const std::optional<double> Error;
	/// @brief Derivative of the squared error over parameters in array
	const std::optional<ParameterArray<double>> ErrorDerivatives;
};

#endif // !COMPLEX_KERNEL_H
