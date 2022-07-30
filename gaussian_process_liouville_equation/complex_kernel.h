/// @file complex_kernel.h
/// @brief The kernel for Complex Gaussian Progress for Regression (CGPR)

#ifndef COMPLEX_KERNEL_H
#define COMPLEX_KERNEL_H

#include "stdafx.h"

#include "kernel.h"

/// @brief The kernel for CGPR
/// @details The CGPR consists of two parts: kernel @f$ k(x,x')=k_R(x,x')+k_I(x,x') @f$,
/// and pseudo-kernel @f$ \tilde{k}(x,x')=k_R(x,x')-k_I(x,x')+2\mathrm{i}k_C(x,x') @f$.
class ComplexKernel final: public KernelBase
{
public:
	static constexpr std::size_t NumTotalParameters = 1 + 2 * (Kernel::NumTotalParameters - 1) + 1; ///< The overall number of parameters, including 1 for magnitude Gaussian and @p PhaseDim for characteristic length of real, imaginary and correlation Gaussian, and 1 for noise
	static constexpr std::size_t NumKernels = 2; ///< Real and Imaginary kernels

	/// The unpacked parameters for the complex kernels. @n
	/// First the global magnitude, then the magnitude and the characteristic lengths of real and imaginary kernel, third the noise
	using KernelParameter = std::tuple<double, std::array<std::tuple<double, ClassicalPhaseVector>, 2>, double>;
	/// The serialized type for parameters of kernel. Generally for derivatives
	/// @tparam T The type of the item whose derivative is calculated
	template <typename T>
	using ParameterArray = std::array<T, NumTotalParameters>;

	ComplexKernel() = delete;

	/// @brief The constructor for feature and label, and average is generally calculated
	/// @param[in] Parameter Parameters used in the kernel
	/// @param[in] feature The left feature
	/// @param[in] label The label of kernels if used
	/// @param[in] IsCalculateAverage Whether to calculate averages (@<1@>, @<r@>, and purity) or not
	/// @param[in] IsCalculateDerivative Whether to calculate derivative of each kernel or not
	ComplexKernel(
		const ParameterVector& Parameter,
		const PhasePoints& feature,
		const Eigen::VectorXcd& label,
		const bool IsToCalculateError,
		const bool IsToCalculateAverage,
		const bool IsToCalculateDerivative);

	/// @brief The constructor for different features
	/// @param[in] Parameter Parameters used in the kernel
	/// @param[in] TestFeature The feature for test set
	/// @param[in] TrainingKernel The kernel of training set
	/// @param[in] IsToCalculateDerivative Whether to calculate derivative or not
	/// @param[in] TestLabel The label corresponding to the feature. Once given, the squared error and derivatives (if needed) is calculated
	ComplexKernel(
		const PhasePoints& TestFeature,
		const ComplexKernel& TrainingKernel,
		const bool IsToCalculateDerivative,
		const std::optional<Eigen::VectorXcd> TestLabel = std::nullopt);

	/// @brief To get the parameters
	/// @return The parameters in std::vector
	const ParameterVector& get_parameters(void) const
	{
		assert(Params.size() == NumTotalParameters);
		return Params;
	}

	/// @brief To get the proper magnitude
	/// @return The magnitude maximizing likelihood
	double get_magnitude(void) const
	{
		assert(UpperPartOfAugmentedInverseLabel.has_value());
		const double within_sqrt = Label->dot(UpperPartOfAugmentedInverseLabel.value()).real() / Label->size();
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

	/// @brief To get the prediction of the test input
	/// @return The prediction
	const Eigen::VectorXcd& get_prediction(void) const
	{
		assert(Prediction.has_value());
		return Prediction.value();
	}

	/// @brief To get the variance of the test input
	/// @return The variance of each input, or the diagonal elements of the covariance matrix
	const Eigen::VectorXd& get_variance(void) const
	{
		assert(ElementwiseVariance.has_value());
		return ElementwiseVariance.value();
	}

	/// @brief To get the cutoff prediction of the test input by compare with variance
	/// @return The cutoff prediction
	const Eigen::VectorXcd& get_prediction_compared_with_variance(void) const
	{
		assert(CutoffPrediction.has_value());
		return CutoffPrediction.value();
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

	/// @brief Virtual destructor
	virtual ~ComplexKernel(void)
	{
	}

private:
	/// @brief The constructor for same features without label, generally used for variance
	/// @param[in] TrainingKernel The kernel for the training set
	/// @param[in] left_feature The left feature
	/// @param[in] right_feature The right feature
	/// @param[in] IsToCalculateDerivative Whether to calculate derivative of each kernel or not
	ComplexKernel(
		const ComplexKernel& TrainingKernel,
		const PhasePoints& left_feature,
		const PhasePoints& right_feature);

	const ParameterVector Params;																///< Parameters in vector
	const KernelParameter KernelParams;															///< All parameters used in kernel
	const Kernel::KernelParameter RealParams;													///< Parameters for the real kernel
	const Kernel::KernelParameter ImagParams;													///< Parameters for the imag kernel
	const Kernel::KernelParameter CorrParams;													///< Parameters for the corr kernel
	const Kernel RealKernel;																	///< The real kernel @f$ k_R(x,x') @f$
	const Kernel ImagKernel;																	///< The imaginary kernel @f$ k_I(x,x') @f$
	const Kernel CorrKernel;																	///< The correlation kernel @f$ k_C(x,x') @f$
	const Eigen::MatrixXd KernelMatrix;															///< The covariance matrix @f$ K @f$
	const Eigen::MatrixXcd PseudoKernelMatrix;													///< The pseudo-covariance matrix @f$ \tilde{K} @f$
	const std::optional<Eigen::VectorXcd> Label;												///< The label used for training set
	const std::optional<Eigen::LLT<Eigen::MatrixXcd>> DecompositionOfKernel;					///< The Cholesky decomposition of the kernel matrix
	const std::optional<Eigen::MatrixXcd> KernelInversePseudoConjugate;							///< @f$ K^{-1}\tilde{K}^* @f$
	const std::optional<Eigen::MatrixXcd> UpperLeftBlockOfAugmentedKernelInverse;				///< The upper left block of the inverse of augmented kernel
	const std::optional<Eigen::MatrixXcd> LowerLeftBlockOfAugmentedKernelInverse;				///< The lower left block of the inverse of augmented kernel
	const std::optional<Eigen::VectorXcd> UpperPartOfAugmentedInverseLabel;						///< The upper part of augmented inverse times augmented label
	const std::optional<Eigen::VectorXcd> Prediction;											///< Prediction from regression
	const std::optional<Eigen::VectorXd> ElementwiseVariance;									///< Variance for each input feature
	const std::optional<Eigen::VectorXcd> CutoffPrediction;										///< A cutoff version of prediction
	const std::optional<double> Error;															///< Training set LOOCV-SE or squared error for test set
	const std::optional<Kernel::KernelParameter> PurityAuxiliaryRealParams;						///< Parameters for the real kernel
	const std::optional<Kernel::KernelParameter> PurityAuxiliaryImagParams;						///< Parameters for the imag kernel
	const std::optional<Kernel::KernelParameter> PurityAuxiliaryCorrParams;						///< Parameters for the corr kernel
	const std::optional<Kernel::KernelParameter> PurityAuxiliaryRealCorrParams;					///< Parameters for the real kernel
	const std::optional<Kernel::KernelParameter> PurityAuxiliaryImagCorrParams;					///< Parameters for the imag kernel
	const std::optional<Kernel> PurityAuxiliaryRealKernel;										///< The real kernel with sqrt(2) times characteristic length
	const std::optional<Kernel> PurityAuxiliaryImagKernel;										///< The imag kernel with sqrt(2) times characteristic length
	const std::optional<Kernel> PurityAuxiliaryCorrKernel;										///< The cor kernel with sqrt(2) times characteristic length
	const std::optional<Kernel> PurityAuxiliaryRealCorrKernel;									///< The real-corr kernel with root mean square char length
	const std::optional<Kernel> PurityAuxiliaryImagCorrKernel;									///< The imag-corr kernel with root mean square char length
	const std::optional<double> Purity;															///< The purity calculated by parameters
	const std::optional<ParameterArray<Eigen::MatrixXd>> Derivatives;							///< Derivatives of kernel matrix over parameters
	const std::optional<ParameterArray<Eigen::MatrixXcd>> PseudoDerivatives;					///< Derivatives of pseudo-kernel matrix over parameters
	const std::optional<ParameterArray<Eigen::MatrixXcd>> UpperLeftAugmentedInverseDerivatives; ///< Derivatives of @p UpperLeftBlockOfAugmentedKernelInverse
	const std::optional<ParameterArray<Eigen::MatrixXcd>> LowerLeftAugmentedInverseDerivatives; ///< Derivatives of @p LowerLeftBlockOfAugmentedKernelInverse
	const std::optional<ParameterArray<Eigen::VectorXcd>> UpperAugmentedInvLblDerivatives;		///< Derivatives of @p UpperPartOfAugmentedInverseLabel
	const std::optional<ParameterArray<double>> ErrorDerivatives;								///< Derivative of the squared error over parameters in array
	const std::optional<ParameterArray<double>> PurityDerivatives;								///< Derivatives of purity over parameters in array
};

#endif // !COMPLEX_KERNEL_H
