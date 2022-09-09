/// @file complex_kernel.h
/// @brief The kernel for Complex Gaussian Progress for Regression (CGPR)

#ifndef COMPLEX_KERNEL_H
#define COMPLEX_KERNEL_H

#include "stdafx.h"

#include "kernel.h"

/// @brief The kernel for CGPR
/// @details The CGPR consists of two parts: kernel @f$ k(x,x')=k_R(x,x')+k_I(x,x') @f$,
/// and pseudo-kernel @f$ \tilde{k}(x,x')=k_R(x,x')-k_I(x,x')+2\mathrm{i}k_C(x,x') @f$.
class ComplexKernel final
{
public:
	/// @brief The overall number of parameters, including 1 for magnitude Gaussian
	/// and @p PhaseDim for characteristic length of real, imaginary and correlation Gaussian,
	/// and 1 for noise
	static constexpr std::size_t NumTotalParameters = 1 + 2 * (Kernel::NumTotalParameters - 1) + 1;
	/// @brief Real and Imaginary kernels
	static constexpr std::size_t NumKernels = 2;

	/// @brief The unpacked parameters for the complex kernels. @n
	/// First the global magnitude, then the magnitude and the characteristic lengths of real and imaginary kernel, third the noise
	using KernelParameter = std::tuple<double, std::array<std::tuple<double, ClassicalPhaseVector>, 2>, double>;
	/// @brief The serialized type for parameters of kernel. Generally for derivatives
	/// @tparam T The type of the item whose derivative is calculated
	template <typename T>
	using ParameterArray = std::array<T, NumTotalParameters>;

	ComplexKernel() = delete;

	/// @brief The constructor for feature and label, and average is generally calculated
	/// @param[in] Parameter Parameters used in the kernel
	/// @param[in] TrainingSet The feature and label for training
	/// @param[in] IsToCalculateError Whether to calculate LOOCV squared error or not
	/// @param[in] IsToCalculateAverage Whether to calculate averages (@<1@>, @<r@>, and purity) or not
	/// @param[in] IsToCalculateDerivative Whether to calculate derivative of each kernel or not
	ComplexKernel(
		const ParameterVector& Parameter,
		const ElementTrainingSet& TrainingSet,
		const bool IsToCalculateError,
		const bool IsToCalculateAverage,
		const bool IsToCalculateDerivative
	);

	/// @brief The constructor for different features
	/// @param[in] TestFeature The feature for test set
	/// @param[in] TrainingKernel The kernel of training set
	/// @param[in] IsToCalculateDerivative Whether to calculate derivative or not
	/// @param[in] TestLabel The label corresponding to the feature. Once given, the squared error and derivatives (if needed) is calculated
	ComplexKernel(
		const PhasePoints& TestFeature,
		const ComplexKernel& TrainingKernel,
		const bool IsToCalculateDerivative,
		const std::optional<Eigen::VectorXcd> TestLabel = std::nullopt
	);

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
	ComplexKernel(
		const ComplexKernel& TrainingKernel,
		const PhasePoints& left_feature,
		const PhasePoints& right_feature
	);

	/// @brief The left feature
	const PhasePoints LeftFeature;
	/// @brief The right feature
	const PhasePoints RightFeature;
	/// @brief Parameters in vector
	const ParameterVector Params;
	/// @brief All parameters used in kernel
	const KernelParameter KernelParams;
	/// @brief Parameters for the real kernel
	const Kernel::KernelParameter RealParams;
	/// @brief Parameters for the imaginary kernel
	const Kernel::KernelParameter ImagParams;
	/// @brief Parameters for the correlation kernel
	const Kernel::KernelParameter CorrParams;
	/// @brief The real kernel @f$ k_R(x,x') @f$
	const Kernel RealKernel;
	/// @brief The imaginary kernel @f$ k_I(x,x') @f$
	const Kernel ImagKernel;
	/// @brief The correlation kernel @f$ k_C(x,x') @f$
	const Kernel CorrKernel;
	/// @brief The covariance matrix @f$ K @f$
	const Eigen::MatrixXd KernelMatrix;
	/// @brief The pseudo-covariance matrix @f$ \tilde{K} @f$
	const Eigen::MatrixXcd PseudoKernelMatrix;
	/// @brief The label used for training set
	const std::optional<Eigen::VectorXcd> Label;
	/// @brief The Cholesky decomposition of the kernel matrix
	const std::optional<Eigen::LDLT<Eigen::MatrixXcd>> DecompositionOfKernel;
	/// @brief @f$ K^{-1}\tilde{K}^* @f$
	const std::optional<Eigen::MatrixXcd> KernelInversePseudoConjugate;
	/// @brief The upper left block of the inverse of augmented kernel
	const std::optional<Eigen::MatrixXcd> UpperLeftBlockOfAugmentedKernelInverse;
	/// @brief The lower left block of the inverse of augmented kernel
	const std::optional<Eigen::MatrixXcd> LowerLeftBlockOfAugmentedKernelInverse;
	/// @brief The upper part of augmented inverse times augmented label
	const std::optional<Eigen::VectorXcd> UpperPartOfAugmentedInverseLabel;
	/// @brief Prediction from regression
	const std::optional<Eigen::VectorXcd> Prediction;
	/// @brief Variance for each input feature
	const std::optional<Eigen::VectorXd> ElementwiseVariance;
	/// @brief A cutoff version of prediction
	const std::optional<Eigen::VectorXcd> CutoffPrediction;
	/// @brief Training set LOOCV squared error or squared error for test set
	const std::optional<double> Error;
	/// @brief Parameters for the auxiliary real kernel
	const std::optional<Kernel::KernelParameter> PurityAuxiliaryRealParams;
	/// @brief Parameters for the auxiliary imaginary kernel
	const std::optional<Kernel::KernelParameter> PurityAuxiliaryImagParams;
	/// @brief Parameters for the auxiliary correlation kernel
	const std::optional<Kernel::KernelParameter> PurityAuxiliaryCorrParams;
	/// @brief Parameters for the auxiliary real-correlation kernel
	const std::optional<Kernel::KernelParameter> PurityAuxiliaryRealCorrParams;
	/// @brief Parameters for the auxiliary imaginary-correlation kernel
	const std::optional<Kernel::KernelParameter> PurityAuxiliaryImagCorrParams;
	/// @brief The real kernel with sqrt(2) times characteristic length
	const std::optional<Kernel> PurityAuxiliaryRealKernel;
	/// @brief The imag kernel with sqrt(2) times characteristic length
	const std::optional<Kernel> PurityAuxiliaryImagKernel;
	/// @brief The cor kernel with sqrt(2) times characteristic length
	const std::optional<Kernel> PurityAuxiliaryCorrKernel;
	/// @brief The real-corr kernel with root mean square characteristic length
	const std::optional<Kernel> PurityAuxiliaryRealCorrKernel;
	/// @brief The imag-corr kernel with root mean square characteristic length
	const std::optional<Kernel> PurityAuxiliaryImagCorrKernel;
	/// @brief The purity calculated by parameters
	const std::optional<double> Purity;
	/// @brief Derivatives of kernel matrix over parameters
	const std::optional<ParameterArray<Eigen::MatrixXd>> Derivatives;
	/// @brief Derivatives of pseudo-kernel matrix over parameters
	const std::optional<ParameterArray<Eigen::MatrixXcd>> PseudoDerivatives;
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

#endif // !COMPLEX_KERNEL_H
