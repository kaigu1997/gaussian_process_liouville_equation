/// @file kernel.h
/// @brief Interface to the kernel and the related

#ifndef KERNEL_H
#define KERNEL_H

#include "stdafx.h"

/// @brief The vector containing parameters (or similarly: bounds, gradient, etc)
using ParameterVector = std::vector<double>;
/// @brief The training set for parameter optimization of one element of density matrix
/// passed as parameter to the optimization function.
/// First is feature, second is label
using ElementTrainingSet = std::tuple<PhasePoints, Eigen::VectorXcd>;

static constexpr double ConnectingPoint = 2.0;

/// @brief To calculate the Kronecker delta kernel @f$ k(x,x')=delta_{x,x'} @f$
/// @param[in] LeftFeature Corresponding to @f$ x @f$
/// @param[in] RightFeature Corresponding to @f$ x' @f$
/// @return The kernel matrix
Eigen::MatrixXd delta_kernel(const PhasePoints& LeftFeature, const PhasePoints& RightFeature);

/// @brief The basic kernel, only contains parameters, features, kernel matrix, and (if needed) kernel matrix derivative
/// @details @f$ k(x_1,x_2)=\sigma_f^2\left(\mathrm{exp}
/// \left[-\frac{1}{2}\sum_i\left(\frac{x_{1,i}-x_{2,i}}{l_i}\right)^2\right]
/// +\sigma_n^2\delta(x_1-x_2)\right) @f$ @n
/// where @f$ \sigma_f @f$ is the magnitude, @f$ l_i @f$ are the characteristic lengths, and @f$ \sigma_n @f$ is the noise.
class KernelBase
{
public:
	/// @brief The overall number of parameters, including 1 for noise, 1 for magnitude of Gaussian and phasedim for characteristic length of Gaussian
	static constexpr std::size_t NumTotalParameters = 1 + PhaseDim + 1;
	/// @brief The reference for rescaling; maximum value is rescaled to this
	/// @details Rescaling is done on the labels and thus the @p InvLbl and error, and their derivative. @n
	/// Averages, purity, predictions, and their derivative will be scaled back.
	static constexpr double RescaleMaximum = 10.0;

	/// @brief The deserialized parameters for the kernels. @n
	/// First is the magnitude, then the characteristic lengths of Gaussian kernel, third the noise
	using KernelParameter = std::tuple<double, ClassicalPhaseVector, double>;
	/// @brief The serialized type for parameters of kernel. Generally for derivative
	/// @tparam T The type of the item whose derivative is calculated
	template <typename T>
	using ParameterArray = std::array<T, NumTotalParameters>;

	/// @brief The constructor for same features without label, generally used for variance and complex kernels
	/// @param[in] Parameter Parameters used in the kernel
	/// @param[in] left_feature The left feature
	/// @param[in] right_feature The right feature
	/// @param[in] IsToCalculateDerivative Whether to calculate derivative of each kernel or not
	KernelBase(
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

	/// @brief To get the derivative over each of the parameter
	/// @return The derivative of the kernel matrix over each of the parameter
	const ParameterArray<Eigen::MatrixXd>& get_derivative(void) const
	{
		assert(Derivatives.has_value());
		return Derivatives.value();
	}

private:
	/// @brief All parameters used in the kernel
	const KernelParameter KernelParams;
	/// @brief The left feature
	const PhasePoints LeftFeature;
	/// @brief The right feature
	const PhasePoints RightFeature;
	/// @brief The kernel matrix
	const Eigen::MatrixXd KernelMatrix;
	/// @brief Derivatives of @p KernelMatrix over parameters
	const std::optional<ParameterArray<Eigen::MatrixXd>> Derivatives;
};

/// @brief The class for kernel for training set, including its inverse, label, expected average, and their derivative. @n
/// Label, its product with inverse, LOOCV error and their derivatives are rescaled. @n
/// Averages, purity and their derivatives are at original scale.
class TrainingKernel final: public KernelBase
{
public:
	/// @brief Alias name from base class
	static constexpr std::size_t NumTotalParameters = KernelBase::NumTotalParameters;

	/// @brief Alias name from base class
	/// @tparam T The type of the item whose derivative is calculated
	template <typename T>
	using ParameterArray = KernelBase::ParameterArray<T>;

	/// @brief The constructor for training set, and average is generally calculated
	/// @param[in] Parameter Parameters used in the kernel
	/// @param[in] TrainingSet The feature and label for training
	/// @param[in] IsToCalculateError Whether to calculate LOOCV squared error or not
	/// @param[in] IsToCalculateAverage Whether to calculate averages (@<1@>, @<r@>, and purity) or not
	/// @param[in] IsToCalculateDerivative Whether to calculate derivative of each kernel or not
	TrainingKernel(
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

	/// @brief To get the inverse of kernel
	/// @return The product of inverse of kernel
	const Eigen::MatrixXd& get_inverse(void) const
	{
		return Inverse;
	}

	/// @brief To get the inverse of kernel times label
	/// @return The product of inverse of kernel and label
	const Eigen::VectorXd& get_inverse_times_label(void) const
	{
		return InvLbl;
	}

	/// @brief To get the proper magnitude
	/// @return The magnitude maximizing likelihood
	double get_magnitude(void) const
	{
		const double within_sqrt = Label.dot(InvLbl) / Label.size();
		if (within_sqrt < 0)
		{
			spdlog::warn("{}Magnitude square of real kernel is negative, {}. Using its absval instead.", indents<3>::apply(), within_sqrt);
			return std::sqrt(-within_sqrt);
		}
		else
		{
			return std::sqrt(within_sqrt);
		}
	}

	/// @brief To get the squared error (LOOCV or comparsion with label)
	/// @return The error
	double get_error(void) const
	{
		assert(Error.has_value());
		return Error.value();
	}

	/// @brief To get the population integration
	/// @return The population
	double get_population(void) const
	{
		assert(Population.has_value());
		return Population.value();
	}

	/// @brief To get the @<x@> and @<p@> integration
	/// @return The @<x@> and @<p@>
	const ClassicalPhaseVector& get_1st_order_average(void) const
	{
		assert(FirstOrderAverage.has_value());
		return FirstOrderAverage.value();
	}

	/// @brief To get the purity integration
	/// @return The purity
	double get_purity(void) const
	{
		assert(Purity.has_value());
		return Purity.value();
	}

	/// @brief To get the derivative of inverse times label over parameters
	/// @return The derivative of inverse times label over parameters
	const ParameterArray<Eigen::VectorXd>& get_inverse_times_label_derivative(void) const
	{
		assert(InvLblDerivatives.has_value());
		return InvLblDerivatives.value();
	}

	/// @brief To get the derivative of error over parameters
	/// @return The derivative of error over parameters
	const ParameterArray<double>& get_error_derivative(void) const
	{
		assert(ErrorDerivatives.has_value());
		return ErrorDerivatives.value();
	}

	/// @brief To get the derivative of population over parameters
	/// @return The derivative of population over parameters
	const ParameterArray<double>& get_population_derivative(void) const
	{
		assert(PopulationDerivatives.has_value());
		return PopulationDerivatives.value();
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
	/// @brief The factor to rescale the labels
	const double RescaleFactor;
	/// @brief The labels corresponding to feature when the features are the same
	const Eigen::VectorXd Label;
	/// @brief The Cholesky decomposition of the kernel matrix (if necessary)
	const Eigen::LDLT<Eigen::MatrixXd> DecompositionOfKernel;
	/// @brief The inverse of the kernel
	const Eigen::MatrixXd Inverse;
	/// @brief The @p Inverse times @p label
	const Eigen::VectorXd InvLbl;
	/// @brief LOOCV squared error for training set or squared error for test set
	const std::optional<double> Error;
	/// @brief The population calculated by parameters
	const std::optional<double> Population;
	/// @brief The @<r@> calculated by parameters
	const std::optional<ClassicalPhaseVector> FirstOrderAverage;
	/// @brief Parameters for the auxiliary kernel
	const std::optional<KernelBase::KernelParameter> PurityAuxiliaryParams;
	/// @brief The kernel whose characteristic length is sqrt(2) times bigger for purity
	const std::optional<KernelBase> PurityAuxiliaryKernel;
	/// @brief The purity calculated by parameters
	const std::optional<double> Purity;
	/// @brief Derivatives of @p Inverse over parameters
	const std::optional<ParameterArray<Eigen::MatrixXd>> InverseDerivatives;
	/// @brief Derivatives of @p InvLbl over parameters
	const std::optional<ParameterArray<Eigen::VectorXd>> InvLblDerivatives;
	/// @brief Derivatives of @p Error over parameters in array
	const std::optional<ParameterArray<double>> ErrorDerivatives;
	/// @brief Derivatives of @p Population over parameters in array
	const std::optional<ParameterArray<double>> PopulationDerivatives;
	/// @brief Derivatives of @p Purity over parameters in array
	const std::optional<ParameterArray<double>> PurityDerivatives;
};

/// @brief To calculate the parameters of purity auxiliary kernel (r', i', or c')
/// @param[in] OriginalParams The parameters for the corresponding kernel (r for r', i for i', c for c')
/// @return Parameters purity auxiliary kernel
inline KernelBase::KernelParameter construct_purity_auxiliary_kernel_params(const KernelBase::KernelParameter& OriginalParams)
{
	[[maybe_unused]] const auto& [OriginalMagnitude, OriginalCharLength, OriginalNoise] = OriginalParams;
	KernelBase::KernelParameter result;
	auto& [magnitude, char_length, noise] = result;
	magnitude = power<2>(OriginalMagnitude) * std::sqrt(OriginalCharLength.prod());
	char_length = std::numbers::sqrt2 * OriginalCharLength;
	noise = 0.0;
	return result;
}

/// @brief To calculate the cutoff factor of each input point
/// @tparam T Type of data (and kernel), could be @p double or @p complex<double>
/// @param[in] Prediction The prediction by GPR
/// @param[in] Variance The variance (not standard deviation) by GPR
/// @return The cutoff factor (in [0,1]) of each prediction
template <typename T>
Eigen::VectorXd cutoff_factor(const Eigen::Matrix<T, Eigen::Dynamic, 1>& Prediction, const Eigen::VectorXd& Variance)
{
	assert(Prediction.size() == Variance.size());
	const std::size_t NumPoints = Prediction.size();
	const auto indices = xt::arange(NumPoints);
	Eigen::VectorXd result(NumPoints);
	std::for_each(
		std::execution::par_unseq,
		indices.cbegin(),
		indices.cend(),
		[&Prediction, &Variance, &result](const std::size_t iPoint) -> void
		{
			const double pred_square = std::norm(Prediction[iPoint]);
			if (pred_square >= power<2>(ConnectingPoint) * Variance[iPoint])
			{
				result[iPoint] = 1.0;
			}
			else if (pred_square <= Variance[iPoint])
			{
				result[iPoint] = 0;
			}
			else
			{
				const double AbsoluteRelativePrediction = std::abs(Prediction[iPoint]) / std::sqrt(Variance[iPoint]);
				result[iPoint] = (3.0 * ConnectingPoint - 2.0 * AbsoluteRelativePrediction - 1.0)
					* power<2>(AbsoluteRelativePrediction - 1) / power<3>(ConnectingPoint - 1);
			}
		}
	);
	return result;
}

/// @brief The class of kernels doing predictions. The left and right feature is generally different. @n
/// Cutoff prediction and its derivatives are exact. Error and its derivatives are rescaled.
class PredictiveKernel: public KernelBase
{
public:
	/// @brief Alias name from base class
	static constexpr std::size_t NumTotalParameters = KernelBase::NumTotalParameters;
	/// @brief Alias name from base class
	/// @tparam T The type of the item whose derivative is calculated
	template <typename T>
	using ParameterArray = KernelBase::ParameterArray<T>;

	/// @brief The constructor for different features
	/// @param[in] TestFeature The feature for test set
	/// @param[in] kernel The kernel of training set
	/// @param[in] IsToCalculateDerivative Whether to calculate derivative or not
	/// @param[in] TestLabel The label corresponding to the feature. Once given, the squared error and derivative (if needed) is calculated
	PredictiveKernel(
		const PhasePoints& TestFeature,
		const TrainingKernel& kernel,
		const bool IsToCalculateDerivative,
		const std::optional<Eigen::VectorXd> TestLabel = std::nullopt
	);

	/// @brief To get the variance of the test input
	/// @return The variance of each input, or the diagonal elements of the covariance matrix
	const Eigen::VectorXd& get_variance(void) const
	{
		return ElementwiseVariance;
	}

	/// @brief To get the cutoff prediction of the test input by compared with variance
	/// @return The cutoff prediction
	const Eigen::VectorXd& get_cutoff_prediction(void) const
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
	/// @brief The factor to rescale the labels
	const double RescaleFactor;
	/// @brief Prediction from regression
	const Eigen::VectorXd Prediction;
	/// @brief Variance for each input feature
	const Eigen::VectorXd ElementwiseVariance;
	/// @brief A cutoff version of prediction
	const Eigen::VectorXd CutoffPrediction;
	/// @brief The labels corresponding to feature when the features are the same
	const std::optional<Eigen::VectorXd> Label;
	/// @brief Squared error for all test inputs
	const std::optional<double> Error;
	/// @brief Derivatives of @p Error over parameters in array
	const std::optional<ParameterArray<double>> ErrorDerivatives;
};

#endif // !KERNEL_H
