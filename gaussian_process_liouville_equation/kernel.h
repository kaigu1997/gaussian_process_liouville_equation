/// @file kernel.h
/// @brief Interface to the kernel and the related

#ifndef KERNEL_H
#define KERNEL_H

#include "stdafx.h"

/// The vector containing parameters (or similarly: bounds, gradient, etc)
using ParameterVector = std::vector<double>;
/// The training set for parameter optimization of one element of density matrix
/// passed as parameter to the optimization function.
/// First is feature, second is label
using ElementTrainingSet = std::tuple<PhasePoints, Eigen::VectorXcd>;

static constexpr double ConnectingPoint = 2.0;

/// @brief To calculate the Kronecker delta kernel @f$ k(x,x')=delta_{x,x'} @f$
/// @param[in] LeftFeature Corresponding to @f$ x @f$
/// @param[in] RightFeature Corresponding to @f$ x' @f$
/// @return The kernel matrix
Eigen::MatrixXd delta_kernel(const PhasePoints& LeftFeature, const PhasePoints& RightFeature);

/// @brief To calculate the cutoff factor of each input point
/// @tparam T Type of data (and kernel), could be @p double or @p complex<double>
/// @param[in] Prediction The prediction by GPR
/// @param[in] Variance The variance (not standard deviation) by GPR
/// @return The cutoff factor (in [0,1]) of each prediction
template <typename T>
Eigen::VectorXd cutoff_factor(const Eigen::Matrix<T, Eigen::Dynamic, 1>& Prediction, const Eigen::VectorXd& Variance)
{
	const std::size_t NumPoints = Prediction.size();
	const auto indices = xt::arange(NumPoints);
	Eigen::VectorXd result(NumPoints);
	std::for_each(
		std::execution::par_unseq,
		indices.cbegin(),
		indices.cend(),
		[&Prediction, &Variance, &result](std::size_t i) -> void
		{
			const double pred_square = std::norm(Prediction[i]);
			if (pred_square >= power<2>(ConnectingPoint) * Variance[i])
			{
				result[i] = 1.0;
			}
			else if (pred_square <= Variance[i])
			{
				result[i] = 0;
			}
			else
			{
				const double AbsoluteRelativePrediction = std::abs(Prediction[i]) / std::sqrt(Variance[i]);
				result[i] = (3.0 * ConnectingPoint - 2.0 * AbsoluteRelativePrediction - 1.0)
					* power<2>(AbsoluteRelativePrediction - 1) / power<3>(ConnectingPoint - 1);
			}
		}
	);
	return result;
}

/// @brief The class for kernel, including parameters, kernel matrix, its inverse, expected average, and derivatives
/// @details @f$ k(x_1,x_2)=\sigma_f^2\left(\mathrm{exp}
/// \left[-\frac{1}{2}\sum_i\left(\frac{x_{1,i}-x_{2,i}}{l_i}\right)^2\right]
/// +\sigma_n^2\delta(x_1-x_2)\right) @f$
/// where @f$ \sigma_f @f$ is the magnitude, @f$ l_i @f$ are the characteristic lengths, and @f$ \sigma_n @f$ is the noise.
class Kernel final
{
public:
	/// @brief The overall number of parameters, including 1 for noise, 1 for magnitude of Gaussian and phasedim for characteristic length of Gaussian
	static constexpr std::size_t NumTotalParameters = 1 + PhaseDim + 1;

	/// The deserialized parameters for the kernels. @n
	/// First is the magnitude, then the characteristic lengths of Gaussian kernel, third the noise
	using KernelParameter = std::tuple<double, ClassicalPhaseVector, double>;
	/// The serialized type for parameters of kernel. Generally for derivatives
	/// @tparam T The type of the item whose derivative is calculated
	template <typename T>
	using ParameterArray = std::array<T, NumTotalParameters>;

	Kernel() = delete;

	/// @brief The constructor for training set, and average is generally calculated
	/// @param[in] Parameter Parameters used in the kernel
	/// @param[in] TrainingSet The feature and label for training
	/// @param[in] IsToCalculateError Whether to calculate LOOCV squared error or not
	/// @param[in] IsToCalculateAverage Whether to calculate averages (@<1@>, @<r@>, and purity) or not
	/// @param[in] IsToCalculateDerivative Whether to calculate derivative of each kernel or not
	Kernel(
		const ParameterVector& Parameter,
		const ElementTrainingSet& TrainingSet,
		const bool IsToCalculateError,
		const bool IsToCalculateAverage,
		const bool IsToCalculateDerivative
	);

	/// @brief The constructor for same features without label, generally used for variance and complex kernels
	/// @param[in] Parameter Parameters used in the kernel
	/// @param[in] left_feature The left feature
	/// @param[in] right_feature The right feature
	/// @param[in] IsToCalculateDerivative Whether to calculate derivative of each kernel or not
	Kernel(
		const KernelParameter& Parameter,
		const PhasePoints& left_feature,
		const PhasePoints& right_feature,
		const bool IsToCalculateDerivative
	);

	/// @brief The constructor for different features
	/// @param[in] TestFeature The feature for test set
	/// @param[in] TrainingKernel The kernel of training set
	/// @param[in] IsToCalculateDerivative Whether to calculate derivative or not
	/// @param[in] TestLabel The label corresponding to the feature. Once given, the squared error and derivatives (if needed) is calculated
	Kernel(
		const PhasePoints& TestFeature,
		const Kernel& TrainingKernel,
		const bool IsToCalculateDerivative,
		const std::optional<Eigen::VectorXd> TestLabel = std::nullopt
	);

	/// @brief To get the parameters
	/// @return The parameters in std::vector
	const ParameterVector& get_parameters(void) const
	{
		assert(Params.size() == NumTotalParameters);
		return Params;
	}

	/// @brief To get the kernel matrix
	/// @return The kernel matrix
	const Eigen::MatrixXd& get_kernel(void) const
	{
		return KernelMatrix;
	}

	/// @brief To get the proper magnitude
	/// @return The magnitude maximizing likelihood
	double get_magnitude(void) const
	{
		assert(InvLbl.has_value());
		const double within_sqrt = Label->dot(InvLbl.value()) / Label->size();
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

	/// @brief To get the prediction of the test input
	/// @return The prediction
	const Eigen::VectorXd& get_prediction(void) const
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
	const Eigen::VectorXd& get_prediction_compared_with_variance(void) const
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

	/// @brief To get the derivatives over each of the parameter
	/// @return The derivative of the kernel matrix over each of the parameter
	const ParameterArray<Eigen::MatrixXd>& get_derivatives(void) const
	{
		assert(Derivatives.has_value());
		return Derivatives.value();
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

	/// @brief Virtual destructor
	virtual ~Kernel(void)
	{
	}

private:
	/// @brief The left feature
	const PhasePoints LeftFeature;
	/// @brief The right feature
	const PhasePoints RightFeature;
	/// @brief Parameters in vector
	const ParameterVector Params;
	/// @brief All parameters used in the kernel
	const KernelParameter KernelParams;
	/// @brief The kernel matrix
	const Eigen::MatrixXd KernelMatrix;
	/// @brief The labels corresponding to feature when the features are the same
	const std::optional<Eigen::VectorXd> Label;
	/// @brief The Cholesky decomposition of the kernel matrix (if necessary)
	const std::optional<Eigen::LDLT<Eigen::MatrixXd>> DecompositionOfKernel;
	/// @brief The inverse of the kernel
	const std::optional<Eigen::MatrixXd> Inverse;
	/// @brief The @p Inverse times @p label
	const std::optional<Eigen::VectorXd> InvLbl;
	/// @brief Prediction from regression
	const std::optional<Eigen::VectorXd> Prediction;
	/// @brief Variance for each input feature
	const std::optional<Eigen::VectorXd> ElementwiseVariance;
	/// @brief A cutoff version of prediction
	const std::optional<Eigen::VectorXd> CutoffPrediction;
	/// @brief LOOCV squared error for training set or squared error for test set
	const std::optional<double> Error;
	/// @brief The population calculated by parameters
	const std::optional<double> Population;
	/// @brief The @<r@> calculated by parameters
	const std::optional<ClassicalPhaseVector> FirstOrderAverage;
	/// @brief Parameters for the auxiliary kernel
	const std::optional<Kernel::KernelParameter> PurityAuxiliaryParams;
	/// @brief The kernel whose characteristic length is sqrt(2) times bigger for purity
	const std::unique_ptr<Kernel> PurityAuxiliaryKernel;
	/// @brief The purity calculated by parameters
	const std::optional<double> Purity;
	/// @brief Derivatives of @p KernelMatrix over parameters
	const std::optional<ParameterArray<Eigen::MatrixXd>> Derivatives;
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
inline Kernel::KernelParameter construct_purity_auxiliary_kernel_params(const Kernel::KernelParameter& OriginalParams)
{
	[[maybe_unused]] const auto& [OriginalMagnitude, OriginalCharLength, OriginalNoise] = OriginalParams;
	Kernel::KernelParameter result;
	auto& [magnitude, char_length, noise] = result;
	magnitude = power<2>(OriginalMagnitude) * std::sqrt(OriginalCharLength.prod());
	char_length = M_SQRT2 * OriginalCharLength;
	noise = 0.0;
	return result;
}

#endif // !KERNEL_H
