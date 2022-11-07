/// @file kernel.cpp
/// @brief Implementation of kernel.h

#include "stdafx.h"

#include "kernel.h"

Eigen::MatrixXd delta_kernel(const PhasePoints& LeftFeature, const PhasePoints& RightFeature)
{
	const std::size_t Rows = LeftFeature.cols(), Cols = RightFeature.cols();
	if (LeftFeature.data() == RightFeature.data())
	{
		// simple test for training set
		return Eigen::MatrixXd::Identity(Rows, Cols);
	}
	Eigen::MatrixXd result = Eigen::MatrixXd::Zero(Rows, Cols);
	const auto indices = xt::arange(Cols);
	std::for_each(
		std::execution::par_unseq,
		indices.cbegin(),
		indices.cend(),
		[&result, &LeftFeature, &RightFeature, Rows](const std::size_t iCol) -> void
		{
			for (const std::size_t iRow : std::ranges::iota_view{0ul, Rows})
			{
				result(iRow, iCol) = static_cast<double>(LeftFeature.col(iRow) == RightFeature.col(iCol));
			}
		}
	);
	return result;
}

/// @brief To calculate the Gaussian kernel
/// @param[in] CharacteristicLength The characteristic lengths for all dimensions
/// @param[in] LeftFeature The left feature
/// @param[in] RightFeature The right feature
/// @return The Gaussian kernel matrix
static Eigen::MatrixXd gaussian_kernel(
	const Eigen::VectorXd& CharacteristicLength,
	const PhasePoints& LeftFeature,
	const PhasePoints& RightFeature
)
{
	auto calculate_element = [&CharacteristicLength, &LeftFeature, &RightFeature](const std::size_t iRow, const std::size_t iCol) -> double
	{
		const auto& diff = (LeftFeature.col(iRow) - RightFeature.col(iCol)).array() / CharacteristicLength.array();
		return std::exp(-diff.square().sum() / 2.0);
	};
	const std::size_t Rows = LeftFeature.cols(), Cols = RightFeature.cols();
	const bool IsTrainingSet = LeftFeature.data() == RightFeature.data() && Rows == Cols;
	Eigen::MatrixXd result = Eigen::MatrixXd::Zero(Rows, Cols);
	const auto indices = xt::arange(Cols);
	std::for_each(
		std::execution::par_unseq,
		indices.cbegin(),
		indices.cend(),
		[&result, &calculate_element, Rows, IsTrainingSet](const std::size_t iCol) -> void
		{
			// column major
			if (IsTrainingSet)
			{
				// training set, the matrix is symmetric
				for (const std::size_t iRow : std::ranges::iota_view{iCol + 1, Rows})
				{
					result(iRow, iCol) += calculate_element(iRow, iCol);
				}
			}
			else
			{
				// general rectangular
				for (const std::size_t iRow : std::ranges::iota_view{0ul, Rows})
				{
					result(iRow, iCol) += calculate_element(iRow, iCol);
				}
			}
		}
	);
	if (IsTrainingSet)
	{
		// training set, the matrix is symmetric
		result.diagonal() = Eigen::VectorXd::Ones(Rows);
		result = result.selfadjointView<Eigen::Lower>();
	}
	return result;
}

/// @brief To calculate the derivative of gaussian kernel over characteristic lengths
/// @param[in] CharacteristicLength The characteristic length
/// @param[in] LeftFeature The left feature
/// @param[in] RightFeature The right feature
/// @param[in] GaussianKernelMatrix Same as @code gaussian_kernel(CharacteristicLength, LeftFeature, RightFeature) @endcode, for saving time
/// @return The derivative of gaussian kernel over characteristic lengths
/// @details Denote @f$ x_{mni}=\frac{[\mathbf{x}_m]_i-[\mathbf{x}_n]_i}{l_i} @f$,
/// for the derivative of element @f$ K_{mn} @f$ over characteristic length @f$ l_j @f$ @n
/// @f[
/// \frac{\partial K_{mn}}{\partial l_j}=\mathrm{exp}\left(-\frac{1}{2}\sum_i x_{mni}^2\right)\frac{x_{mnj}^2}{l_j}
/// @f] @n
/// and thus the derivative matrix is constructed.
static std::array<Eigen::MatrixXd, PhaseDim> gaussian_derivative_over_char_length(
	const ClassicalPhaseVector& CharacteristicLength,
	const PhasePoints& LeftFeature,
	const PhasePoints& RightFeature,
	const Eigen::MatrixXd& GaussianKernelMatrix
)
{
	auto calculate_factor = [&CharacteristicLength, &LeftFeature, &RightFeature](const std::size_t iRow, const std::size_t iCol) -> ClassicalPhaseVector
	{
		const auto& diff = (LeftFeature.col(iRow) - RightFeature.col(iCol)).array() / CharacteristicLength.array();
		return diff.square() / CharacteristicLength.array();
	};
	const std::size_t Rows = LeftFeature.cols(), Cols = RightFeature.cols();
	const bool IsTrainingSet = LeftFeature.data() == RightFeature.data() && Rows == Cols;
	const auto indices = xt::arange(Cols);
	std::array<Eigen::MatrixXd, PhaseDim> result;
	result.fill(GaussianKernelMatrix);
	std::for_each(
		std::execution::par_unseq,
		indices.cbegin(),
		indices.cend(),
		[&result, &calculate_factor, Rows, IsTrainingSet](const std::size_t iCol) -> void
		{
			// column major
			if (IsTrainingSet)
			{
				// training set, the matrix is symmetric
				for (const std::size_t iRow : std::ranges::iota_view{iCol + 1, Rows})
				{
					const ClassicalPhaseVector& factor = calculate_factor(iRow, iCol);
					for (const std::size_t iDim : std::ranges::iota_view{0ul, PhaseDim})
					{
						result[iDim](iRow, iCol) *= factor[iDim];
					}
				}
			}
			else
			{
				// general rectangular
				for (const std::size_t iRow : std::ranges::iota_view{0ul, Rows})
				{
					const ClassicalPhaseVector& factor = calculate_factor(iRow, iCol);
					for (const std::size_t iDim : std::ranges::iota_view{0ul, PhaseDim})
					{
						result[iDim](iRow, iCol) *= factor[iDim];
					}
				}
			}
		}
	);
	if (IsTrainingSet)
	{
		// training set, the matrix is symmetric
		for (const std::size_t iDim : std::ranges::iota_view{0ul, PhaseDim})
		{
			// as it is symmetric, only lower part of the matrix are necessary to calculate
			result[iDim].diagonal() = Eigen::VectorXd::Zero(Rows);
			result[iDim] = result[iDim].selfadjointView<Eigen::Lower>();
		}
	}
	return result;
}

/// @brief To calculate the derivative over parameters
/// @param[in] KernelParams Deserialized parameters of gaussian kernel
/// @param[in] LeftFeature The left feature
/// @param[in] RightFeature The right feature
/// @param[in] KernelMatrix The kernel matrix
/// @return The derivative of kernel matrix over parameters
static KernelBase::ParameterArray<Eigen::MatrixXd> calculate_derivative(
	const KernelBase::KernelParameter& KernelParams,
	const PhasePoints& LeftFeature,
	const PhasePoints& RightFeature,
	const Eigen::MatrixXd& KernelMatrix
)
{
	const std::size_t Rows = LeftFeature.cols(), Cols = RightFeature.cols();
	[[maybe_unused]] const auto& [Magnitude, CharLength, Noise] = KernelParams;
	KernelBase::ParameterArray<Eigen::MatrixXd> result;
	result.fill(KernelMatrix);
	std::size_t iParam = 0;
	// first, magnitude
	result[iParam] *= 2.0 / Magnitude;
	iParam++;
	// second, Gaussian
	const std::array<Eigen::MatrixXd, PhaseDim> GaussianDerivOverCharLength =
		[&LeftFeature, &RightFeature, &KernelMatrix, &CharLength, Noise = Magnitude * Noise](void) -> std::array<Eigen::MatrixXd, PhaseDim>
	{
		if (LeftFeature.data() == RightFeature.data())
		{
			// training set, kernel need to subtract noise
			const Eigen::MatrixXd& GaussianMatrix = KernelMatrix - power<2>(Noise) * Eigen::MatrixXd::Identity(LeftFeature.cols(), LeftFeature.cols());
			return gaussian_derivative_over_char_length(CharLength, LeftFeature, RightFeature, GaussianMatrix);
		}
		else
		{
			// test set, no noise
			return gaussian_derivative_over_char_length(CharLength, LeftFeature, RightFeature, KernelMatrix);
		}
	}();
	for (const std::size_t iDim : std::ranges::iota_view{0ul, PhaseDim})
	{
		result[iDim + iParam] = GaussianDerivOverCharLength[iDim];
	}
	iParam += PhaseDim;
	// third, noise
	if (LeftFeature.data() == RightFeature.data())
	{
		result[iParam] = 2.0 * power<2>(Magnitude) * Noise * Eigen::MatrixXd::Identity(Rows, Cols);
	}
	else
	{
		result[iParam] = Eigen::MatrixXd::Zero(Rows, Cols);
	}
	iParam++;
	return result;
}

KernelBase::KernelBase(
	const KernelParameter& Parameter,
	const PhasePoints& left_feature,
	const PhasePoints& right_feature,
	const bool IsToCalculateDerivative
):
	// general calculations
	KernelParams(Parameter),
	LeftFeature(left_feature),
	RightFeature(right_feature),
	KernelMatrix(power<2>(std::get<0>(KernelParams)) * (gaussian_kernel(std::get<1>(KernelParams), LeftFeature, RightFeature) + power<2>(std::get<2>(KernelParams)) * delta_kernel(left_feature, right_feature))),
	// calculation only for derivative
	Derivatives(
		IsToCalculateDerivative
			? std::optional<ParameterArray<Eigen::MatrixXd>>(
				calculate_derivative(
					KernelParams,
					left_feature,
					right_feature,
					KernelMatrix
				)
			)
			: std::nullopt
	)
{
}

TrainingKernel::TrainingKernel(
	const ParameterVector& Parameter,
	const ElementTrainingSet& TrainingSet,
	const bool IsToCalculateError,
	const bool IsToCalculateAverage,
	const bool IsToCalculateDerivative
):
	// general calculations
	KernelBase(
		[&Parameter](void) -> KernelBase::KernelParameter
		{
			assert(Parameter.size() == NumTotalParameters);
			// have all the parameters
			KernelBase::KernelParameter result;
			auto& [magnitude, char_length, noise] = result;
			std::size_t iParam = 0;
			// first, magnitude
			magnitude = Parameter[iParam];
			iParam++;
			// second, Gaussian
			for (const std::size_t iDim : std::ranges::iota_view{0ul, PhaseDim})
			{
				char_length[iDim] = Parameter[iDim + iParam];
			}
			iParam += PhaseDim;
			// third, noise
			noise = Parameter[iParam];
			iParam++;
			return result;
		}(),
		std::get<0>(TrainingSet),
		std::get<0>(TrainingSet),
		IsToCalculateDerivative
	),
	Params(Parameter),
	RescaleFactor(KernelBase::RescaleMaximum / std::get<1>(TrainingSet).real().array().abs().maxCoeff()),
	Label(std::get<1>(TrainingSet).real() * RescaleFactor),
	DecompositionOfKernel(KernelBase::get_kernel()),
	Inverse(DecompositionOfKernel.solve(Eigen::MatrixXd::Identity(Label.size(), Label.size()))),
	InvLbl(DecompositionOfKernel.solve(Label)),
	// calculation only for error/averages (averages are only for training set set only)
	Error(IsToCalculateError ? std::optional<double>((InvLbl.array() / Inverse.diagonal().array()).square().sum()) : std::nullopt),
	Population(
		IsToCalculateAverage
			? std::optional<double>(
				[&KernelParams = KernelBase::get_formatted_parameters(), &v = InvLbl, RescaleFactor = RescaleFactor](void) -> double
				{
					static constexpr double GlobalFactor = power<Dim>(2.0 * std::numbers::pi);
					[[maybe_unused]] const auto& [Magnitude, CharLength, Noise] = KernelParams;
					return GlobalFactor * power<2>(Magnitude) * CharLength.prod() * v.sum() / RescaleFactor;
				}()
			)
			: std::nullopt
	),
	FirstOrderAverage(
		IsToCalculateAverage
			? std::optional<ClassicalPhaseVector>(
				[&feature = std::get<0>(TrainingSet),
				 &KernelParams = KernelBase::get_formatted_parameters(),
				 &v = InvLbl,
				 RescaleFactor = RescaleFactor](void) -> ClassicalPhaseVector
				{
					static constexpr double GlobalFactor = power<Dim>(2.0 * std::numbers::pi);
					[[maybe_unused]] const auto& [Magnitude, CharLength, Noise] = KernelParams;
					return GlobalFactor * power<2>(Magnitude) * CharLength.prod() * feature * v / RescaleFactor;
				}()
			)
			: std::nullopt
	),
	PurityAuxiliaryParams(construct_purity_auxiliary_kernel_params(KernelBase::get_formatted_parameters())),
	PurityAuxiliaryKernel(
		IsToCalculateAverage
			? std::optional<KernelBase>(
				std::in_place,
				PurityAuxiliaryParams.value(),
				std::get<0>(TrainingSet),
				std::get<0>(TrainingSet),
				IsToCalculateDerivative
			)
			: std::nullopt
	),
	Purity(
		IsToCalculateAverage
			? std::optional<double>(
				[&v = InvLbl, &K1 = PurityAuxiliaryKernel->get_kernel(), RescaleFactor = RescaleFactor](void) -> double
				{
					static constexpr double GlobalFactor = PurityFactor * power<Dim>(std::numbers::pi);
					return GlobalFactor * (v.transpose() * K1 * v).value() / power<2>(RescaleFactor);
				}()
			)
			: std::nullopt
	),
	// calculation only for derivative (items apart from derivative of kernel are for training set only)
	InverseDerivatives(
		IsToCalculateDerivative
			? std::optional<ParameterArray<Eigen::MatrixXd>>(
				[&KernelParams = KernelBase::get_formatted_parameters(),
				 &Inverse = Inverse,
				 &Derivatives = KernelBase::get_derivative()](void) -> ParameterArray<Eigen::MatrixXd>
				{
					[[maybe_unused]] const auto& [Magnitude, CharLength, Noise] = KernelParams;
					ParameterArray<Eigen::MatrixXd> result;
					result.fill(Inverse);
					// first, magnitude
					std::size_t iParam = 0;
					result[iParam] *= -2.0 / Magnitude;
					iParam++;
					// second, Gaussian
					for (const std::size_t iDim : std::ranges::iota_view{0ul, PhaseDim})
					{
						result[iDim + iParam] = -Inverse * Derivatives[iDim + iParam] * Inverse;
					}
					iParam += PhaseDim;
					// third, noise
					result[iParam] *= -2 * power<2>(Magnitude) * Noise * Inverse;
					iParam++;
					return result;
				}()
			)
			: std::nullopt
	),
	InvLblDerivatives(
		IsToCalculateDerivative
			? std::optional<ParameterArray<Eigen::VectorXd>>(
				[&y = Label, &InverseDerivatives = InverseDerivatives.value()](void) -> ParameterArray<Eigen::VectorXd>
				{
					ParameterArray<Eigen::VectorXd> result;
					for (const std::size_t iParam : std::ranges::iota_view{0ul, NumTotalParameters})
					{
						result[iParam] = InverseDerivatives[iParam] * y;
					}
					return result;
				}()
			)
			: std::nullopt
	),
	// calculation for both error/average and derivative
	ErrorDerivatives(
		IsToCalculateError && IsToCalculateDerivative
			? std::optional<ParameterArray<double>>(
				[&Inverse = Inverse,
				 &v = InvLbl,
				 &InverseDerivatives = InverseDerivatives.value(),
				 &vDerivatives = InvLblDerivatives.value()](void) -> ParameterArray<double>
				{
					ParameterArray<double> result;
					const auto& InvDiag = Inverse.diagonal().array();
					const auto& Diff = v.array() / InvDiag;
					for (const std::size_t iParam : std::ranges::iota_view{0ul, NumTotalParameters})
					{
						result[iParam] = 2.0 * (Diff / InvDiag * (vDerivatives[iParam].array() - Diff * InverseDerivatives[iParam].diagonal().array())).sum();
					}
					return result;
				}()
			)
			: std::nullopt
	),
	PopulationDerivatives(
		IsToCalculateAverage && IsToCalculateDerivative
			? std::optional<ParameterArray<double>>(
				[&KernelParams = KernelBase::get_formatted_parameters(),
				 &v = InvLbl,
				 &vDerivatives = InvLblDerivatives.value(),
				 RescaleFactor = RescaleFactor](void) -> ParameterArray<double>
				{
					static constexpr double GlobalFactor = power<Dim>(2.0 * std::numbers::pi);
					[[maybe_unused]] const auto& [Magnitude, CharLength, Noise] = KernelParams;
					const double ThisTimeFactor = GlobalFactor * power<2>(Magnitude) * CharLength.prod();
					ParameterArray<double> result;
					std::size_t iParam = 0;
					// first, magnitude
					result[iParam] = 0.0;
					iParam++;
					// second, Gaussian
					for (const std::size_t iDim : std::ranges::iota_view{0ul, PhaseDim})
					{
						result[iParam + iDim] = ThisTimeFactor * (v.sum() / CharLength[iDim] + vDerivatives[iParam + iDim].sum());
					}
					iParam += PhaseDim;
					// third, noise
					result[iParam] = ThisTimeFactor * vDerivatives[iParam].sum();
					iParam++;
					// divide over rescale factor
					for (double& d : result)
					{
						d /= RescaleFactor;
					}
					return result;
				}()
			)
			: std::nullopt
	),
	PurityDerivatives(
		IsToCalculateAverage && IsToCalculateDerivative
			? std::optional<ParameterArray<double>>(
				[&KernelParams = KernelBase::get_formatted_parameters(),
				 &v = InvLbl,
				 &K1 = PurityAuxiliaryKernel->get_kernel(),
				 &vDerivatives = InvLblDerivatives.value(),
				 &K1Derivatives = PurityAuxiliaryKernel->get_derivative(),
				 RescaleFactor = RescaleFactor](void) -> ParameterArray<double>
				{
					static constexpr double GlobalFactor = PurityFactor * power<Dim>(std::numbers::pi);
					[[maybe_unused]] const auto& [Magnitude, CharLength, Noise] = KernelParams;
					ParameterArray<double> result;
					std::size_t iParam = 0;
					// first, magnitude
					result[iParam] = 0.0;
					iParam++;
					// second, Gaussian
					for (const std::size_t iDim : std::ranges::iota_view{0ul, PhaseDim})
					{
						const Eigen::VectorXd& v_deriv = vDerivatives[iDim + iParam];
						// as the characteristic lengths is changed in K1,
						// the result needed to multiply with the amplitude factor, sqrt(2)
						result[iDim + iParam] =
							(v.transpose() * (K1 / CharLength[iDim] + std::numbers::sqrt2 * K1Derivatives[iDim + iParam]) * v).value()
							+ 2.0 * (v_deriv.transpose() * K1 * v).value();
						result[iDim + iParam] *= GlobalFactor;
					}
					iParam += PhaseDim;
					// third, noise
					result[iParam] = 2.0 * GlobalFactor * (vDerivatives[iParam].transpose() * K1 * v).value();
					iParam++;
					// divide over rescale factor
					for (double& d : result)
					{
						d /= power<2>(RescaleFactor);
					}
					return result;
				}()
			)
			: std::nullopt
	)
{
}

PredictiveKernel::PredictiveKernel(
	const PhasePoints& TestFeature,
	const TrainingKernel& kernel,
	const bool IsToCalculateDerivative,
	const std::optional<Eigen::VectorXd> TestLabel
):
	// general calculations
	KernelBase(
		kernel.get_formatted_parameters(),
		TestFeature,
		kernel.get_left_feature(),
		IsToCalculateDerivative
	),
	RescaleFactor(kernel.get_rescale_factor()),
	Prediction(KernelBase::get_kernel() * kernel.get_inverse_times_label()),
	ElementwiseVariance(
		[&TestFeature,
		 &KernelParams = KernelBase::get_formatted_parameters(),
		 &KernelMatrix = KernelBase::get_kernel(),
		 &TrainingKernelInverse = kernel.get_inverse()](void) -> Eigen::VectorXd
		{
			const std::size_t NumPoints = TestFeature.cols();
			const auto indices = xt::arange(NumPoints);
			Eigen::VectorXd result(NumPoints);
			std::for_each(
				std::execution::par_unseq,
				indices.cbegin(),
				indices.cend(),
				[&result, &KernelParams, &TestFeature, &KernelMatrix, &TrainingKernelInverse](const std::size_t iCol) -> void
				{
					const auto& col = ClassicalPhaseVector::Map(TestFeature.col(iCol).data());
					result(iCol) = KernelBase(KernelParams, col, col, false).get_kernel().value()
						- (KernelMatrix.row(iCol) * TrainingKernelInverse * KernelMatrix.row(iCol).transpose()).value();
				}
			);
			return result;
		}()
	),
	CutoffPrediction(Prediction.array() * cutoff_factor(Prediction, ElementwiseVariance).array() / RescaleFactor),
	// calculation only for error
	Label(TestLabel.has_value() ? std::optional<Eigen::VectorXd>(TestLabel.value() * RescaleFactor) : std::nullopt),
	Error(Label.has_value() ? std::optional<double>((Prediction - Label.value()).squaredNorm()) : std::nullopt),
	// calculation for both error and derivative
	ErrorDerivatives(
		Label.has_value() && IsToCalculateDerivative
			? std::optional<ParameterArray<double>>(
				[PredictionDifference = CutoffPrediction * RescaleFactor - Label.value(),
				 &KernelMatrix = KernelBase::get_kernel(),
				 &v = kernel.get_inverse_times_label(),
				 &Derivatives = KernelBase::get_derivative(),
				 &vDerivatives = kernel.get_inverse_times_label_derivative()](void) -> ParameterArray<double>
				{
					ParameterArray<double> result;
					for (const std::size_t iParam : std::ranges::iota_view{0ul, NumTotalParameters})
					{
						result[iParam] = 2.0 * PredictionDifference.dot(Derivatives[iParam] * v + KernelMatrix * vDerivatives[iParam]);
					}
					return result;
				}()
			)
			: std::nullopt
	)
{
}
