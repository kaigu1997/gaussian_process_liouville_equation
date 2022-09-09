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
		[&result, &LeftFeature, &RightFeature, Rows](std::size_t iCol) -> void
		{
			for (std::size_t iRow = 0; iRow < Rows; iRow++)
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
		[&result, &calculate_element, Rows, IsTrainingSet](std::size_t iCol) -> void
		{
			// column major
			if (IsTrainingSet)
			{
				// training set, the matrix is symmetric
				for (std::size_t iRow = iCol + 1; iRow < Rows; iRow++)
				{
					result(iRow, iCol) += calculate_element(iRow, iCol);
				}
			}
			else
			{
				// general rectangular
				for (std::size_t iRow = 0; iRow < Rows; iRow++)
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
		[&result, &calculate_factor, Rows, IsTrainingSet](std::size_t iCol) -> void
		{
			// column major
			if (IsTrainingSet)
			{
				// training set, the matrix is symmetric
				for (std::size_t iRow = iCol + 1; iRow < Rows; iRow++)
				{
					const ClassicalPhaseVector& factor = calculate_factor(iRow, iCol);
					for (std::size_t iDim = 0; iDim < PhaseDim; iDim++)
					{
						result[iDim](iRow, iCol) *= factor[iDim];
					}
				}
			}
			else
			{
				// general rectangular
				for (std::size_t iRow = 0; iRow < Rows; iRow++)
				{
					const ClassicalPhaseVector& factor = calculate_factor(iRow, iCol);
					for (std::size_t iDim = 0; iDim < PhaseDim; iDim++)
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
		for (std::size_t iDim = 0; iDim < PhaseDim; iDim++)
		{
			// as it is symmetric, only lower part of the matrix are necessary to calculate
			result[iDim].diagonal() = Eigen::VectorXd::Zero(Rows);
			result[iDim] = result[iDim].selfadjointView<Eigen::Lower>();
		}
	}
	return result;
}

/// @brief To calculate the derivatives over parameters
/// @param[in] KernelParams Deserialized parameters of gaussian kernel
/// @param[in] LeftFeature The left feature
/// @param[in] RightFeature The right feature
/// @param[in] KernelMatrix The kernel matrix
/// @return The derivative of kernel matrix over parameters
static Kernel::ParameterArray<Eigen::MatrixXd> calculate_derivatives(
	const Kernel::KernelParameter& KernelParams,
	const PhasePoints& LeftFeature,
	const PhasePoints& RightFeature,
	const Eigen::MatrixXd& KernelMatrix
)
{
	const std::size_t Rows = LeftFeature.cols(), Cols = RightFeature.cols();
	[[maybe_unused]] const auto& [Magnitude, CharLength, Noise] = KernelParams;
	Kernel::ParameterArray<Eigen::MatrixXd> result;
	result.fill(KernelMatrix);
	std::size_t iParam = 0;
	// first, magnitude
	result[iParam] *= 2.0 / Magnitude;
	iParam++;
	// second, Gaussian
	const std::array<Eigen::MatrixXd, PhaseDim> GaussianDerivOverCharLength = [&LeftFeature, &RightFeature, &KernelMatrix, &CharLength = CharLength, Noise = Magnitude * Noise](void) -> std::array<Eigen::MatrixXd, PhaseDim>
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
	for (std::size_t iDim = 0; iDim < PhaseDim; iDim++)
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

Kernel::Kernel(
	const ParameterVector& Parameter,
	const ElementTrainingSet& TrainingSet,
	const bool IsToCalculateError,
	const bool IsToCalculateAverage,
	const bool IsToCalculateDerivative
):
	// general calculations
	LeftFeature(std::get<0>(TrainingSet)),
	RightFeature(std::get<0>(TrainingSet)),
	Params(Parameter),
	KernelParams(
		[&Parameter](void) -> KernelParameter
		{
			assert(Parameter.size() == NumTotalParameters);
			// have all the parameters
			KernelParameter result;
			auto& [magnitude, char_length, noise] = result;
			std::size_t iParam = 0;
			// first, magnitude
			magnitude = Parameter[iParam];
			iParam++;
			// second, Gaussian
			for (std::size_t iDim = 0; iDim < PhaseDim; iDim++)
			{
				char_length[iDim] = Parameter[iDim + iParam];
			}
			iParam += PhaseDim;
			// third, noise
			noise = Parameter[iParam];
			iParam++;
			return result;
		}()
	),
	KernelMatrix(power<2>(std::get<0>(KernelParams)) * (gaussian_kernel(std::get<1>(KernelParams), LeftFeature, RightFeature) + power<2>(std::get<2>(KernelParams)) * Eigen::MatrixXd::Identity(LeftFeature.cols(), RightFeature.cols()))),
	// calculations only for training set
	Label(std::get<1>(TrainingSet).real()),
	DecompositionOfKernel(Eigen::LDLT<Eigen::MatrixXd>(KernelMatrix)),
	Inverse(DecompositionOfKernel->solve(Eigen::MatrixXd::Identity(KernelMatrix.rows(), KernelMatrix.cols()))),
	InvLbl(DecompositionOfKernel->solve(Label.value())),
	// calculation only for test set
	Prediction(std::nullopt),
	ElementwiseVariance(std::nullopt),
	CutoffPrediction(std::nullopt),
	// calculation only for error/averages (averages are only for training set set only)
	Error(IsToCalculateError ? std::optional<double>((InvLbl->array() / Inverse->diagonal().array()).square().sum()) : std::nullopt),
	Population(
		IsToCalculateAverage
			? std::optional<double>(
				[&KernelParams = KernelParams, &v = InvLbl.value()](void) -> double
				{
					static constexpr double GlobalFactor = power<Dim>(2.0 * M_PI);
					[[maybe_unused]] const auto& [Magnitude, CharLength, Noise] = KernelParams;
					return GlobalFactor * power<2>(Magnitude) * CharLength.prod() * v.sum();
				}()
			)
			: std::nullopt
	),
	FirstOrderAverage(
		IsToCalculateAverage
			? std::optional<ClassicalPhaseVector>(
				[&feature = LeftFeature, &KernelParams = KernelParams, &v = InvLbl.value()](void) -> ClassicalPhaseVector
				{
					static constexpr double GlobalFactor = power<Dim>(2.0 * M_PI);
					[[maybe_unused]] const auto& [Magnitude, CharLength, Noise] = KernelParams;
					return GlobalFactor * power<2>(Magnitude) * CharLength.prod() * feature * v;
				}()
			)
			: std::nullopt
	),
	PurityAuxiliaryParams(construct_purity_auxiliary_kernel_params(KernelParams)),
	PurityAuxiliaryKernel(
		IsToCalculateAverage
			? std::make_unique<Kernel>(
				PurityAuxiliaryParams.value(),
				LeftFeature,
				RightFeature,
				IsToCalculateDerivative
			)
			: nullptr
	),
	Purity(
		IsToCalculateAverage
			? std::optional<double>(
				[&v = InvLbl.value(), &K1 = PurityAuxiliaryKernel->KernelMatrix](void) -> double
				{
					static constexpr double GlobalFactor = PurityFactor * power<Dim>(M_PI);
					return GlobalFactor * (v.transpose() * K1 * v).value();
				}()
			)
			: std::nullopt
	),
	// calculation only for derivatives (items apart from derivative of kernel are for training set only)
	Derivatives(
		IsToCalculateDerivative
			? std::optional<ParameterArray<Eigen::MatrixXd>>(calculate_derivatives(
				KernelParams,
				std::get<0>(TrainingSet),
				std::get<0>(TrainingSet),
				KernelMatrix
			)
			)
			: std::nullopt
	),
	InverseDerivatives(
		IsToCalculateDerivative
			? std::optional<ParameterArray<Eigen::MatrixXd>>(
				[&KernelParams = KernelParams, &Inverse = Inverse.value(), &Derivatives = Derivatives.value()](void) -> ParameterArray<Eigen::MatrixXd>
				{
					[[maybe_unused]] const auto& [Magnitude, CharLength, Noise] = KernelParams;
					ParameterArray<Eigen::MatrixXd> result;
					result.fill(Inverse);
					// first, magnitude
					std::size_t iParam = 0;
					result[iParam] *= -2.0 / Magnitude;
					iParam++;
					// second, Gaussian
					for (std::size_t iDim = 0; iDim < PhaseDim; iDim++)
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
				[&y = Label.value(), &InverseDerivatives = InverseDerivatives.value()](void) -> ParameterArray<Eigen::VectorXd>
				{
					ParameterArray<Eigen::VectorXd> result;
					for (std::size_t iParam = 0; iParam < NumTotalParameters; iParam++)
					{
						result[iParam] = InverseDerivatives[iParam] * y;
					}
					return result;
				}()
			)
			: std::nullopt
	),
	// calculation for both error/average and derivatives
	ErrorDerivatives(
		IsToCalculateError && IsToCalculateDerivative
			? std::optional<ParameterArray<double>>(
				[&Inverse = Inverse.value(),
				 &v = InvLbl.value(),
				 &InverseDerivatives = InverseDerivatives.value(),
				 &vDerivatives = InvLblDerivatives.value()](void) -> ParameterArray<double>
				{
					ParameterArray<double> result;
					const auto& InvDiag = Inverse.diagonal().array();
					const auto& Diff = v.array() / InvDiag;
					for (std::size_t iParam = 0; iParam < NumTotalParameters; iParam++)
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
				[&KernelParams = KernelParams, &v = InvLbl.value(), &vDerivatives = InvLblDerivatives.value()](void) -> ParameterArray<double>
				{
					static constexpr double GlobalFactor = power<Dim>(2.0 * M_PI);
					[[maybe_unused]] const auto& [Magnitude, CharLength, Noise] = KernelParams;
					const double ThisTimeFactor = GlobalFactor * power<2>(Magnitude) * CharLength.prod();
					ParameterArray<double> result;
					std::size_t iParam = 0;
					// first, magnitude
					result[iParam] = 0.0;
					iParam++;
					// second, Gaussian
					for (std::size_t iDim = 0; iDim < PhaseDim; iDim++)
					{
						result[iParam] = ThisTimeFactor * (v.sum() / CharLength[iDim] + vDerivatives[iParam].sum());
						iParam++;
					}
					// third, noise
					result[iParam] = ThisTimeFactor * vDerivatives[iParam].sum();
					iParam++;
					return result;
				}()
			)
			: std::nullopt
	),
	PurityDerivatives(
		IsToCalculateAverage && IsToCalculateDerivative
			? std::optional<ParameterArray<double>>(
				[&KernelParams = KernelParams,
				 &v = InvLbl.value(),
				 &K1 = PurityAuxiliaryKernel->KernelMatrix,
				 &vDerivatives = InvLblDerivatives.value(),
				 &K1Derivatives = PurityAuxiliaryKernel->Derivatives.value()](void) -> ParameterArray<double>
				{
					static constexpr double GlobalFactor = PurityFactor * power<Dim>(M_PI);
					[[maybe_unused]] const auto& [Magnitude, CharLength, Noise] = KernelParams;
					ParameterArray<double> result;
					std::size_t iParam = 0;
					// first, magnitude
					result[iParam] = 0.0;
					iParam++;
					// second, Gaussian
					for (std::size_t iDim = 0; iDim < PhaseDim; iDim++)
					{
						const Eigen::VectorXd& v_deriv = vDerivatives[iDim + iParam];
						// as the characteristic lengths is changed in K1,
						// the result needed to multiply with the amplitude factor, sqrt(2)
						result[iDim + iParam] = GlobalFactor * (v.transpose() * (K1 / CharLength[iDim] + M_SQRT2 * K1Derivatives[iDim + iParam]) * v).value()
							+ GlobalFactor * 2.0 * (v_deriv.transpose() * K1 * v).value();
					}
					iParam += PhaseDim;
					// third, noise
					result[iParam] = 2.0 * GlobalFactor * (vDerivatives[iParam].transpose() * K1 * v).value();
					iParam++;
					return result;
				}()
			)
			: std::nullopt
	)
{
}

Kernel::Kernel(
	const KernelParameter& Parameter,
	const PhasePoints& left_feature,
	const PhasePoints& right_feature,
	const bool IsToCalculateDerivative
):
	// general calculations
	LeftFeature(left_feature),
	RightFeature(right_feature),
	KernelParams(Parameter),
	KernelMatrix(power<2>(std::get<0>(KernelParams)) * (gaussian_kernel(std::get<1>(KernelParams), LeftFeature, RightFeature) + power<2>(std::get<2>(KernelParams)) * delta_kernel(left_feature, right_feature))),
	// calculations only for training set
	// calculation only for test set
	// calculation only for error/averages (averages are only for training set set only)
	// calculation only for derivatives (items apart from derivative of kernel are for training set only)
	Derivatives(
		IsToCalculateDerivative
			? std::optional<ParameterArray<Eigen::MatrixXd>>(calculate_derivatives(
				KernelParams,
				LeftFeature,
				RightFeature,
				KernelMatrix
			))
			: std::nullopt
	)
// calculation for both average and derivatives
{
}

Kernel::Kernel(
	const PhasePoints& TestFeature,
	const Kernel& TrainingKernel,
	const bool IsToCalculateDerivative,
	const std::optional<Eigen::VectorXd> TestLabel
):
	// general calculations
	LeftFeature(TestFeature),
	RightFeature(TrainingKernel.LeftFeature),
	Params(TrainingKernel.Params),
	KernelParams(TrainingKernel.KernelParams),
	KernelMatrix(power<2>(std::get<0>(KernelParams)) * (gaussian_kernel(std::get<1>(KernelParams), LeftFeature, RightFeature) + power<2>(std::get<2>(KernelParams)) * delta_kernel(LeftFeature, RightFeature))),
	// calculations only for training set
	Label(TestLabel),
	// calculation only for test set
	Prediction(KernelMatrix * TrainingKernel.InvLbl.value()),
	ElementwiseVariance(
		[&TestFeature,
		 &KernelParams = KernelParams,
		 &KernelMatrix = KernelMatrix,
		 &TrainingKernelInverse = TrainingKernel.Inverse.value()](void) -> Eigen::VectorXd
		{
			const std::size_t NumPoints = TestFeature.cols();
			const auto indices = xt::arange(NumPoints);
			Eigen::VectorXd result(NumPoints);
			std::for_each(
				std::execution::par_unseq,
				indices.cbegin(),
				indices.cend(),
				[&result, &KernelParams, &TestFeature, &KernelMatrix, &TrainingKernelInverse](std::size_t iCol) -> void
				{
					const auto& col = PhasePoints::Map(TestFeature.col(iCol).data(), PhaseDim, 1);
					result(iCol) = Kernel(KernelParams, col, col, false).KernelMatrix.value()
						- (KernelMatrix.row(iCol) * TrainingKernelInverse * KernelMatrix.row(iCol).transpose()).value();
				}
			);
			return result;
		}()
	),
	CutoffPrediction(Prediction.value().array() * cutoff_factor(Prediction.value(), ElementwiseVariance.value()).array()),
	// calculation only for error/averages (averages are only for training set set only)
	Error(Label.has_value() ? std::optional<double>((Prediction.value() - Label.value()).squaredNorm()) : std::nullopt),
	// calculation only for derivatives (items apart from derivative of kernel are for training set only)
	Derivatives(
		IsToCalculateDerivative
			? std::optional<ParameterArray<Eigen::MatrixXd>>(calculate_derivatives(
				KernelParams,
				LeftFeature,
				RightFeature,
				KernelMatrix
			))
			: std::nullopt
	),
	// calculation for both average and derivatives
	ErrorDerivatives(
		Label.has_value() && IsToCalculateDerivative
			? std::optional<ParameterArray<double>>(
				[PredictionDifference = Prediction.value().real() - Label.value(),
				 &KernelMatrix = KernelMatrix,
				 &v = TrainingKernel.InvLbl.value(),
				 &Derivatives = Derivatives.value(),
				 &vDerivatives = TrainingKernel.InvLblDerivatives.value()](void) -> ParameterArray<double>
				{
					ParameterArray<double> result;
					for (std::size_t iParam = 0; iParam < NumTotalParameters; iParam++)
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
