/// @file complex_kernel.cpp
/// @brief Implementation of complex_kernel.h

#include "stdafx.h"

#include "complex_kernel.h"

#include "kernel.h"

using namespace std::literals::complex_literals;

/// @brief To calculate the parameters of purity auxiliary mixed kernel (rc or ic)
/// @param[in] FirstParams The parameters for one of the mixed kernel
/// @param[in] SecondParams The parameters for another kernel. The order does not matter
/// @return Parameters purity auxiliary kernel
static inline Kernel::KernelParameter construct_purity_auxiliary_mixed_kernel_params(
	const Kernel::KernelParameter& FirstParams,
	const Kernel::KernelParameter& SecondParams
)
{
	[[maybe_unused]] const auto& [FirstMagnitude, FirstCharLength, FirstNoise] = FirstParams;
	[[maybe_unused]] const auto& [SecondMagnitude, SecondCharLength, SecondNoise] = SecondParams;
	Kernel::KernelParameter result;
	auto& [magnitude, char_length, noise] = result;
	magnitude = FirstMagnitude * SecondMagnitude / std::sqrt(std::sqrt((0.5 * (FirstCharLength.array().inverse().square() + SecondCharLength.array().inverse().square())).prod()));
	char_length = (FirstCharLength.array().square() + SecondCharLength.array().square()).sqrt();
	noise = 0.0;
	return result;
}

/// @brief To get the derivatives of covariance matrix over parameters
/// @param[in] KernelParams Deserialized parameters of complex kernel
/// @param[in] LeftFeature The left feature
/// @param[in] RightFeature The right feature
/// @param[in] KernelMatrix The kernel matrix
/// @param[in] RealDerivatives The derivatives of kernel for real part over real parameters
/// @param[in] ImagDerivatives The derivatives of kernel for imaginary part over imaginary parameters
/// @return The derivative of kernel matrix over parameters
static ComplexKernel::ParameterArray<Eigen::MatrixXd> calculate_derivatives(
	const ComplexKernel::KernelParameter& KernelParams,
	const PhasePoints& LeftFeature,
	const PhasePoints& RightFeature,
	const Eigen::MatrixXd& KernelMatrix,
	const Kernel::ParameterArray<Eigen::MatrixXd>& RealDerivatives,
	const Kernel::ParameterArray<Eigen::MatrixXd>& ImagDerivatives
)
{
	const std::size_t Rows = LeftFeature.cols(), Cols = RightFeature.cols();
	[[maybe_unused]] const auto& [Magnitude, Params, Noise] = KernelParams;
	ComplexKernel::ParameterArray<Eigen::MatrixXd> result;
	std::size_t iParam = 0;
	// first, magnitude
	result[iParam] = 2.0 / Magnitude * KernelMatrix;
	iParam++;
	// then, real kernel
	for (std::size_t iRealParam = 0; iRealParam < Kernel::NumTotalParameters - 1; iRealParam++)
	{
		result[iParam + iRealParam] = RealDerivatives[iRealParam];
	}
	iParam += Kernel::NumTotalParameters - 1;
	// then, imaginary kernel
	for (std::size_t iImagParam = 0; iImagParam < Kernel::NumTotalParameters - 1; iImagParam++)
	{
		result[iParam + iImagParam] = ImagDerivatives[iImagParam];
	}
	iParam += Kernel::NumTotalParameters - 1;
	// finally, noise
	if (LeftFeature.data() == RightFeature.data())
	{
		result[iParam] = 2.0 * Noise * Eigen::MatrixXd::Identity(Rows, Cols);
	}
	else
	{
		result[iParam] = Eigen::MatrixXd::Zero(Rows, Cols);
	}
	iParam++;
	return result;
}

/// @brief To get the derivatives of pseudo-covariance matrix over parameters
/// @param[in] KernelParams Deserialized parameters of complex kernel
/// @param[in] LeftFeature The left feature
/// @param[in] RightFeature The right feature
/// @param[in] PseudoKernelMatrix The pseudo kernel matrix
/// @param[in] CorrKernelMatrix The kernel matrix for correlation kernel
/// @param[in] RealParams Parameters for the real kernel
/// @param[in] ImagParams Parameters for the imaginary kernel
/// @param[in] CorrParams Parameters for the correlation kernel
/// @param[in] RealDerivatives The derivatives of real kernel over real parameters
/// @param[in] ImagDerivatives The derivatives of imaginary kernel over imaginary parameters
/// @param[in] CorrDerivatives The derivatives of correlation kernel over correlation parameters
/// @return The derivative of pseudo-kernel matrix over parameters
static ComplexKernel::ParameterArray<Eigen::MatrixXcd> calculate_pseudo_derivatives(
	const ComplexKernel::KernelParameter& KernelParams,
	const PhasePoints& LeftFeature,
	const PhasePoints& RightFeature,
	const Eigen::MatrixXcd& PseudoKernelMatrix,
	const Eigen::MatrixXd& CorrKernelMatrix,
	const Kernel::KernelParameter& RealParams,
	const Kernel::KernelParameter& ImagParams,
	const Kernel::KernelParameter& CorrParams,
	const Kernel::ParameterArray<Eigen::MatrixXd>& RealDerivatives,
	const Kernel::ParameterArray<Eigen::MatrixXd>& ImagDerivatives,
	const Kernel::ParameterArray<Eigen::MatrixXd>& CorrDerivatives
)
{
	const std::size_t Rows = LeftFeature.cols(), Cols = RightFeature.cols();
	[[maybe_unused]] const auto& [Magnitude, Params, Noise] = KernelParams; // only magnitude is used
	[[maybe_unused]] const auto& [CorrMagnitude, CorrCharLength, CorrNoise] = CorrParams;
	ComplexKernel::ParameterArray<Eigen::MatrixXcd> result;
	std::size_t iParam = 0;
	// first, magnitude
	result[iParam] = 2.0 / Magnitude * PseudoKernelMatrix;
	iParam++;
	// then, real kernel
	{
		const auto& [RealMagnitude, RealCharLength, RealNoise] = RealParams;
		std::size_t iRealParam = 0;
		// real magnitude
		result[iParam + iRealParam] = RealDerivatives[iRealParam] + 2.0i / RealMagnitude * CorrKernelMatrix;
		iRealParam++;
		// real characteristic lengths
		for (std::size_t iRealDim = 0; iRealDim < PhaseDim; iRealDim++)
		{
			result[iParam + iRealParam + iRealDim] = RealDerivatives[iRealParam + iRealDim]
				+ 2.0i * (1.0 / RealCharLength[iRealDim] - RealCharLength[iRealDim] / power<2>(CorrCharLength[iRealDim])) * CorrKernelMatrix
				+ 1.0i * RealCharLength[iRealDim] / CorrCharLength[iRealDim] * CorrDerivatives[iRealParam + iRealDim];
		}
	}
	iParam += Kernel::NumTotalParameters - 1;
	// then, imaginary kernel
	{
		const auto& [ImagMagnitude, ImagCharLength, ImagNoise] = ImagParams;
		std::size_t iImagParam = 0;
		// imaginary magnitude
		result[iParam + iImagParam] = -ImagDerivatives[iImagParam] + 2.0i / ImagMagnitude * CorrKernelMatrix;
		iImagParam++;
		// imaginary characteristic lengths
		for (std::size_t iImagDim = 0; iImagDim < PhaseDim; iImagDim++)
		{
			result[iParam + iImagParam + iImagDim] = -ImagDerivatives[iImagParam + iImagDim]
				+ 2.0i * (1.0 / ImagCharLength[iImagDim] - ImagCharLength[iImagDim] / power<2>(CorrCharLength[iImagDim])) * CorrKernelMatrix
				+ 1.0i * ImagCharLength[iImagDim] / CorrCharLength[iImagDim] * CorrDerivatives[iImagParam + iImagDim];
		}
	}
	iParam += Kernel::NumTotalParameters - 1;
	// finally, noise
	result[iParam] = Eigen::MatrixXd::Zero(Rows, Cols);
	iParam++;
	return result;
}

ComplexKernel::ComplexKernel(
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
			KernelParameter result;
			auto& [magnitude, kernels_param, noise] = result;
			std::size_t iParam = 0;
			// first, magnitude
			magnitude = Parameter[iParam];
			iParam++;
			// going over all kernels: real, imaginary, and correlation
			for (auto& [magnitude, char_length] : kernels_param)
			{
				// first, magnitude
				magnitude = Parameter[iParam];
				iParam++;
				// second, Gaussian
				for (std::size_t iDim = 0; iDim < PhaseDim; iDim++)
				{
					char_length[iDim] = Parameter[iDim + iParam];
				}
				iParam += PhaseDim;
			}
			// finally, noise
			noise = Parameter[iParam];
			iParam++;
			return result;
		}()
	),
	RealParams(std::tuple_cat(std::get<1>(KernelParams)[0], std::make_tuple(0.0))),
	ImagParams(std::tuple_cat(std::get<1>(KernelParams)[1], std::make_tuple(0.0))),
	CorrParams(
		[&RealParams = RealParams, &ImagParams = ImagParams](void) -> Kernel::KernelParameter
		{
			[[maybe_unused]] const auto& [RealMagnitude, RealCharLength, RealNoise] = RealParams;
			[[maybe_unused]] const auto& [ImagMagnitude, ImagCharLength, ImagNoise] = ImagParams;
			const auto& SquareSum = RealCharLength.array().square() + ImagCharLength.array().square();
			Kernel::KernelParameter result;
			auto& [corr_magnitude, corr_char_length, corr_noise] = result;
			corr_magnitude = std::sqrt(RealMagnitude * ImagMagnitude * (2.0 * RealCharLength.array() * ImagCharLength.array() / SquareSum).prod());
			corr_char_length = (SquareSum / 2.0).sqrt();
			corr_noise = 0.0;
			return result;
		}()
	),
	RealKernel(RealParams, LeftFeature, RightFeature, IsToCalculateDerivative),
	ImagKernel(ImagParams, LeftFeature, RightFeature, IsToCalculateDerivative),
	CorrKernel(CorrParams, LeftFeature, RightFeature, IsToCalculateDerivative),
	KernelMatrix(power<2>(std::get<0>(KernelParams)) * (RealKernel.get_kernel() + ImagKernel.get_kernel() + power<2>(std::get<2>(KernelParams)) * Eigen::MatrixXd::Identity(LeftFeature.cols(), RightFeature.cols()))),
	PseudoKernelMatrix(power<2>(std::get<0>(KernelParams)) * (RealKernel.get_kernel() - ImagKernel.get_kernel() + 2.0i * CorrKernel.get_kernel())),
	// calculations only for training set
	Label(std::get<1>(TrainingSet)),
	DecompositionOfKernel(Eigen::LDLT<Eigen::MatrixXcd>(KernelMatrix)),
	KernelInversePseudoConjugate(DecompositionOfKernel->solve(PseudoKernelMatrix.conjugate())),
	UpperLeftBlockOfAugmentedKernelInverse((KernelMatrix - PseudoKernelMatrix * KernelInversePseudoConjugate.value()).selfadjointView<Eigen::Lower>().ldlt().solve(Eigen::MatrixXcd::Identity(LeftFeature.cols(), RightFeature.cols())).selfadjointView<Eigen::Lower>()),
	LowerLeftBlockOfAugmentedKernelInverse(-KernelInversePseudoConjugate.value() * UpperLeftBlockOfAugmentedKernelInverse.value()),
	UpperPartOfAugmentedInverseLabel(UpperLeftBlockOfAugmentedKernelInverse.value() * Label.value() + (LowerLeftBlockOfAugmentedKernelInverse.value() * Label.value()).conjugate()),
	// calculation only for test set
	// calculation only for error/averages (purity is only for training set only)
	Error(
		IsToCalculateError
			? std::optional<double>(
				[&P = UpperLeftBlockOfAugmentedKernelInverse.value(),
				 &Q = LowerLeftBlockOfAugmentedKernelInverse.value(),
				 &v = UpperPartOfAugmentedInverseLabel.value()](void) -> double
				{
					const auto& P_diag = P.diagonal().array();
					const auto& Q_diag = Q.diagonal().array();
					const auto& P_diag_square = P_diag.real().square();
					const auto& Q_diag_square = Q_diag.abs2();
					const auto& diff = (P_diag * v.array() - (Q_diag * v.array()).conjugate()) / (P_diag_square - Q_diag_square);
					return diff.abs2().sum();
				}()
			)
			: std::nullopt
	),
	PurityAuxiliaryRealParams(construct_purity_auxiliary_kernel_params(RealParams)),
	PurityAuxiliaryImagParams(construct_purity_auxiliary_kernel_params(ImagParams)),
	PurityAuxiliaryCorrParams(construct_purity_auxiliary_kernel_params(CorrParams)),
	PurityAuxiliaryRealCorrParams(construct_purity_auxiliary_mixed_kernel_params(RealParams, CorrParams)),
	PurityAuxiliaryImagCorrParams(construct_purity_auxiliary_mixed_kernel_params(ImagParams, CorrParams)),
	PurityAuxiliaryRealKernel(
		IsToCalculateAverage
			? std::optional<Kernel>(
				std::in_place,
				PurityAuxiliaryRealParams.value(),
				LeftFeature,
				RightFeature,
				IsToCalculateDerivative
			)
			: std::nullopt
	),
	PurityAuxiliaryImagKernel(
		IsToCalculateAverage
			? std::optional<Kernel>(
				std::in_place,
				PurityAuxiliaryImagParams.value(),
				LeftFeature,
				RightFeature,
				IsToCalculateDerivative
			)
			: std::nullopt
	),
	PurityAuxiliaryCorrKernel(
		IsToCalculateAverage
			? std::optional<Kernel>(
				std::in_place,
				PurityAuxiliaryCorrParams.value(),
				LeftFeature,
				RightFeature,
				IsToCalculateDerivative
			)
			: std::nullopt
	),
	PurityAuxiliaryRealCorrKernel(
		IsToCalculateAverage
			? std::optional<Kernel>(
				std::in_place,
				PurityAuxiliaryRealCorrParams.value(),
				LeftFeature,
				RightFeature,
				IsToCalculateDerivative
			)
			: std::nullopt
	),
	PurityAuxiliaryImagCorrKernel(
		IsToCalculateAverage
			? std::optional<Kernel>(
				std::in_place,
				PurityAuxiliaryImagCorrParams.value(),
				LeftFeature,
				RightFeature,
				IsToCalculateDerivative
			)
			: std::nullopt
	),
	Purity(
		IsToCalculateAverage
			? std::optional<double>(
				[&KernelParams = KernelParams,
				 &v = UpperPartOfAugmentedInverseLabel.value(),
				 &KRprime = PurityAuxiliaryRealKernel->get_kernel(),
				 &KIprime = PurityAuxiliaryImagKernel->get_kernel(),
				 &KCprime = PurityAuxiliaryCorrKernel->get_kernel(),
				 &KRC = PurityAuxiliaryRealCorrKernel->get_kernel(),
				 &KIC = PurityAuxiliaryImagCorrKernel->get_kernel()](void) -> double
				{
					static constexpr double GlobalFactor = PurityFactor * 2.0 * power<Dim>(M_PI);
					const double ThisTimeFactor = GlobalFactor * power<4>(std::get<0>(KernelParams));
					const Eigen::MatrixXd K1 = KRprime + KIprime + 2.0 * KCprime;
					const Eigen::MatrixXcd K2 = KRprime - KIprime - 2.0i * (KRC + KIC);
					return ThisTimeFactor * ((v.adjoint() * K1 * v).value().real() + (v.transpose() * K2 * v).value().real());
				}()
			)
			: std::nullopt
	),
	// calculation only for derivatives (items apart from derivative of (pseudo-)kernel are for training set only)
	Derivatives(
		IsToCalculateDerivative
			? std::optional<ParameterArray<Eigen::MatrixXd>>(calculate_derivatives(
				KernelParams,
				std::get<0>(TrainingSet),
				std::get<0>(TrainingSet),
				KernelMatrix,
				RealKernel.get_derivatives(),
				ImagKernel.get_derivatives()
			))
			: std::nullopt
	),
	PseudoDerivatives(
		IsToCalculateDerivative
			? std::optional<ParameterArray<Eigen::MatrixXcd>>(calculate_pseudo_derivatives(
				KernelParams,
				std::get<0>(TrainingSet),
				std::get<0>(TrainingSet),
				PseudoKernelMatrix,
				CorrKernel.get_kernel(),
				RealParams,
				ImagParams,
				CorrParams,
				RealKernel.get_derivatives(),
				ImagKernel.get_derivatives(),
				CorrKernel.get_derivatives()
			))
			: std::nullopt
	),
	UpperLeftAugmentedInverseDerivatives(
		IsToCalculateDerivative
			? std::optional<ParameterArray<Eigen::MatrixXcd>>(
				[&P = UpperLeftBlockOfAugmentedKernelInverse.value(),
				 &Q = LowerLeftBlockOfAugmentedKernelInverse.value(),
				 &Derivatives = Derivatives.value(),
				 &PseudoDerivatives = PseudoDerivatives.value()](void) -> ParameterArray<Eigen::MatrixXcd>
				{
					ParameterArray<Eigen::MatrixXcd> result;
					for (std::size_t iParam = 0; iParam < NumTotalParameters; iParam++)
					{
						result[iParam] = (-(P * Derivatives[iParam] * P + Q.adjoint() * Derivatives[iParam] * Q + P * PseudoDerivatives[iParam] * Q + Q.adjoint() * PseudoDerivatives[iParam].conjugate() * P)).selfadjointView<Eigen::Lower>();
					}
					return result;
				}()
			)
			: std::nullopt
	),
	LowerLeftAugmentedInverseDerivatives(
		IsToCalculateDerivative
			? std::optional<ParameterArray<Eigen::MatrixXcd>>(
				[&Cholesky = DecompositionOfKernel.value(),
				 &KKTildeStar = KernelInversePseudoConjugate.value(),
				 &Derivatives = Derivatives.value(),
				 &PseudoDerivatives = PseudoDerivatives.value(),
				 &P = UpperLeftBlockOfAugmentedKernelInverse.value(),
				 &Q = LowerLeftBlockOfAugmentedKernelInverse.value(),
				 &PDerivatives = UpperLeftAugmentedInverseDerivatives.value()](void) -> ParameterArray<Eigen::MatrixXcd>
				{
					ParameterArray<Eigen::MatrixXcd> result;
					for (std::size_t iParam = 0; iParam < NumTotalParameters; iParam++)
					{
						const auto& K_deriv = Derivatives[iParam];
						const auto& K_tilde_deriv_star = PseudoDerivatives[iParam].conjugate();
						const auto& P_deriv = PDerivatives[iParam];
						result[iParam] = -Cholesky.solve(K_deriv * Q + K_tilde_deriv_star * P) - KKTildeStar * P_deriv;
					}
					return result;
				}()
			)
			: std::nullopt
	),
	UpperAugmentedInvLblDerivatives(
		IsToCalculateDerivative
			? std::optional<ParameterArray<Eigen::VectorXcd>>(
				[&y = Label.value(),
				 &PDerivatives = UpperLeftAugmentedInverseDerivatives.value(),
				 &QDerivatives = LowerLeftAugmentedInverseDerivatives.value()](void) -> ParameterArray<Eigen::VectorXcd>
				{
					ParameterArray<Eigen::VectorXcd> result;
					for (std::size_t iParam = 0; iParam < NumTotalParameters; iParam++)
					{
						result[iParam] = PDerivatives[iParam] * y + (QDerivatives[iParam] * y).conjugate();
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
				[&P = UpperLeftBlockOfAugmentedKernelInverse.value(),
				 &Q = LowerLeftBlockOfAugmentedKernelInverse.value(),
				 &v = UpperPartOfAugmentedInverseLabel.value(),
				 &PDerivatives = UpperLeftAugmentedInverseDerivatives.value(),
				 &QDerivatives = LowerLeftAugmentedInverseDerivatives.value(),
				 &vDerivatives = UpperAugmentedInvLblDerivatives.value()](void) -> std::optional<ParameterArray<double>>
				{
					ParameterArray<double> result;
					const auto& P_diag = P.diagonal().array();
					const auto& Q_diag = Q.diagonal().array();
					const auto& P_diag_square = P_diag.real().square();
					const auto& Q_diag_square = Q_diag.abs2();
					const auto& square_diff = P_diag_square - Q_diag_square;
					const auto& diff = (P_diag * v.array() - (Q_diag * v.array()).conjugate()) / square_diff;
					for (std::size_t iParam = 0; iParam < NumTotalParameters; iParam++)
					{
						const auto& P_diag_deriv = PDerivatives[iParam].diagonal().array();
						const auto& Q_diag_deriv = QDerivatives[iParam].diagonal().array();
						const auto& v_deriv = vDerivatives[iParam].array();
						const auto& numerator_deriv = diff.conjugate() * (P_diag_deriv * v.array() + P_diag * v_deriv - (Q_diag_deriv * v.array() + Q_diag * v_deriv).conjugate());
						const auto& denominator_deriv = -2.0 * diff.abs2() * (P_diag * P_diag_deriv - (Q_diag.conjugate() * Q_diag_deriv).real().array());
						result[iParam] = 2.0 * ((numerator_deriv + denominator_deriv) / square_diff).real().sum();
					}
					return result;
				}()
			)
			: std::nullopt
	),
	PurityDerivatives(
		IsToCalculateAverage && IsToCalculateDerivative
			? std::optional<ParameterArray<double>>(
				[&RealParams = RealParams,
				 &ImagParams = ImagParams,
				 &CorrParams = CorrParams,
				 &PurityAuxiliaryRealCorrParams = PurityAuxiliaryRealCorrParams.value(),
				 &PurityAuxiliaryImagCorrParams = PurityAuxiliaryImagCorrParams.value(),
				 &v = UpperPartOfAugmentedInverseLabel.value(),
				 &KRprime = PurityAuxiliaryRealKernel->get_kernel(),
				 &KIprime = PurityAuxiliaryImagKernel->get_kernel(),
				 &KCprime = PurityAuxiliaryCorrKernel->get_kernel(),
				 &KRC = PurityAuxiliaryRealCorrKernel->get_kernel(),
				 &KIC = PurityAuxiliaryImagCorrKernel->get_kernel(),
				 &vDerivatives = UpperAugmentedInvLblDerivatives.value(),
				 &KRprimeDerivatives = PurityAuxiliaryRealKernel->get_derivatives(),
				 &KIprimeDerivatives = PurityAuxiliaryImagKernel->get_derivatives(),
				 &KCprimeDerivatives = PurityAuxiliaryCorrKernel->get_derivatives(),
				 &KRCDerivatives = PurityAuxiliaryRealCorrKernel->get_derivatives(),
				 &KICDerivatives = PurityAuxiliaryImagCorrKernel->get_derivatives()](void) -> std::optional<ParameterArray<double>>
				{
					static constexpr double GlobalFactor = PurityFactor * 2.0 * power<Dim>(M_PI);
					// parameters
					[[maybe_unused]] const auto& [RealMagnitude, RealCharLength, RealNoise] = RealParams;
					[[maybe_unused]] const auto& [ImagMagnitude, ImagCharLength, ImagNoise] = ImagParams;
					[[maybe_unused]] const auto& [CorrMagnitude, CorrCharLength, CorrNoise] = CorrParams;
					[[maybe_unused]] const auto& [RealCorrMagnitude, RealCorrCharLength, RealCorrNoise] = PurityAuxiliaryRealCorrParams;
					[[maybe_unused]] const auto& [ImagCorrMagnitude, ImagCorrCharLength, ImagCorrNoise] = PurityAuxiliaryImagCorrParams;
					// auxiliary variables
					const Eigen::MatrixXd Sum2RC1IC = 2.0 * KRC + KIC, Sum1RC2IC = KRC + 2.0 * KIC;
					const ClassicalPhaseVector
						ROverC = RealCharLength.array() / CorrCharLength.array(),
						IOverC = ImagCharLength.array() / CorrCharLength.array(),
						ROverCSquare = RealCharLength.array() / CorrCharLength.array().square(),
						IOverCSquare = ImagCharLength.array() / CorrCharLength.array().square(),
						ROverRC = RealCharLength.array() / RealCorrCharLength.array(),
						ROverIC = RealCharLength.array() / ImagCorrCharLength.array(),
						IOverRC = ImagCharLength.array() / RealCorrCharLength.array(),
						IOverIC = ImagCharLength.array() / ImagCorrCharLength.array();
					const std::size_t size = v.size();
					ParameterArray<Eigen::MatrixXd> KRprime_deriv, KIprime_deriv, KCprime_deriv, KRC_deriv, KIC_deriv;
					{
						std::size_t iParam = 0;
						// first, magnitude
						KRprime_deriv[iParam] = KIprime_deriv[iParam] = KCprime_deriv[iParam] = KRC_deriv[iParam] = KIC_deriv[iParam] = Eigen::MatrixXd::Zero(size, size);
						iParam++;
						// real kernel
						{
							// magnitude
							KRprime_deriv[iParam] = 4.0 / RealMagnitude * KRprime;
							KIprime_deriv[iParam] = Eigen::MatrixXd::Zero(size, size);
							KCprime_deriv[iParam] = 2.0 / RealMagnitude * KCprime;
							KRC_deriv[iParam] = 3.0 / RealMagnitude * KRC;
							KIC_deriv[iParam] = 1.0 / RealMagnitude * KIC;
							iParam++;
							// characteristic lengths
							for (std::size_t iDim = 0; iDim < PhaseDim; iDim++)
							{
								KRprime_deriv[iDim + iParam] = KRprime / RealCharLength[iDim] + M_SQRT2 * KRprimeDerivatives[iDim + iParam];
								KIprime_deriv[iDim + iParam] = Eigen::MatrixXd::Zero(size, size);
								KCprime_deriv[iDim + iParam] = (2.0 / RealCharLength[iDim] - 3.0 * ROverCSquare[iDim] / 2.0) * KCprime
									+ M_SQRT1_2 * ROverC[iDim] * KCprimeDerivatives[iDim + iParam];
								KRC_deriv[iDim + iParam] = (2.0 / RealCharLength[iDim] - ROverCSquare[iDim] / 2.0) * KRC
									+ 1.5 * ROverRC[iDim] * (KRCDerivatives[iDim + iParam] - KRC / RealCorrCharLength[iDim]);
								KIC_deriv[iDim + iParam] = (1.0 / RealCharLength[iDim] - ROverCSquare[iDim] / 2.0) * KIC
									+ ROverIC[iDim] / 2.0 * (KICDerivatives[iDim + iParam] - KIC / ImagCorrCharLength[iDim]);
							}
							iParam += PhaseDim;
						}
						// imaginary kernel
						{
							// magnitude
							KRprime_deriv[iParam] = Eigen::MatrixXd::Zero(size, size);
							KIprime_deriv[iParam] = 4.0 / ImagMagnitude * KIprime;
							KCprime_deriv[iParam] = 2.0 / ImagMagnitude * KCprime;
							KRC_deriv[iParam] = 1.0 / ImagMagnitude * KRC;
							KIC_deriv[iParam] = 3.0 / ImagMagnitude * KIC;
							iParam++;
							// characteristic lengths
							for (std::size_t iDim = 0; iDim < PhaseDim; iDim++)
							{
								KRprime_deriv[iDim + iParam] = Eigen::MatrixXd::Zero(size, size);
								KIprime_deriv[iDim + iParam] = KIprime / ImagCharLength[iDim] + M_SQRT2 * KIprimeDerivatives[iDim + iParam - (Kernel::NumTotalParameters - 1)];
								KCprime_deriv[iDim + iParam] = (2.0 / ImagCharLength[iDim] - 3.0 * IOverCSquare[iDim] / 2.0) * KCprime
									+ M_SQRT1_2 * IOverC[iDim] * KCprimeDerivatives[iDim + iParam - (Kernel::NumTotalParameters - 1)];
								KRC_deriv[iDim + iParam] = (1.0 / ImagCharLength[iDim] - IOverCSquare[iDim] / 2.0) * KRC
									+ IOverRC[iDim] / 2.0 * (KRCDerivatives[iDim + iParam - (Kernel::NumTotalParameters - 1)] - KRC / RealCorrCharLength[iDim]);
								KIC_deriv[iDim + iParam] = (2.0 / ImagCharLength[iDim] - IOverCSquare[iDim] / 2.0) * KIC
									+ 1.5 * IOverIC[iDim] * (KICDerivatives[iDim + iParam - (Kernel::NumTotalParameters - 1)] - KIC / ImagCorrCharLength[iDim]);
							}
							iParam += PhaseDim;
						}
						// last, noise
						KRprime_deriv[iParam] = KIprime_deriv[iParam] = KCprime_deriv[iParam] = KRC_deriv[iParam] = KIC_deriv[iParam] = Eigen::MatrixXd::Zero(size, size);
					}
					const Eigen::MatrixXd K1 = KRprime + KIprime + 2.0 * KCprime;
					const Eigen::MatrixXcd K2 = KRprime - KIprime - 2.0i * (KRC + KIC);
					ParameterArray<Eigen::MatrixXd> K1_deriv;
					ParameterArray<Eigen::MatrixXcd> K2_deriv;
					ParameterArray<double> result;
					for (std::size_t iParam = 0; iParam < NumTotalParameters; iParam++)
					{
						K1_deriv[iParam] = KRprime_deriv[iParam] + KIprime_deriv[iParam] + 2.0 * KCprime_deriv[iParam];
						K2_deriv[iParam] = KRprime_deriv[iParam] - KIprime_deriv[iParam] - 2.0i * (KRC_deriv[iParam] + KIC_deriv[iParam]);
						result[iParam] = 2.0 * (v.adjoint() * K1 * vDerivatives[iParam]).value().real()
							+ (v.adjoint() * K1_deriv[iParam] * v).value().real()
							+ 2.0 * (v.transpose() * K2 * vDerivatives[iParam]).value().real()
							+ (v.transpose() * K2_deriv[iParam] * v).value().real();
						result[iParam] *= GlobalFactor;
					}
					return result;
				}()
			)
			: std::nullopt
	)
{
}

ComplexKernel::ComplexKernel(
	const ComplexKernel& TrainingKernel,
	const PhasePoints& left_feature,
	const PhasePoints& right_feature
):
	// general calculations
	// calculation only for derivatives (items apart from derivative of (pseudo-)kernel are for training set only)
	LeftFeature(left_feature),
	RightFeature(right_feature),
	Params(TrainingKernel.Params),
	KernelParams(TrainingKernel.KernelParams),
	RealParams(TrainingKernel.RealParams),
	ImagParams(TrainingKernel.ImagParams),
	CorrParams(TrainingKernel.CorrParams),
	RealKernel(RealParams, LeftFeature, RightFeature, false),
	ImagKernel(ImagParams, LeftFeature, RightFeature, false),
	CorrKernel(CorrParams, LeftFeature, RightFeature, false),
	KernelMatrix(power<2>(std::get<0>(KernelParams)) * (RealKernel.get_kernel() + ImagKernel.get_kernel() + power<2>(std::get<2>(KernelParams)) * delta_kernel(left_feature, right_feature))),
	PseudoKernelMatrix(power<2>(std::get<0>(KernelParams)) * (RealKernel.get_kernel() - ImagKernel.get_kernel() + 2.0i * CorrKernel.get_kernel()))
// calculations only for training set
// calculation only for test set
// calculation only for error/averages (purity is only for training set only)
// calculation only for derivatives (items apart from derivative of (pseudo-)kernel are for training set only)
// calculation for both error/average and derivatives
{
}

ComplexKernel::ComplexKernel(
	const PhasePoints& TestFeature,
	const ComplexKernel& TrainingKernel,
	const bool IsToCalculateDerivative,
	const std::optional<Eigen::VectorXcd> TestLabel
):
	// general calculations
	LeftFeature(TestFeature),
	RightFeature(TrainingKernel.LeftFeature),
	Params(TrainingKernel.Params),
	KernelParams(TrainingKernel.KernelParams),
	RealParams(TrainingKernel.RealParams),
	ImagParams(TrainingKernel.ImagParams),
	CorrParams(TrainingKernel.CorrParams),
	RealKernel(RealParams, LeftFeature, RightFeature, IsToCalculateDerivative),
	ImagKernel(ImagParams, LeftFeature, RightFeature, IsToCalculateDerivative),
	CorrKernel(CorrParams, LeftFeature, RightFeature, IsToCalculateDerivative),
	KernelMatrix(power<2>(std::get<0>(KernelParams)) * (RealKernel.get_kernel() + ImagKernel.get_kernel() + power<2>(std::get<2>(KernelParams)) * delta_kernel(LeftFeature, RightFeature))),
	PseudoKernelMatrix(power<2>(std::get<0>(KernelParams)) * (RealKernel.get_kernel() - ImagKernel.get_kernel() + 2.0i * CorrKernel.get_kernel())),
	// calculations only for training set
	Label(TestLabel),
	// calculation only for test set
	Prediction(KernelMatrix * TrainingKernel.UpperPartOfAugmentedInverseLabel.value() + PseudoKernelMatrix * TrainingKernel.UpperPartOfAugmentedInverseLabel.value().conjugate()),
	ElementwiseVariance(
		[&TestFeature,
		 &TrainingKernel,
		 &KernelMatrix = KernelMatrix,
		 &PseudoKernelMatrix = PseudoKernelMatrix,
		 &P = TrainingKernel.UpperLeftBlockOfAugmentedKernelInverse.value(),
		 &Q = TrainingKernel.LowerLeftBlockOfAugmentedKernelInverse.value()](void) -> Eigen::VectorXd
		{
			const std::size_t NumPoints = TestFeature.cols();
			const auto indices = xt::arange(NumPoints);
			Eigen::VectorXd result(NumPoints);
			std::for_each(
				std::execution::par_unseq,
				indices.cbegin(),
				indices.cend(),
				[&result, &TrainingKernel, &TestFeature, &KernelMatrix, &PseudoKernelMatrix, &P, &Q](std::size_t iCol) -> void
				{
					const ClassicalPhaseVector& col = TestFeature.col(iCol);
					const auto& kernel_row = KernelMatrix.row(iCol);
					const auto& kernel_col = kernel_row.transpose();
					const auto& pseudo_row = PseudoKernelMatrix.row(iCol);
					const auto& pseudo_col = pseudo_row.adjoint();
					result(iCol) =
						(ComplexKernel(TrainingKernel, col, col).KernelMatrix.value()
						 - (kernel_row * P * kernel_col).value()
						 - (pseudo_row * P.conjugate() * pseudo_col).value()
						 - (pseudo_row * Q * kernel_col).value()
						 - (kernel_row * Q.conjugate() * pseudo_col).value())
							.real();
				}
			);
			return result;
		}()
	),
	CutoffPrediction(Prediction.value().array() * cutoff_factor(Prediction.value(), ElementwiseVariance.value()).array()),
	// calculation only for error/averages (purity is only for training set only)
	Error(Label.has_value() ? std::optional<double>((Prediction.value() - Label.value()).squaredNorm()) : std::nullopt),
	// calculation only for derivatives (items apart from derivative of (pseudo-)kernel are for training set only)
	Derivatives(
		IsToCalculateDerivative
			? std::optional<ParameterArray<Eigen::MatrixXd>>(calculate_derivatives(
				KernelParams,
				LeftFeature,
				RightFeature,
				KernelMatrix,
				RealKernel.get_derivatives(),
				ImagKernel.get_derivatives()
			))
			: std::nullopt
	),
	PseudoDerivatives(
		IsToCalculateDerivative
			? std::optional<ParameterArray<Eigen::MatrixXcd>>(calculate_pseudo_derivatives(
				KernelParams,
				LeftFeature,
				RightFeature,
				PseudoKernelMatrix,
				CorrKernel.get_kernel(),
				RealParams,
				ImagParams,
				CorrParams,
				RealKernel.get_derivatives(),
				ImagKernel.get_derivatives(),
				CorrKernel.get_derivatives()
			))
			: std::nullopt
	),
	// calculation for both error/average and derivatives
	ErrorDerivatives(
		Label.has_value() && IsToCalculateDerivative
			? std::optional<ParameterArray<double>>(
				[PredictionDifference = Prediction.value() - Label.value(),
				 &KernelMatrix = KernelMatrix,
				 &PseudoKernelMatrix = PseudoKernelMatrix,
				 &v = TrainingKernel.UpperPartOfAugmentedInverseLabel.value(),
				 &Derivatives = Derivatives.value(),
				 &PseudoDerivatives = PseudoDerivatives.value(),
				 &vDerivatives = TrainingKernel.UpperAugmentedInvLblDerivatives.value()](void) -> ParameterArray<double>
				{
					ParameterArray<double> result;
					for (std::size_t iParam = 0; iParam < NumTotalParameters; iParam++)
					{
						result[iParam] = 2.0 * PredictionDifference.dot(Derivatives[iParam] * v + KernelMatrix * vDerivatives[iParam] + PseudoDerivatives[iParam] * v.conjugate() + PseudoKernelMatrix * vDerivatives[iParam].conjugate()).real();
					}
					return result;
				}()
			)
			: std::nullopt
	)
{
}
