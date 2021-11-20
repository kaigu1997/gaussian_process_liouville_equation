/// @file kernel.cpp
/// @brief Implementation of kernel.h

#include "stdafx.h"

#include "kernel.h"

/// @brief To unpack the parameters to components
/// @param[in] Parameter All the parameters
/// @return The unpacked parameters
static Kernel::KernelParameter unpack_parameters(const ParameterVector& Parameter)
{
	assert(Parameter.size() == Kernel::NumTotalParameters);
	// have all the parameters
	Kernel::KernelParameter result;
	auto& [noise, gaussian] = result;
	auto& [magnitude, char_length] = gaussian;
	std::size_t iParam = 0;
	// first, noise
	noise = Parameter[iParam++];
	// second, Gaussian
	magnitude = Parameter[iParam++];
	for (int iDim = 0; iDim < PhaseDim; iDim++)
	{
		char_length[iDim] = Parameter[iDim + iParam];
	}
	iParam += PhaseDim;
	return result;
}

/// @brief To calculate the Gaussian kernel
/// @param[in] CharacteristicLength The characteristic lengths for all dimensions
/// @param[in] LeftFeature The left feature
/// @param[in] RightFeature The right feature
/// @return The Gaussian kernel matrix
static Eigen::MatrixXd gaussian_kernel(
	const Eigen::VectorXd& CharacteristicLength,
	const Eigen::MatrixXd& LeftFeature,
	const Eigen::MatrixXd& RightFeature)
{
	const int Rows = LeftFeature.cols(), Cols = RightFeature.cols();
	Eigen::MatrixXd result = Eigen::MatrixXd::Zero(Rows, Cols);
	if (LeftFeature.data() == RightFeature.data())
	{
		// training set, the matrix is symmetric
		if (static_cast<bool>(result.IsRowMajor) == true)
		{
#pragma omp parallel for
			for (int iRow = 1; iRow < Rows; iRow++)
			{
				for (int iCol = 0; iCol < iRow; iCol++)
				{
					const auto& diff = ((LeftFeature.col(iRow) - RightFeature.col(iCol)).array() / CharacteristicLength.array()).matrix();
					result(iRow, iCol) += std::exp(-diff.squaredNorm() / 2.0);
				}
			}
		}
		else
		{
#pragma omp parallel for
			for (int iCol = 0; iCol < Cols - 1; iCol++)
			{
				for (int iRow = iCol + 1; iRow < Rows; iRow++)
				{
					const auto& diff = ((LeftFeature.col(iRow) - RightFeature.col(iCol)).array() / CharacteristicLength.array()).matrix();
					result(iRow, iCol) += std::exp(-diff.squaredNorm() / 2.0);
				}
			}
		}
		result.diagonal() = Eigen::VectorXd::Ones(Rows);
		result = result.selfadjointView<Eigen::Lower>();
	}
	else
	{
		if (static_cast<bool>(result.IsRowMajor) == true)
		{
#pragma omp parallel for
			for (int iRow = 0; iRow < Rows; iRow++)
			{
				for (int iCol = 0; iCol < Cols; iCol++)
				{
					const auto& diff = ((LeftFeature.col(iRow) - RightFeature.col(iCol)).array() / CharacteristicLength.array()).matrix();
					result(iRow, iCol) += std::exp(-diff.squaredNorm() / 2.0);
				}
			}
		}
		else
		{
#pragma omp parallel for
			for (int iCol = 0; iCol < Cols; iCol++)
			{
				for (int iRow = 0; iRow < Rows; iRow++)
				{
					const auto& diff = ((LeftFeature.col(iRow) - RightFeature.col(iCol)).array() / CharacteristicLength.array()).matrix();
					result(iRow, iCol) += std::exp(-diff.squaredNorm() / 2.0);
				}
			}
		}
	}
	return result;
}

/// @brief To calculate the kernel matrix, \f$ K(X_1,X_2) \f$
/// @param[in] KernelParams The unpacked parameters for all kernels
/// @param[in] LeftFeature The left feature
/// @param[in] RightFeature The right feature
/// @return The kernel matrix
/// @details This function calculates the kernel matrix with noise. Denote \f$ x_{mni}=\frac{[\mathbf{x}_m]_i-[\mathbf{x}_n]_i}{l_i} \f$,
///
/// \f[
/// k(\mathbf{x}_1,\mathbf{x}_2)=\sigma_f^2\mathrm{exp}\left(-\frac{1}{2}\sum_ix_{12i}^2\right)+\sigma_n^2\delta(\mathbf{x}_1-\mathbf{x}_2)
/// \f]
///
/// where \f$ M \f$ is the characteristic matrix in the form of a diagonal matrix, whose elements are the characteristic length of each dimention.
///
/// When there are more than one feature, the kernel matrix follows \f$ K_{ij}=k(X_{1_{\cdot i}}, X_{2_{\cdot j}}) \f$.
static inline Eigen::MatrixXd get_kernel_matrix(
	const Kernel::KernelParameter& KernelParams,
	const Eigen::MatrixXd& LeftFeature,
	const Eigen::MatrixXd& RightFeature)
{
	assert(LeftFeature.rows() == PhaseDim && RightFeature.rows() == PhaseDim);
	const int Rows = LeftFeature.cols(), Cols = RightFeature.cols();
	Eigen::MatrixXd result = Eigen::MatrixXd::Zero(Rows, Cols);
	const auto& [Noise, Gaussian] = KernelParams;
	const auto& [Magnitude, CharLength] = Gaussian;
	// first, noise
	if (LeftFeature.data() == RightFeature.data())
	{
		// in case when it is not training feature, the diagonal kernel (working as noise) is not needed
		result += std::pow(Noise, 2) * Eigen::MatrixXd::Identity(Rows, Cols);
	}
	// second, Gaussian
	result += std::pow(Magnitude, 2) * gaussian_kernel(CharLength, LeftFeature, RightFeature);
	return result;
}

/// @brief To calculate the population of one diagonal element by analytic integration of parameters
/// @param[in] KernelParams The unpacked parameters for all kernels
/// @param[in] InvLbl The inverse of kernel matrix of training features times the training labels
/// @return The integral of \f$ \langle\rho\rangle \f$
/// @details For the current kernel,
///
/// \f[
/// \langle\rho\rangle=(2\pi)^D\sigma_f^2\prod_i\abs{l_i}\sum_i[K^{-1}\mathbf{y}]_i
/// \f]
static inline double calculate_population(const Kernel::KernelParameter& KernelParams, const Eigen::VectorXd& InvLbl)
{
	static const double GlobalFactor = std::pow(2.0 * M_PI, Dim);
	[[maybe_unused]] const auto& [Noise, Gaussian] = KernelParams;
	const auto& [Magnitude, CharLength] = Gaussian;
	double ppl = 0.0;
	ppl += GlobalFactor * std::pow(Magnitude, 2) * CharLength.prod() * InvLbl.sum();
	return ppl;
}

/// @brief To calculate the average position and momentum of one diagonal element by analytic integration of parameters
/// @param[in] KernelParams The unpacked parameters for all kernels
/// @param[in] Features The features
/// @param[in] InvLbl The inverse of kernel matrix of training features times the training labels
/// @return Average positions and momenta
/// @details For the current kernel, for \f$ x_0\f$ for example,
///
/// \f[
/// \langle x_0\rangle=(2\pi)^D\sigma_f^2\prod_i\abs{l_i}\sum_i[\mathbf{x}_i]_0[K^{-1}\mathbf{y}]_i
/// \f]
///
/// where
/// and similar for other \f$ x_i \f$ and \f$ p_i \f$.
static inline ClassicalPhaseVector calculate_1st_order_average(
	const Kernel::KernelParameter& KernelParams,
	const Eigen::MatrixXd& Features,
	const Eigen::VectorXd& InvLbl)
{
	static const double GlobalFactor = std::pow(2.0 * M_PI, Dim);
	[[maybe_unused]] const auto& [Noise, Gaussian] = KernelParams;
	const auto& [Magnitude, CharLength] = Gaussian;
	ClassicalPhaseVector r = ClassicalPhaseVector::Zero();
	r += GlobalFactor * std::pow(Magnitude, 2) * CharLength.prod() * Features * InvLbl;
	return r;
}

/// @brief To calculate the average position and momentum of one diagonal element by analytic integration of parameters
/// @param[in] KernelParams The unpacked parameters for all kernels
/// @param[in] Features The features
/// @param[in] InvLbl The inverse of kernel matrix of training features times the training labels
/// @return The integral of \f$ (2\pi\hbar)^{\frac{d}{2}}\langle\rho^2\rangle \f$
/// @details For the current kernel,
///
/// \f[
/// (2\pi\hbar)^{\frac{d}{2}}\langle\rho^2\rangle=(\pi)^D\sigma_f^4\prod_i\abs{l_i}\mathbf{y}^{\mathsf{T}}K^{-1}K_1K^{-1}\mathbf{y}
/// \f]
///
/// where \f$ [K_1]_{mn}=\mathrm{exp}\left(-\frac{1}{4}\sum_i x_{mni}^2\right) \f$
/// if denote \f$ x_{mni}=\frac{[\mathbf{x}_m]_i-[\mathbf{x}_n]_i}{l_i} \f$.
static inline double calculate_purity(
	const Kernel::KernelParameter& KernelParams,
	const Eigen::MatrixXd& Features,
	const Eigen::VectorXd& InvLbl)
{
	static const double GlobalFactor = std::pow(2.0 * M_PI * hbar, Dim) * std::pow(M_PI, Dim);
	[[maybe_unused]] const auto& [Noise, Gaussian] = KernelParams;
	const auto& [Magnitude, CharLength] = Gaussian;
	double result = 0.0;
	result += GlobalFactor * std::pow(Magnitude, 4) * CharLength.prod() * (InvLbl.transpose() * gaussian_kernel(M_SQRT2 * CharLength, Features, Features) * InvLbl).value();
	return result;
}

/// @brief To calculate the derivative of gaussian kernel over characteristic lengths
/// @param[in] CharLegnth The characteristic length
/// @param[in] Feature The feature
/// @param[in] GaussianKernelMatrix The gaussian kernel matrix, for less calculation
/// @return The derivative of gaussian kernel over characteristic lengths
/// @details Denote \f$ x_{mni}=\frac{[\mathbf{x}_m]_i-[\mathbf{x}_n]_i}{l_i} \f$,
/// for the derivative of element \f$ K_{mn} \f$ over characteristic length \f$ l_j \f$
///
/// \f[
/// \frac{\partial K_{mn}}{\partial l_j}=\mathrm{exp}\left(-\frac{1}{2}\sum_i x_{mni}^2\right)\frac{x_{mnj}^2}{l_j}
/// \f]
///
/// and thus the derivative matrix is constructed.
static EigenVector<Eigen::MatrixXd> gaussian_derivative_over_char_length(
	const ClassicalPhaseVector& CharLength,
	const Eigen::MatrixXd& Feature,
	const Eigen::MatrixXd& GaussianKernelMatrix)
{
	EigenVector<Eigen::MatrixXd> result(PhaseDim, GaussianKernelMatrix);
	const int Rows = Feature.cols();
	if (static_cast<bool>(GaussianKernelMatrix.IsRowMajor) == true)
	{
#pragma omp parallel for
		for (int iRow = 1; iRow < Rows; iRow++)
		{
			for (int iCol = 0; iCol < iRow; iCol++)
			{
				const auto& diff = ((Feature.col(iRow) - Feature.col(iCol)).array() / CharLength.array()).matrix();
				for (int iDim = 0; iDim < PhaseDim; iDim++)
				{
					result[iDim](iRow, iCol) *= std::pow(diff[iDim], 2) / CharLength[iDim];
				}
			}
		}
	}
	else
	{
#pragma omp parallel for
		for (int iCol = 0; iCol < Rows - 1; iCol++)
		{
			for (int iRow = iCol + 1; iRow < Rows; iRow++)
			{
				const auto& diff = ((Feature.col(iRow) - Feature.col(iCol)).array() / CharLength.array()).matrix();
				for (int iDim = 0; iDim < PhaseDim; iDim++)
				{
					result[iDim](iRow, iCol) *= std::pow(diff[iDim], 2) / CharLength[iDim];
				}
			}
		}
	}
	for (int iDim = 0; iDim < PhaseDim; iDim++)
	{
		// as it is symmetric, only lower part of the matrix are necessary to calculate
		result[iDim].diagonal() = Eigen::VectorXd::Zero(Rows);
		result[iDim] = result[iDim].selfadjointView<Eigen::Lower>();
	}
	return result;
}

/// @brief To calculate the derivatives over parameters
/// @param[in] KernelParams The unpacked parameters for all kernels
/// @param[in] Feature The feature
/// @return The derivative of kernel matrix over parameters
static EigenVector<Eigen::MatrixXd> calculate_derivatives(
	const Kernel::KernelParameter& KernelParams,
	const Eigen::MatrixXd& Feature)
{
	const int Rows = Feature.cols();
	const auto& [Noise, Gaussian] = KernelParams;
	const auto& [Magnitude, CharLength] = Gaussian;
	const Eigen::MatrixXd GaussianKernelMatrix = gaussian_kernel(CharLength, Feature, Feature);
	EigenVector<Eigen::MatrixXd> result;
	// first, noise
	result.push_back(2.0 * Noise * Eigen::MatrixXd::Identity(Rows, Rows));
	// second, Gaussian
	result.push_back(2.0 * Magnitude * GaussianKernelMatrix);
	const EigenVector<Eigen::MatrixXd> GaussianDerivOverCharLength = gaussian_derivative_over_char_length(CharLength, Feature, GaussianKernelMatrix);
	const double MagSquare = std::pow(Magnitude, 2);
	for (const auto& Mat : GaussianDerivOverCharLength)
	{
		result.push_back(MagSquare * Mat);
	}
	return result;
}

/// @brief To calculate the inverse of kernel times the derivatives of kernel
/// @param[in] Inverse The inverse of the kernel matrix
/// @param[in] Derivatives The derivatives of the kernel matrix over parameters
/// @return The product
static EigenVector<Eigen::MatrixXd> inverse_times_derivatives(const Eigen::MatrixXd& Inverse, const EigenVector<Eigen::MatrixXd>& Derivatives)
{
	EigenVector<Eigen::MatrixXd> result(Kernel::NumTotalParameters, Inverse);
	for (std::size_t iParam = 0; iParam < Kernel::NumTotalParameters; iParam++)
	{
		result[iParam] *= Derivatives[iParam];
	}
	return result;
}

/// @brief To calculate the product of \f$ -K^{-1}\frac{\partial K}{\partial\theta}K^{-1} \f$
/// @param[in] InvDeriv The inverse of kernel times the derivatives
/// @param[in] Inverse The inverse of the kernel matirx
/// @return \f$ \frac{\partial K^{-1}}{\partial\theta}=-K^{-1}\frac{\partial K}{\partial\theta}K^{-1} \f$
static EigenVector<Eigen::MatrixXd> negative_inverse_derivative_inverse(const EigenVector<Eigen::MatrixXd>& InvDeriv, const Eigen::MatrixXd& Inverse)
{
	EigenVector<Eigen::MatrixXd> result = InvDeriv;
	for (auto& Mat : result)
	{
		Mat *= -Inverse;
	}
	return result;
}

/// @brief To calculate the product of \f$ -K^{-1}\frac{\partial K}{\partial\theta}K^{-1}\mathbf{y} \f$
/// @param[in] InvDeriv The inverse of kernel times the derivatives
/// @param[in] InvLbl The inverse of kernel matrix of training features times the training labels
/// @return \f$ \frac{\partial(K^{-1}\mathbf{y})}{\partial\theta}=-K^{-1}\frac{\partial K}{\partial\theta}K^{-1}\mathbf{y} \f$
static EigenVector<Eigen::VectorXd> negative_inverse_derivative_inverse_label(const EigenVector<Eigen::MatrixXd>& InvDeriv, const Eigen::VectorXd& InvLbl)
{
	EigenVector<Eigen::VectorXd> result(Kernel::NumTotalParameters, InvLbl);
	for (std::size_t iParam = 0; iParam < Kernel::NumTotalParameters; iParam++)
	{
		result[iParam] = -InvDeriv[iParam] * InvLbl;
	}
	return result;
}

/// @brief To calculate the derivative of \f$ \langle\rho\rangle \f$ over all parameters
/// @param[in] KernelParams The unpacked parameters for all kernels
/// @param[in] InvDerivInvLbl The inverse
/// @param[in] InvLbl The inverse of kernel matrix of training features times the training labels
/// @return The derivative of \f$ \langle\rho\rangle \f$ over all parameters
static ParameterVector population_derivatives(
	const Kernel::KernelParameter& KernelParams,
	const EigenVector<Eigen::VectorXd>& NegInvDerivInvLbl,
	const Eigen::VectorXd& InvLbl)
{
	static const double GlobalFactor = std::pow(2.0 * M_PI, Dim);
	[[maybe_unused]] const auto& [Noise, Gaussian] = KernelParams;
	const auto& [Magnitude, CharLength] = Gaussian;
	const double ThisTimeFactor = GlobalFactor * std::pow(Magnitude, 2) * CharLength.prod();
	ParameterVector result(Kernel::NumTotalParameters);
	std::size_t iParam = 0;
	// first, noise
	result[iParam] = ThisTimeFactor * NegInvDerivInvLbl[iParam].sum();
	iParam++;
	// second, Gaussian
	result[iParam] = ThisTimeFactor * (2.0 / Magnitude * InvLbl.sum() + NegInvDerivInvLbl[iParam].sum());
	iParam++;
	for (int iDim = 0; iDim < PhaseDim; iDim++)
	{
		result[iParam] = ThisTimeFactor * (InvLbl.sum() / CharLength[iDim] + NegInvDerivInvLbl[iParam].sum());
		iParam++;
	}
	return result;
}

/// @brief To calculate the derivative of \f$ \langle\rho^2\rangle \f$ over all parameters
/// @param[in] KernelParams The unpacked parameters for all kernels
/// @param[in] Feature The feature
/// @param[in] InvDerivInvLbl The inverse
/// @param[in] InvLbl The inverse of kernel matrix of training features times the training labels
/// @return The derivative of \f$ \langle\rho\rangle \f$ over all parameters
static ParameterVector purity_derivatives(
	const Kernel::KernelParameter& KernelParams,
	const Eigen::MatrixXd& Feature,
	const EigenVector<Eigen::VectorXd>& NegInvDerivInvLbl,
	const Eigen::VectorXd& InvLbl)
{
	static const double GlobalFactor = std::pow(2.0 * M_PI * hbar, Dim) * std::pow(M_PI, Dim);
	[[maybe_unused]] const auto& [Noise, Gaussian] = KernelParams;
	const auto& [Magnitude, CharLength] = Gaussian;
	const double ThisTimeFactor = GlobalFactor * std::pow(Magnitude, 4) * CharLength.prod();
	ParameterVector result(Kernel::NumTotalParameters);
	std::size_t iParam = 0;
	const Eigen::MatrixXd ModifiedGaussianKernel = gaussian_kernel(M_SQRT2 * CharLength, Feature, Feature);
	// first, noise
	result[iParam] = ThisTimeFactor
		* ((NegInvDerivInvLbl[iParam].transpose() * ModifiedGaussianKernel * InvLbl).value()
			+ (InvLbl.transpose() * ModifiedGaussianKernel * NegInvDerivInvLbl[iParam]).value());
	iParam++;
	// second, Gaussian
	result[iParam] = ThisTimeFactor
		* (4.0 / Magnitude * (InvLbl.transpose() * ModifiedGaussianKernel * InvLbl).value()
			+ (NegInvDerivInvLbl[iParam].transpose() * ModifiedGaussianKernel * InvLbl).value()
			+ (InvLbl.transpose() * ModifiedGaussianKernel * NegInvDerivInvLbl[iParam]).value());
	iParam++;
	const EigenVector<Eigen::MatrixXd> ModifiedGaussianKernelDerivatives = [&, CharLength = CharLength](void) -> EigenVector<Eigen::MatrixXd>
	{
		// as the characteristic lengths is changed in the modified Gaussian kernel,
		// the
		EigenVector<Eigen::MatrixXd> result = gaussian_derivative_over_char_length(CharLength, Feature, ModifiedGaussianKernel);
		for (Eigen::MatrixXd& deriv : result)
		{
			deriv /= 2.0;
		}
		return result;
	}();
	for (int iDim = 0; iDim < PhaseDim; iDim++)
	{
		result[iParam] = ThisTimeFactor
			* ((InvLbl.transpose() * ModifiedGaussianKernel * InvLbl).value() / CharLength[iDim]
				+ (NegInvDerivInvLbl[iParam].transpose() * ModifiedGaussianKernel * InvLbl).value()
				+ (InvLbl.transpose() * ModifiedGaussianKernelDerivatives[iDim] * InvLbl).value()
				+ (InvLbl.transpose() * ModifiedGaussianKernel * NegInvDerivInvLbl[iParam]).value());
		iParam++;
	}
	return result;
}

const std::size_t Kernel::NumTotalParameters = 1 + 1 + PhaseDim;
static const double InitialNoise = 1.0;			 ///< The initial weight for noise
static const double InitialGaussianWeight = 1e4; ///< The initial weight for Gaussian kernel

ParameterVector Kernel::set_initial_parameters(const ClassicalVector<double>& xSigma, const ClassicalVector<double>& pSigma)
{
	ParameterVector result(Kernel::NumTotalParameters);
	std::size_t iParam = 0;
	// first, noise
	result[iParam++] = InitialNoise; // diagonal weight is fixed at 1.0
	// second, Gaussian
	result[iParam++] = InitialGaussianWeight;
	for (int iDim = 0; iDim < Dim; iDim++)
	{
		result[iParam++] = xSigma[iDim];
	}
	// dealing with p
	for (int iDim = 0; iDim < Dim; iDim++)
	{
		result[iParam++] = pSigma[iDim];
	}
	return result;
}

std::array<ParameterVector, 2> Kernel::set_parameter_bounds(
	const ClassicalVector<double>& xSize,
	const ClassicalVector<double>& pSize)
{
	static const double GaussKerMinCharLength = 1.0 / 100.0; // Minimal characteristic length for Gaussian kernel
	std::array<ParameterVector, 2> result;
	auto& [LowerBound, UpperBound] = result;
	{
		std::size_t iParam = 0;
		// first, noise
		// diagonal weight is fixed
		LowerBound.push_back(InitialNoise);
		UpperBound.push_back(InitialNoise);
		iParam++;
		// second, Gaussian
		LowerBound.push_back(InitialNoise);
		UpperBound.push_back(std::pow(InitialGaussianWeight, 2) / InitialNoise);
		iParam++;
		// dealing with x
		for (int iDim = 0; iDim < Dim; iDim++)
		{
			LowerBound.push_back(GaussKerMinCharLength);
			UpperBound.push_back(xSize[iDim]);
			iParam++;
		}
		// dealing with p
		for (int iDim = 0; iDim < Dim; iDim++)
		{
			LowerBound.push_back(GaussKerMinCharLength);
			UpperBound.push_back(pSize[iDim]);
			iParam++;
		}
	}
	return result;
}

/// @details To calculate the kernel matrix based on the given features.
///
/// If this is the kernel for training set by compare the two features,
/// calculate the Cholesky decomposition of kernel matrix, and
/// calculate derivative matrices if necessary.
///
/// Simiarly, \f$ K^{-1}\bold{y} \f$ is calculated only when label is
/// given and this is training set.
Kernel::Kernel(
	const ParameterVector& Parameter,
	const Eigen::MatrixXd& left_feature,
	const Eigen::MatrixXd& right_feature,
	const std::optional<Eigen::VectorXd>& label,
	const bool IsCalculateAverage,
	const bool IsCalculateDerivative) :
	// general calculations
	KernelParams(unpack_parameters(Parameter)),
	LeftFeature(left_feature),
	RightFeature(right_feature),
	KernelMatrix(get_kernel_matrix(KernelParams, left_feature, right_feature)),
	// calculations only for training set
	IsTrainingSet(left_feature.size() != 0 && left_feature.data() == right_feature.data()),
	Label(IsTrainingSet == true ? label : std::nullopt),
	DecompOfKernel(IsTrainingSet == true
			? decltype(DecompOfKernel)(std::in_place, KernelMatrix)
			: std::nullopt),
	LogDeterminant(IsTrainingSet == true
			? decltype(LogDeterminant)(2.0 * Eigen::MatrixXd(DecompOfKernel->matrixL()).diagonal().array().log().sum())
			: std::nullopt),
	Inverse(IsTrainingSet == true
			? decltype(Inverse)(DecompOfKernel->solve(Eigen::MatrixXd::Identity(KernelMatrix.rows(), KernelMatrix.cols())))
			: std::nullopt),
	InvLbl(Label.has_value() == true ? decltype(InvLbl)(DecompOfKernel->solve(Label.value())) : std::nullopt),
	// calculation only for averages (which is training set only too)
	IsAverageCalculated(InvLbl.has_value() == true && IsCalculateAverage == true),
	Population(IsAverageCalculated == true ? decltype(Population)(calculate_population(KernelParams, InvLbl.value())) : std::nullopt),
	FirstOrderAverage(IsAverageCalculated == true
			? decltype(FirstOrderAverage)(calculate_1st_order_average(KernelParams, LeftFeature, InvLbl.value()))
			: std::nullopt),
	Purity(IsAverageCalculated == true ? decltype(Purity)(calculate_purity(KernelParams, LeftFeature, InvLbl.value())) : std::nullopt),
	// calculation only for derivatives (which is training set only too)
	IsDerivativeCalculated(IsTrainingSet == true && IsCalculateDerivative == true),
	Derivatives(IsDerivativeCalculated == true ? decltype(Derivatives)(calculate_derivatives(KernelParams, LeftFeature)) : std::nullopt),
	InvDeriv(IsDerivativeCalculated == true ? decltype(InvDeriv)(inverse_times_derivatives(Inverse.value(), Derivatives.value())) : std::nullopt),
	NegInvDerivInv(IsDerivativeCalculated == true ? decltype(NegInvDerivInv)(negative_inverse_derivative_inverse(InvDeriv.value(), Inverse.value())) : std::nullopt),
	NegInvDerivInvLbl(IsDerivativeCalculated == true && Label.has_value() == true
			? decltype(NegInvDerivInvLbl)(negative_inverse_derivative_inverse_label(InvDeriv.value(), InvLbl.value()))
			: std::nullopt),
	// calculation for both average and derivatives
	IsAverageDerivativeCalculated(IsAverageCalculated == true && IsDerivativeCalculated == true && Label.has_value() == true),
	PopulationDeriv(IsAverageDerivativeCalculated == true
			? decltype(PopulationDeriv)(population_derivatives(KernelParams, NegInvDerivInvLbl.value(), InvLbl.value()))
			: std::nullopt),
	PurityDeriv(IsAverageDerivativeCalculated == true
			? decltype(PopulationDeriv)(purity_derivatives(KernelParams, LeftFeature, NegInvDerivInvLbl.value(), InvLbl.value()))
			: std::nullopt)
{
}