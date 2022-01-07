/// @file kernel.cpp
/// @brief Implementation of kernel.h

#include "stdafx.h"

#include "kernel.h"

const std::size_t Kernel::NumTotalParameters = 1 + 1 + PhaseDim;

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
	auto calculate_element = [&CharacteristicLength, &LeftFeature, &RightFeature](const int iRow, const int iCol) -> double
	{
		const auto& diff = ((LeftFeature.col(iRow) - RightFeature.col(iCol)).array() / CharacteristicLength.array()).matrix();
		return std::exp(-diff.squaredNorm() / 2.0);
	};
	const int Rows = LeftFeature.cols(), Cols = RightFeature.cols();
	const bool IsTrainingSet = LeftFeature.data() == RightFeature.data() && Rows == Cols;
	Eigen::MatrixXd result = Eigen::MatrixXd::Zero(Rows, Cols);
	const std::vector<int> indices = get_indices(static_cast<bool>(result.IsRowMajor) ? Rows : Cols);
	std::for_each(
		std::execution::par_unseq,
		indices.cbegin(),
		indices.cend(),
		[&result, &calculate_element, Rows, Cols, IsTrainingSet](int i) -> void
		{
			if (static_cast<bool>(result.IsRowMajor))
			{
				// row major
				const int iRow = i;
				if (IsTrainingSet)
				{
					// training set, the matrix is symmetric
					for (int iCol = 0; iCol < iRow; iCol++)
					{
						result(iRow, iCol) += calculate_element(iRow, iCol);
					}
				}
				else
				{
					// general rectangular
					for (int iCol = 0; iCol < Cols; iCol++)
					{
						result(iRow, iCol) += calculate_element(iRow, iCol);
					}
				}
			}
			else
			{
				// column major
				const int iCol = i;
				if (IsTrainingSet)
				{
					// training set, the matrix is symmetric
					for (int iRow = iCol + 1; iRow < Rows; iRow++)
					{
						result(iRow, iCol) += calculate_element(iRow, iCol);
					}
				}
				else
				{
					// general rectangular
					for (int iRow = 0; iRow < Rows; iRow++)
					{
						result(iRow, iCol) += calculate_element(iRow, iCol);
					}
				}
			}
		});
	if (IsTrainingSet)
	{
		// training set, the matrix is symmetric
		result.diagonal() = Eigen::VectorXd::Ones(Rows);
		result = result.selfadjointView<Eigen::Lower>();
	}
	return result;
}

/// @brief To calculate the kernel matrix, @f$ K(X_1,X_2) @f$
/// @param[in] KernelParams The unpacked parameters for all kernels
/// @param[in] LeftFeature The left feature
/// @param[in] RightFeature The right feature
/// @return The kernel matrix
/// @details This function calculates the kernel matrix with noise. Denote @f$ x_{mni}=\frac{[\mathbf{x}_m]_i-[\mathbf{x}_n]_i}{l_i} @f$,
///
/// @f[
/// k(\mathbf{x}_1,\mathbf{x}_2)=\sigma_f^2\mathrm{exp}\left(-\frac{1}{2}\sum_ix_{12i}^2\right)+\sigma_n^2\delta(\mathbf{x}_1-\mathbf{x}_2)
/// @f]
///
/// where M is the characteristic matrix in the form of a diagonal matrix, whose elements are the characteristic length of each dimention.
///
/// When there are more than one feature, the kernel matrix follows @f$ K_{ij}=k(X_{1_{\cdot i}}, X_{2_{\cdot j}}) @f$.
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
/// @return The integral of @f$ \langle\rho\rangle @f$
/// @details For the current kernel,
///
/// @f[
/// \langle\rho\rangle=(2\pi)^D\sigma_f^2\prod_i\abs{l_i}\sum_i[K^{-1}\mathbf{y}]_i
/// @f]
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
/// @details For the current kernel, for @f$ x_0 @f$ for example,
///
/// @f[
/// \langle x_0\rangle=(2\pi)^D\sigma_f^2\prod_i\abs{l_i}\sum_i[\mathbf{x}_i]_0[K^{-1}\mathbf{y}]_i
/// @f]
///
/// and similar for other @f$ x_i @f$ and @f$ p_i @f$.
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
/// @return The integral of @f$ (2\pi\hbar)^{\frac{d}{2}}\langle\rho^2\rangle @f$
/// @details For the current kernel,
///
/// @f[
/// (2\pi\hbar)^{\frac{d}{2}}\langle\rho^2\rangle=(\pi)^D\sigma_f^4\prod_i\abs{l_i}\mathbf{y}^{\mathsf{T}}K^{-1}K_1K^{-1}\mathbf{y}
/// @f]
///
/// where @f$ [K_1]_{mn}=\mathrm{exp}\left(-\frac{1}{4}\sum_i x_{mni}^2\right) @f$
/// if denote @f$ x_{mni}=\frac{[\mathbf{x}_m]_i-[\mathbf{x}_n]_i}{l_i} @f$.
static inline double calculate_purity(
	const Kernel::KernelParameter& KernelParams,
	const Eigen::MatrixXd& Features,
	const Eigen::VectorXd& InvLbl)
{
	static const double GlobalFactor = PurityFactor * std::pow(M_PI, Dim);
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
/// @details Denote @f$ x_{mni}=\frac{[\mathbf{x}_m]_i-[\mathbf{x}_n]_i}{l_i} @f$,
/// for the derivative of element @f$ K_{mn} @f$ over characteristic length @f$ l_j @f$
///
/// @f[
/// \frac{\partial K_{mn}}{\partial l_j}=\mathrm{exp}\left(-\frac{1}{2}\sum_i x_{mni}^2\right)\frac{x_{mnj}^2}{l_j}
/// @f]
///
/// and thus the derivative matrix is constructed.
static EigenVector<Eigen::MatrixXd> gaussian_derivative_over_char_length(
	const ClassicalPhaseVector& CharacteristicLength,
	const Eigen::MatrixXd& LeftFeature,
	const Eigen::MatrixXd& RightFeature,
	const Eigen::MatrixXd& GaussianKernelMatrix)
{
	auto calculate_factor = [&CharacteristicLength, &LeftFeature, &RightFeature](const int iRow, const int iCol) -> ClassicalPhaseVector
	{
		const auto& diff = (LeftFeature.col(iRow) - RightFeature.col(iCol)).array() / CharacteristicLength.array();
		return diff.square() / CharacteristicLength.array();
	};
	EigenVector<Eigen::MatrixXd> result(PhaseDim, GaussianKernelMatrix);
	const int Rows = LeftFeature.cols(), Cols = RightFeature.cols();
	const bool IsRowMajor = static_cast<bool>(GaussianKernelMatrix.IsRowMajor);
	const bool IsTrainingSet = LeftFeature.data() == RightFeature.data() && Rows == Cols;
	const std::vector<int> indices = get_indices(IsRowMajor ? Rows : Cols);
	std::for_each(
		std::execution::par_unseq,
		indices.cbegin(),
		indices.cend(),
		[&result, &calculate_factor, Rows, Cols, IsRowMajor, IsTrainingSet](int i) -> void
		{
			if (IsRowMajor)
			{
				// row major
				const int iRow = i;
				if (IsTrainingSet)
				{
					// training set, the matrix is symmetric
					for (int iCol = 0; iCol < iRow; iCol++)
					{
						const ClassicalPhaseVector& factor = calculate_factor(iRow, iCol);
						for (int iDim = 0; iDim < PhaseDim; iDim++)
						{
							result[iDim](iRow, iCol) *= factor[iDim];
						}
					}
				}
				else
				{
					// general rectangular
					for (int iCol = 0; iCol < Cols; iCol++)
					{
						const ClassicalPhaseVector& factor = calculate_factor(iRow, iCol);
						for (int iDim = 0; iDim < PhaseDim; iDim++)
						{
							result[iDim](iRow, iCol) *= factor[iDim];
						}
					}
				}
			}
			else
			{
				// column major
				const int iCol = i;
				if (IsTrainingSet)
				{
					// training set, the matrix is symmetric
					for (int iRow = iCol + 1; iRow < Rows; iRow++)
					{
						const ClassicalPhaseVector& factor = calculate_factor(iRow, iCol);
						for (int iDim = 0; iDim < PhaseDim; iDim++)
						{
							result[iDim](iRow, iCol) *= factor[iDim];
						}
					}
				}
				else
				{
					// general rectangular
					for (int iRow = 0; iRow < Rows; iRow++)
					{
						const ClassicalPhaseVector& factor = calculate_factor(iRow, iCol);
						for (int iDim = 0; iDim < PhaseDim; iDim++)
						{
							result[iDim](iRow, iCol) *= factor[iDim];
						}
					}
				}
			}
		});
	if (LeftFeature.data() == RightFeature.data() && Rows == Cols)
	{
		// training set, the matrix is symmetric
		for (int iDim = 0; iDim < PhaseDim; iDim++)
		{
			// as it is symmetric, only lower part of the matrix are necessary to calculate
			result[iDim].diagonal() = Eigen::VectorXd::Zero(Rows);
			result[iDim] = result[iDim].selfadjointView<Eigen::Lower>();
		}
	}
	return result;
}

/// @brief To calculate the derivatives over parameters
/// @param[in] KernelParams The unpacked parameters for all kernels
/// @param[in] LeftFeature The feature on the left
/// @param[in] RightFeature The feature on the right
/// @return The derivative of kernel matrix over parameters
static EigenVector<Eigen::MatrixXd> calculate_derivatives(
	const Kernel::KernelParameter& KernelParams,
	const Eigen::MatrixXd& LeftFeature,
	const Eigen::MatrixXd& RightFeature)
{
	assert(LeftFeature.rows() == PhaseDim && RightFeature.rows() == PhaseDim);
	const int Rows = LeftFeature.cols(), Cols = RightFeature.cols();
	const auto& [Noise, Gaussian] = KernelParams;
	const auto& [Magnitude, CharLength] = Gaussian;
	const Eigen::MatrixXd GaussianKernelMatrix = gaussian_kernel(CharLength, LeftFeature, RightFeature);
	EigenVector<Eigen::MatrixXd> result;
	// first, noise
	if (LeftFeature.data() == RightFeature.data())
	{
		result.push_back(2.0 * Noise * Eigen::MatrixXd::Identity(Rows, Cols));
	}
	else
	{
		result.push_back(Eigen::MatrixXd::Zero(Rows, Cols));
	}
	// second, Gaussian
	result.push_back(2.0 * Magnitude * GaussianKernelMatrix);
	const EigenVector<Eigen::MatrixXd> GaussianDerivOverCharLength = gaussian_derivative_over_char_length(CharLength, LeftFeature, RightFeature, GaussianKernelMatrix);
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

/// @brief To calculate the product of @f$ -K^{-1}\frac{\partial K}{\partial\theta}K^{-1} @f$
/// @param[in] InvDeriv The inverse of kernel times the derivatives
/// @param[in] Inverse The inverse of the kernel matirx
/// @return @f$ \frac{\partial K^{-1}}{\partial\theta}=-K^{-1}\frac{\partial K}{\partial\theta}K^{-1} @f$
static EigenVector<Eigen::MatrixXd> negative_inverse_derivative_inverse(const EigenVector<Eigen::MatrixXd>& InvDeriv, const Eigen::MatrixXd& Inverse)
{
	EigenVector<Eigen::MatrixXd> result = InvDeriv;
	for (auto& Mat : result)
	{
		Mat *= -Inverse;
	}
	return result;
}

/// @brief To calculate the product of @f$ -K^{-1}\frac{\partial K}{\partial\theta}K^{-1}\mathbf{y} @f$
/// @param[in] InvDeriv The inverse of kernel times the derivatives
/// @param[in] InvLbl The inverse of kernel matrix of training features times the training labels
/// @return @f$ \frac{\partial(K^{-1}\mathbf{y})}{\partial\theta}=-K^{-1}\frac{\partial K}{\partial\theta}K^{-1}\mathbf{y} @f$
static EigenVector<Eigen::VectorXd> negative_inverse_derivative_inverse_label(const EigenVector<Eigen::MatrixXd>& InvDeriv, const Eigen::VectorXd& InvLbl)
{
	EigenVector<Eigen::VectorXd> result(Kernel::NumTotalParameters, InvLbl);
	for (std::size_t iParam = 0; iParam < Kernel::NumTotalParameters; iParam++)
	{
		result[iParam] = -InvDeriv[iParam] * InvLbl;
	}
	return result;
}

/// @brief To calculate the derivative of @f$ \langle\rho\rangle @f$ over all parameters
/// @param[in] KernelParams The unpacked parameters for all kernels
/// @param[in] InvDerivInvLbl The inverse
/// @param[in] InvLbl The inverse of kernel matrix of training features times the training labels
/// @return The derivative of @f$ \langle\rho\rangle @f$ over all parameters
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

/// @brief To calculate the derivative of @f$ \langle\rho^2\rangle @f$ over all parameters
/// @param[in] KernelParams The unpacked parameters for all kernels
/// @param[in] Feature The feature
/// @param[in] InvDerivInvLbl The inverse
/// @param[in] InvLbl The inverse of kernel matrix of training features times the training labels
/// @return The derivative of @f$ \langle\rho\rangle @f$ over all parameters
static ParameterVector purity_derivatives(
	const Kernel::KernelParameter& KernelParams,
	const Eigen::MatrixXd& Feature,
	const EigenVector<Eigen::VectorXd>& NegInvDerivInvLbl,
	const Eigen::VectorXd& InvLbl)
{
	static const double GlobalFactor = PurityFactor * std::pow(M_PI, Dim);
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
	const EigenVector<Eigen::MatrixXd> ModifiedGaussianKernelDerivatives = [&Feature, &ModifiedGaussianKernel, CharLength = CharLength](void) -> EigenVector<Eigen::MatrixXd>
	{
		// as the characteristic lengths is changed in the modified Gaussian kernel,
		// the
		EigenVector<Eigen::MatrixXd> result = gaussian_derivative_over_char_length(CharLength, Feature, Feature, ModifiedGaussianKernel);
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

Kernel::Kernel(
	const ParameterVector& Parameter,
	const Eigen::MatrixXd& feature,
	const Eigen::VectorXd& label,
	const bool IsCalculateAverage,
	const bool IsCalculateDerivative) :
	// general calculations
	Params(Parameter),
	KernelParams(unpack_parameters(Parameter)),
	LeftFeature(feature),
	RightFeature(feature),
	KernelMatrix(get_kernel_matrix(KernelParams, feature, feature)),
	// calculations only for training set
	Label(label),
	DecompOfKernel(Eigen::LLT<Eigen::MatrixXd>(KernelMatrix)),
	LogDeterminant(2.0 * Eigen::MatrixXd(DecompOfKernel->matrixL()).diagonal().array().log().sum()),
	Inverse(DecompOfKernel->solve(Eigen::MatrixXd::Identity(KernelMatrix.rows(), KernelMatrix.cols()))),
	InvLbl(DecompOfKernel->solve(Label.value())),
	// calculation only for averages (which is training set only too)
	Population(IsCalculateAverage ? decltype(Population)(calculate_population(KernelParams, InvLbl.value())) : std::nullopt),
	FirstOrderAverage(IsCalculateAverage
			? decltype(FirstOrderAverage)(calculate_1st_order_average(KernelParams, feature, InvLbl.value()) / Population.value())
			: std::nullopt),
	Purity(IsCalculateAverage ? decltype(Purity)(calculate_purity(KernelParams, feature, InvLbl.value())) : std::nullopt),
	// calculation only for derivatives (which is training set only too)
	Derivatives(IsCalculateDerivative ? decltype(Derivatives)(calculate_derivatives(KernelParams, feature, feature)) : std::nullopt),
	InvDeriv(IsCalculateDerivative ? decltype(InvDeriv)(inverse_times_derivatives(Inverse.value(), Derivatives.value())) : std::nullopt),
	NegInvDerivInv(IsCalculateDerivative ? decltype(NegInvDerivInv)(negative_inverse_derivative_inverse(InvDeriv.value(), Inverse.value())) : std::nullopt),
	NegInvDerivInvLbl(IsCalculateDerivative
			? decltype(NegInvDerivInvLbl)(negative_inverse_derivative_inverse_label(InvDeriv.value(), InvLbl.value()))
			: std::nullopt),
	// calculation for both average and derivatives
	PopulationDeriv(IsCalculateAverage && IsCalculateDerivative
			? decltype(PopulationDeriv)(population_derivatives(KernelParams, NegInvDerivInvLbl.value(), InvLbl.value()))
			: std::nullopt),
	PurityDeriv(IsCalculateAverage && IsCalculateDerivative
			? decltype(PopulationDeriv)(purity_derivatives(KernelParams, feature, NegInvDerivInvLbl.value(), InvLbl.value()))
			: std::nullopt)
{
}

Kernel::Kernel(
	const ParameterVector& Parameter,
	const Eigen::MatrixXd& left_feature,
	const Eigen::MatrixXd& right_feature,
	const bool IsCalculateDerivative) :
	Params(Parameter),
	KernelParams(unpack_parameters(Parameter)),
	LeftFeature(left_feature),
	RightFeature(right_feature),
	KernelMatrix(get_kernel_matrix(KernelParams, LeftFeature, RightFeature)),
	// calculations only for training set
	Label(std::nullopt),
	DecompOfKernel(std::nullopt),
	LogDeterminant(std::nullopt),
	Inverse(std::nullopt),
	InvLbl(std::nullopt),
	// calculation only for averages (which is training set only too)
	Population(std::nullopt),
	FirstOrderAverage(std::nullopt),
	Purity(std::nullopt),
	// calculation only for derivatives (which is training set only too)
	Derivatives(IsCalculateDerivative ? decltype(Derivatives)(calculate_derivatives(KernelParams, LeftFeature, RightFeature)) : std::nullopt),
	InvDeriv(std::nullopt),
	NegInvDerivInv(std::nullopt),
	NegInvDerivInvLbl(std::nullopt),
	// calculation for both average and derivatives
	PopulationDeriv(std::nullopt),
	PurityDeriv(std::nullopt)
{
}
