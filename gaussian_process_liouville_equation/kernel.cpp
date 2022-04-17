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
	auto& [magnitude, char_length, noise] = result;
	std::size_t iParam = 0;
	// first, magnitude
	magnitude = Parameter[iParam++];
	// second, Gaussian
	for (std::size_t iDim = 0; iDim < PhaseDim; iDim++)
	{
		char_length[iDim] = Parameter[iDim + iParam];
	}
	iParam += PhaseDim;
	// third, noise
	noise = Parameter[iParam++];
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
	auto calculate_element = [&CharacteristicLength, &LeftFeature, &RightFeature](const std::size_t iRow, const std::size_t iCol) -> double
	{
		const auto& diff = ((LeftFeature.col(iRow) - RightFeature.col(iCol)).array() / CharacteristicLength.array()).matrix();
		return std::exp(-diff.squaredNorm() / 2.0);
	};
	const std::size_t Rows = LeftFeature.cols(), Cols = RightFeature.cols();
	const bool IsTrainingSet = LeftFeature.data() == RightFeature.data() && Rows == Cols;
	Eigen::MatrixXd result = Eigen::MatrixXd::Zero(Rows, Cols);
	const std::vector<std::size_t> indices = get_indices(static_cast<bool>(result.IsRowMajor) ? Rows : Cols);
	std::for_each(
		std::execution::par_unseq,
		indices.cbegin(),
		indices.cend(),
		[&result, &calculate_element, Rows, Cols, IsTrainingSet](std::size_t i) -> void
		{
			if (static_cast<bool>(result.IsRowMajor))
			{
				// row major
				const std::size_t iRow = i;
				if (IsTrainingSet)
				{
					// training set, the matrix is symmetric
					for (std::size_t iCol = 0; iCol < iRow; iCol++)
					{
						result(iRow, iCol) += calculate_element(iRow, iCol);
					}
				}
				else
				{
					// general rectangular
					for (std::size_t iCol = 0; iCol < Cols; iCol++)
					{
						result(iRow, iCol) += calculate_element(iRow, iCol);
					}
				}
			}
			else
			{
				// column major
				const std::size_t iCol = i;
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
/// @details This function calculates the kernel matrix with noise. Denote @f$ x_{mni}=\frac{[\mathbf{x}_m]_i-[\mathbf{x}_n]_i}{l_i} @f$, @n
/// @f[
/// k(\mathbf{x}_1,\mathbf{x}_2)=\sigma_f^2\mathrm{exp}\left(-\frac{1}{2}\sum_ix_{12i}^2\right)+\sigma_n^2\delta(\mathbf{x}_1-\mathbf{x}_2)
/// @f] @n
/// where M is the characteristic matrix in the form of a diagonal matrix, whose elements are the characteristic length of each dimention. @n
/// When there are more than one feature, the kernel matrix follows @f$ K_{ij}=k(X_{1_{\cdot i}}, X_{2_{\cdot j}}) @f$.
static inline Eigen::MatrixXd get_kernel_matrix(
	const Kernel::KernelParameter& KernelParams,
	const Eigen::MatrixXd& LeftFeature,
	const Eigen::MatrixXd& RightFeature)
{
	assert(LeftFeature.rows() == PhaseDim && RightFeature.rows() == PhaseDim);
	const std::size_t Rows = LeftFeature.cols(), Cols = RightFeature.cols();
	Eigen::MatrixXd result = Eigen::MatrixXd::Zero(Rows, Cols);
	const auto& [Magnitude, CharLength, Noise] = KernelParams;
	// second, Gaussian
	result += gaussian_kernel(CharLength, LeftFeature, RightFeature);
	// third, noise
	if (LeftFeature.data() == RightFeature.data())
	{
		// in case when it is not training feature, the diagonal kernel (working as noise) is not needed
		result += power<2>(Noise) * Eigen::MatrixXd::Identity(Rows, Cols);
	}
	// "first", magnitude
	result *= power<2>(Magnitude);
	return result;
}

/// @brief To calculate the population of one diagonal element by analytic integration of parameters
/// @param[in] KernelParams The unpacked parameters for all kernels
/// @param[in] InvLbl The inverse of kernel matrix of training features times the training labels
/// @return The integral of @f$ \langle\rho\rangle @f$
/// @details For the current kernel, @n
/// @f[
/// \langle\rho\rangle=(2\pi)^D\sigma_f^2\prod_i\left|l_i\right|\sum_i[K^{-1}\mathbf{y}]_i
/// @f]
static inline double calculate_population(const Kernel::KernelParameter& KernelParams, const Eigen::VectorXd& InvLbl)
{
	static constexpr double GlobalFactor = power<Dim>(2.0 * M_PI);
	[[maybe_unused]] const auto& [Magnitude, CharLength, Noise] = KernelParams;
	double ppl = 0.0;
	ppl += GlobalFactor * power<2>(Magnitude) * CharLength.prod() * InvLbl.sum();
	return ppl;
}

/// @brief To calculate the average position and momentum of one diagonal element by analytic integration of parameters
/// @param[in] KernelParams The unpacked parameters for all kernels
/// @param[in] Features The features
/// @param[in] InvLbl The inverse of kernel matrix of training features times the training labels
/// @return Average positions and momenta
/// @details For the current kernel, for @f$ x_0 @f$ for example, @n
/// @f[
/// \langle x_0\rangle=(2\pi)^D\sigma_f^2\prod_i\left|l_i\right|\sum_i[\mathbf{x}_i]_0[K^{-1}\mathbf{y}]_i
/// @f] @n
/// and similar for other @f$ x_i @f$ and @f$ p_i @f$.
static inline ClassicalPhaseVector calculate_1st_order_average(
	const Kernel::KernelParameter& KernelParams,
	const Eigen::MatrixXd& Features,
	const Eigen::VectorXd& InvLbl)
{
	static constexpr double GlobalFactor = power<Dim>(2.0 * M_PI);
	[[maybe_unused]] const auto& [Magnitude, CharLength, Noise] = KernelParams;
	ClassicalPhaseVector r = ClassicalPhaseVector::Zero();
	r += GlobalFactor * power<2>(Magnitude) * CharLength.prod() * Features * InvLbl;
	return r;
}

/// @brief To calculate the average position and momentum of one diagonal element by analytic integration of parameters
/// @param[in] KernelParams The unpacked parameters for all kernels
/// @param[in] Features The features
/// @param[in] InvLbl The inverse of kernel matrix of training features times the training labels
/// @return The integral of @f$ (2\pi\hbar)^{\frac{d}{2}}\langle\rho^2\rangle @f$
/// @details For the current kernel, @n
/// @f[
/// (2\pi\hbar)^{\frac{d}{2}}\langle\rho^2\rangle=(\pi)^D\sigma_f^4\prod_i\left|l_i\right|\mathbf{y}^{\mathsf{T}}K^{-1}K_1K^{-1}\mathbf{y}
/// @f] @n
/// where @f$ [K_1]_{mn}=\mathrm{exp}\left(-\frac{1}{4}\sum_i x_{mni}^2\right) @f$
/// if denote @f$ x_{mni}=\frac{[\mathbf{x}_m]_i-[\mathbf{x}_n]_i}{l_i} @f$.
static inline double calculate_purity(
	const Kernel::KernelParameter& KernelParams,
	const Eigen::MatrixXd& Features,
	const Eigen::VectorXd& InvLbl)
{
	static constexpr double GlobalFactor = PurityFactor * power<Dim>(M_PI);
	[[maybe_unused]] const auto& [Magnitude, CharLength, Noise] = KernelParams;
	double result = 0.0;
	result += GlobalFactor * power<4>(Magnitude) * CharLength.prod() * (InvLbl.transpose() * gaussian_kernel(M_SQRT2 * CharLength, Features, Features) * InvLbl).value();
	return result;
}

/// @brief To calculate the derivative of gaussian kernel over characteristic lengths
/// @param[in] CharacteristicLength The characteristic length
/// @param[in] LeftFeature The left feature
/// @param[in] RightFeature The right feature
/// @return The derivative of gaussian kernel over characteristic lengths
/// @details Denote @f$ x_{mni}=\frac{[\mathbf{x}_m]_i-[\mathbf{x}_n]_i}{l_i} @f$,
/// for the derivative of element @f$ K_{mn} @f$ over characteristic length @f$ l_j @f$ @n
/// @f[
/// \frac{\partial K_{mn}}{\partial l_j}=\mathrm{exp}\left(-\frac{1}{2}\sum_i x_{mni}^2\right)\frac{x_{mnj}^2}{l_j}
/// @f] @n
/// and thus the derivative matrix is constructed.
static EigenVector<Eigen::MatrixXd> gaussian_derivative_over_char_length(
	const ClassicalPhaseVector& CharacteristicLength,
	const Eigen::MatrixXd& LeftFeature,
	const Eigen::MatrixXd& RightFeature)
{
	const Eigen::MatrixXd& GaussianKernelMatrix = gaussian_kernel(CharacteristicLength, LeftFeature, RightFeature);
	auto calculate_factor = [&CharacteristicLength, &LeftFeature, &RightFeature](const std::size_t iRow, const std::size_t iCol) -> ClassicalPhaseVector
	{
		const auto& diff = (LeftFeature.col(iRow) - RightFeature.col(iCol)).array() / CharacteristicLength.array();
		return diff.square() / CharacteristicLength.array();
	};
	EigenVector<Eigen::MatrixXd> result(PhaseDim, GaussianKernelMatrix);
	const std::size_t Rows = LeftFeature.cols(), Cols = RightFeature.cols();
	const bool IsRowMajor = static_cast<bool>(GaussianKernelMatrix.IsRowMajor);
	const bool IsTrainingSet = LeftFeature.data() == RightFeature.data() && Rows == Cols;
	const std::vector<std::size_t> indices = get_indices(IsRowMajor ? Rows : Cols);
	std::for_each(
		std::execution::par_unseq,
		indices.cbegin(),
		indices.cend(),
		[&result, &calculate_factor, Rows, Cols, IsRowMajor, IsTrainingSet](std::size_t i) -> void
		{
			if (IsRowMajor)
			{
				// row major
				const std::size_t iRow = i;
				if (IsTrainingSet)
				{
					// training set, the matrix is symmetric
					for (std::size_t iCol = 0; iCol < iRow; iCol++)
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
					for (std::size_t iCol = 0; iCol < Cols; iCol++)
					{
						const ClassicalPhaseVector& factor = calculate_factor(iRow, iCol);
						for (std::size_t iDim = 0; iDim < PhaseDim; iDim++)
						{
							result[iDim](iRow, iCol) *= factor[iDim];
						}
					}
				}
			}
			else
			{
				// column major
				const std::size_t iCol = i;
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
		});
	if (LeftFeature.data() == RightFeature.data() && Rows == Cols)
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
/// @param[in] KernelParams The unpacked parameters for all kernels
/// @param[in] LeftFeature The left feature
/// @param[in] RightFeature The right feature
/// @param[in] KernelMatrix The kernel matrix
/// @return The derivative of kernel matrix over parameters
static EigenVector<Eigen::MatrixXd> calculate_derivatives(
	const Kernel::KernelParameter& KernelParams,
	const Eigen::MatrixXd& LeftFeature,
	const Eigen::MatrixXd& RightFeature,
	const Eigen::MatrixXd& KernelMatrix)
{
	assert(LeftFeature.rows() == PhaseDim && RightFeature.rows() == PhaseDim);
	const std::size_t Rows = LeftFeature.cols(), Cols = RightFeature.cols();
	[[maybe_unused]] const auto& [Magnitude, CharLength, Noise] = KernelParams;
	EigenVector<Eigen::MatrixXd> result;
	// first, magnitude
	result.push_back(2.0 / Magnitude * KernelMatrix);
	// second, Gaussian
	const EigenVector<Eigen::MatrixXd> GaussianDerivOverCharLength = gaussian_derivative_over_char_length(CharLength, LeftFeature, RightFeature);
	const double MagSquare = power<2>(Magnitude);
	// third, noise
	for (const auto& Mat : GaussianDerivOverCharLength)
	{
		result.push_back(MagSquare * Mat);
	}
	if (LeftFeature.data() == RightFeature.data())
	{
		result.push_back(2.0 * MagSquare * Noise * Eigen::MatrixXd::Identity(Rows, Cols));
	}
	else
	{
		result.push_back(Eigen::MatrixXd::Zero(Rows, Cols));
	}
	return result;
}

/// @brief To calculate the inverse of kernel times the derivatives of kernel
/// @param[in] KernelParams The unpacked parameters for all kernels
/// @param[in] Inverse The inverse of the kernel matrix
/// @param[in] Derivatives The derivatives of the kernel matrix over parameters
/// @return The product
static EigenVector<Eigen::MatrixXd> inverse_times_derivatives(
	const Kernel::KernelParameter& KernelParams,
	const Eigen::MatrixXd& Inverse,
	const EigenVector<Eigen::MatrixXd>& Derivatives)
{
	[[maybe_unused]] const auto& [Magnitude, CharLength, Noise] = KernelParams;
	EigenVector<Eigen::MatrixXd> result(Kernel::NumTotalParameters, Inverse);
	// first, magnitude
	std::size_t iParam = 0;
	result[iParam] = 2.0 / Magnitude * Eigen::MatrixXd::Identity(Inverse.rows(), Inverse.cols());
	iParam++;
	// second, Gaussian
	for (std::size_t iDim = 0; iDim < PhaseDim; iDim++)
	{
		result[iDim + iParam] *= Derivatives[iDim + iParam];
	}
	iParam += PhaseDim;
	// third, noise
	result[iParam] = 2 * power<2>(Magnitude) * Noise * Inverse;
	iParam++;
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
/// @param[in] NegInvDerivInvLbl @f$ -K^{-1}\frac{\partial K}{\partial \theta}K^{-1} @f$
/// @param[in] InvLbl The inverse of kernel matrix of training features times the training labels
/// @return The derivative of @f$ \langle\rho\rangle @f$ over all parameters
static ParameterVector population_derivatives(
	const Kernel::KernelParameter& KernelParams,
	const EigenVector<Eigen::VectorXd>& NegInvDerivInvLbl,
	const Eigen::VectorXd& InvLbl)
{
	static constexpr double GlobalFactor = power<Dim>(2.0 * M_PI);
	[[maybe_unused]] const auto& [Magnitude, CharLength, Noise] = KernelParams;
	const double ThisTimeFactor = GlobalFactor * power<2>(Magnitude) * CharLength.prod();
	ParameterVector result(Kernel::NumTotalParameters);
	std::size_t iParam = 0;
	// first, magnitude
	result[iParam] = 0.0;
	iParam++;
	// second, Gaussian
	for (std::size_t iDim = 0; iDim < PhaseDim; iDim++)
	{
		result[iParam] = ThisTimeFactor * (InvLbl.sum() / CharLength[iDim] + NegInvDerivInvLbl[iParam].sum());
		iParam++;
	}
	// third, noise
	result[iParam] = ThisTimeFactor * NegInvDerivInvLbl[iParam].sum();
	iParam++;
	return result;
}

/// @brief To calculate the derivative of @f$ \langle\rho^2\rangle @f$ over all parameters
/// @param[in] KernelParams The unpacked parameters for all kernels
/// @param[in] Feature The feature
/// @param[in] NegInvDerivInvLbl @f$ -K^{-1}\frac{\partial K}{\partial \theta}K^{-1} @f$
/// @param[in] InvLbl The inverse of kernel matrix of training features times the training labels
/// @return The derivative of @f$ \langle\rho\rangle @f$ over all parameters
static ParameterVector purity_derivatives(
	const Kernel::KernelParameter& KernelParams,
	const Eigen::MatrixXd& Feature,
	const EigenVector<Eigen::VectorXd>& NegInvDerivInvLbl,
	const Eigen::VectorXd& InvLbl)
{
	static constexpr double GlobalFactor = PurityFactor * power<Dim>(M_PI);
	[[maybe_unused]] const auto& [Magnitude, CharLength, Noise] = KernelParams;
	const double ThisTimeFactor = GlobalFactor * power<4>(Magnitude) * CharLength.prod();
	ParameterVector result(Kernel::NumTotalParameters);
	std::size_t iParam = 0;
	const ClassicalPhaseVector ModifiedCharLength = M_SQRT2 * CharLength;
	const Eigen::MatrixXd ModifiedGaussianKernel = gaussian_kernel(ModifiedCharLength, Feature, Feature); // K_1 in document
	// first, magnitude
	result[iParam] = 0.0;
	iParam++;
	// second, Gaussian
	const EigenVector<Eigen::MatrixXd> ModifiedGaussianKernelDerivatives = [&Feature, &ModifiedCharLength](void) -> EigenVector<Eigen::MatrixXd>
	{
		// as the characteristic lengths is changed in the modified Gaussian kernel,
		// the result needed to multiply with the amplitude factor
		EigenVector<Eigen::MatrixXd> result = gaussian_derivative_over_char_length(ModifiedCharLength, Feature, Feature);
		for (Eigen::MatrixXd& deriv : result)
		{
			deriv *= M_SQRT2;
		}
		return result;
	}();
	for (std::size_t iDim = 0; iDim < PhaseDim; iDim++)
	{
		result[iParam] = ThisTimeFactor
			* ((InvLbl.transpose() * ModifiedGaussianKernel * InvLbl).value() / CharLength[iDim]
				+ (NegInvDerivInvLbl[iParam].transpose() * ModifiedGaussianKernel * InvLbl).value()
				+ (InvLbl.transpose() * ModifiedGaussianKernelDerivatives[iDim] * InvLbl).value()
				+ (InvLbl.transpose() * ModifiedGaussianKernel * NegInvDerivInvLbl[iParam]).value());
		iParam++;
	}
	// third, noise
	result[iParam] = ThisTimeFactor
		* ((NegInvDerivInvLbl[iParam].transpose() * ModifiedGaussianKernel * InvLbl).value()
			+ (InvLbl.transpose() * ModifiedGaussianKernel * NegInvDerivInvLbl[iParam]).value());
	iParam++;
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
			? decltype(FirstOrderAverage)(calculate_1st_order_average(KernelParams, feature, InvLbl.value()))
			: std::nullopt),
	Purity(IsCalculateAverage ? decltype(Purity)(calculate_purity(KernelParams, feature, InvLbl.value())) : std::nullopt),
	// calculation only for derivatives (which is training set only too)
	Derivatives(IsCalculateDerivative ? decltype(Derivatives)(calculate_derivatives(KernelParams, feature, feature, KernelMatrix)) : std::nullopt),
	InvDeriv(IsCalculateDerivative ? decltype(InvDeriv)(inverse_times_derivatives(KernelParams, Inverse.value(), Derivatives.value())) : std::nullopt),
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
	Derivatives(IsCalculateDerivative ? decltype(Derivatives)(calculate_derivatives(KernelParams, LeftFeature, RightFeature, KernelMatrix)) : std::nullopt),
	InvDeriv(std::nullopt),
	NegInvDerivInv(std::nullopt),
	NegInvDerivInvLbl(std::nullopt),
	// calculation for both average and derivatives
	PopulationDeriv(std::nullopt),
	PurityDeriv(std::nullopt)
{
}
