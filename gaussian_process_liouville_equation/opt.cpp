/// @file opt.cpp
/// @brief Implementation of opt.h

#include "stdafx.h"

#include "opt.h"

#include "mc.h"
#include "pes.h"

/// The unpacked parameters for the kernels.
/// First is the noise, then the Gaussian kernel parameters: magnitude and characteristic lengths
using KernelParameter = std::tuple<double, std::tuple<double, ClassicalPhaseVector>>;
/// The training set for parameter optimization of one element of density matrix
/// passed as parameter to the optimization function.
/// First is feature, second is label.
using ElementTrainingSet = std::tuple<Eigen::MatrixXd, Eigen::VectorXd>;
/// The training set for optimization of all diagonal elements
/// altogether with regularization of population and energy.
/// First is separate training set for each diagonal elements,
/// then the energy for each PES and the total energy (last term in array)
/// then the purity. If purity is negative, no purity condition needed
using DiagonalTrainingSet = std::tuple<std::array<ElementTrainingSet, NumDiagonalElements>, std::array<double, NumDiagonalElements + 1>, double>;
/// The training set for optimization of all off-diagonal elements
/// First is all the training set, second is the purity,
/// third is the diagonal parameters, last is the diagonal \f$ K^{-1}y \f$ s
using OffDiagonalTrainingSet = std::tuple<std::array<ElementTrainingSet, NumElements>, double, ParameterVector, Eigen::MatrixXd>;

static const int NumOverallParameters = 1 + 1 + PhaseDim; ///< The overall number of parameters, include 1 for noise, 1 for magnitude of Gaussian and phasedim for characteristic length of Gaussian

/// @brief To unpack the parameters to components
/// @param[in] Parameter All the parameters
/// @return The unpacked parameters
static KernelParameter unpack_parameters(const ParameterVector& Parameter)
{
	assert(Parameter.size() == NumOverallParameters);
	// have all the parameters
	KernelParameter result;
	auto& [noise, gaussian] = result;
	auto& [magnitude, char_length] = gaussian;
	int iParam = 0;
	// first, noise
	noise = Parameter[iParam++];
	// second, Gaussian
	magnitude = std::exp2(Parameter[iParam++]);
	for (int iDim = 0; iDim < PhaseDim; iDim++)
	{
		char_length[iDim] = Parameter[iDim + iParam];
	}
	iParam += PhaseDim;
	return result;
}

/// @brief To calculate the Gaussian kernel
/// @param[in] LeftFeature The feature on the left, \f$ X_1 \f$
/// @param[in] RightFeature The feature on the right, \f$ X_2 \f$
/// @param[in] CharacteristicLength The characteristic lengths for all dimensions
/// @return The Gaussian kernel matrix
static Eigen::MatrixXd gaussian_kernel(
	const Eigen::MatrixXd& LeftFeature,
	const Eigen::MatrixXd& RightFeature,
	const Eigen::VectorXd& CharacteristicLength)
{
	const int Rows = LeftFeature.cols(), Cols = RightFeature.cols();
	Eigen::MatrixXd result = Eigen::MatrixXd::Zero(Rows, Cols);
	if (Rows >= Cols)
	{
#pragma omp parallel for
		for (int iRow = 0; iRow < Rows; iRow++)
		{
			for (int iCol = 0; iCol < Cols; iCol++)
			{
				const auto diff = LeftFeature.col(iRow) - RightFeature.col(iCol);
				result(iRow, iCol) += std::exp(-(diff.transpose() * CharacteristicLength.array().abs2().inverse().matrix().asDiagonal() * diff).value() / 2.0);
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
				const auto diff = LeftFeature.col(iRow) - RightFeature.col(iCol);
				result(iRow, iCol) += std::exp(-(diff.transpose() * CharacteristicLength.array().abs2().inverse().matrix().asDiagonal() * diff).value() / 2.0);
			}
		}
	}
	return result;
};

/// @brief To calculate the kernel matrix, \f$ K(X_1,X_2) \f$
/// @param[in] LeftFeature The feature on the left, \f$ X_1 \f$
/// @param[in] RightFeature The feature on the right, \f$ X_2 \f$
/// @param[in] Parameter The parameters for all kernels
/// @return The kernel matrix
/// @details This function calculates the kernel matrix with noise
///
/// \f[
/// k(\mathbf{x}_1,\mathbf{x}_2)=\sigma_f^2\mathrm{exp}\left(-\frac{(\mathbf{x}_1-\mathbf{x}_2)^\top M^{-1} (\mathbf{x}_1-\mathbf{x}_2)}{2}\right)+\sigma_n^2\delta(\mathbf{x}_1-\mathbf{x}_2)
/// \f]
///
/// where \f$ M \f$ is the characteristic matrix in the form of a diagonal matrix, whose elements are the characteristic length of each dimention.
///
/// When there are more than one feature, the kernel matrix follows \f$ K_{ij}=k(X_{1_{\cdot i}}, X_{2_{\cdot j}}) \f$.
static inline Eigen::MatrixXd get_kernel_matrix(
	const Eigen::MatrixXd& LeftFeature,
	const Eigen::MatrixXd& RightFeature,
	const ParameterVector& Parameter)
{
	assert(LeftFeature.rows() == PhaseDim && RightFeature.rows() == PhaseDim);
	const int Rows = LeftFeature.cols(), Cols = RightFeature.cols();
	Eigen::MatrixXd result = Eigen::MatrixXd::Zero(Rows, Cols);
	const auto& [Noise, Gaussian] = unpack_parameters(Parameter);
	const auto& [Magnitude, CharLength] = Gaussian;
	// first, noise
	if (LeftFeature.data() == RightFeature.data())
	{
		// in case when it is not training feature, the diagonal kernel (working as noise) is not needed
		result += std::pow(Noise, 2) * Eigen::MatrixXd::Identity(Rows, Cols);
	}
	// second, Gaussian
	result += std::pow(Magnitude, 2) * gaussian_kernel(LeftFeature, RightFeature, CharLength);
	return result;
}

/// @brief The function for nlopt optimizer to minimize, return the negative log likelihood
/// @param[in] x The input parameters, need to calculate the function value and gradient at this point
/// @param[out] grad The gradient at the given point. It will not be used
/// @param[in] params Other parameters. Here it is combination of kernel types and selected training set
/// @return The -ln(marginal likelihood + phasedim/2*ln(2*pi))
[[maybe_unused]] static double negative_log_marginal_likelihood(const ParameterVector& x, [[maybe_unused]] ParameterVector& grad, void* params)
{
	// get the parameters
	const auto& [TrainingFeature, TrainingLabel] = *static_cast<ElementTrainingSet*>(params);
	// get kernel
	const Eigen::MatrixXd& KernelMatrix = get_kernel_matrix(TrainingFeature, TrainingFeature, x);
	// calculate
	const Eigen::LLT<Eigen::MatrixXd> DecompOfKernel(KernelMatrix);
	const Eigen::MatrixXd& L = DecompOfKernel.matrixL();
	const Eigen::VectorXd& KInvLbl = DecompOfKernel.solve(TrainingLabel); // K^{-1}*y
	return (TrainingLabel.adjoint() * KInvLbl).value() / 2.0 + L.diagonal().array().abs().log().sum();
}

/// @brief The function for nlopt optimizer to minimize, return the LOOCV MSE
/// @param[in] x The input parameters, need to calculate the function value and gradient at this point
/// @param[out] grad The gradient at the given point. It will not be used
/// @param[in] params Other parameters. Here it is the training set
/// @return The squared error of LOOCV
[[maybe_unused]] static double leave_one_out_cross_validation(const ParameterVector& x, [[maybe_unused]] ParameterVector& grad, void* params)
{
	// get the parameters
	const auto& [TrainingFeature, TrainingLabel] = *static_cast<ElementTrainingSet*>(params);
	// get kernel
	const Eigen::MatrixXd& KernelMatrix = get_kernel_matrix(TrainingFeature, TrainingFeature, x);
	// prediction: mu_i=y_i-[K^{-1}*y]_i/[K^{-1}]_{ii}
	const Eigen::LLT<Eigen::MatrixXd> DecompOfKernel(KernelMatrix);
	const Eigen::MatrixXd& KInv = DecompOfKernel.solve(Eigen::MatrixXd::Identity(KernelMatrix.rows(), KernelMatrix.cols())); // K^{-1}
	const Eigen::VectorXd& KInvLbl = DecompOfKernel.solve(TrainingLabel);													 // K^{-1}*y
	return (KInvLbl.array() / KInv.diagonal().array()).abs2().sum();
}

/// @brief To calculate the population of one diagonal element by analytic integration of parameters
/// @param[in] Kernel The kernels with parameters set for the given element
/// @param[in] KInvLbl The inverse of kernel matrix of training features times the training labels
/// @return Population of the given element
static double calculate_population(const ParameterVector& Parameter, const Eigen::VectorXd& KInvLbl)
{
	static const double GlobalFactor = std::pow(2.0 * M_PI, Dim);
	double ppl = 0.0;
	[[maybe_unused]] const auto& [Noise, Gaussian] = unpack_parameters(Parameter);
	const auto& [Magnitude, CharLength] = Gaussian;
	ppl += GlobalFactor * std::pow(Magnitude, 2) * CharLength.prod() * KInvLbl.sum();
	return ppl;
}

/// @brief To calculate the average position and momentum of one diagonal element by analytic integration of parameters
/// @param[in] Features The training features (phase space coordinates) of the given density matrix element
/// @param[in] Kernel The kernels with parameters set for the given elements of density matrix
/// @param[in] KInvLbl The inverse of kernel matrix of training features times the training labels
/// @return Average position and momentum
static ClassicalPhaseVector calculate_1st_order_average(const Eigen::MatrixXd& Features, const ParameterVector& Parameter, const Eigen::VectorXd& KInvLbl)
{
	static const double GlobalFactor = std::pow(2.0 * M_PI, Dim);
	ClassicalPhaseVector r = ClassicalPhaseVector::Zero();
	[[maybe_unused]] const auto& [Noise, Gaussian] = unpack_parameters(Parameter);
	const auto& [Magnitude, CharLength] = Gaussian;
	r += GlobalFactor * std::pow(Magnitude, 2) * CharLength.prod() * Features * KInvLbl;
	return r;
}

/// @brief To calculate the average position and momentum of one diagonal element by monte carlo integration
/// @param[in] density The selected density matrices
/// @param[in] PESIndex The index of the potential energy surface, corresponding to (PESIndex, PESIndex) in density matrix
/// @return Average position and momentum
static ClassicalPhaseVector calculate_1st_order_average(const EvolvingDensity& density, const int PESIndex)
{
	const int NumPoints = density.size() / NumElements;
	const int BeginIndex = PESIndex * (NumPES + 1) * NumPoints;
	ClassicalPhaseVector r = ClassicalPhaseVector::Zero();
	for (int iPoint = BeginIndex; iPoint < BeginIndex + NumPoints; iPoint++)
	{
		const auto& [x, p, rho] = density[iPoint];
		for (int iDim = 0; iDim < Dim; iDim++)
		{
			r[iDim] += x[iDim];
			r[iDim + Dim] += p[iDim];
		}
	}
	return r / NumPoints;
}

/// @brief To calculate the average kinetic energy of one diagonal element by monte carlo integration
/// @param[in] density The selected density matrices
/// @param[in] PESIndex The index of the potential energy surface, corresponding to (PESIndex, PESIndex) in density matrix
/// @param[in] mass Mass of classical degree of freedom
/// @return Average kinetic energy
static double calculate_kinetic_average(const EvolvingDensity& density, const int PESIndex, const ClassicalDoubleVector& mass)
{
	const int NumPoints = density.size() / NumElements;
	const int BeginIndex = PESIndex * (NumPES + 1) * NumPoints;
	double T = 0.0;
	for (int iPoint = BeginIndex; iPoint < BeginIndex + NumPoints; iPoint++)
	{
		const auto& [x, p, rho] = density[iPoint];
		T += p.dot((p.array() / mass.array()).matrix()) / 2.0;
	}
	return T / NumPoints;
}

/// @brief To calculate the average potential energy of one diagonal element by monte carlo integration
/// @param[in] density The selected density matrices
/// @param[in] PESIndex The index of the potential energy surface, corresponding to (PESIndex, PESIndex) in density matrix
/// @return Average potential energy
static double calculate_potential_average(const EvolvingDensity& density, const int PESIndex)
{
	const int NumPoints = density.size() / NumElements;
	const int BeginIndex = PESIndex * (NumPES + 1) * NumPoints;
	double V = 0.0;
	for (int iPoint = BeginIndex; iPoint < BeginIndex + NumPoints; iPoint++)
	{
		const auto& [x, p, rho] = density[iPoint];
		V += adiabatic_potential(x)[PESIndex];
	}
	return V / NumPoints;
}

/// @brief To calculate the average position and momentum of one diagonal element by analytic integration of parameters
/// @param[in] Features The training features (phase space coordinates) of the given density matrix element
/// @param[in] Parameter The parameters of all kernels (magnitude and other)
/// @param[in] KInvLbl The inverse of kernel matrix of training features times the training labels
/// @param[in] IsDiagonal Whether the element is diagonal or not; if not, double the result
/// @return Average position and momentum
static double calculate_element_purity(
	const Eigen::MatrixXd& Features,
	const ParameterVector& Parameter,
	const Eigen::VectorXd& KInvLbl,
	const bool IsDiagonal = true)
{
	static const double Coefficient = std::pow(2.0 * M_PI * hbar, Dim) * std::pow(M_PI, Dim); // Coefficient for purity
	[[maybe_unused]] const auto& [Noise, Gaussian] = unpack_parameters(Parameter);
	const auto& [Magnitude, CharLength] = Gaussian;
	double result = Coefficient * std::pow(Magnitude, 4) * CharLength.prod() * (KInvLbl.transpose() * gaussian_kernel(Features, Features, M_SQRT2 * CharLength) * KInvLbl).value();
	if (IsDiagonal == true)
	{
		return result;
	}
	else
	{
		return 2.0 * result;
	}
}

/// @brief The function for nlopt optimizer to minimize, return the LOOCV MSE + Regularization of Normalization, Energy, and Purity (if applicable)
/// @param[in] x The input parameters of all diagonal elements, need to calculate the function value and gradient at this point
/// @param[out] grad The gradient at the given point. It will not be used
/// @param[in] params Other parameters. Here it is combination of selected training set, energies and purity
/// @return The squared error of LOOCV and the magnified squared error of normalization, energy, and purity (if applicable)
static double diagonal_loocv_with_constraints(const ParameterVector& x, ParameterVector& grad, void* params)
{
	static const double NormRegCoe = 1e3;	// Enlargement coefficient of normalization constraint
	static const double EngRegCoe = 1e3;	// Enlargement coefficient of energy constraint
	static const double PurityRegCoe = 1e3; // Enlargement coefficient of purity constraint
	double err = 0.0;
	double ppl = 0.0;
	double eng = 0.0;
	double purity = 0.0;
	// get the parameters
	const auto& [TrainingSets, Energies, Purity] = *static_cast<DiagonalTrainingSet*>(params);
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		const auto& [features, labels] = TrainingSets[iPES];
		if (features.size() != 0)
		{
			const int ParamBeginIndex = iPES * NumOverallParameters;
			const ParameterVector param(x.cbegin() + ParamBeginIndex, x.cbegin() + ParamBeginIndex + NumOverallParameters);
			ParameterVector element_grad(NumOverallParameters);
			// get loocv error
			err += leave_one_out_cross_validation(param, element_grad, const_cast<ElementTrainingSet*>(&TrainingSets[iPES]));
			if (grad.empty() == false)
			{
				std::copy(element_grad.cbegin(), element_grad.cend(), grad.begin() + ParamBeginIndex);
			}
			// get population
			const Eigen::VectorXd& KInvLbl = get_kernel_matrix(features, features, param).llt().solve(labels);
			const double ppl_i = calculate_population(param, KInvLbl);
			ppl += ppl_i;
			eng += ppl_i * Energies[iPES];
			// get purity, if necessary
			if (Purity > 0)
			{
				purity += calculate_element_purity(features, param, KInvLbl);
			}
		}
	}
	if (Purity < 0)
	{
		// no purity constraint necessary
		purity = Purity;
	}
	return err + NormRegCoe * std::pow(1.0 - ppl, 2) + EngRegCoe * std::pow(Energies[NumPES] - eng, 2) + PurityRegCoe * std::pow(Purity - purity, 2.0);
}

/// @brief The function for nlopt optimizer to minimize, return the LOOCV MSE + Regularization of Purity
/// @param[in] x The input parameters of all offdiagonal elements, need to calculate the function value and gradient at this point
/// @param[out] grad The gradient at the given point. It will not be used
/// @param[in] params Other parameters. Here it is combination of selected training set, purity, and diagonal element information
/// @return The squared error of LOOCV and the magnified squared error of purity
static double offdiagonal_loocv_with_purity(const ParameterVector& x, ParameterVector& grad, void* params)
{
	static const double PurityRegCoe = 1e3; // Enlargement coefficient of purity regularization
	double err = 0.0;
	double purity = 0.0;
	// get the parameters
	const auto& [TrainingSets, Purity, DiagonalParameters, DiagonalKInvLbls] = *static_cast<OffDiagonalTrainingSet*>(params);
	for (int iElement = 0; iElement < NumElements; iElement++)
	{
		const int iPES = iElement / NumPES, jPES = iElement % NumPES;
		const auto& [features, labels] = TrainingSets[iElement];
		if (features.size() != 0)
		{
			if (iPES == jPES)
			{
				// for diagonal elements, simply add its purity
				const int ParamBeginIndex = iPES * NumOverallParameters;
				const ParameterVector param(DiagonalParameters.cbegin() + ParamBeginIndex, DiagonalParameters.cbegin() + ParamBeginIndex + NumOverallParameters);
				const Eigen::VectorXd KInvLbl = DiagonalKInvLbls.col(iPES);
				purity += calculate_element_purity(features, param, KInvLbl, iPES == jPES);
			}
			else
			{
				// for off-diagonal elements, set the parameters, calculate loocv and purity
				const int ElementIndex = iPES * (NumPES - 1) + jPES + (iPES < jPES ? -1 : 0);
				const int ParamBeginIndex = ElementIndex * NumOverallParameters;
				const ParameterVector param(x.cbegin() + ParamBeginIndex, x.cbegin() + ParamBeginIndex + NumOverallParameters);
				ParameterVector element_grad(NumOverallParameters);
				// get loocv error
				err += leave_one_out_cross_validation(param, element_grad, const_cast<ElementTrainingSet*>(&TrainingSets[ElementIndex]));
				if (grad.empty() == false)
				{
					std::copy(element_grad.cbegin(), element_grad.cend(), grad.begin() + ParamBeginIndex);
				}
				// get purity
				const Eigen::VectorXd KInvLbl = get_kernel_matrix(features, features, param).llt().solve(labels);
				purity += calculate_element_purity(features, param, KInvLbl, iPES == jPES);
			}
		}
	}
	return err + PurityRegCoe * std::pow(Purity - purity, 2);
}

/// @brief To set up the initial parameters for one element
/// @param[in] xSigma The variance of position
/// @param[in] pSigma The variance of momentum
/// @return Vector containing all parameters for one element
static ParameterVector set_initial_parameter(
	const ClassicalDoubleVector& xSigma,
	const ClassicalDoubleVector& pSigma)
{
	ParameterVector result(NumOverallParameters);
	int iParam = 0;
	// first, noise
	result[iParam++] = std::exp2(-15.0); // diagonal weight is fixed at 1.0
	// second, Gaussian
	result[iParam++] = 0.0; // Gaussian weight is base-2 logarithmic
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

/// @brief To generate local and global optimizers that have been already set up
/// @param[inout] local_optimizer The optimizer with local optimization algorithm
/// @param[inout] global_optimizer The optimizer with global optimization algorithm
/// @param[in] LowerBound The lower boundary
/// @param[in] UpperBound The upper boundary
/// @param[in] Tolerance The relative tolerance of x and f
/// @param[in] InitialStepSize The size of initial step for local optimizer
/// @param[in] NumPopulation The number of initial points for stochastic global algorithm
static inline void optimizers_setup(
	nlopt::opt& local_optimizer,
	nlopt::opt& global_optimizer,
	const ParameterVector& LowerBound,
	const ParameterVector& UpperBound,
	const double Tolerance,
	const double InitialStepSize,
	const int NumPopulation)
{
	// construct a function for code reuse
	auto core_step = [&](nlopt::opt& optimizer) -> void
	{
		optimizer.set_lower_bounds(LowerBound);
		optimizer.set_upper_bounds(UpperBound);
		optimizer.set_xtol_rel(Tolerance);
		optimizer.set_ftol_rel(Tolerance);
	};
	// set up local optimizer
	core_step(local_optimizer);
	local_optimizer.set_initial_step(InitialStepSize);
	// set up global optimizer
	core_step(global_optimizer);
	switch (global_optimizer.get_algorithm())
	{
		case nlopt::algorithm::G_MLSL:
		case nlopt::algorithm::GN_MLSL:
		case nlopt::algorithm::GD_MLSL:
		case nlopt::algorithm::G_MLSL_LDS:
		case nlopt::algorithm::GN_MLSL_LDS:
		case nlopt::algorithm::GD_MLSL_LDS:
			global_optimizer.set_local_optimizer(local_optimizer);
			[[fallthrough]];
		case nlopt::algorithm::GN_CRS2_LM:
		case nlopt::algorithm::GN_ISRES:
			global_optimizer.set_population(NumPopulation);
			break;
		default:
			break;
	}
}

Optimization::Optimization(
	const Parameters& Params,
	const nlopt::algorithm LocalAlgorithm,
	const nlopt::algorithm GlobalAlgorithm):
	NumPoints(Params.get_number_of_selected_points()),
	InitialParameter(set_initial_parameter(Params.get_sigma_x0(), Params.get_sigma_p0()))
{
	static const double GaussKerMinCharLength = 1.0 / 100.0; // Minimal characteristic length for Gaussian kernel
	static const double Tolerance = 1e-5;					 // Absolute tolerance of independent variable (x) in optimization
	static const double InitialStepSize = 0.5;				 // Initial step size in optimization
	static const int PointsPerDimension = 10;				 // The number of points to search per parameter
	// set up bounds and parameters
	ParameterVector LowerBound(NumOverallParameters, std::numeric_limits<float>::min());
	ParameterVector UpperBound(NumOverallParameters, std::numeric_limits<float>::max());
	ParameterVectors[0].resize(NumOverallParameters, 0.0);
	// set up the bounds
	const ClassicalDoubleVector xSize = Params.get_xmax() - Params.get_xmin();
	const ClassicalDoubleVector pSize = Params.get_pmax() - Params.get_pmin();
	{
		int iParam = 0;
		// first, noise
		// diagonal weight is fixed
		LowerBound[iParam] = InitialParameter[iParam];
		UpperBound[iParam] = InitialParameter[iParam];
		iParam++;
		// second, Gaussian
		// weight is base-2 logarithmic
		LowerBound[iParam] = std::log2(InitialParameter[iParam - 1]);
		UpperBound[iParam] = 2.0 * InitialParameter[iParam] - LowerBound[iParam];
		iParam++;
		// dealing with x
		for (int iDim = 0; iDim < Dim; iDim++)
		{
			LowerBound[iParam] = GaussKerMinCharLength;
			UpperBound[iParam] = xSize[iDim];
			iParam++;
		}
		// dealing with p
		for (int iDim = 0; iDim < Dim; iDim++)
		{
			LowerBound[iParam] = GaussKerMinCharLength;
			UpperBound[iParam] = pSize[iDim];
			iParam++;
		}
	}
	// set up parameters and minimizers
	for (int iElement = 0; iElement < NumElements; iElement++)
	{
		ParameterVectors[iElement] = InitialParameter;
		NLOptLocalMinimizers.push_back(nlopt::opt(LocalAlgorithm, NumOverallParameters));
		NLOptGlobalMinimizers.push_back(nlopt::opt(GlobalAlgorithm, NumOverallParameters));
		optimizers_setup(
			*NLOptLocalMinimizers.rbegin(),
			*NLOptGlobalMinimizers.rbegin(),
			LowerBound,
			UpperBound,
			Tolerance,
			InitialStepSize,
			PointsPerDimension * NumOverallParameters);
	}
	// minimizer for all diagonal elements with regularization (population + energy)
	ParameterVector LowerBounds = LowerBound;
	ParameterVector UpperBounds = UpperBound;
	for (int iPES = 1; iPES < NumDiagonalElements; iPES++)
	{
		LowerBounds.insert(LowerBounds.cend(), LowerBound.cbegin(), LowerBound.cend());
		UpperBounds.insert(UpperBounds.cend(), UpperBound.cbegin(), UpperBound.cend());
	}
	NLOptLocalMinimizers.push_back(nlopt::opt(LocalAlgorithm, NumOverallParameters * NumDiagonalElements));
	NLOptGlobalMinimizers.push_back(nlopt::opt(GlobalAlgorithm, NumOverallParameters * NumDiagonalElements));
	optimizers_setup(
		*NLOptLocalMinimizers.rbegin(),
		*NLOptGlobalMinimizers.rbegin(),
		LowerBounds,
		UpperBounds,
		Tolerance,
		InitialStepSize,
		PointsPerDimension * NumOverallParameters * NumDiagonalElements);
	// minimizer for all off-diagonal elements with regularization (purity)
	if (NumPES != 1)
	{
		for (int iElement = NumDiagonalElements; iElement < NumOffDiagonalElements; iElement++)
		{
			LowerBounds.insert(LowerBounds.cend(), LowerBound.cbegin(), LowerBound.cend());
			UpperBounds.insert(UpperBounds.cend(), UpperBound.cbegin(), UpperBound.cend());
		}
		NLOptLocalMinimizers.push_back(nlopt::opt(LocalAlgorithm, NumOverallParameters * NumOffDiagonalElements));
		NLOptGlobalMinimizers.push_back(nlopt::opt(GlobalAlgorithm, NumOverallParameters * NumOffDiagonalElements));
		optimizers_setup(
			*NLOptLocalMinimizers.rbegin(),
			*NLOptGlobalMinimizers.rbegin(),
			LowerBounds,
			UpperBounds,
			Tolerance,
			InitialStepSize,
			PointsPerDimension * NumOverallParameters * NumOffDiagonalElements);
	}
}

/// @brief To construct the training set for single element
/// @param[in] density The vector containing all known density matrix
/// @param[in] ElementIndex The index of the element in density matrix
/// @param[in] NumPoints The number of points selected for parameter optimization
/// @return The training set of the corresponding element in density matrix
static ElementTrainingSet construct_element_training_set(
	const EvolvingDensity& density,
	const int ElementIndex,
	const int NumPoints)
{
	ElementTrainingSet result;
	const int BeginIndex = ElementIndex * NumPoints;
	// construct training feature (PhaseDim*N) and training labels (N*1)
	auto& [feature, label] = result;
	feature.resize(PhaseDim, NumPoints);
	label.resize(NumPoints);
	for (int iPoint = 0; iPoint < NumPoints; iPoint++)
	{
		const auto& [x, p, rho] = density[BeginIndex + iPoint];
		feature.col(iPoint) << x, p;
		label[iPoint] = get_density_matrix_element(rho, ElementIndex);
	}
	return result;
}

std::vector<int> Optimization::optimize_elementwise(
	const EvolvingDensity& density,
	const QuantumBoolMatrix& IsSmall,
	std::vector<nlopt::opt>& minimizers)
{
	static const nlopt::vfunc minimizing_function = leave_one_out_cross_validation;
	std::vector<int> result(NumElements, 0);
	// optimize element by element
	for (int iElement = 0; iElement < NumElements; iElement++)
	{
		// first, check if all points are very small or not
		if (IsSmall(iElement / NumPES, iElement % NumPES) == false)
		{
			ElementTrainingSet ts = construct_element_training_set(density, iElement, NumPoints);
			minimizers[iElement].set_min_objective(minimizing_function, &ts);
			// set variable for saving parameters and function value (marginal likelihood)
			double err = 0.0;
			try
			{
				minimizers[iElement].optimize(ParameterVectors[iElement], err);
			}
#ifdef NDEBUG
			catch (...)
			{
			}
#else
			catch (std::exception& e)
			{
				std::cerr << e.what() << std::endl;
			}
#endif
			// save the number of steps
			result[iElement] = minimizers[iElement].get_numevals();
			// set up kernels
			const auto& [feature, label] = ts;
			TrainingFeatures[iElement] = feature;
			KInvLbls[iElement] = get_kernel_matrix(feature, feature, ParameterVectors[iElement]).llt().solve(label);
		}
	}
	return result;
}

std::tuple<double, int> Optimization::optimize_diagonal(
	const EvolvingDensity& density,
	const ClassicalDoubleVector& mass,
	const double TotalEnergy,
	const double Purity,
	const QuantumBoolMatrix& IsSmall,
	nlopt::opt& minimizer)
{
	static const nlopt::vfunc minimizing_function = diagonal_loocv_with_constraints;
	// construct training set
	DiagonalTrainingSet dts;
	auto& [diagonal_element_ts, energies, purity] = dts;
	energies[NumPES] = TotalEnergy;
	purity = Purity;
	ParameterVector DiagonalParameters;
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		const int ElementIndex = iPES * (NumPES + 1);
		if (IsSmall(iPES, iPES) == false)
		{
			energies[iPES] = calculate_kinetic_average(density, iPES, mass) + calculate_potential_average(density, iPES);
			diagonal_element_ts[iPES] = construct_element_training_set(density, iPES * (NumPES + 1), NumPoints);
		}
		DiagonalParameters.insert(
			DiagonalParameters.cend(),
			ParameterVectors[ElementIndex].cbegin(),
			ParameterVectors[ElementIndex].cend());
	}
	minimizer.set_min_objective(minimizing_function, &dts);
	ParameterVector grad;
	// set variable for saving parameters and function value (marginal likelihood)
	double err = 0.0;
	try
	{
		minimizer.optimize(DiagonalParameters, err);
	}
#ifdef NDEBUG
	catch (...)
	{
	}
#else
	catch (std::exception& e)
	{
		std::cerr << e.what() << std::endl;
	}
#endif
	// set up kernels
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		if (IsSmall(iPES, iPES) == false)
		{
			const auto& [feature, label] = diagonal_element_ts[iPES];
			const int ElementIndex = iPES * (NumPES + 1);
			const int ParamIndex = iPES * NumOverallParameters;
			ParameterVector(DiagonalParameters.cbegin() + ParamIndex, DiagonalParameters.cbegin() + ParamIndex + NumOverallParameters).swap(ParameterVectors[ElementIndex]);
			KInvLbls[ElementIndex] = get_kernel_matrix(feature, feature, ParameterVectors[ElementIndex]).llt().solve(label);
		}
	}
	return std::make_tuple(err, minimizer.get_numevals());
}

std::tuple<double, int> Optimization::optimize_offdiagonal(
	const EvolvingDensity& density,
	const double Purity,
	const QuantumBoolMatrix& IsSmall,
	nlopt::opt& minimizer)
{
	static const nlopt::vfunc minimizing_function = offdiagonal_loocv_with_purity;// construct training set
	OffDiagonalTrainingSet odts;
	auto& [elements_ts, purity, DiagonalParameters, DiagonalKInvLbls] = odts;
	purity = Purity;
	DiagonalKInvLbls.resize(NumPoints, NumDiagonalElements);
	ParameterVector OffDiagonalParameters;
	for (int iElement = 0; iElement < NumElements; iElement++)
	{
		const int iPES = iElement / NumPES, jPES = iElement % NumPES;
		if (IsSmall(iPES, jPES) == false)
		{
			elements_ts[iElement] = construct_element_training_set(density, iElement, NumPoints);
		}
		if (iPES != jPES)
		{
			OffDiagonalParameters.insert(
				OffDiagonalParameters.cend(),
				ParameterVectors[iElement].cbegin(),
				ParameterVectors[iElement].cend());
		}
		else
		{
			DiagonalParameters.insert(
				DiagonalParameters.cend(),
				ParameterVectors[iElement].cbegin(),
				ParameterVectors[iElement].cend());
			DiagonalKInvLbls.col(iPES) << KInvLbls[iElement];
		}
	}
	minimizer.set_min_objective(minimizing_function, &odts);
	ParameterVector grad;
	// set variable for saving parameters and function value (marginal likelihood)
	double err = 0.0;
	try
	{
		minimizer.optimize(OffDiagonalParameters, err);
	}
#ifdef NDEBUG
	catch (...)
	{
	}
#else
	catch (std::exception& e)
	{
		std::cerr << e.what() << std::endl;
	}
#endif
	// set up kernels
	for (int iElement = 0; iElement < NumElements; iElement++)
	{
		const int iPES = iElement / NumPES, jPES = iElement % NumPES;
		if (iPES != jPES && IsSmall(iPES, jPES) == false)
		{
			const auto& [feature, label] = elements_ts[iElement];
			const int ElementIndex = iPES * (NumPES - 1) + jPES + (iPES < jPES ? -1 : 0);
			const int ParamIndex = ElementIndex * NumOverallParameters;
			ParameterVector(OffDiagonalParameters.cbegin() + ParamIndex, OffDiagonalParameters.cbegin() + ParamIndex + NumOverallParameters).swap(ParameterVectors[ElementIndex]);
			KInvLbls[ElementIndex] = get_kernel_matrix(feature, feature, ParameterVectors[ElementIndex]).llt().solve(label);
		}
	}
	return std::make_tuple(err, minimizer.get_numevals());
}

/// @details This function calculates the average position and momentum of each surfaces
/// by parameters and by MC average and compares them to judge the reoptimization.
bool Optimization::is_reoptimize(const EvolvingDensity& density, const QuantumBoolMatrix& IsSmall)
{
	static const double AbsoluteTolerance = 0.2;
	static const double RelativeTolerance = 0.1;
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		if (IsSmall(iPES, iPES) == false)
		{
			const int ElementIndex = iPES * (NumPES + 1);
			const Eigen::MatrixXd& Feature = TrainingFeatures[ElementIndex];
			const ParameterVector& Parameter = ParameterVectors[ElementIndex];
			const Eigen::VectorXd& KInvLbl = KInvLbls[ElementIndex];
			const ClassicalPhaseVector r_mc = calculate_1st_order_average(density, iPES);
			const ClassicalPhaseVector r_prm = calculate_1st_order_average(Feature, Parameter, KInvLbl)
				/ calculate_population(Parameter, KInvLbl);
			if (((r_prm - r_mc).array() / r_mc.array() > RelativeTolerance).any()
				&& ((r_prm - r_mc).array().abs() > AbsoluteTolerance).any())
			{
				return true;
			}
		}
	}
	return false;
}

/// @details Using Gaussian Process Regression (GPR) to depict phase space distribution.
/// First optimize elementwise, then optimize the diagonal with normalization and energy conservation
std::tuple<double, std::vector<int>> Optimization::optimize(
	const EvolvingDensity& density,
	const ClassicalDoubleVector& mass,
	const double TotalEnergy,
	const double Purity,
	const QuantumBoolMatrix& IsSmall)
{
	// any offdiagonal element is not small
	const bool OffDiagonalOptimization = [&](void) -> bool
	{
		for (int iElement = 0; iElement < NumElements; iElement++)
		{
			const int iPES = iElement / NumPES, jPES = iElement % NumPES;
			if (iPES != jPES && IsSmall(iPES, jPES) == false)
			{
				return true;
			}
		}
		return false;
	}();
	// construct a function for code reuse
	auto core_steps = [&](void) -> std::tuple<double, std::vector<int>>
	{
		std::tuple<double, std::vector<int>> result;
		auto& [err, steps] = result;
		// first, optimize with last step parameters
		steps = optimize_elementwise(density, IsSmall, NLOptLocalMinimizers);
		if (OffDiagonalOptimization == true)
		{
			// do not add purity condition for diagonal, so give a negative purity to indicate no necessity
			const auto& [DiagonalError, DiagonalStep] = optimize_diagonal(density, mass, TotalEnergy, -Purity, IsSmall, NLOptLocalMinimizers[NumElements]);
			const auto& [OffDiagonalError, OffDiagonalStep] = optimize_offdiagonal(density, Purity, IsSmall, NLOptLocalMinimizers[NumElements + 1]);
			err += DiagonalError + OffDiagonalError;
			steps.push_back(DiagonalStep);
			steps.push_back(OffDiagonalStep);
		}
		else
		{
			const auto& [DiagonalError, DiagonalStep] = optimize_diagonal(density, mass, TotalEnergy, Purity, IsSmall, NLOptLocalMinimizers[NumElements]);
			err += DiagonalError;
			steps.push_back(DiagonalStep);
			steps.push_back(0);
		}
		return result;
	};

	std::tuple<double, std::vector<int>> result = core_steps();
	// next, check if the averages meet; otherwise, reopt with initial parameter
	if (is_reoptimize(density, IsSmall) == true)
	{
		for (int iElement = 0; iElement < NumElements; iElement++)
		{
			ParameterVectors[iElement] = InitialParameter;
		}
		result = core_steps();
	}
	else
	{
		return result;
	}
	// // if the initial parameter does not work either, go to global optimizers
	// if (is_reoptimize(density, IsSmall) == true)
	// {
	// 	for (int iElement = 0; iElement < NumElements; iElement++)
	// 	{
	// 		ParameterVectors[iElement] = InitialParameter;
	// 	}
	// 	optimize_elementwise(density, IsSmall, NLOptGlobalMinimizers);
	// 	result = core_steps();
	// }
	return result;
}

/// @details First, calculate the total population before normalization by analytical integration.
///
/// Then, divide the whole density matrix over the total population to normalize
double Optimization::normalize(EvolvingDensity& density, const QuantumBoolMatrix& IsSmall) const
{
	double population = 0.0;
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		if (IsSmall(iPES, iPES) == false)
		{
			population += calculate_population(ParameterVectors[iPES * (NumPES + 1)], KInvLbls[iPES * (NumPES + 1)]);
		}
	}
	for (auto& [x, p, rho] : density)
	{
		rho /= population;
	}
	return population;
}

void Optimization::update_training_set(const EvolvingDensity& density, const QuantumBoolMatrix& IsSmall)
{
	for (int iElement = 0; iElement < NumElements; iElement++)
	{
		// first, check if all points are very small or not
		if (IsSmall(iElement / NumPES, iElement % NumPES) == false)
		{
			// select points for optimization
			const int BeginIndex = iElement * NumPoints;
			// construct training feature (PhaseDim*N) and training labels (N*1)
			Eigen::MatrixXd feature(PhaseDim, NumPoints);
			Eigen::VectorXd label(NumPoints);
			for (int iPoint = 0; iPoint < NumPoints; iPoint++)
			{
				const auto& [x, p, rho] = density[BeginIndex + iPoint];
				label[iPoint] = get_density_matrix_element(rho, iElement);
				feature.col(iPoint) << x, p;
			}
			// set up kernels
			TrainingFeatures[iElement] = feature;
			KInvLbls[iElement] = get_kernel_matrix(feature, feature, ParameterVectors[iElement]).llt().solve(label);
		}
	}
}

/// @details Using Gaussian Process Regression to predict by
///
/// \f[
/// E[p(\mathbf{f}_*|X_*,X,\mathbf{y})]=K(X_*,X)[K(X,X)+\sigma_n^2I]^{-1}\mathbf{y}
/// \f]
///
/// where \f$ X_* \f$ is the test features, \f$ X \f$ is the training features, and \f$ \mathbf{y} \f$ is the training labels.
///
/// Warning: must call optimize() first before callings of this function!
double Optimization::predict_element(
	const QuantumBoolMatrix& IsSmall,
	const ClassicalDoubleVector& x,
	const ClassicalDoubleVector& p,
	const int ElementIndex) const
{
	if (IsSmall(ElementIndex / NumPES, ElementIndex % NumPES) == false)
	{
		// not small, have something to do
		// generate feature
		Eigen::MatrixXd test_feat(PhaseDim, 1);
		test_feat << x, p;
		// predict
		// not using the saved kernels because of multi-threading, and this is const member function
		return (get_kernel_matrix(test_feat, TrainingFeatures[ElementIndex], ParameterVectors[ElementIndex]) * KInvLbls[ElementIndex]).value();
	}
	else
	{
		return 0;
	}
}

QuantumComplexMatrix Optimization::predict_matrix(
	const QuantumBoolMatrix& IsSmall,
	const ClassicalDoubleVector& x,
	const ClassicalDoubleVector& p) const
{
	using namespace std::literals::complex_literals;
	assert(x.size() == p.size());
	QuantumComplexMatrix result = QuantumComplexMatrix::Zero();
	for (int iElement = 0; iElement < NumElements; iElement++)
	{
		const int iPES = iElement / NumPES, jPES = iElement % NumPES;
		const double ElementValue = predict_element(IsSmall, x, p, iElement);
		if (iPES <= jPES)
		{
			result(jPES, iPES) += ElementValue;
		}
		else
		{
			result(iPES, jPES) += 1.0i * ElementValue;
		}
	}
	return result.selfadjointView<Eigen::Lower>();
}

Eigen::VectorXd Optimization::predict_elements(
	const QuantumBoolMatrix& IsSmall,
	const ClassicalVectors& x,
	const ClassicalVectors& p,
	const int ElementIndex) const
{
	assert(x.size() == p.size());
	const int NumPoints = x.size();
	if (IsSmall(ElementIndex / NumPES, ElementIndex % NumPES) == false)
	{
		// not small, have something to do
		// generate feature
		Eigen::MatrixXd test_feat(PhaseDim, NumPoints);
		for (int i = 0; i < NumPoints; i++)
		{
			test_feat.col(i) << x[i], p[i];
		}
		// predict
		// not using the saved kernels because of multi-threading, and this is const member function
		return get_kernel_matrix(test_feat, TrainingFeatures[ElementIndex], ParameterVectors[ElementIndex]) * KInvLbls[ElementIndex];
	}
	else
	{
		return Eigen::VectorXd::Zero(NumPoints);
	}
}

Eigen::VectorXd Optimization::print_element(const QuantumBoolMatrix& IsSmall, const Eigen::MatrixXd& PhaseGrids, const int ElementIndex) const
{
	if (IsSmall(ElementIndex / NumPES, ElementIndex % NumPES) == false)
	{
		// not using the saved kernels because this is const member function
		return get_kernel_matrix(PhaseGrids, TrainingFeatures[ElementIndex], ParameterVectors[ElementIndex]) * KInvLbls[ElementIndex];
	}
	else
	{
		return Eigen::VectorXd::Zero(PhaseGrids.cols());
	}
}

/// @details This function sums the purity of each element, which is
///
/// \f[
/// (2\pi\hbar)^{\frac{d}{2}}\sum_{i=0,j=0}^N{(2-\delta_{ij})}(\pi)^{\frac{d}{2}}(\sigma_f^{ij})^4\prod_{d=0}^{D-1}\left|{l_d^{ij}}\right|\mathbf{y}_{ij}^TK_{ij}^{-1}K^{\prime}_{ij}K_{ij}^{-1}\mathbf{y}
/// \f]
///
/// where \f$ (K^{\prime}_{ij})_{kl}=\mathrm{exp}\left[-\sum_{d=0}^{D-1}(((x_{ij})_{k,d}-(x_{ij})_{l,d})/2l_d)^2\right] \f$.
double Optimization::calculate_purity(const QuantumBoolMatrix& IsSmall) const
{
	double result = 0.0;
	for (int iElement = 0; iElement < NumElements; iElement++)
	{
		if (IsSmall(iElement / NumPES, iElement % NumPES) == false)
		{
			result += calculate_element_purity(
				TrainingFeatures[iElement],
				ParameterVectors[iElement],
				KInvLbls[iElement],
				iElement % NumPES == iElement / NumPES);
		}
	}
	return result;
}

/**
 * @details To calculate averages,
 *
 * \f{eqnarray*}{
 * \langle\rho\rangle_{ii}&=&(2\pi)^{\frac{d}{2}}(\sigma_f^{ii})^2\left|\Lambda^{ii}\right|^{-1}(1\ 1\ \dots\ 1)K_{ii}^{-1}\mathbf{y}_{ii} \\
 * \langle r\rangle_{ii}&=&(2\pi)^{\frac{d}{2}}(\sigma_f^{ii})^2\left|\Lambda^{ii}\right|^{-1}(r_0\ r_1\ \dots\ r_n)K_{ii}^{-1}\mathbf{y}_{ii} \\
 * \langle p^2\rangle_{ii}&=&(2\pi)^{\frac{d}{2}}(\sigma_f^{ii})^2\left|\Lambda^{ii}\right|^{-1}(p_0^2+l_{p_0}^2\ p_1^2+l_{p_1}^2\ \dots\ p_n^2+l_{p_n}^2)K_{ii}^{-1}\mathbf{y}_{ii}
 * \f}
 *
 * where the \f$ \Lambda \f$ matrix is the lower-triangular matrix used in Gaussian ARD kernel, \f$ r = x,p \f$,
 * \f$ l_p \f$ is the characteristic length of momentum.
 *
 * To integrate with Monte Carlo, \f$ I=\int f(x)dx \f$, one select a weight function \f$ w(x) \f$,
 * so that the integration becomes \f$ I=\int \frac{f(x)}{w(x)} w(x)dx \f$.
 *
 * To do that, one samples \f$ N \f$ points with the weight function, then \f$ I\approx\frac{1}{N}\sum\frac{f(x_i)}{w(x_i)} \f$.
 *
 * To calculate average of \f$ f=x,p,V,T \f$, one integrate \f$ <f>=\int f(x)w(x)dx \f$, or \f$ <f>\approx\frac{1}{N}\sum f(x_i) \f$
 */
Averages calculate_average(
	const Optimization& optimizer,
	const EvolvingDensity& density,
	const ClassicalDoubleVector& mass,
	const QuantumBoolMatrix& IsSmall,
	const int PESIndex)
{
	assert(PESIndex >= 0 && PESIndex < NumPES);
	const int ElementIndex = PESIndex * (NumPES + 1);
	if (IsSmall(PESIndex, PESIndex) == false)
	{
		const ParameterVector& Parameter = optimizer.ParameterVectors[ElementIndex];
		const Eigen::MatrixXd& Features = optimizer.TrainingFeatures[ElementIndex]; // get feature matrix, PhaseDim-by-NumPoints
		const Eigen::VectorXd& KInvLbl = optimizer.KInvLbls[ElementIndex];
		if (Features.size() != 0 && KInvLbl.size() != 0)
		{
			// not small, calculate by exact parameter integration
			return std::make_tuple(
				calculate_population(Parameter, KInvLbl),
				calculate_1st_order_average(Features, Parameter, KInvLbl),
				calculate_1st_order_average(density, PESIndex),
				calculate_kinetic_average(density, PESIndex, mass),
				calculate_potential_average(density, PESIndex));
		}
		else
		{
			// not initialized, only calculate the MC averages
			return std::make_tuple(
				0.0,
				ClassicalPhaseVector::Zero(),
				calculate_1st_order_average(density, PESIndex),
				calculate_kinetic_average(density, PESIndex, mass),
				calculate_potential_average(density, PESIndex));
		}
	}
	else
	{
		return std::make_tuple(0.0, ClassicalPhaseVector::Zero(), ClassicalPhaseVector::Zero(), 0.0, 0.0);
	}
}
