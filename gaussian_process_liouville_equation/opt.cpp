/// @file opt.cpp
/// @brief Implementation of opt.h

#include "stdafx.h"

#include "opt.h"

#include "input.h"
#include "kernel.h"
#include "mc_predict.h"

/// The lower and upper bound
using Bounds = std::array<ParameterVector, 2>;
/// The training sets of all elements
using AllTrainingSets = QuantumArray<ElementTrainingSet>;
/// The parameters passed to the optimization subroutine of a single element
using ElementTrainingParameters = std::tuple<const ElementTrainingSet&, const ElementTrainingSet&>;
/// The parameters passed to the optimization subroutine of all (diagonal) elements
/// including the training sets and an extra training set
using AnalyticalLooseFunctionParameters = std::tuple<const AllTrainingSets&, const AllTrainingSets&>;
/// The parameters used for population conservation
/// by analytical integral, including the training sets
using AnalyticalPopulationConstraintParameters = std::tuple<const AllTrainingSets&>;
/// The parameters used for energy conservation by analytical integral,
/// including the training sets, the energy on each surfaces and the total energy
using AnalyticalEnergyConstraintParameters = std::tuple<const AllTrainingSets&, const QuantumVector<double>&, const double>;
/// The parameters used for purity conservation by analytical integral,
/// including the training sets, and the exact initial purity
using AnalyticalPurityConstraintParameters = std::tuple<const AllTrainingSets&, const double>;

/// @brief To get the element of density matrix
/// @param[in] DensityMatrix The density matrix
/// @param[in] iPES The index of the row of the density matrix element
/// @param[in] jPES The index of the column of the density matrix element
/// @return The element of density matrix
/// @details If the index corresponds to upper triangular elements, gives real part. @n
/// If the index corresponds to lower triangular elements, gives imaginary part. @n
/// If the index corresponds to diagonal elements, gives the original value.
static inline double get_density_matrix_element(const QuantumMatrix<std::complex<double>>& DensityMatrix, const std::size_t iPES, const std::size_t jPES)
{
	if (iPES <= jPES)
	{
		return DensityMatrix(iPES, jPES).real();
	}
	else
	{
		return DensityMatrix(iPES, jPES).imag();
	}
}

/// @brief To construct the training set for all elements
/// @param[in] density The selected points in phase space for each element of density matrices
/// @return The training set of all elements in density matrix
static AllTrainingSets construct_training_sets(const AllPoints& density)
{
	auto construct_element_training_set = [&density](const std::size_t iPES, const std::size_t jPES) -> ElementTrainingSet
	{
		const ElementPoints& ElementDensity = density[iPES * NumPES + jPES];
		const std::size_t NumPoints = ElementDensity.size();
		// construct training feature (PhaseDim*N) and training labels (N*1)
		Eigen::MatrixXd feature = Eigen::MatrixXd::Zero(PhaseDim, NumPoints);
		Eigen::VectorXd label = Eigen::VectorXd::Zero(NumPoints);
		// first insert original points
		const std::vector<std::size_t> indices = get_indices(NumPoints);
		std::for_each(
			std::execution::par_unseq,
			indices.cbegin(),
			indices.cend(),
			[&ElementDensity, &feature, &label, iPES, jPES](std::size_t iPoint) -> void
			{
				const auto& [r, rho] = ElementDensity[iPoint];
				feature.col(iPoint) = r;
				label[iPoint] = get_density_matrix_element(rho, iPES, jPES);
			});
		// if all points are small, remove this element
		if ((label.array().abs() < epsilon).all())
		{
			feature.resize(0, 0);
			label.resize(0);
		}
		// this happens for coherence, where only real/imaginary part is important
		return std::make_tuple(feature, label);
	};
	AllTrainingSets result;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES < NumPES; jPES++)
		{
			const std::size_t ElementIndex = iPES * NumPES + jPES;
			if (!density[ElementIndex].empty())
			{
				result[ElementIndex] = construct_element_training_set(iPES, jPES);
			}
		}
	}
	return result;
}

/// @brief To update the kernels using the training sets
/// @param[in] ParameterVectors The parameters for all elements of density matrix
/// @param[in] TrainingSets The training sets of all elements of density matrix
/// @param[in] IsCalculateAverage Whether the kernels are used to calculate analytical averages or not
/// @param[in] IsCalculateDerivative Whether the kernels are used to calculate derivatives or not
/// @return Array of kernels
static OptionalKernels construct_predictors(
	const QuantumArray<ParameterVector>& ParameterVectors,
	const AllTrainingSets& TrainingSets,
	const bool IsCalculateAverage,
	const bool IsCalculateDerivative)
{
	OptionalKernels result;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES < NumPES; jPES++)
		{
			const std::size_t ElementIndex = iPES * NumPES + jPES;
			const auto& [feature, label] = TrainingSets[ElementIndex];
			if (!ParameterVectors[ElementIndex].empty() && feature.size() != 0)
			{
				result[ElementIndex].emplace(ParameterVectors[ElementIndex], feature, label, IsCalculateAverage, IsCalculateDerivative);
			}
			else
			{
				result[ElementIndex] = std::nullopt;
			}
		}
	}
	return result;
}

/// @details As there are constant members in Kernel class, the assignment is unavailable
void construct_predictors(
	OptionalKernels& kernels,
	const QuantumArray<ParameterVector>& ParameterVectors,
	const AllPoints& density)
{
	OptionalKernels k = construct_predictors(ParameterVectors, construct_training_sets(density), true, false);
	for (std::size_t iElement = 0; iElement < NumElements; iElement++)
	{
		if (k[iElement].has_value())
		{
			kernels[iElement].emplace(k[iElement].value());
		}
		else
		{
			kernels[iElement].reset();
		}
	}
}

static const double InitialMagnitude = 1.0; ///< The initial weight for all kernels
static const double InitialNoise = 1e-2;	///< The initial weight for noise

/// @brief To set up the values for the initial parameter
/// @param[in] rSigma The variance of all classical degree of freedom
/// @return Vector containing all parameters for one element
static ParameterVector set_initial_parameters(const ClassicalPhaseVector& rSigma)
{
	ParameterVector result(Kernel::NumTotalParameters);
	std::size_t iParam = 0;
	// first, magnitude
	result[iParam] = InitialMagnitude;
	iParam++;
	// second, Gaussian
	for (std::size_t iDim = 0; iDim < PhaseDim; iDim++)
	{
		result[iParam] = rSigma[iDim];
		iParam++;
	}
	// thrid, noise
	result[iParam++] = InitialNoise;
	return result;
}

/// @brief To set up the bounds for the parameter
/// @param[in] rSize The size of the box on all classical directions
/// @return Lower and upper bounds
static Bounds calculate_bounds(const ClassicalPhaseVector& rSize)
{
	static const double GaussKerMinCharLength = 1.0 / 100.0; // Minimal characteristic length for Gaussian kernel
	Bounds result;
	auto& [LowerBound, UpperBound] = result;
	// first, magnitude
	LowerBound.push_back(InitialMagnitude);
	UpperBound.push_back(InitialMagnitude);
	// second, Gaussian
	for (std::size_t iDim = 0; iDim < PhaseDim; iDim++)
	{
		LowerBound.push_back(GaussKerMinCharLength);
		UpperBound.push_back(rSize[iDim]);
	}
	// third, noise
	LowerBound.push_back(InitialNoise * InitialNoise / InitialMagnitude);
	UpperBound.push_back(std::sqrt(InitialNoise));
	return result;
}

/// @brief To set up the bounds for the parameter
/// @param[in] density The selected points in phase space of one element of density matrices
/// @return Lower and upper bounds
static Bounds calculate_bounds(const ElementPoints& density)
{
	assert(!density.empty());
	Bounds result;
	auto& [LowerBound, UpperBound] = result;
	// first, magnitude
	LowerBound.push_back(InitialMagnitude);
	UpperBound.push_back(InitialMagnitude);
	// second, Gaussian
	// lower bound of characteristic length, which is half the minimum distance between points
	const ClassicalPhaseVector stddev = calculate_standard_deviation_one_surface(density);
	const ClassicalPhaseVector lb = stddev / std::sqrt(density.size()), ub = 2.0 * stddev;
	LowerBound.insert(LowerBound.cend(), lb.data(), lb.data() + PhaseDim);
	UpperBound.insert(UpperBound.cend(), ub.data(), ub.data() + PhaseDim);
	// third, noise
	// diagonal weight is fixed
	LowerBound.push_back(InitialNoise * InitialNoise / InitialMagnitude);
	UpperBound.push_back(std::sqrt(InitialNoise));
	return result;
}

/// @brief To transform local parameters to global parameters by ln
/// @param[in] param The parameters for local optimizer, i.e., the normal parameter
/// @return The parameters for global optimizer, i.e., the changed parameter
static inline ParameterVector local_parameter_to_global(const ParameterVector& param)
{
	assert(param.size() == Kernel::NumTotalParameters);
	ParameterVector result = param;
	std::size_t iParam = 0;
	// first, magnitude
	iParam++;
	// second, gaussian
	iParam += PhaseDim;
	// third, noise
	result[iParam] = std::log(result[iParam]);
	iParam++;
	return result;
}

/// @brief To transform local gradient to global gradient
/// @param[in] param The parameters for local optimizer, i.e., the normal parameter
/// @param[in] grad The gradient from local optimizer, i.e., the normal gradient
/// @return The gradient for global optimizer, i.e., the changed gradient
/// @details Since for the magnitude of global optimizer is the ln of the local optimizer, @n
/// @f[
/// \frac{\partial}{\partial\mathrm{ln}x}=x\frac{\partial}{\partial x}
/// @f] @n
/// the multiplication of the normal magnitude is needed.
static inline ParameterVector local_gradient_to_global(const ParameterVector& param, const ParameterVector& grad)
{
	assert(param.size() == Kernel::NumTotalParameters);
	assert(grad.size() == Kernel::NumTotalParameters || grad.empty());
	ParameterVector result = grad;
	if (!result.empty())
	{
		std::size_t iParam = 0;
		// first, magnitude
		iParam++;
		// second, gaussian
		iParam += PhaseDim;
		// third, noise
		if (!result.empty())
		{
			result[iParam] *= param[iParam];
		}
		iParam++;
	}
	return result;
}

/// @brief To transform local parameters to global parameters by exp
/// @param[in] param The parameters for global optimizer, i.e., the changed parameter
/// @return The parameters for local optimizer, i.e., the normal parameter
static inline ParameterVector global_parameter_to_local(const ParameterVector& param)
{
	assert(param.size() == Kernel::NumTotalParameters);
	ParameterVector result = param;
	std::size_t iParam = 0;
	// first, magnitude
	iParam++;
	// second, gaussian
	iParam += PhaseDim;
	// third, noise
	result[iParam] = std::exp(result[iParam]);
	iParam++;
	return result;
}

/// @brief To set up bounds for optimizers
/// @param[inout] optimizers The local/global optimizers
/// @param[in] Bounds The lower and upper bounds for each element
/// @details The number of global optimizers must be exactly NumElements.
/// The number of local optimizers must be exactly 2 more than NumElements.
void set_optimizer_bounds(std::vector<nlopt::opt>& optimizers, const QuantumArray<Bounds>& Bounds)
{
	assert(optimizers.size() == NumElements || optimizers.size() == NumElements + 2);
	const bool IsLocalOptimizer = (optimizers.size() != NumElements);
	ParameterVector diagonal_lower_bound, full_lower_bound, diagonal_upper_bound, full_upper_bound;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES < NumPES; jPES++)
		{
			const std::size_t ElementIndex = iPES * NumPES + jPES;
			const auto& [LowerBound, UpperBound] = Bounds[ElementIndex];
			assert(LowerBound.size() == Kernel::NumTotalParameters && UpperBound.size() == Kernel::NumTotalParameters);
			if (IsLocalOptimizer)
			{
				optimizers[ElementIndex].set_lower_bounds(LowerBound);
				optimizers[ElementIndex].set_upper_bounds(UpperBound);
				if (iPES == jPES)
				{
					diagonal_lower_bound.insert(diagonal_lower_bound.cend(), LowerBound.cbegin(), LowerBound.cend());
					diagonal_upper_bound.insert(diagonal_upper_bound.cend(), UpperBound.cbegin(), UpperBound.cend());
				}
				full_lower_bound.insert(full_lower_bound.cend(), LowerBound.cbegin(), LowerBound.cend());
				full_upper_bound.insert(full_upper_bound.cend(), UpperBound.cbegin(), UpperBound.cend());
			}
			else
			{
				optimizers[ElementIndex].set_lower_bounds(local_parameter_to_global(LowerBound));
				optimizers[ElementIndex].set_upper_bounds(local_parameter_to_global(UpperBound));
			}
		}
	}
	if (IsLocalOptimizer)
	{
		optimizers[NumElements].set_lower_bounds(diagonal_lower_bound);
		optimizers[NumElements].set_upper_bounds(diagonal_upper_bound);
		optimizers[NumElements + 1].set_lower_bounds(full_lower_bound);
		optimizers[NumElements + 1].set_upper_bounds(full_upper_bound);
	}
}

Optimization::Optimization(
	const InitialParameters& InitParams,
	const double InitialTotalEnergy,
	const double InitialPurity,
	const nlopt::algorithm LocalAlgorithm,
	const nlopt::algorithm ConstraintAlgorithm,
	const nlopt::algorithm GlobalAlgorithm,
	const nlopt::algorithm GlobalSubAlgorithm) :
	TotalEnergy(InitialTotalEnergy),
	Purity(InitialPurity),
	mass(InitParams.get_mass()),
	InitialParameter(set_initial_parameters(InitParams.get_sigma_r0())),
	InitialParameterForGlobalOptimizer(local_parameter_to_global(InitialParameter))
{
	static constexpr std::size_t MaximumEvaluations = 100000; // Maximum evaluations for global optimizer

	// set up parameters and minimizers
	auto set_optimizer = [](nlopt::opt& optimizer) -> void
	{
		static constexpr double RelativeTolerance = 1e-5;  // Relative tolerance
		static constexpr double AbsoluteTolerance = 1e-15; // Relative tolerance
		static constexpr double InitialStepSize = 0.5;	   // Initial step size in optimization
		optimizer.set_xtol_rel(RelativeTolerance);
		optimizer.set_ftol_rel(RelativeTolerance);
		optimizer.set_xtol_abs(AbsoluteTolerance);
		optimizer.set_ftol_abs(AbsoluteTolerance);
		if (std::string(optimizer.get_algorithm_name()).find("(local, no-derivative)") != std::string::npos)
		{
			optimizer.set_initial_step(InitialStepSize);
		}
	};
	// set up a local optimizer using given algorithm
	auto set_subsidiary_optimizer = [&set_optimizer](nlopt::opt& optimizer, const nlopt::algorithm Algorithm) -> void
	{
		nlopt::opt result(Algorithm, optimizer.get_dimension());
		set_optimizer(result);
		optimizer.set_local_optimizer(result);
	};

	// minimizer for each element
	for (std::size_t iElement = 0; iElement < NumElements; iElement++)
	{
		ParameterVectors[iElement] = InitialParameter;
		NLOptLocalMinimizers.push_back(nlopt::opt(LocalAlgorithm, Kernel::NumTotalParameters));
		set_optimizer(NLOptLocalMinimizers.back());
		NLOptGlobalMinimizers.push_back(nlopt::opt(GlobalAlgorithm, Kernel::NumTotalParameters));
		set_optimizer(NLOptGlobalMinimizers.back());
		NLOptGlobalMinimizers.back().set_maxeval(MaximumEvaluations);
		switch (NLOptGlobalMinimizers.back().get_algorithm())
		{
		case nlopt::algorithm::G_MLSL:
		case nlopt::algorithm::GN_MLSL:
		case nlopt::algorithm::GD_MLSL:
		case nlopt::algorithm::G_MLSL_LDS:
		case nlopt::algorithm::GN_MLSL_LDS:
		case nlopt::algorithm::GD_MLSL_LDS:
			set_subsidiary_optimizer(NLOptGlobalMinimizers.back(), GlobalSubAlgorithm);
			break;
		default:
			break;
		}
	}
	// minimizer for all diagonal elements with regularization (population + energy + purity)
	NLOptLocalMinimizers.push_back(nlopt::opt(nlopt::algorithm::AUGLAG_EQ, Kernel::NumTotalParameters * NumPES));
	set_optimizer(NLOptLocalMinimizers.back());
	set_subsidiary_optimizer(NLOptLocalMinimizers.back(), ConstraintAlgorithm);
	// minimizer for all off-diagonal elements with regularization (purity)
	// the dimension is set to be all elements, but the diagonal elements have no effect
	// this is for simpler code, while the memory cost is larger but acceptable
	NLOptLocalMinimizers.push_back(nlopt::opt(nlopt::algorithm::AUGLAG_EQ, Kernel::NumTotalParameters * NumElements));
	set_optimizer(NLOptLocalMinimizers.back());
	set_subsidiary_optimizer(NLOptLocalMinimizers.back(), ConstraintAlgorithm);
	// set up bounds
	QuantumArray<Bounds> bounds;
	bounds.fill(calculate_bounds(InitParams.get_rmax() - InitParams.get_rmin()));
	set_optimizer_bounds(NLOptLocalMinimizers, bounds);
	set_optimizer_bounds(NLOptGlobalMinimizers, bounds);
}

/// @brief If the value is abnormal (inf, nan), set it to be the extreme
/// @param[inout] d The value to judge
static inline void make_normal(double& d)
{
	switch (std::fpclassify(d))
	{
	case FP_NAN:
	case FP_INFINITE:
		d = std::numeric_limits<double>::max();
		break;
	default: // FP_NORMAL, FP_SUBNORMAL and FP_ZERO
		break;
	}
}

/// @brief The function for nlopt optimizer to minimize
/// @param[in] x The input parameters, need to calculate the function value and gradient at this point
/// @param[out] grad The gradient at the given point.
/// @param[in] params Other parameters. Here it is the training set and the extra training set
/// @return The error
/// @details The error is the squared error. @n
/// For the training set, difference is @f$ [K^{-1}\vec{y}]_i/[K^{-1}]_{ii} @f$; @n
/// For the extra training set, difference is @f$ K(\vec{x}_i,X)K^{-1}\vec{y}-y_i @f$. @n
static double loose_function(const ParameterVector& x, ParameterVector& grad, void* params)
{
	// unpack the parameters
	const auto& [TrainingSet, ExtraTrainingSet] = *static_cast<ElementTrainingParameters*>(params);
	const auto& [TrainingFeature, TrainingLabel] = TrainingSet;
	const auto& [ExtraTrainingFeature, ExtraTrainingLabel] = ExtraTrainingSet;
	// construct the kernel, then pass it to the error function
	const Kernel kernel(x, TrainingFeature, TrainingLabel, false, !grad.empty());
	const Kernel ExtraKernel(x, ExtraTrainingFeature, TrainingFeature, !grad.empty());
	// about the training part
	const Eigen::VectorXd& InvLbl = kernel.get_inverse_label();
	const Eigen::MatrixXd& Inverse = kernel.get_inverse();
	// and its array wrappers
	const auto& InvLblArr = InvLbl.array();
	const auto& InvDiagArr = Inverse.diagonal().array();
	// about the extra training part
	const Eigen::MatrixXd& K_Extra_Training = ExtraKernel.get_kernel();
	const auto& PredDiff = K_Extra_Training * InvLbl - ExtraTrainingLabel;
	double result = (InvLblArr / InvDiagArr).square().sum() + PredDiff.squaredNorm();
	if (!grad.empty())
	{
		const EigenVector<Eigen::MatrixXd>& NegInvDerivInvVec = kernel.get_negative_inv_deriv_inv();
		const EigenVector<Eigen::VectorXd>& NegInvDerivInvLblVec = kernel.get_negative_inv_deriv_inv_lbl();
		const EigenVector<Eigen::MatrixXd>& ExtraKernelDeriv = ExtraKernel.get_derivatives();
		for (std::size_t iParam = 0; iParam < Kernel::NumTotalParameters; iParam++)
		{
			const auto& NegInvDerivInvDiagArr = NegInvDerivInvVec[iParam].diagonal().array();
			const Eigen::VectorXd& NegInvDerivInvLbl = NegInvDerivInvLblVec[iParam];
			const auto& NegInvDerivInvLblArr = NegInvDerivInvLbl.array();
			grad[iParam] = 2.0 * ((InvLblArr * NegInvDerivInvLblArr * InvLblArr - InvLblArr.square() * NegInvDerivInvDiagArr) / InvDiagArr.pow(3)).sum()
				+ 2.0 * PredDiff.dot((ExtraKernelDeriv[iParam] * InvLbl + K_Extra_Training * NegInvDerivInvLbl));
		}
	}
	make_normal(result);
	for (double& d : grad)
	{
		make_normal(d);
	}
	return result;
}

/// @brief The wrapper function for nlopt global optimizer to minimize
/// @param[in] x The input parameters
/// @param[out] grad The gradient at the given point.
/// @param[in] params Other parameters. Here it is the training set
/// @return The error
static double loose_function_global_wrapper(const ParameterVector& x, ParameterVector& grad, void* params)
{
	// save local gradient
	ParameterVector grad_local = grad;
	const ParameterVector x_local = global_parameter_to_local(x);
	const double result = loose_function(x_local, grad_local, params);
	grad = local_gradient_to_global(x_local, grad_local);
	return result;
}

/// @brief To get the name of the element
/// @param[in] iPES The index of the row of the density matrix element
/// @param[in] jPES The index of the column of the density matrix element
/// @return The name of the element
/// @details For diagonal elements, return exactly its name. @n
/// For strict-lower elements, return the imaginary part of its name. @n
/// For strict-upper elements, return the real part of its tranpose element.
static inline std::string get_element_name(const std::size_t iPES, const std::size_t jPES)
{
	if (iPES == jPES)
	{
		// diagonal elements
		return "rho[" + std::to_string(iPES) + "][" + std::to_string(jPES) + "]";
	}
	else if (iPES < jPES)
	{
		// strict upper elements
		return "Re(rho[" + std::to_string(jPES) + "][" + std::to_string(iPES) + "])";
	}
	else
	{
		// strict lower elements
		return "Im(rho[" + std::to_string(iPES) + "][" + std::to_string(jPES) + "])";
	}
}

/// @brief To optimize parameters of each density matrix element based on the given density
/// @param[in] TrainingSets The training features and labels of all elements
/// @param[in] ExtraTrainingSets The extra features and labels that only used for training (but not for prediction)
/// @param[inout] minimizers The minimizers to use
/// @param[inout] ParameterVectors The parameters for all elements of density matrix
/// @return The total error and the optimization steps of each element
static Optimization::Result optimize_elementwise(
	const AllTrainingSets& TrainingSets,
	const AllTrainingSets& ExtraTrainingSets,
	std::vector<nlopt::opt>& minimizers,
	QuantumArray<ParameterVector>& ParameterVectors)
{
	auto is_global = [](const nlopt::opt& optimizer) -> bool
	{
		return std::string(optimizer.get_algorithm_name()).find("global") != std::string::npos || optimizer.get_algorithm() == nlopt::algorithm::GN_ESCH;
	}; // judge if it is global optimizer by checking its name
	double total_error = 0.0;
	std::vector<std::size_t> num_steps(NumElements, 0);
	// optimize element by element
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES < NumPES; jPES++)
		{
			const std::size_t ElementIndex = iPES * NumPES + jPES;
			ElementTrainingParameters etp = std::tie(TrainingSets[ElementIndex], ExtraTrainingSets[ElementIndex]);
			if (std::get<0>(TrainingSets[ElementIndex]).size() != 0)
			{
				if (is_global(minimizers[ElementIndex]))
				{
					minimizers[ElementIndex].set_min_objective(loose_function_global_wrapper, static_cast<void*>(&etp));
				}
				else
				{
					minimizers[ElementIndex].set_min_objective(loose_function, static_cast<void*>(&etp));
				}
				double err = 0.0;
				try
				{
					minimizers[ElementIndex].optimize(ParameterVectors[ElementIndex], err);
				}
#ifdef NDEBUG
				catch (...)
				{
				}
#else
				catch (std::exception& e)
				{
					spdlog::error("Optimization of {} failed by {}", get_element_name(iPES, jPES), e.what());
				}
#endif
				spdlog::info("Error of {} = {}.", get_element_name(iPES, jPES), err);
				total_error += err;
				num_steps[ElementIndex] = minimizers[ElementIndex].get_numevals();
			}
			else
			{
				spdlog::info("{} is small.", get_element_name(iPES, jPES));
			}
		}
	}
	return std::make_tuple(total_error, num_steps);
}

/// @brief To calculate the overall loose for diagonal elements
/// @param[in] x The input parameters, need to calculate the function value and gradient at this point
/// @param[out] grad The gradient at the given point
/// @param[in] params Other parameters. Here it is the diagonal training sets
/// @return The diagonal errors
static double diagonal_loose(const ParameterVector& x, ParameterVector& grad, void* params)
{
	const auto& [TrainingSets, ExtraTrainingSets] = *static_cast<AnalyticalLooseFunctionParameters*>(params);
	double err = 0.0;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		const std::size_t ElementIndex = iPES * NumPES + iPES;
		[[maybe_unused]] const auto& [feature, label] = TrainingSets[ElementIndex];
		const std::size_t BeginIndex = iPES * Kernel::NumTotalParameters;
		const std::size_t EndIndex = BeginIndex + Kernel::NumTotalParameters;
		if (feature.size() != 0)
		{
			const ParameterVector x_element(x.cbegin() + BeginIndex, x.cbegin() + EndIndex);
			ParameterVector grad_element(!grad.empty() ? Kernel::NumTotalParameters : 0);
			ElementTrainingParameters etp = std::tie(TrainingSets[ElementIndex], ExtraTrainingSets[ElementIndex]);
			err += loose_function(x_element, grad_element, static_cast<void*>(&etp));
			if (!grad.empty())
			{
				std::copy(grad_element.cbegin(), grad_element.cend(), grad.begin() + BeginIndex);
			}
		}
		else
		{
			// element is small while gradient is needed, set gradient 0
			if (!grad.empty())
			{
				std::fill(grad.begin() + BeginIndex, grad.begin() + EndIndex, 0.0);
			}
		}
	}
	make_normal(err);
	for (double& d : grad)
	{
		make_normal(d);
	}
	return err;
}

/// @brief To construct parameters for all elements from parameters of diagonal elements
/// @param[in] x The parameters of diagonal elements
/// @return Parameters of all elements, leaving off-diagonal parameters blank
static QuantumArray<ParameterVector> construct_all_parameters_from_diagonal(const ParameterVector& x)
{
	assert(x.size() == NumPES * Kernel::NumTotalParameters);
	QuantumArray<ParameterVector> result;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES < NumPES; jPES++)
		{
			if (iPES == jPES)
			{
				result[iPES * NumPES + jPES] = ParameterVector(
					x.cbegin() + iPES * Kernel::NumTotalParameters,
					x.cbegin() + (iPES + 1) * Kernel::NumTotalParameters);
			}
			else
			{
				result[iPES * NumPES + jPES] = ParameterVector(Kernel::NumTotalParameters, 0.0);
			}
		}
	}
	return result;
}

/// @brief To calculate the error on population constraint by analytical integral of parameters
/// @param[in] x The input parameters, need to calculate the function value and gradient at this point
/// @param[out] grad The gradient at the given point
/// @param[in] params Other parameters. Here it is the diagonal training sets
/// @return @f$ \langle\rho\rangle_{\mathrm{parameters}} - 1 @f$
[[maybe_unused]] static double diagonal_analytical_integral_population_constraint(const ParameterVector& x, ParameterVector& grad, void* params)
{
	[[maybe_unused]] const auto& [TrainingSets] = *static_cast<AnalyticalPopulationConstraintParameters*>(params);
	// construct predictors
	const QuantumArray<ParameterVector> all_params = construct_all_parameters_from_diagonal(x);
	const OptionalKernels Kernels = construct_predictors(all_params, TrainingSets, true, !grad.empty());
	// calculate population of each surfaces
	double result = calculate_population(Kernels) - 1.0;
	// calculate_derivatives
	if (!grad.empty())
	{
		grad = population_derivative(Kernels);
	}
	make_normal(result);
	for (double& d : grad)
	{
		make_normal(d);
	}
	return result;
}

/// @brief To calculate the error on energy constraint by analytical integral of parameters
/// @param[in] x The input parameters, need to calculate the function value and gradient at this point
/// @param[out] grad The gradient at the given point
/// @param[in] params Other parameters. Here it is the diagonal training sets, total energy and mass
/// @return @f$ \langle E\rangle_{\mathrm{parameters}} - E_0 @f$
[[maybe_unused]] static double diagonal_analytical_integral_energy_constraint(const ParameterVector& x, ParameterVector& grad, void* params)
{
	const auto& [TrainingSets, Energies, TotalEnergy] = *static_cast<AnalyticalEnergyConstraintParameters*>(params);
	// construct predictors
	const QuantumArray<ParameterVector> all_params = construct_all_parameters_from_diagonal(x);
	const OptionalKernels Kernels = construct_predictors(all_params, TrainingSets, true, !grad.empty());
	// calculate population of each surfaces
	double result = calculate_total_energy_average(Kernels, Energies) - TotalEnergy;
	// calculate_derivatives
	if (!grad.empty())
	{
		grad = total_energy_derivative(Kernels, Energies);
	}
	make_normal(result);
	for (double& d : grad)
	{
		make_normal(d);
	}
	return result;
}

/// @brief To calculate the error on purity conservation constraint by analytical integral parameter
/// @param[in] x The input parameters, need to calculate the function value and gradient at this point
/// @param[out] grad The gradient at the given point
/// @param[in] params Other parameters. Here it is the diagonal training sets and purity
/// @return @f$ \langle\rho^2\rangle_{\mathrm{parameters}} - \langle\rho^2\rangle_{t=0} @f$
[[maybe_unused]] static double diagonal_analytical_integral_purity_constraint(const ParameterVector& x, ParameterVector& grad, void* params)
{
	const auto& [TrainingSets, Purity] = *static_cast<AnalyticalPurityConstraintParameters*>(params);
	// construct predictors
	const QuantumArray<ParameterVector> all_params = construct_all_parameters_from_diagonal(x);
	const OptionalKernels Kernels = construct_predictors(all_params, TrainingSets, true, !grad.empty());
	// calculate population of each surfaces
	double result = calculate_purity(Kernels) - Purity;
	// calculate_derivatives
	if (!grad.empty())
	{
		ParameterVector all_deriv = purity_derivative(Kernels);
		for (std::size_t iPES = 0; iPES < NumPES; iPES++)
		{
			std::copy(
				all_deriv.cbegin() + (iPES * NumPES + iPES) * Kernel::NumTotalParameters,
				all_deriv.cbegin() + (iPES * NumPES + iPES + 1) * Kernel::NumTotalParameters,
				grad.begin() + iPES * Kernel::NumTotalParameters);
		}
	}
	make_normal(result);
	for (double& d : grad)
	{
		make_normal(d);
	}
	return result;
}

/// @brief To optimize parameters of diagonal elements based on the given density and constraint regularizations
/// @param[in] TrainingSets The training features and labels of all elements
/// @param[in] ExtraTrainingSets The extra features and labels that only used for training (but not for prediction)
/// @param[in] Energies The energy on each surfaces
/// @param[in] TotalEnergy The total energy of the partial Wigner transformed density matrix
/// @param[in] Purity The purity of the partial Wigner transformed density matrix
/// @param[inout] minimizer The minimizer to use
/// @param[inout] ParameterVectors The parameters for all elements of density matrix
/// @return The total error and the number of steps
static Optimization::Result optimize_diagonal(
	const AllTrainingSets& TrainingSets,
	const AllTrainingSets& ExtraTrainingSets,
	[[maybe_unused]] const QuantumVector<double>& Energies,
	[[maybe_unused]] const double TotalEnergy,
	[[maybe_unused]] const double Purity,
	nlopt::opt& minimizer,
	QuantumArray<ParameterVector>& ParameterVectors)
{
	// construct training set and parameters
	AllTrainingSets ats = TrainingSets, extra_ats = ExtraTrainingSets;
	ParameterVector diagonal_parameters;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES < NumPES; jPES++)
		{
			const std::size_t ElementIndex = iPES * NumPES + jPES;
			if (iPES == jPES)
			{
				// diagonal elements, save parameters
				diagonal_parameters.insert(
					diagonal_parameters.cend(),
					ParameterVectors[ElementIndex].cbegin(),
					ParameterVectors[ElementIndex].cend());
			}
			else
			{
				// off-diagonal elements, remove training set
				auto& [feature, label] = ats[ElementIndex];
				feature = Eigen::MatrixXd(0, 0);
				label = Eigen::VectorXd(0);
				auto& [extra_feature, extra_label] = extra_ats[ElementIndex];
				extra_feature = Eigen::MatrixXd(0, 0);
				extra_label = Eigen::VectorXd(0);
			}
		}
	}
	AnalyticalLooseFunctionParameters alfp = std::tie(ats, extra_ats);
	minimizer.set_min_objective(diagonal_loose, static_cast<void*>(&alfp));
	// set up constraints
	AnalyticalPopulationConstraintParameters applcp = std::tie(ats);
	minimizer.add_equality_constraint(diagonal_analytical_integral_population_constraint, static_cast<void*>(&applcp));
	AnalyticalEnergyConstraintParameters aengcp = std::tie(ats, Energies, TotalEnergy);
	minimizer.add_equality_constraint(diagonal_analytical_integral_energy_constraint, static_cast<void*>(&aengcp));
	AnalyticalPurityConstraintParameters aprtcp = std::tie(ats, Purity); // definition must be out of if, or memory will be released
	if (Purity > 0)
	{
		minimizer.add_equality_constraint(diagonal_analytical_integral_purity_constraint, static_cast<void*>(&aprtcp));
	}
	// optimize
	double err = 0.0;
	try
	{
		minimizer.optimize(diagonal_parameters, err);
	}
#ifdef NDEBUG
	catch (...)
	{
	}
#else
	catch (std::exception& e)
	{
		spdlog::error("Diagonal elements optimization failed by {}", e.what());
	}
#endif
	minimizer.remove_equality_constraints();
	// set up parameters
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		const std::size_t ElementIndex = iPES * NumPES + iPES;
		[[maybe_unused]] const auto& [feature, label] = TrainingSets[ElementIndex];
		if (feature.size() != 0)
		{
			ParameterVectors[ElementIndex] = ParameterVector(
				diagonal_parameters.cbegin() + iPES * Kernel::NumTotalParameters,
				diagonal_parameters.cbegin() + (iPES + 1) * Kernel::NumTotalParameters);
			spdlog::info(
				"Parameter of {}: {}.",
				get_element_name(iPES, iPES),
				ElementParameter::Map(ParameterVectors[ElementIndex].data()).format(VectorFormatter));
		}
		else
		{
			spdlog::info("{} is small.", get_element_name(iPES, iPES));
		}
	}
	const OptionalKernels kernels = construct_predictors(ParameterVectors, ats, true, false);
	spdlog::info(
		"Diagonal error = {}; Population = {}; Energy = {}; Diagonal purity = {}",
		err,
		calculate_population(kernels),
		calculate_total_energy_average(kernels, Energies),
		calculate_purity(kernels));
	return std::make_tuple(err, std::vector<std::size_t>(1, minimizer.get_numevals()));
}

/// @brief To construct parameters for all elements from parameters of all elements
/// @param[in] x The combined parameters for all elements
/// @return Parameters of all elements, leaving off-diagonal parameters blank
static QuantumArray<ParameterVector> construct_all_parameters(const ParameterVector& x)
{
	assert(x.size() == NumElements * Kernel::NumTotalParameters);
	QuantumArray<ParameterVector> result;
	for (std::size_t iElement = 0; iElement < NumElements; iElement++)
	{
		result[iElement] = ParameterVector(
			x.cbegin() + iElement * Kernel::NumTotalParameters,
			x.cbegin() + (iElement + 1) * Kernel::NumTotalParameters);
	}
	return result;
}

/// @brief To combine parameters of each elements into one, inverse function of @p construct_all_parameters
/// @param[in] x parameters of all elements
/// @return The combined parameters of all elements
static ParameterVector construct_combined_parameters(const QuantumArray<ParameterVector>& x)
{
	ParameterVector result;
	result.reserve(NumElements * x[0].size());
	for (std::size_t iElement = 0; iElement < NumElements; iElement++)
	{
		result.insert(result.cend(), x[iElement].cbegin(), x[iElement].cend());
	}
	return result;
}

/// @brief To calculate the overall loose for off-diagonal elements
/// @param[in] x The input parameters, need to calculate the function value and gradient at this point
/// @param[out] grad The gradient at the given point
/// @param[in] params Other parameters. Here it is the off-diagonal training sets
/// @return The off-diagonal errors
static double full_loose(const ParameterVector& x, ParameterVector& grad, void* params)
{
	double err = 0.0;
	// get the parameters
	const auto& [TrainingSets, ExtraTrainingSets] = *static_cast<AnalyticalLooseFunctionParameters*>(params);
	const QuantumArray<ParameterVector> AllParams = construct_all_parameters(x);
	QuantumArray<ParameterVector> all_grads;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES < NumPES; jPES++)
		{
			const std::size_t ElementIndex = iPES * NumPES + jPES;
			[[maybe_unused]] const auto& [feature, label] = TrainingSets[ElementIndex];
			all_grads[ElementIndex] = ParameterVector(!grad.empty() ? Kernel::NumTotalParameters : 0);
			if (feature.size() != 0)
			{
				ElementTrainingParameters etp = std::tie(TrainingSets[ElementIndex], ExtraTrainingSets[ElementIndex]);
				err += loose_function(AllParams[ElementIndex], all_grads[ElementIndex], static_cast<void*>(&etp));
			}
		}
	}
	grad = construct_combined_parameters(all_grads);
	make_normal(err);
	for (double& d : grad)
	{
		make_normal(d);
	}
	return err;
}

/// @brief To calculate the error on population constraint by analytical integral of parameters
/// @param[in] x The input parameters, need to calculate the function value and gradient at this point
/// @param[out] grad The gradient at the given point
/// @param[in] params Other parameters. Here it is the diagonal training sets
/// @return @f$ \langle\rho\rangle_{\mathrm{parameters}} - 1 @f$
[[maybe_unused]] static double full_analytical_integral_population_constraint(const ParameterVector& x, ParameterVector& grad, void* params)
{
	[[maybe_unused]] const auto& [TrainingSets] = *static_cast<AnalyticalPopulationConstraintParameters*>(params);
	// construct predictors
	const QuantumArray<ParameterVector> all_params = construct_all_parameters(x);
	const OptionalKernels Kernels = construct_predictors(all_params, TrainingSets, true, !grad.empty());
	// calculate population of each surfaces
	double result = calculate_population(Kernels) - 1.0;
	// calculate_derivatives
	if (!grad.empty())
	{
		grad = construct_combined_parameters(construct_all_parameters_from_diagonal(population_derivative(Kernels)));
	}
	make_normal(result);
	for (double& d : grad)
	{
		make_normal(d);
	}
	return result;
}

/// @brief To calculate the error on energy constraint by analytical integral of parameters
/// @param[in] x The input parameters, need to calculate the function value and gradient at this point
/// @param[out] grad The gradient at the given point
/// @param[in] params Other parameters. Here it is the diagonal training sets, total energy and mass
/// @return @f$ \langle E\rangle_{\mathrm{parameters}} - E_0 @f$
[[maybe_unused]] static double full_analytical_integral_energy_constraint(const ParameterVector& x, ParameterVector& grad, void* params)
{
	const auto& [TrainingSets, Energies, TotalEnergy] = *static_cast<AnalyticalEnergyConstraintParameters*>(params);
	// construct predictors
	const QuantumArray<ParameterVector> all_params = construct_all_parameters(x);
	const OptionalKernels Kernels = construct_predictors(all_params, TrainingSets, true, !grad.empty());
	// calculate population of each surfaces
	double result = calculate_total_energy_average(Kernels, Energies) - TotalEnergy;
	// calculate_derivatives
	if (!grad.empty())
	{
		grad = construct_combined_parameters(construct_all_parameters_from_diagonal(total_energy_derivative(Kernels, Energies)));
	}
	make_normal(result);
	for (double& d : grad)
	{
		make_normal(d);
	}
	return result;
}

/// @brief To calculate the error on purity conservation constraint
/// @param[in] x The input parameters, need to calculate the function value and gradient at this point
/// @param[out] grad The gradient at the given point
/// @param[in] params Other parameters. Here it is the off-diagonal training sets and purity
/// @return @f$ \langle\rho^2\rangle_{\mathrm{parameters}} - \langle\rho^2\rangle_{t=0} @f$
[[maybe_unused]] static double full_analytical_integral_purity_constraint(const ParameterVector& x, ParameterVector& grad, void* params)
{
	const auto& [TrainingSets, Purity] = *static_cast<AnalyticalPurityConstraintParameters*>(params);
	// construct kernels
	const QuantumArray<ParameterVector> AllParams = construct_all_parameters(x);
	const OptionalKernels Kernels = construct_predictors(AllParams, TrainingSets, true, !grad.empty());
	// calculate population of each surfaces
	double result = calculate_purity(Kernels) - Purity;
	// calculate_derivatives
	if (!grad.empty())
	{
		grad = purity_derivative(Kernels);
	}
	make_normal(result);
	for (double& d : grad)
	{
		make_normal(d);
	}
	return result;
}

/// @brief To optimize parameters of off-diagonal element based on the given density and purity
/// @param[in] TrainingSets The training features and labels of all elements
/// @param[in] ExtraTrainingSets The extra features and labels that only used for training (but not for prediction)
/// @param[in] Energies The energy on each surfaces
/// @param[in] TotalEnergy The total energy of the partial Wigner transformed density matrix
/// @param[in] Purity The purity of the partial Wigner transformed density matrix, which should conserve
/// @param[inout] minimizer The minimizer to use
/// @param[inout] ParameterVectors The parameters for all elements of density matrix
/// @return The total error (MSE, log likelihood, etc, including regularizations) of all off-diagonal elements of density matrix
Optimization::Result optimize_full(
	const AllTrainingSets& TrainingSets,
	const AllTrainingSets& ExtraTrainingSets,
	[[maybe_unused]] const QuantumVector<double>& Energies,
	[[maybe_unused]] const double TotalEnergy,
	[[maybe_unused]] const double Purity,
	nlopt::opt& minimizer,
	QuantumArray<ParameterVector>& ParameterVectors)
{
	// construct training set
	ParameterVector full_parameters;
	full_parameters.reserve(Kernel::NumTotalParameters * NumElements);
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES < NumPES; jPES++)
		{
			const std::size_t ElementIndex = iPES * NumPES + jPES;
			full_parameters.insert(
				full_parameters.cend(),
				ParameterVectors[ElementIndex].cbegin(),
				ParameterVectors[ElementIndex].cend());
		}
	}
	AnalyticalLooseFunctionParameters alfp = std::tie(TrainingSets, ExtraTrainingSets);
	minimizer.set_min_objective(full_loose, static_cast<void*>(&alfp));
	// setup constraint
	AnalyticalPopulationConstraintParameters applcp = std::tie(TrainingSets);
	minimizer.add_equality_constraint(full_analytical_integral_population_constraint, static_cast<void*>(&applcp));
	AnalyticalEnergyConstraintParameters aengcp = std::tie(TrainingSets, Energies, TotalEnergy);
	minimizer.add_equality_constraint(full_analytical_integral_energy_constraint, static_cast<void*>(&aengcp));
	AnalyticalPurityConstraintParameters aprtcp = std::tie(TrainingSets, Purity);
	minimizer.add_equality_constraint(full_analytical_integral_purity_constraint, static_cast<void*>(&aprtcp));
	// optimize
	double err = 0.0;
	try
	{
		minimizer.optimize(full_parameters, err);
	}
#ifdef NDEBUG
	catch (...)
	{
	}
#else
	catch (std::exception& e)
	{
		spdlog::error("Off-diagonal elements optimization failed by {}", e.what());
	}
#endif
	minimizer.remove_equality_constraints();
	// set up parameters
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES < NumPES; jPES++)
		{
			const std::size_t ElementIndex = iPES * NumPES + jPES;
			[[maybe_unused]] const auto& [feature, label] = TrainingSets[ElementIndex];
			if (feature.size() != 0)
			{
				ParameterVectors[ElementIndex] = ParameterVector(
					full_parameters.cbegin() + ElementIndex * Kernel::NumTotalParameters,
					full_parameters.cbegin() + (ElementIndex + 1) * Kernel::NumTotalParameters);
				spdlog::info(
					"Parameter of {}: {}.",
					get_element_name(iPES, jPES),
					ElementParameter::Map(ParameterVectors[ElementIndex].data()).format(VectorFormatter));
			}
			else
			{
				spdlog::info("{} is small.", get_element_name(iPES, jPES));
			}
		}
	}
	const OptionalKernels kernels = construct_predictors(ParameterVectors, TrainingSets, true, false);
	spdlog::info(
		"Off-diagonal error = {}; Population = {}; Energy = {}; Purity = {}.",
		err,
		calculate_population(kernels),
		calculate_total_energy_average(kernels, Energies),
		calculate_purity(kernels));
	return std::make_tuple(err, std::vector<std::size_t>(1, minimizer.get_numevals()));
}

/// @brief To check each of the averages (@<r@>, @<1@>, @<E@> and purity) is satisfactory or not
/// @param[in] Kernels The kernels of all elements
/// @param[in] density The vector containing all known density matrix
/// @param[in] Energies The energy of each surfaces and the total energy
/// @param[in] TotalEnergy The energy of the whole partial Wigner-transformed density matrics
/// @param[in] Purity The purity of the partial Wigner transformed density matrix
/// @return Whether each of the averages (@<r@>, @<1@>, @<E@> and purity) is satisfactory or not
/// @details The comparison happens between integral and average for @<r@>,
/// and between integral and exact value for @<1@>, @<E@> and purity.
static Eigen::Vector4d check_averages(
	const OptionalKernels& Kernels,
	const AllPoints& density,
	const QuantumVector<double>& Energies,
	const double TotalEnergy,
	const double Purity)
{
	// to calculate the error that is above the tolerance or not
	// if within tolerance, return 0; otherwise, return relative error
	static auto beyond_tolerance_error = [](const auto& calc, const auto& ref) -> double
	{
		static_assert(std::is_same_v<decltype(calc), decltype(ref)>);
		static_assert(std::is_same_v<decltype(calc), const double&> || std::is_same_v<decltype(calc), const ClassicalPhaseVector&>);
		if constexpr (std::is_same_v<decltype(calc), const double&>)
		{
			static constexpr double AverageTolerance = 2e-2;
			const double err = std::abs((calc / ref) - 1.0);
			if (err < AverageTolerance)
			{
				return 0.0;
			}
			else
			{
				return err;
			}
		}
		else // if constexpr (std::is_same_v<decltype(calc), const ClassicalPhaseVector&>)
		{
			static constexpr double AbsoluteTolerance = 0.2;
			static constexpr double RelativeTolerance = 0.1;
			const auto err = ((calc.array() / ref.array()).abs() - 1.0).abs();
			if ((err < RelativeTolerance).all() && ((calc - ref).array().abs() < AbsoluteTolerance).all())
			{
				return 0.0;
			}
			else
			{
				return err.sum();
			}
		}
	};

	static auto check_and_output = [](const double error, const std::string& name) -> void
	{
		if (error == 0.0)
		{
			spdlog::info(name + " passes checking.");
		}
		else
		{
			spdlog::info(name + " does not pass checking and has total relative error of {}.", error);
		}
	};

	Eigen::Vector4d result = Eigen::Vector4d::Zero();
	// <r>
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		const std::size_t ElementIndex = iPES * NumPES + iPES;
		if (Kernels[ElementIndex].has_value() && !density[ElementIndex].empty())
		{
			const ClassicalPhaseVector r_ave = calculate_1st_order_average_one_surface(density[ElementIndex]);
			const ClassicalPhaseVector r_prm = calculate_1st_order_average_one_surface(Kernels[ElementIndex].value()); // prm = parameters
			spdlog::info(
				"On surface {}, Exact <r> = {}, Analytical integrated <r> = {}.",
				iPES,
				r_ave.format(VectorFormatter),
				r_prm.format(VectorFormatter));
			result[0] += beyond_tolerance_error(r_prm, r_ave);
		}
	}
	check_and_output(result[0], "<r>");
	// <1>
	const double ppl_prm = calculate_population(Kernels);
	spdlog::info("Exact population = {}, analytical integrated population = {}.", 1.0, ppl_prm);
	result[1] += beyond_tolerance_error(ppl_prm, 1.0);
	check_and_output(result[1], "Population");
	// <E>
	const double eng_prm_ave = calculate_total_energy_average(Kernels, Energies);
	spdlog::info("Exact energy = {}, analytical integrated energy = {}", TotalEnergy, eng_prm_ave);
	result[2] += beyond_tolerance_error(eng_prm_ave, TotalEnergy);
	check_and_output(result[2], "Energy");
	// purity
	const double prt_prm = calculate_purity(Kernels);
	spdlog::info("Exact purity = {}, analytical integrated purity = {}.", Purity, prt_prm);
	result[3] += beyond_tolerance_error(prt_prm, Purity);
	check_and_output(result[3], "Purity");
	return result;
}

/// @details Using Gaussian Process Regression (GPR) to depict phase space distribution.
/// First optimize elementwise, then optimize the diagonal with normalization and energy conservation
Optimization::Result Optimization::optimize(
	const AllPoints& density,
	const AllPoints& extra_points)
{
	// construct training sets
	const AllTrainingSets TrainingSets = construct_training_sets(density), ExtraTrainingSets = construct_training_sets(extra_points);
	// set bounds for current bounds, and calculate energy on each surfaces
	const QuantumVector<double> Energies = calculate_total_energy_average_each_surface(density, mass);
	auto move_into_bounds = [](ParameterVector& params, const Bounds& bounds)
	{
		const auto& [LowerBound, UpperBound] = bounds;
		for (std::size_t iParam = 0; iParam < Kernel::NumTotalParameters; iParam++)
		{
			params[iParam] = std::max(params[iParam], LowerBound[iParam]);
			params[iParam] = std::min(params[iParam], UpperBound[iParam]);
		}
	};
	QuantumArray<Bounds> param_bounds;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES < NumPES; jPES++)
		{
			const std::size_t ElementIndex = iPES * NumPES + jPES;
			if (iPES <= jPES)
			{
				// only get upper-triangular bounds
				if (!density[ElementIndex].empty())
				{
					param_bounds[ElementIndex] = calculate_bounds(density[ElementIndex]);
				}
				else
				{
					auto& [lower_bound, upper_bound] = param_bounds[ElementIndex];
					lower_bound = NLOptLocalMinimizers[ElementIndex].get_lower_bounds();
					upper_bound = NLOptLocalMinimizers[ElementIndex].get_upper_bounds();
				}
			}
			else
			{
				// for strict lower-triangular bounds, copy from upper
				param_bounds[ElementIndex] = param_bounds[jPES * NumPES + iPES];
			}
			// compare each element; if they are out of bounds, move it in
			move_into_bounds(ParameterVectors[ElementIndex], param_bounds[ElementIndex]);
		}
	}
	set_optimizer_bounds(NLOptLocalMinimizers, param_bounds);
	set_optimizer_bounds(NLOptGlobalMinimizers, param_bounds);
	// any offdiagonal element is not small
	const bool OffDiagonalOptimization = [&density](void) -> bool
	{
		if constexpr (NumOffDiagonalElements != 0)
		{
			for (std::size_t iPES = 1; iPES < NumPES; iPES++)
			{
				for (std::size_t jPES = 0; jPES < iPES; jPES++)
				{
					if (!density[iPES * NumPES + jPES].empty())
					{
						return true;
					}
				}
			}
		}
		return false;
	}();

	// construct a function for code reuse
	auto optimize_and_check =
		[this, &density, &TrainingSets, &ExtraTrainingSets, &Energies, OffDiagonalOptimization](
			QuantumArray<ParameterVector>& parameter_vectors) -> std::tuple<Optimization::Result, Eigen::Vector4d>
	{
		// initially, set the magnitude to be 1
		for (std::size_t iElement = 0; iElement < NumElements; iElement++)
		{
			parameter_vectors[iElement][0] = InitialMagnitude;
		}
		Optimization::Result result = optimize_elementwise(TrainingSets, ExtraTrainingSets, NLOptLocalMinimizers, parameter_vectors);
		auto& [err, steps] = result;
		if (OffDiagonalOptimization)
		{
			// do not add purity condition for diagonal, so give a negative purity to indicate no necessity
			const auto& [DiagonalError, DiagonalStep] = optimize_diagonal(
				TrainingSets,
				ExtraTrainingSets,
				Energies,
				TotalEnergy,
				NAN,
				NLOptLocalMinimizers[NumElements],
				parameter_vectors);
			const auto& [OffDiagonalError, OffDiagonalStep] = optimize_full(
				TrainingSets,
				ExtraTrainingSets,
				Energies,
				TotalEnergy,
				Purity,
				NLOptLocalMinimizers[NumElements + 1],
				parameter_vectors);
			err = DiagonalError + OffDiagonalError;
			steps.insert(steps.cend(), DiagonalStep.cbegin(), DiagonalStep.cend());
			steps.insert(steps.cend(), OffDiagonalStep.cbegin(), OffDiagonalStep.cend());
		}
		else
		{
			const auto& [DiagonalError, DiagonalStep] = optimize_diagonal(
				TrainingSets,
				ExtraTrainingSets,
				Energies,
				TotalEnergy,
				Purity,
				NLOptLocalMinimizers[NumElements],
				parameter_vectors);
			err = DiagonalError;
			steps.insert(steps.cend(), DiagonalStep.cbegin(), DiagonalStep.cend());
			steps.push_back(0);
		}
		// afterwards, calculate the magnitude
		for (std::size_t iElement = 0; iElement < NumElements; iElement++)
		{
			const auto& [feature, label] = TrainingSets[iElement];
			if (feature.size() != 0)
			{
				const Kernel kernel(parameter_vectors[iElement], feature, label, true, false);
				parameter_vectors[iElement][0] = std::sqrt(label.dot(kernel.get_inverse_label()) / label.size());
			}
		}
		spdlog::info("error = {}", std::get<0>(result));
		return std::make_tuple(result, check_averages(construct_predictors(parameter_vectors, TrainingSets, true, false), density, Energies, TotalEnergy, Purity));
	};

	auto compare_and_overwrite =
		[this](
			Optimization::Result& result,
			Eigen::Vector4d& check_result,
			const std::string& old_name,
			const Optimization::Result& result_new,
			const Eigen::Vector4d& check_new,
			const std::string& new_name,
			const QuantumArray<ParameterVector>& param_vec_new) -> void
	{
		auto& [error, steps] = result;
		const auto& [error_new, steps_new] = result_new;
		const std::size_t BetterResults = (check_new.array() <= check_result.array()).count();
		if (BetterResults > 2)
		{
			spdlog::info(new_name + " is better because of better averages.");
			ParameterVectors = param_vec_new;
			error = error_new;
			for (std::size_t iElement = 0; iElement < NumElements + 2; iElement++)
			{
				steps[iElement] += steps_new[iElement];
			}
			check_result = check_new;
		}
		else if (BetterResults == 2 && error_new < error)
		{
			spdlog::info(new_name + " is better because of smaller error.");
			ParameterVectors = param_vec_new;
			error = error_new;
			for (std::size_t iElement = 0; iElement < NumElements + 2; iElement++)
			{
				steps[iElement] += steps_new[iElement];
			}
			check_result = check_new;
		}
		else
		{
			spdlog::info(old_name + " is better.");
		}
	};

	// 1. optimize locally with parameters from last step
	spdlog::info("Local optimization with previous parameters.");
	auto [result, check_result] = optimize_and_check(ParameterVectors);
	if ((check_result.array() == 0.0).all())
	{
		spdlog::info("Local optimization with previous parameters succeeded.");
		return result;
	}
	// 2. optimize locally with initial parameter
	spdlog::warn("Local optimization with previous parameters failed. Retry with local optimization with initial parameters.");
	QuantumArray<ParameterVector> param_vec_initial;
	for (std::size_t iElement = 0; iElement < NumElements; iElement++)
	{
		param_vec_initial[iElement] = InitialParameter;
	}
	const auto& [result_initial, check_initial] = optimize_and_check(param_vec_initial);
	compare_and_overwrite(
		result,
		check_result,
		"Local optimization with previous parameters",
		result_initial,
		check_initial,
		"Local optimization with initial parameters",
		param_vec_initial);
	if ((check_result.array() == 0.0).all())
	{
		spdlog::info("Local optimization succeeded.");
		return result;
	}
	// 3. optimize globally
	spdlog::warn("Local optimization failed. Retry with global optimization.");
	QuantumArray<ParameterVector> param_vec_global, param_vec_local;
	for (std::size_t iElement = 0; iElement < NumElements; iElement++)
	{
		param_vec_global[iElement] = InitialParameterForGlobalOptimizer;
	}
	[[maybe_unused]] const auto& [error_global_elem, steps_global_elem] = optimize_elementwise(TrainingSets, ExtraTrainingSets, NLOptGlobalMinimizers, param_vec_global);
	for (std::size_t iElement = 0; iElement < NumElements; iElement++)
	{
		param_vec_local[iElement] = global_parameter_to_local(param_vec_global[iElement]);
	}
	auto [result_global, check_global] = optimize_and_check(param_vec_local);
	for (std::size_t iElement = 0; iElement < NumElements; iElement++)
	{
		std::get<1>(result_global)[iElement] += steps_global_elem[iElement];
	}
	compare_and_overwrite(
		result,
		check_result,
		"Local optimization",
		result_global,
		check_global,
		"Global optimization",
		param_vec_local);
	if ((check_result.array() == 0.0).all())
	{
		spdlog::info("Optimization succeeded.");
		return result;
	}
	// 4. end
	spdlog::warn("Optimization failed.");
	return result;
}

QuantumArray<ParameterVector> Optimization::get_lower_bounds(void) const
{
	QuantumArray<ParameterVector> result;
	for (std::size_t iElement = 0; iElement < NumElements; iElement++)
	{
		result[iElement] = NLOptLocalMinimizers[iElement].get_lower_bounds();
	}
	return result;
}

QuantumArray<ParameterVector> Optimization::get_upper_bounds(void) const
{
	QuantumArray<ParameterVector> result;
	for (std::size_t iElement = 0; iElement < NumElements; iElement++)
	{
		result[iElement] = NLOptLocalMinimizers[iElement].get_upper_bounds();
	}
	return result;
}