/// @file opt.cpp
/// @brief Implementation of opt.h

#include "stdafx.h"

#include "opt.h"

#include "complex_kernel.h"
#include "input.h"
#include "kernel.h"
#include "predict.h"

/// The lower and upper bound
using Bounds = std::array<ParameterVector, 2>;
/// The training set for parameter optimization of one element of density matrix
/// passed as parameter to the optimization function.
/// First is feature, second is label
using ElementTrainingSet = std::tuple<PhasePoints, Eigen::VectorXcd>;
/// The training sets of all elements
using AllTrainingSets = QuantumArray<ElementTrainingSet>;
/// The parameters passed to the optimization subroutine of a single element
using ElementTrainingParameters = std::tuple<const ElementTrainingSet&, const ElementTrainingSet&>;
/// The parameters passed to the optimization subroutine of all (diagonal) elements
/// including the training sets and an extra training set
using AnalyticalLooseFunctionParameters = std::tuple<const AllTrainingSets&, const AllTrainingSets&>;
/// The parameters used for population, energy and purity conservation by analytical integral,
/// including the training sets, the energy on each surfaces, the total energy, and the exact initial purity
using AnalyticalConstraintParameters = std::tuple<const AllTrainingSets&, const QuantumVector<double>&, const double, const double>;

static constexpr double InitialMagnitude = 1.0;																											///< The initial weight for all kernels
static constexpr double InitialNoise = 1e-2;																											///< The initial weight for noise
static constexpr std::size_t NumTotalParameters = Kernel::NumTotalParameters * NumPES + ComplexKernel::NumTotalParameters * NumOffDiagonalElements / 2; ///< The number of parameters for all elements

QuantumMatrix<bool> is_very_small(const AllPoints& density)
{
	// squared value below this are regarded as 0
	QuantumMatrix<bool> result = QuantumMatrix<bool>::Ones();
	// as density is symmetric, only lower-triangular elements needs evaluating
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES <= iPES; jPES++)
		{
			const std::size_t ElementIndex = iPES * NumPES + jPES;
			result(iPES, jPES) = std::all_of(
				std::execution::par_unseq,
				density[ElementIndex].cbegin(),
				density[ElementIndex].cend(),
				[iPES, jPES](const PhaseSpacePoint& psp) -> bool
				{
					return std::get<1>(psp)(iPES, jPES) == 0.0;
				});
		}
	}
	return result.selfadjointView<Eigen::Lower>();
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
		PhasePoints feature = PhasePoints::Zero(PhaseDim, NumPoints);
		Eigen::VectorXcd label = Eigen::VectorXd::Zero(NumPoints);
		// first insert original points
		const auto indices = xt::arange(NumPoints);
		std::for_each(
			std::execution::par_unseq,
			indices.cbegin(),
			indices.cend(),
			[&ElementDensity, &feature, &label, iPES, jPES](std::size_t iPoint) -> void
			{
				const auto& [r, rho] = ElementDensity[iPoint];
				feature.col(iPoint) = r;
				label[iPoint] = rho(iPES, jPES);
			});
		// this happens for coherence, where only real/imaginary part is important
		return std::make_tuple(feature, label);
	};
	AllTrainingSets result;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES <= iPES; jPES++)
		{
			result[iPES * NumPES + jPES] = construct_element_training_set(iPES, jPES);
		}
	}
	return result;
}

/// @brief To update the kernels using the training sets
/// @param[in] ParameterVectors The parameters for all elements of density matrix
/// @param[in] TrainingSets The training sets of all elements of density matrix
/// @param[in] IsToCalculateError Whether to calculate LOOCV squared error or not
/// @param[in] IsToCalculateAverage Whether to calculate averages (@<1@>, @<r@>, and purity) or not
/// @param[in] IsToCalculateDerivative Whether to calculate derivative of each kernel or not
/// @return Array of kernels
static OptionalKernels construct_predictors(
	const QuantumArray<ParameterVector>& ParameterVectors,
	const AllTrainingSets& TrainingSets,
	const bool IsToCalculateError,
	const bool IsToCalculateAverage,
	const bool IsToCalculateDerivative)
{
	OptionalKernels result;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES <= iPES; jPES++)
		{
			const std::size_t ElementIndex = iPES * NumPES + jPES;
			const auto& [feature, label] = TrainingSets[ElementIndex];
			if (iPES == jPES)
			{
				result[ElementIndex] = std::make_unique<Kernel>(ParameterVectors[ElementIndex], feature, label.real(), IsToCalculateError, IsToCalculateAverage, IsToCalculateDerivative);
			}
			else
			{
				if (std::all_of(ParameterVectors[ElementIndex].cbegin(), ParameterVectors[ElementIndex].cend(), std::bind(std::equal_to<double>{},std::placeholders::_1, 0)))
				{
					// when doing diagonal optimization, off-diagonal elements are not used, and the parameters are set 0
					// in that case, we need to reset the label to be 0 but the parameters are non-zero
					result[ElementIndex] = std::make_unique<ComplexKernel>(ParameterVector(ComplexKernel::NumTotalParameters, 1.0), feature, Eigen::VectorXcd::Zero(label.size()), IsToCalculateError, IsToCalculateAverage, IsToCalculateDerivative);
				}
				else
				{
					result[ElementIndex] = std::make_unique<ComplexKernel>(ParameterVectors[ElementIndex], feature, label, IsToCalculateError, IsToCalculateAverage, IsToCalculateDerivative);
				}
			}
		}
	}
	return result;
}

OptionalKernels construct_predictors(
	const QuantumArray<ParameterVector>& ParameterVectors,
	const AllPoints& density)
{
	return construct_predictors(ParameterVectors, construct_training_sets(density), true, true, false);
}

/// @brief To calculate the lower and upper bound for kernel
/// @param[in] CharLengthLowerBound The lower bound for characteristic lengths
/// @param[in] CharLengthUpperBound The upper bound for characteristic lengths
/// @return The bounds for kernel
static Bounds calculate_kernel_bounds(
	const ClassicalPhaseVector& CharLengthLowerBound,
	const ClassicalPhaseVector& CharLengthUpperBound)
{
	const Bounds& KernelBounds = [&CharLengthLowerBound, &CharLengthUpperBound](void) -> Bounds
	{
		Bounds result {ParameterVector(Kernel::NumTotalParameters), ParameterVector(Kernel::NumTotalParameters)};
		auto& [lb, ub] = result;
		std::size_t iParam = 0;
		// first, magnitude
		lb[iParam] = InitialMagnitude;
		ub[iParam] = InitialMagnitude;
		iParam++;
		// second, Gaussian
		for (std::size_t iDim = 0; iDim < PhaseDim; iDim++)
		{
			lb[iParam] = CharLengthLowerBound[iDim];
			ub[iParam] = CharLengthUpperBound[iDim];
			iParam++;
		}
		// third, noise
		lb[iParam] = InitialNoise;
		ub[iParam] = InitialNoise;
		iParam++;
		return result;
	}();
	return KernelBounds;
}

/// @brief To calculate the lower and upper bound for complex kernel
/// @param[in] CharLengthLowerBound The lower bound for characteristic lengths
/// @param[in] CharLengthUpperBound The upper bound for characteristic lengths
/// @return The bounds for complex kernel
static Bounds calculate_complex_kernel_bounds(
	const ClassicalPhaseVector& CharLengthLowerBound,
	const ClassicalPhaseVector& CharLengthUpperBound)
{
	const Bounds& ComplexKernelBounds = [&CharLengthLowerBound, &CharLengthUpperBound](void) -> Bounds
	{
		Bounds result {ParameterVector(ComplexKernel::NumTotalParameters), ParameterVector(ComplexKernel::NumTotalParameters)};
		auto& [lb, ub] = result;
		std::size_t iParam = 0;
		// complex kernel, similarly
		// first, magnitude
		lb[iParam] = InitialMagnitude;
		ub[iParam] = InitialMagnitude;
		iParam++;
		// second, kernels
		for (std::size_t iKernel = 0; iKernel < ComplexKernel::NumKernels; iKernel++)
		{
			// first, magnitude
			lb[iParam] = std::sqrt(InitialMagnitude * InitialNoise);
			ub[iParam] = std::sqrt(InitialMagnitude / InitialNoise);
			iParam++;
			// second, Gaussian
			for (std::size_t iDim = 0; iDim < PhaseDim; iDim++)
			{
				lb[iParam] = CharLengthLowerBound[iDim];
				ub[iParam] = CharLengthUpperBound[iDim];
				iParam++;
			}
		}
		// third, noise
		lb[iParam] = InitialNoise;
		ub[iParam] = InitialNoise;
		iParam++;
		return result;
	}();
	return ComplexKernelBounds;
}

/// @brief To set up the bounds for the parameter
/// @param[in] rSize The size of the box on all classical directions
/// @return Lower and upper bounds
static QuantumArray<Bounds> calculate_bounds(const ClassicalPhaseVector& rSize)
{
	static constexpr double GaussKerMinCharLength = 1.0 / 100.0; // Minimal characteristic length for Gaussian kernel
	const Bounds& KernelBounds = calculate_kernel_bounds(ClassicalPhaseVector::Ones() * GaussKerMinCharLength, rSize);
	const Bounds& ComplexKernelBounds = calculate_complex_kernel_bounds(ClassicalPhaseVector::Ones() * GaussKerMinCharLength, rSize);
	// then assignment
	QuantumArray<Bounds> result;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES <= iPES; jPES++)
		{
			const std::size_t ElementIndex = iPES * NumPES + jPES;
			if (iPES == jPES)
			{
				result[ElementIndex] = KernelBounds;
			}
			else
			{
				result[ElementIndex] = ComplexKernelBounds;
			}
		}
	}
	return result;
}

/// @brief To set up the bounds for the parameter
/// @param[in] density The selected points in phase space of one element of density matrices
/// @return Lower and upper bounds
static QuantumArray<Bounds> calculate_bounds(const AllPoints& density)
{
	QuantumArray<Bounds> result;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES <= iPES; jPES++)
		{
			const std::size_t ElementIndex = iPES * NumPES + jPES;
			const ClassicalPhaseVector StdDev = calculate_standard_deviation_one_surface(density[ElementIndex]);
			const ClassicalPhaseVector CharLengthLB = StdDev / std::sqrt(density[ElementIndex].size()), CharLengthUB = 2.0 * StdDev;
			if (iPES == jPES)
			{
				result[ElementIndex] = calculate_kernel_bounds(CharLengthLB, CharLengthUB);
			}
			else
			{
				result[ElementIndex] = calculate_complex_kernel_bounds(CharLengthLB, CharLengthUB);
			}
		}
	}
	return result;
}

/// @brief To transform local parameters to global parameters by ln
/// @param[in] param The parameters for local optimizer, i.e., the normal parameter
/// @return The parameters for global optimizer, i.e., the changed parameter
static inline ParameterVector local_parameter_to_global(const ParameterVector& param)
{
	assert(param.size() == Kernel::NumTotalParameters || param.size() == ComplexKernel::NumTotalParameters);
	ParameterVector result = param;
	std::size_t iParam = 0;
	if (param.size() == ComplexKernel::NumTotalParameters)
	{
		// real, imaginary and noise to log
		// first, magnitude
		iParam++;
		// second, real and imaginary kernel
		for (std::size_t iKernel = 0; iKernel < ComplexKernel::NumKernels; iKernel++)
		{
			// magnitude
			result[iParam] = std::log(result[iParam]);
			iParam++;
			// and characteristic lengths
			iParam += PhaseDim;
		}
		// third, noise
		result[iParam] = std::log(result[iParam]);
		iParam++;
	}
	else // if (param.size() == Kernel::NumTotalParameters)
	{
		// noise to log
		// first, magnitude
		iParam++;
		// second, gaussian
		iParam += PhaseDim;
		// third, noise
		result[iParam] = std::log(result[iParam]);
		iParam++;
	}
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
	assert(param.size() == Kernel::NumTotalParameters || param.size() == ComplexKernel::NumTotalParameters);
	assert(grad.size() == param.size() || grad.empty());
	ParameterVector result = grad;
	std::size_t iParam = 0;
	if (result.size() == ComplexKernel::NumTotalParameters)
	{
		// real, imaginary and noise to log
		// first, magnitude
		iParam++;
		// second, real and imaginary kernel
		for (std::size_t iKernel = 0; iKernel < ComplexKernel::NumKernels; iKernel++)
		{
			// magnitude
			result[iParam] *= param[iParam];
			iParam++;
			// and characteristic lengths
			iParam += PhaseDim;
		}
		// third, noise
		result[iParam] *= param[iParam];
		iParam++;
	}
	else if (result.size() == Kernel::NumTotalParameters)
	{
		std::size_t iParam = 0;
		// first, magnitude
		iParam++;
		// second, gaussian
		iParam += PhaseDim;
		// third, noise
		result[iParam] *= param[iParam];
		iParam++;
	}
	// else if (result.empty());
	return result;
}

/// @brief To transform local parameters to global parameters by exp
/// @param[in] param The parameters for global optimizer, i.e., the changed parameter
/// @return The parameters for local optimizer, i.e., the normal parameter
static inline ParameterVector global_parameter_to_local(const ParameterVector& param)
{
	assert(param.size() == Kernel::NumTotalParameters || param.size() == ComplexKernel::NumTotalParameters);
	ParameterVector result = param;
	std::size_t iParam = 0;
	if (param.size() == ComplexKernel::NumTotalParameters)
	{
		// real, imaginary and noise to log
		// first, magnitude
		iParam++;
		// second, real and imaginary kernel
		for (std::size_t iKernel = 0; iKernel < ComplexKernel::NumKernels; iKernel++)
		{
			// magnitude
			result[iParam] = std::exp(result[iParam]);
			iParam++;
			// and characteristic lengths
			iParam += PhaseDim;
		}
		// third, noise
		result[iParam] = std::exp(result[iParam]);
		iParam++;
	}
	else // if (param.size() == Kernel::NumTotalParameters)
	{
		// noise to log
		// first, magnitude
		iParam++;
		// second, gaussian
		iParam += PhaseDim;
		// third, noise
		result[iParam] = std::exp(result[iParam]);
		iParam++;
	}
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
		for (std::size_t jPES = 0; jPES <= iPES; jPES++)
		{
			const std::size_t ElementIndex = iPES * NumPES + jPES;
			const auto& [LowerBound, UpperBound] = Bounds[ElementIndex];
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
	InitialKernelParameter([&rSigma = InitParams.get_sigma_r0()](void) -> ParameterVector
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
			result[iParam] = InitialNoise;
			iParam++;
			return result;
		}()),
	InitialComplexKernelParameter([&rSigma = InitParams.get_sigma_r0()](void) -> ParameterVector
		{
			ParameterVector result(ComplexKernel::NumTotalParameters);
			std::size_t iParam = 0;
			// first, magnitude
			result[iParam] = InitialMagnitude;
			iParam++;
			// second, real and imaginary kernel
			for (std::size_t iKernel = 0; iKernel < ComplexKernel::NumKernels; iKernel++)
			{
				// first, magnitude
				result[iParam] = InitialMagnitude;
				iParam++;
				// second, Gaussian
				for (std::size_t iDim = 0; iDim < PhaseDim; iDim++)
				{
					result[iParam] = rSigma[iDim];
					iParam++;
				}
			}
			// finally, noise
			result[iParam] = InitialNoise;
			iParam++;
			return result;
		}())
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
		if (std::string_view(optimizer.get_algorithm_name()).find("(local, no-derivative)") != std::string_view::npos)
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
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES < NumPES; jPES++)
		{
			const std::size_t ElementIndex = iPES * NumPES + jPES;
			if (iPES < jPES)
			{
				// upper-triangular, not used
				NLOptLocalMinimizers.push_back(nlopt::opt(LocalAlgorithm, 0));
				NLOptGlobalMinimizers.push_back(nlopt::opt(GlobalAlgorithm, 0));
			}
			else if (iPES > jPES)
			{
				// lower-triangular, used for off-diagonal optimization
				ParameterVectors[ElementIndex] = InitialComplexKernelParameter;
				NLOptLocalMinimizers.push_back(nlopt::opt(LocalAlgorithm, ComplexKernel::NumTotalParameters));
				NLOptGlobalMinimizers.push_back(nlopt::opt(GlobalAlgorithm, ComplexKernel::NumTotalParameters));
			}
			else
			{
				// diagonal optimization
				ParameterVectors[ElementIndex] = InitialKernelParameter;
				NLOptLocalMinimizers.push_back(nlopt::opt(LocalAlgorithm, Kernel::NumTotalParameters));
				NLOptGlobalMinimizers.push_back(nlopt::opt(GlobalAlgorithm, Kernel::NumTotalParameters));
			}
			set_optimizer(NLOptLocalMinimizers.back());
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
	}
	// minimizer for all diagonal elements with regularization (population + energy + purity)
	NLOptLocalMinimizers.push_back(nlopt::opt(nlopt::algorithm::AUGLAG_EQ, Kernel::NumTotalParameters * NumPES));
	set_optimizer(NLOptLocalMinimizers.back());
	set_subsidiary_optimizer(NLOptLocalMinimizers.back(), ConstraintAlgorithm);
	// minimizer for all elements with regularization (population, energy and purity)
	NLOptLocalMinimizers.push_back(nlopt::opt(nlopt::algorithm::AUGLAG_EQ, NumTotalParameters));
	set_optimizer(NLOptLocalMinimizers.back());
	set_subsidiary_optimizer(NLOptLocalMinimizers.back(), ConstraintAlgorithm);
	// set up bounds
	const QuantumArray<Bounds> bounds = calculate_bounds(InitParams.get_rmax() - InitParams.get_rmin());
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
	double result = 0.0;
	if (x.size() == Kernel::NumTotalParameters)
	{
		const Kernel kernel(x, TrainingFeature, TrainingLabel.real(), true, false, !grad.empty());
		const Kernel ExtraKernel(ExtraTrainingFeature, kernel, !grad.empty(), ExtraTrainingLabel.real());
		result = kernel.get_error() + ExtraKernel.get_error();
		if (!grad.empty())
		{
			for (std::size_t iParam = 0; iParam < Kernel::NumTotalParameters; iParam++)
			{
				grad[iParam] = kernel.get_error_derivative()[iParam] + ExtraKernel.get_error_derivative()[iParam];
			}
		}
	}
	else // x.size() == ComplexKernel::NumTotalParameters
	{
		const ComplexKernel kernel(x, TrainingFeature, TrainingLabel, true, false, !grad.empty());
		const ComplexKernel ExtraKernel(ExtraTrainingFeature, kernel, !grad.empty(), ExtraTrainingLabel);
		result = kernel.get_error() + ExtraKernel.get_error();
		if (!grad.empty())
		{
			for (std::size_t iParam = 0; iParam < ComplexKernel::NumTotalParameters; iParam++)
			{
				grad[iParam] = kernel.get_error_derivative()[iParam] + ExtraKernel.get_error_derivative()[iParam];
			}
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
static inline double loose_function_global_wrapper(const ParameterVector& x, ParameterVector& grad, void* params)
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
	return "rho[" + std::to_string(iPES) + "][" + std::to_string(jPES) + "]";
}

/// @brief To optimize parameters of each density matrix element based on the given density
/// @param[in] TrainingSets The training features and labels of all elements
/// @param[in] ExtraTrainingSets The extra features and labels that only used for training (but not for prediction)
/// @param[in] IsSmall Whether each element is small or not
/// @param[inout] minimizers The minimizers to use
/// @param[inout] ParameterVectors The parameters for all elements of density matrix
/// @return The total error and the optimization steps of each element
static Optimization::Result optimize_elementwise(
	const AllTrainingSets& TrainingSets,
	const AllTrainingSets& ExtraTrainingSets,
	const QuantumMatrix<bool>& IsSmall,
	std::vector<nlopt::opt>& minimizers,
	QuantumArray<ParameterVector>& ParameterVectors)
{
	auto is_global = [](const nlopt::opt& optimizer) -> bool
	{
		return std::string_view(optimizer.get_algorithm_name()).find("global") != std::string_view::npos || optimizer.get_algorithm() == nlopt::algorithm::GN_ESCH;
	}; // judge if it is global optimizer by checking its name
	double total_error = 0.0;
	std::vector<std::size_t> num_steps(NumElements, 0);
	// optimize element by element
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES <= iPES; jPES++)
		{
			if (!IsSmall(iPES, jPES))
			{
				const std::size_t ElementIndex = iPES * NumPES + jPES;
				ElementTrainingParameters etp = std::tie(TrainingSets[ElementIndex], ExtraTrainingSets[ElementIndex]);
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
					spdlog::error("{}Optimization of {} failed by {}", indents<2>::apply(), get_element_name(iPES, jPES), e.what());
				}
	#endif
				spdlog::info(
					"{}Parameter of {}: {}.",
					indents<3>::apply(),
					get_element_name(iPES, jPES),
					Eigen::VectorXd::Map(ParameterVectors[ElementIndex].data(), ParameterVectors[ElementIndex].size()).format(VectorFormatter));
				spdlog::info(
					"{}Error of {} = {}, using {} steps.",
					indents<2>::apply(),
					get_element_name(iPES, jPES),
					err,
					minimizers[ElementIndex].get_numevals());
				total_error += err;
				num_steps[ElementIndex] = minimizers[ElementIndex].get_numevals();
			}
			else
			{
				spdlog::info("{}{} is 0 everywhere.", indents<2>::apply(), get_element_name(iPES, jPES));
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
		const std::size_t BeginIndex = iPES * Kernel::NumTotalParameters;
		const ParameterVector x_element(x.cbegin() + BeginIndex, x.cbegin() + BeginIndex + Kernel::NumTotalParameters);
		ParameterVector grad_element(!grad.empty() ? Kernel::NumTotalParameters : 0);
		ElementTrainingParameters etp = std::tie(TrainingSets[ElementIndex], ExtraTrainingSets[ElementIndex]);
		err += loose_function(x_element, grad_element, static_cast<void*>(&etp));
		std::copy(grad_element.cbegin(), grad_element.cend(), grad.begin() + BeginIndex);
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
static QuantumArray<ParameterVector> construct_all_parameters_from_diagonal(const double* x)
{
	QuantumArray<ParameterVector> result;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES <= iPES; jPES++)
		{
			const std::size_t ElementIndex = iPES * NumPES + jPES;
			if (iPES == jPES)
			{
				result[ElementIndex] = ParameterVector(x + iPES * Kernel::NumTotalParameters, x + (iPES + 1) * Kernel::NumTotalParameters);
			}
			else
			{
				result[ElementIndex] = ParameterVector(ComplexKernel::NumTotalParameters, 0.0);
			}
		}
	}
	return result;
}

/// @brief To calculate the error on all constraints (population, energy, and maybe purity) by analytical integral of parameters
/// @param[in] NumConstraints The number of constraints, could be 2 (population + energy) or 3 (+ purity)
/// @param[out] result The error of each constraints
/// @param[in] NumParams The number of parameters, which should be @p NumPES * @p Kernel::NumTotalParameters
/// @param[in] x The input parameters, need to calculate the function value and gradient at this point
/// @param[out] grad The gradient at the given point
/// @param[in] params Other parameters. Here it is the training sets, energies on each surface, total energy and purity
static void diagonal_constraints(
	const unsigned NumConstraints,
	double* result,
	[[maybe_unused]] const unsigned NumParams,
	const double* x,
	double* grad,
	void* params)
{
	[[maybe_unused]] const auto& [TrainingSets, Energies, TotalEnergy, Purity] = *static_cast<AnalyticalConstraintParameters*>(params);
	// construct predictors
	const QuantumArray<ParameterVector> all_params = construct_all_parameters_from_diagonal(x);
	const OptionalKernels Kernels = construct_predictors(all_params, TrainingSets, false, true, grad != nullptr);
	// calculate population and energy
	result[0] = calculate_population(Kernels) - 1.0;
	result[1] = calculate_total_energy_average(Kernels, Energies) - TotalEnergy;
	if (NumConstraints == 3)
	{
		result[2] = calculate_purity(Kernels) - Purity;
	}
	if (grad != nullptr)
	{
		std::size_t iParam = 0;
		// population derivative
		{
			const ParameterVector& PplDeriv = population_derivative(Kernels);
			std::copy(PplDeriv.cbegin(), PplDeriv.cend(), grad + iParam);
			iParam += NumPES * Kernel::NumTotalParameters;
		}
		// energy derivative
		{
			const ParameterVector& EngDeriv = total_energy_derivative(Kernels, Energies);
			std::copy(EngDeriv.cbegin(), EngDeriv.cend(), grad + iParam);
			iParam += NumPES * Kernel::NumTotalParameters;
		}
		// purity derivative
		if (NumConstraints == 3)
		{
			const ParameterVector& PrtDeriv = purity_derivative(Kernels);
			std::size_t iPrtParam = 0;
			for (std::size_t iPES = 0; iPES < NumPES; iPES++)
			{
				for (std::size_t jPES = 0; jPES <= iPES; jPES++)
				{
					if (iPES == jPES)
					{
						std::copy(PrtDeriv.cbegin() + iPrtParam, PrtDeriv.cbegin() + iPrtParam + Kernel::NumTotalParameters, grad + iParam);
						iPrtParam += Kernel::NumTotalParameters;
						iParam += Kernel::NumTotalParameters;
					}
					else
					{
						iPrtParam += ComplexKernel::NumTotalParameters;
					}
				}
			}
		}
	}
	// prevent result from inf or nan
	for (std::size_t i = 0; i < NumConstraints; i++)
	{
		make_normal(result[i]);
	}
	if (grad != nullptr)
	{
		for (std::size_t i = 0; i < NumConstraints * NumParams; i++)
		{
			make_normal(grad[i]);
		}
	}
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
		}
	}
	AnalyticalLooseFunctionParameters alfp = std::tie(TrainingSets, ExtraTrainingSets);
	minimizer.set_min_objective(diagonal_loose, static_cast<void*>(&alfp));
	// set up constraints
	AnalyticalConstraintParameters acp = std::tie(TrainingSets, Energies, TotalEnergy, Purity);
	minimizer.add_equality_mconstraint(diagonal_constraints, static_cast<void*>(&acp), ParameterVector(Purity > 0 ? 3 : 2, 0.0));
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
		spdlog::error("{}Diagonal elements optimization failed by {}", indents<2>::apply(), e.what());
	}
#endif
	minimizer.remove_equality_constraints();
	// set up parameters
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		const std::size_t ElementIndex = iPES * NumPES + iPES;
		ParameterVectors[ElementIndex] = ParameterVector(
			diagonal_parameters.cbegin() + iPES * Kernel::NumTotalParameters,
			diagonal_parameters.cbegin() + (iPES + 1) * Kernel::NumTotalParameters);
		spdlog::info(
			"{}Parameter of {}: {}.",
			indents<3>::apply(),
			get_element_name(iPES, iPES),
			Eigen::VectorXd::Map(ParameterVectors[ElementIndex].data(), ParameterVectors[ElementIndex].size()).format(VectorFormatter));
	}
	const OptionalKernels kernels = construct_predictors(ParameterVectors, TrainingSets, false, true, false);
	spdlog::info(
		"{}Diagonal error = {}; Population = {}; Energy = {}; Purity = {}; using {} steps",
		indents<2>::apply(),
		err,
		calculate_population(kernels),
		calculate_total_energy_average(kernels, Energies),
		calculate_purity(kernels),
		minimizer.get_numevals());
	return std::make_tuple(err, std::vector<std::size_t>(1, minimizer.get_numevals()));
}

/// @brief To construct parameters for all elements from parameters of all elements
/// @param[in] x The combined parameters for all elements
/// @return Parameters of all elements, leaving off-diagonal parameters blank
static QuantumArray<ParameterVector> construct_all_parameters(const double* x)
{
	QuantumArray<ParameterVector> result;
	std::size_t iParam = 0;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES <= iPES; jPES++)
		{
			const std::size_t ElementIndex = iPES * NumPES + jPES;
			if (iPES == jPES)
			{
				result[ElementIndex].resize(Kernel::NumTotalParameters, 0);
				std::copy(x + iParam, x + iParam + Kernel::NumTotalParameters, result[ElementIndex].begin());
				iParam += Kernel::NumTotalParameters;
			}
			else
			{
				result[ElementIndex].resize(ComplexKernel::NumTotalParameters, 0);
				std::copy(x + iParam, x + iParam + ComplexKernel::NumTotalParameters, result[ElementIndex].begin());
				iParam += ComplexKernel::NumTotalParameters;
			}
		}
	}
	return result;
}

/// @brief To combine parameters of each elements into one, inverse function of @p construct_all_parameters
/// @param[in] x parameters of all elements
/// @return The combined parameters of all elements
static ParameterVector construct_combined_parameters(const QuantumArray<ParameterVector>& x)
{
	ParameterVector result;
	result.reserve(NumTotalParameters);
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES <= iPES; jPES++)
		{
			const std::size_t ElementIndex = iPES * NumPES + jPES;
			result.insert(result.cend(), x[ElementIndex].cbegin(), x[ElementIndex].cend());
		}
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
	const QuantumArray<ParameterVector> AllParams = construct_all_parameters(x.data());
	QuantumArray<ParameterVector> all_grads;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES <= iPES; jPES++)
		{
			const std::size_t ElementIndex = iPES * NumPES + jPES;
			ElementTrainingParameters etp = std::tie(TrainingSets[ElementIndex], ExtraTrainingSets[ElementIndex]);
			all_grads[ElementIndex] = ParameterVector(!grad.empty() ? (iPES == jPES ? Kernel::NumTotalParameters : ComplexKernel::NumTotalParameters) : 0);
			err += loose_function(AllParams[ElementIndex], all_grads[ElementIndex], static_cast<void*>(&etp));
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

/// @brief To calculate the error on all constraints (population, energy, and maybe purity) by analytical integral of parameters
/// @param[in] NumConstraints The number of constraints, should be 3
/// @param[out] result The error of each constraints
/// @param[in] NumParams The number of parameters, which should be @p NumTotalParameters
/// @param[in] x The input parameters, need to calculate the function value and gradient at this point
/// @param[out] grad The gradient at the given point
/// @param[in] params Other parameters. Here it is the training sets, energies on each surface, total energy and purity
static void full_constraints(
	const unsigned NumConstraints,
	double* result,
	[[maybe_unused]] const unsigned NumParams,
	const double* x,
	double* grad,
	void* params)
{
	[[maybe_unused]] const auto& [TrainingSets, Energies, TotalEnergy, Purity] = *static_cast<AnalyticalConstraintParameters*>(params);
	// construct predictors
	const QuantumArray<ParameterVector> all_params = construct_all_parameters(x);
	const OptionalKernels Kernels = construct_predictors(all_params, TrainingSets, false, true, grad != nullptr);
	// calculate population and energy
	result[0] = calculate_population(Kernels) - 1.0;
	result[1] = calculate_total_energy_average(Kernels, Energies) - TotalEnergy;
	result[2] = calculate_purity(Kernels) - Purity;
	if (grad != nullptr)
	{
		std::size_t iParam = 0;
		// population derivative
		{
			const ParameterVector& PplDeriv = construct_combined_parameters(construct_all_parameters_from_diagonal(population_derivative(Kernels).data()));
			std::copy(PplDeriv.cbegin(), PplDeriv.cend(), grad + iParam);
			iParam += NumTotalParameters;
		}
		// energy derivative
		{
			const ParameterVector& EngDeriv = construct_combined_parameters(construct_all_parameters_from_diagonal(total_energy_derivative(Kernels, Energies).data()));
			std::copy(EngDeriv.cbegin(), EngDeriv.cend(), grad + iParam);
			iParam += NumTotalParameters;
		}
		// purity derivative
		{
			const ParameterVector& PrtDeriv = purity_derivative(Kernels);
			std::copy(PrtDeriv.cbegin(), PrtDeriv.cend(), grad + iParam);
			iParam += NumTotalParameters;
		}
	}
	// prevent result from inf or nan
	for (std::size_t i = 0; i < NumConstraints; i++)
	{
		make_normal(result[i]);
	}
	if (grad != nullptr)
	{
		for (std::size_t i = 0; i < NumConstraints * NumParams; i++)
		{
			make_normal(grad[i]);
		}
	}
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
		for (std::size_t jPES = 0; jPES <= iPES; jPES++)
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
	AnalyticalConstraintParameters acp = std::tie(TrainingSets, Energies, TotalEnergy, Purity);
	minimizer.add_equality_mconstraint(full_constraints, static_cast<void*>(&acp), ParameterVector(3, 0.0));
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
		spdlog::error("{}All elements optimization failed by {}", indents<2>::apply(), e.what());
	}
#endif
	minimizer.remove_equality_constraints();
	// set up parameters
	std::size_t iParam = 0;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES <= iPES; jPES++)
		{
			const std::size_t ElementIndex = iPES * NumPES + jPES;
			[[maybe_unused]] const auto& [feature, label] = TrainingSets[ElementIndex];
			const std::size_t ElementTotalParameters = iPES == jPES ? Kernel::NumTotalParameters : ComplexKernel::NumTotalParameters;
			ParameterVectors[ElementIndex] = ParameterVector(full_parameters.cbegin() + iParam, full_parameters.cbegin() + iParam + ElementTotalParameters);
			spdlog::info(
				"{}Parameter of {}: {}.",
				indents<3>::apply(),
				get_element_name(iPES, jPES),
				Eigen::VectorXd::Map(ParameterVectors[ElementIndex].data(), ParameterVectors[ElementIndex].size()).format(VectorFormatter));
			iParam += ElementTotalParameters;
		}
	}
	const OptionalKernels kernels = construct_predictors(ParameterVectors, TrainingSets, false, true, false);
	spdlog::info(
		"{}Total error = {}; Population = {}; Energy = {}; Purity = {}; using {} steps.",
		indents<2>::apply(),
		err,
		calculate_population(kernels),
		calculate_total_energy_average(kernels, Energies),
		calculate_purity(kernels),
		minimizer.get_numevals());
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
		static_assert(std::is_same_v<std::decay_t<decltype(calc)>, std::decay_t<decltype(ref)>>);
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
			static constexpr double AbsoluteTolerance = 0.25;
			static constexpr double RelativeTolerance = 0.1;
			const auto err = ((calc.array() / ref.array()).abs() - 1.0).abs();
			if ((err < RelativeTolerance).all() || ((calc - ref).array().abs() < AbsoluteTolerance).all())
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
			spdlog::info("{}{} passes checking.", indents<2>::apply(), name);
		}
		else
		{
			spdlog::info("{}{} does not pass checking and has total relative error of {}.", indents<2>::apply(), name, error);
		}
	};

	Eigen::Vector4d result = Eigen::Vector4d::Zero();
	// <r>
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		const std::size_t ElementIndex = iPES * NumPES + iPES;
		const ClassicalPhaseVector r_ave = calculate_1st_order_average_one_surface(density[ElementIndex]);
		const ClassicalPhaseVector r_prm = calculate_1st_order_average_one_surface(Kernels[ElementIndex]); // prm = parameters
		spdlog::info(
			"{}On surface {}, Exact <r> = {}, Analytical integrated <r> = {}.",
			indents<3>::apply(),
			iPES,
			r_ave.format(VectorFormatter),
			r_prm.format(VectorFormatter));
		result[0] += beyond_tolerance_error(r_prm, r_ave);
	}
	check_and_output(result[0], "<r>");
	// <1>
	const double ppl_prm = calculate_population(Kernels);
	spdlog::info("{}Exact population = {}, analytical integrated population = {}.", indents<3>::apply(), 1.0, ppl_prm);
	result[1] += beyond_tolerance_error(ppl_prm, 1.0);
	check_and_output(result[1], "Population");
	// <E>
	const double eng_prm_ave = calculate_total_energy_average(Kernels, Energies);
	spdlog::info("{}Exact energy = {}, analytical integrated energy = {}", indents<3>::apply(), TotalEnergy, eng_prm_ave);
	result[2] += beyond_tolerance_error(eng_prm_ave, TotalEnergy);
	check_and_output(result[2], "Energy");
	// purity
	const double prt_prm = calculate_purity(Kernels);
	spdlog::info("{}Exact purity = {}, analytical integrated purity = {}.", indents<3>::apply(), Purity, prt_prm);
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
	// calculate is small or not
	const QuantumMatrix<bool> IsSmall = is_very_small(density);
	// construct training sets
	const AllTrainingSets TrainingSets = construct_training_sets(density), ExtraTrainingSets = construct_training_sets(extra_points);
	// set bounds for current bounds, and calculate energy on each surfaces
	const QuantumVector<double> Energies = calculate_total_energy_average_each_surface(density, mass);
	const QuantumArray<Bounds> ParameterBounds = calculate_bounds(density);
	auto move_into_bounds = [&ParameterBounds](QuantumArray<ParameterVector>& parameter_vectors)
	{
		for (std::size_t iPES = 0; iPES < NumPES; iPES++)
		{
			for (std::size_t jPES = 0; jPES <= iPES; jPES++)
			{
				const std::size_t ElementIndex = iPES * NumPES + jPES;
				const auto& [LowerBound, UpperBound] = ParameterBounds[ElementIndex];
				for (std::size_t iParam = 0; iParam < (iPES == jPES ? Kernel::NumTotalParameters : ComplexKernel::NumTotalParameters); iParam++)
				{
					parameter_vectors[ElementIndex][iParam] = std::max(parameter_vectors[ElementIndex][iParam], LowerBound[iParam]);
					parameter_vectors[ElementIndex][iParam] = std::min(parameter_vectors[ElementIndex][iParam], UpperBound[iParam]);
				}
			}
		}
	};
	set_optimizer_bounds(NLOptLocalMinimizers, ParameterBounds);
	set_optimizer_bounds(NLOptGlobalMinimizers, ParameterBounds);
	// any offdiagonal element is not small
	const bool OffDiagonalOptimization = [&IsSmall](void) -> bool
	{
		if constexpr (NumOffDiagonalElements != 0)
		{
			for (std::size_t iPES = 1; iPES < NumPES; iPES++)
			{
				for (std::size_t jPES = 0; jPES < iPES; jPES++)
				{
					if (!IsSmall(iPES, jPES))
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
		[this, &density, &IsSmall, &TrainingSets, &ExtraTrainingSets, &Energies, OffDiagonalOptimization, &ParameterBounds, &move_into_bounds](
			QuantumArray<ParameterVector>& parameter_vectors) -> std::tuple<Optimization::Result, Eigen::Vector4d>
	{
		// initially, set the magnitude to be 1
		// and move the parameters into bounds
		for (std::size_t iPES = 0; iPES < NumPES; iPES++)
		{
			for (std::size_t jPES = 0; jPES <= iPES; jPES++)
			{
				parameter_vectors[iPES * NumPES + jPES][0] = InitialMagnitude;
			}
		}
		move_into_bounds(parameter_vectors);
		Optimization::Result result = optimize_elementwise(TrainingSets, ExtraTrainingSets, IsSmall, NLOptLocalMinimizers, parameter_vectors);
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
			const auto& [TotalError, TotalStep] = optimize_full(
				TrainingSets,
				ExtraTrainingSets,
				Energies,
				TotalEnergy,
				Purity,
				NLOptLocalMinimizers[NumElements + 1],
				parameter_vectors);
			err = TotalError;
			steps.insert(steps.cend(), DiagonalStep.cbegin(), DiagonalStep.cend());
			steps.insert(steps.cend(), TotalStep.cbegin(), TotalStep.cend());
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
		for (std::size_t iPES = 0; iPES < NumPES; iPES++)
		{
			for (std::size_t jPES = 0; jPES <= iPES; jPES++)
			{
				if (!IsSmall(iPES, jPES))
				{
					const std::size_t ElementIndex = iPES * NumPES + jPES;
					const auto& [feature, label] = TrainingSets[ElementIndex];
					if (iPES == jPES)
					{
						parameter_vectors[ElementIndex][0] = Kernel(parameter_vectors[ElementIndex], feature, label.real(), false, false, false).get_magnitude();
						spdlog::info("{}Magnitude of {} is {}.", indents<2>::apply(), get_element_name(iPES, jPES), parameter_vectors[ElementIndex][0]);
					}
					else
					{
						parameter_vectors[ElementIndex][0] = ComplexKernel(parameter_vectors[ElementIndex], feature, label, false, false, false).get_magnitude();
						spdlog::info("{}Magnitude of {} is {}.", indents<2>::apply(), get_element_name(iPES, jPES), parameter_vectors[ElementIndex][0]);
					}
				}
				else
				{
					spdlog::info("{}Magnitude of {} is not needed.", indents<2>::apply(), get_element_name(iPES, jPES));
				}
			}
		}
		spdlog::info("{}Error = {}", indents<2>::apply(), err);
		return std::make_tuple(result, check_averages(construct_predictors(parameter_vectors, TrainingSets, false, true, false), density, Energies, TotalEnergy, Purity));
	};

	auto compare_and_overwrite =
		[this](
			Optimization::Result& result,
			Eigen::Vector4d& check_result,
			const std::string_view& old_name,
			const Optimization::Result& result_new,
			const Eigen::Vector4d& check_new,
			const std::string_view& new_name,
			const QuantumArray<ParameterVector>& param_vec_new) -> void
	{
		auto& [error, steps] = result;
		const auto& [error_new, steps_new] = result_new;
		const std::size_t BetterResults = (check_new.array() <= check_result.array()).count();
		if (BetterResults > 2)
		{
			spdlog::info("{}{} is better because of better averages.", indents<1>::apply(), new_name);
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
			spdlog::info("{}{} is better because of smaller error.", indents<1>::apply(), new_name);
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
			spdlog::info("{}{} is better.", indents<1>::apply(), old_name);
		}
	};

	// 1. optimize locally with parameters from last step
	spdlog::info("{}Local optimization with previous parameters.", indents<1>::apply());
	auto [result, check_result] = optimize_and_check(ParameterVectors);
	if ((check_result.array() == 0.0).all())
	{
		spdlog::info("{}Local optimization with previous parameters succeeded.", indents<1>::apply());
		return result;
	}
	// 2. optimize locally with initial parameter
	spdlog::warn("{}Local optimization with previous parameters failed. Retry with local optimization with initial parameters.", indents<1>::apply());
	QuantumArray<ParameterVector> param_vec_initial;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES <= iPES; jPES++)
		{
			const std::size_t ElementIndex = iPES * NumPES + jPES;
			if (iPES == jPES)
			{
				param_vec_initial[ElementIndex] = InitialKernelParameter;
			}
			else
			{
				param_vec_initial[ElementIndex] = InitialComplexKernelParameter;
			}
		}
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
		spdlog::info("{}Local optimization succeeded.", indents<1>::apply());
		return result;
	}
	// 3. optimize globally
	spdlog::warn("{}Local optimization failed. Retry with global optimization.", indents<1>::apply());
	QuantumArray<ParameterVector> param_vec_global;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES <= iPES; jPES++)
		{
			const std::size_t ElementIndex = iPES * NumPES + jPES;
			if (iPES == jPES)
			{
				param_vec_global[ElementIndex] = InitialKernelParameter;
			}
			else
			{
				param_vec_global[ElementIndex] = InitialComplexKernelParameter;
			}
		}
	}
	move_into_bounds(param_vec_global);
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES <= iPES; jPES++)
		{
			const std::size_t ElementIndex = iPES * NumPES + jPES;
			param_vec_global[ElementIndex] = local_parameter_to_global(param_vec_global[ElementIndex]);
		}
	}
	[[maybe_unused]] const auto& [error_global_elem, steps_global_elem] =
		optimize_elementwise(TrainingSets, ExtraTrainingSets, IsSmall, NLOptGlobalMinimizers, param_vec_global);
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES <= iPES; jPES++)
		{
			const std::size_t ElementIndex = iPES * NumPES + jPES;
			param_vec_global[ElementIndex] = global_parameter_to_local(param_vec_global[ElementIndex]);
		}
	}
	auto [result_global, check_global] = optimize_and_check(param_vec_global);
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
		param_vec_global);
	if ((check_result.array() == 0.0).all())
	{
		spdlog::info("{}Optimization succeeded.", indents<1>::apply());
		return result;
	}
	// 4. end
	spdlog::warn("{}Optimization failed.", indents<1>::apply());
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
