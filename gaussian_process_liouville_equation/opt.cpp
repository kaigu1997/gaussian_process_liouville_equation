/// @file opt.cpp
/// @brief Implementation of opt.h

#include "stdafx.h"

#include "opt.h"

#include "complex_kernel.h"
#include "input.h"
#include "kernel.h"
#include "predict.h"

/// @brief The lower and upper bound
using Bounds = std::array<ParameterVector, 2>;
/// @brief The parameters passed to the optimization subroutine of a single element
using ElementTrainingParameters = std::tuple<const ElementTrainingSet&, const ElementTrainingSet&>;
/// @brief The parameters passed to the optimization subroutine of all (diagonal) elements
/// including the training sets and an extra training set
using AnalyticalLooseFunctionParameters = std::tuple<const AllTrainingSets&, const AllTrainingSets&>;
/// @brief The parameters used for population, energy and purity conservation by analytical integral,
/// including the training sets, the energy on each surfaces, the total energy, and the exact initial purity
using AnalyticalConstraintParameters = std::tuple<const AllTrainingSets&, const QuantumVector<double>&, const double, const double>;

/// @brief The initial weight for all kernels
static constexpr double InitialMagnitude = 1.0;
/// @brief The initial weight for noise
static constexpr double InitialNoise = 1e-2;
/// @brief The number of parameters for all elements
static constexpr std::size_t NumTotalParameters = Kernel::NumTotalParameters * NumPES + ComplexKernel::NumTotalParameters * NumOffDiagonalElements;

QuantumMatrix<bool> is_very_small(const AllPoints& density)
{
	// squared value below this are regarded as 0
	QuantumMatrix<bool> result;
	// as density is symmetric, only lower-triangular elements needs evaluating
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES <= iPES; jPES++)
		{
			result(iPES, jPES) = std::all_of(
				std::execution::par_unseq,
				density(iPES, jPES).cbegin(),
				density(iPES, jPES).cend(),
				[iPES, jPES](const PhaseSpacePoint& psp) -> bool
				{
					return psp.get<1>() == 0.0;
				}
			);
		}
	}
	return result.selfadjointView<Eigen::Lower>();
}

/// @brief To calculate the lower and upper bound for kernel
/// @param[in] CharLengthLowerBound The lower bound for characteristic lengths
/// @param[in] CharLengthUpperBound The upper bound for characteristic lengths
/// @return The bounds for kernel
static Bounds calculate_kernel_bounds(
	const ClassicalPhaseVector& CharLengthLowerBound,
	const ClassicalPhaseVector& CharLengthUpperBound
)
{
	const Bounds& KernelBounds = [&CharLengthLowerBound, &CharLengthUpperBound](void) -> Bounds
	{
		Bounds result{ParameterVector(Kernel::NumTotalParameters), ParameterVector(Kernel::NumTotalParameters)};
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
	const ClassicalPhaseVector& CharLengthUpperBound
)
{
	const Bounds& ComplexKernelBounds = [&CharLengthLowerBound, &CharLengthUpperBound](void) -> Bounds
	{
		Bounds result{ParameterVector(ComplexKernel::NumTotalParameters), ParameterVector(ComplexKernel::NumTotalParameters)};
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
			lb[iParam] = InitialMagnitude / 10.0;
			ub[iParam] = InitialMagnitude * 10.0;
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
static QuantumStorage<Bounds> calculate_bounds(const ClassicalPhaseVector& rSize)
{
	static constexpr double GaussKerMinCharLength = 1.0 / 100.0; // Minimal characteristic length for Gaussian kernel
	const Bounds& KernelBounds = calculate_kernel_bounds(ClassicalPhaseVector::Ones() * GaussKerMinCharLength, rSize);
	const Bounds& ComplexKernelBounds = calculate_complex_kernel_bounds(ClassicalPhaseVector::Ones() * GaussKerMinCharLength, rSize);
	// then assignment
	QuantumStorage<Bounds> result;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES <= iPES; jPES++)
		{
			if (iPES == jPES)
			{
				result(iPES) = KernelBounds;
			}
			else
			{
				result(iPES, jPES) = ComplexKernelBounds;
			}
		}
	}
	return result;
}

/// @brief To set up the bounds for the parameter
/// @param[in] density The selected points in phase space of one element of density matrices
/// @return Lower and upper bounds
static QuantumStorage<Bounds> calculate_bounds(const AllPoints& density)
{
	QuantumStorage<Bounds> result;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES <= iPES; jPES++)
		{
			const ClassicalPhaseVector StdDev = calculate_standard_deviation_one_surface(density(iPES, jPES));
			const ClassicalPhaseVector CharLengthLB = StdDev / std::sqrt(density(iPES, jPES).size()), CharLengthUB = 2.0 * StdDev;
			if (iPES == jPES)
			{
				result(iPES) = calculate_kernel_bounds(CharLengthLB, CharLengthUB);
			}
			else
			{
				result(iPES, jPES) = calculate_complex_kernel_bounds(CharLengthLB, CharLengthUB);
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
/// @param[in] Bounds The lower and upper bounds for each element
/// @param[inout] local_optimizers The local optimizers
/// @param[inout] diagonal_optimizer Optimizer with diagonal constraints
/// @param[inout] full_optimizer Optimizer with all constraints
/// @param[inout] global_optimizers The global optimizers
void set_optimizer_bounds(
	const QuantumStorage<Bounds>& Bounds,
	QuantumStorage<nlopt::opt>& local_optimizers,
	nlopt::opt& diagonal_optimizer,
	nlopt::opt& full_optimizer,
	QuantumStorage<nlopt::opt>& global_optimizers
)
{
	ParameterVector diagonal_lower_bound, full_lower_bound, diagonal_upper_bound, full_upper_bound;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES <= iPES; jPES++)
		{
			const auto& [LowerBound, UpperBound] = Bounds(iPES, jPES);
			local_optimizers(iPES, jPES).set_lower_bounds(LowerBound);
			local_optimizers(iPES, jPES).set_upper_bounds(UpperBound);
			global_optimizers(iPES, jPES).set_lower_bounds(local_parameter_to_global(LowerBound));
			global_optimizers(iPES, jPES).set_upper_bounds(local_parameter_to_global(UpperBound));
			if (iPES == jPES)
			{
				diagonal_lower_bound.insert(diagonal_lower_bound.cend(), LowerBound.cbegin(), LowerBound.cend());
				diagonal_upper_bound.insert(diagonal_upper_bound.cend(), UpperBound.cbegin(), UpperBound.cend());
			}
			full_lower_bound.insert(full_lower_bound.cend(), LowerBound.cbegin(), LowerBound.cend());
			full_upper_bound.insert(full_upper_bound.cend(), UpperBound.cbegin(), UpperBound.cend());
		}
	}
	diagonal_optimizer.set_lower_bounds(diagonal_lower_bound);
	diagonal_optimizer.set_upper_bounds(diagonal_upper_bound);
	full_optimizer.set_lower_bounds(full_lower_bound);
	full_optimizer.set_upper_bounds(full_upper_bound);
}

Optimization::Optimization(
	const InitialParameters& InitParams,
	const double InitialTotalEnergy,
	const double InitialPurity,
	const nlopt::algorithm LocalAlgorithm,
	const nlopt::algorithm ConstraintAlgorithm,
	const nlopt::algorithm GlobalAlgorithm,
	const nlopt::algorithm GlobalSubAlgorithm
):
	TotalEnergy(InitialTotalEnergy),
	Purity(InitialPurity),
	mass(InitParams.get_mass()),
	InitialKernelParameter(
		[&rSigma = InitParams.get_sigma_r0()](void) -> ParameterVector
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
		}()
	),
	InitialComplexKernelParameter(
		[&rSigma = InitParams.get_sigma_r0()](void) -> ParameterVector
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
		}()
	),
	LocalMinimizers(nlopt::opt(LocalAlgorithm, Kernel::NumTotalParameters), nlopt::opt(LocalAlgorithm, ComplexKernel::NumTotalParameters)),
	DiagonalMinimizer(nlopt::algorithm::AUGLAG_EQ, Kernel::NumTotalParameters * NumPES),
	FullMinimizer(nlopt::algorithm::AUGLAG_EQ, NumTotalParameters),
	GlobalMinimizers(nlopt::opt(GlobalAlgorithm, Kernel::NumTotalParameters), nlopt::opt(GlobalAlgorithm, ComplexKernel::NumTotalParameters)),
	ParameterVectors(InitialKernelParameter, InitialComplexKernelParameter)
{
	static constexpr std::size_t MaximumEvaluations = 100000; // Maximum evaluations for global optimizer

	// set up parameters and minimizers
	auto set_optimizer = [](nlopt::opt& optimizer) -> void
	{
		static constexpr double RelativeTolerance = 1e-5;
		static constexpr double AbsoluteTolerance = 1e-15;
		static constexpr double InitialStepSize = 0.5;
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
		for (std::size_t jPES = 0; jPES <= iPES; jPES++)
		{
			set_optimizer(LocalMinimizers(iPES, jPES));
			set_optimizer(GlobalMinimizers(iPES, jPES));
			GlobalMinimizers(iPES, jPES).set_maxeval(MaximumEvaluations);
			switch (GlobalMinimizers(iPES, jPES).get_algorithm())
			{
			case nlopt::algorithm::G_MLSL:
			case nlopt::algorithm::GN_MLSL:
			case nlopt::algorithm::GD_MLSL:
			case nlopt::algorithm::G_MLSL_LDS:
			case nlopt::algorithm::GN_MLSL_LDS:
			case nlopt::algorithm::GD_MLSL_LDS:
				set_subsidiary_optimizer(GlobalMinimizers(iPES, jPES), GlobalSubAlgorithm);
				break;
			default:
				break;
			}
		}
	}
	// minimizer for all diagonal elements with regularization (population + energy + purity)
	set_optimizer(DiagonalMinimizer);
	set_subsidiary_optimizer(DiagonalMinimizer, ConstraintAlgorithm);
	// minimizer for all elements with regularization (population, energy and purity)
	set_optimizer(FullMinimizer);
	set_subsidiary_optimizer(FullMinimizer, ConstraintAlgorithm);
	// set up bounds
	set_optimizer_bounds(calculate_bounds(InitParams.get_rmax() - InitParams.get_rmin()), LocalMinimizers, DiagonalMinimizer, FullMinimizer, GlobalMinimizers);
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
	const auto& [ExtraTrainingFeature, ExtraTrainingLabel] = ExtraTrainingSet;
	// construct the kernel, then pass it to the error function
	double result = 0.0;
	if (x.size() == Kernel::NumTotalParameters)
	{
		const Kernel kernel(x, TrainingSet, true, false, !grad.empty());
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
		const ComplexKernel kernel(x, TrainingSet, true, false, !grad.empty());
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
	QuantumStorage<nlopt::opt>& minimizers,
	QuantumStorage<ParameterVector>& ParameterVectors
)
{
	const bool is_global = std::string_view(minimizers(0).get_algorithm_name()).find("global") != std::string_view::npos || minimizers(0).get_algorithm() == nlopt::algorithm::GN_ESCH; // judge if it is global optimizer by checking its name
	double total_error = 0.0;
	std::vector<std::size_t> num_steps;
	num_steps.reserve(NumPES + NumOffDiagonalElements);
	// optimize element by element
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES <= iPES; jPES++)
		{
			if (!IsSmall(iPES, jPES))
			{
				ElementTrainingParameters etp = std::tie(TrainingSets(iPES, jPES), ExtraTrainingSets(iPES, jPES));
				if (is_global)
				{
					minimizers(iPES, jPES).set_min_objective(loose_function_global_wrapper, static_cast<void*>(&etp));
				}
				else
				{
					minimizers(iPES, jPES).set_min_objective(loose_function, static_cast<void*>(&etp));
				}
				double err = 0.0;
				try
				{
					minimizers(iPES, jPES).optimize(ParameterVectors(iPES, jPES), err);
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
					Eigen::VectorXd::Map(ParameterVectors(iPES, jPES).data(), ParameterVectors(iPES, jPES).size()).format(VectorFormatter)
				);
				spdlog::info(
					"{}Error of {} = {}, using {} steps.",
					indents<2>::apply(),
					get_element_name(iPES, jPES),
					err,
					minimizers(iPES, jPES).get_numevals()
				);
				total_error += err;
				num_steps.push_back(minimizers(iPES, jPES).get_numevals());
			}
			else
			{
				spdlog::info("{}{} is 0 everywhere.", indents<2>::apply(), get_element_name(iPES, jPES));
				num_steps.push_back(0);
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
		const std::size_t BeginIndex = iPES * Kernel::NumTotalParameters;
		const ParameterVector x_element(x.cbegin() + BeginIndex, x.cbegin() + BeginIndex + Kernel::NumTotalParameters);
		ParameterVector grad_element(!grad.empty() ? Kernel::NumTotalParameters : 0);
		ElementTrainingParameters etp = std::tie(TrainingSets(iPES), ExtraTrainingSets(iPES));
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
static QuantumStorage<ParameterVector> construct_all_parameters_from_diagonal(const double* x)
{
	QuantumStorage<ParameterVector> result;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES <= iPES; jPES++)
		{
			if (iPES == jPES)
			{
				result(iPES) = ParameterVector(x + iPES * Kernel::NumTotalParameters, x + (iPES + 1) * Kernel::NumTotalParameters);
			}
			else
			{
				result(iPES, jPES) = ParameterVector(ComplexKernel::NumTotalParameters, 0.0);
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
	const unsigned NumParams,
	const double* x,
	double* grad,
	void* params
)
{
	[[maybe_unused]] const auto& [TrainingSets, Energies, TotalEnergy, Purity] = *static_cast<AnalyticalConstraintParameters*>(params);
	// construct predictors
	const Kernels AllKernels(construct_all_parameters_from_diagonal(x), TrainingSets, false, true, grad != nullptr);
	// calculate population and energy
	result[0] = AllKernels.calculate_population() - 1.0;
	result[1] = AllKernels.calculate_total_energy_average(Energies) - TotalEnergy;
	if (NumConstraints == 3)
	{
		result[2] = AllKernels.calculate_purity() - Purity;
	}
	if (grad != nullptr)
	{
		std::size_t iParam = 0;
		// population derivative
		{
			const ParameterVector& PplDeriv = AllKernels.population_derivative();
			std::copy(PplDeriv.cbegin(), PplDeriv.cend(), grad + iParam);
			iParam += NumPES * Kernel::NumTotalParameters;
		}
		// energy derivative
		{
			const ParameterVector& EngDeriv = AllKernels.total_energy_derivative(Energies);
			std::copy(EngDeriv.cbegin(), EngDeriv.cend(), grad + iParam);
			iParam += NumPES * Kernel::NumTotalParameters;
		}
		// purity derivative
		if (NumConstraints == 3)
		{
			const ParameterVector& PrtDeriv = AllKernels.purity_derivative();
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
	const QuantumVector<double>& Energies,
	const double TotalEnergy,
	const double Purity,
	nlopt::opt& minimizer,
	QuantumStorage<ParameterVector>& ParameterVectors
)
{
	// construct training set and parameters
	ParameterVector diagonal_parameters;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		// diagonal elements, save parameters
		diagonal_parameters.insert(
			diagonal_parameters.cend(),
			ParameterVectors(iPES).cbegin(),
			ParameterVectors(iPES).cend()
		);
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
		ParameterVectors(iPES) = ParameterVector(
			diagonal_parameters.cbegin() + iPES * Kernel::NumTotalParameters,
			diagonal_parameters.cbegin() + (iPES + 1) * Kernel::NumTotalParameters
		);
		spdlog::info(
			"{}Parameter of {}: {}.",
			indents<3>::apply(),
			get_element_name(iPES, iPES),
			Eigen::VectorXd::Map(ParameterVectors(iPES).data(), Kernel::NumTotalParameters).format(VectorFormatter)
		);
	}
	const Kernels AllKernels(ParameterVectors, TrainingSets, false, true, false);
	spdlog::info(
		"{}Diagonal error = {}; Population = {}; Energy = {}; Purity = {}; using {} steps",
		indents<2>::apply(),
		err,
		AllKernels.calculate_population(),
		AllKernels.calculate_total_energy_average(Energies),
		AllKernels.calculate_purity(),
		minimizer.get_numevals()
	);
	return std::make_tuple(err, std::vector<std::size_t>(1, minimizer.get_numevals()));
}

/// @brief To construct parameters for all elements from parameters of all elements
/// @param[in] x The combined parameters for all elements
/// @return Parameters of all elements, leaving off-diagonal parameters blank
static QuantumStorage<ParameterVector> construct_all_parameters(const double* x)
{
	QuantumStorage<ParameterVector> result;
	std::size_t iParam = 0;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES <= iPES; jPES++)
		{
			if (iPES == jPES)
			{
				result(iPES).resize(Kernel::NumTotalParameters, 0);
				std::copy(x + iParam, x + iParam + Kernel::NumTotalParameters, result(iPES).begin());
				iParam += Kernel::NumTotalParameters;
			}
			else
			{
				result(iPES, jPES).resize(ComplexKernel::NumTotalParameters, 0);
				std::copy(x + iParam, x + iParam + ComplexKernel::NumTotalParameters, result(iPES, jPES).begin());
				iParam += ComplexKernel::NumTotalParameters;
			}
		}
	}
	return result;
}

/// @brief To combine parameters of each elements into one, inverse function of @p construct_all_parameters
/// @param[in] x parameters of all elements
/// @return The combined parameters of all elements
static ParameterVector construct_combined_parameters(const QuantumStorage<ParameterVector>& x)
{
	ParameterVector result;
	result.reserve(NumTotalParameters);
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES <= iPES; jPES++)
		{
			result.insert(result.cend(), x(iPES, jPES).cbegin(), x(iPES, jPES).cend());
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
	const QuantumStorage<ParameterVector> AllParams = construct_all_parameters(x.data());
	QuantumStorage<ParameterVector> all_grads;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES <= iPES; jPES++)
		{
			ElementTrainingParameters etp = std::tie(TrainingSets(iPES, jPES), ExtraTrainingSets(iPES, jPES));
			all_grads(iPES, jPES) = ParameterVector(!grad.empty() ? (iPES == jPES ? Kernel::NumTotalParameters : ComplexKernel::NumTotalParameters) : 0);
			err += loose_function(AllParams(iPES, jPES), all_grads(iPES, jPES), static_cast<void*>(&etp));
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
	const unsigned NumParams,
	const double* x,
	double* grad,
	void* params
)
{
	[[maybe_unused]] const auto& [TrainingSets, Energies, TotalEnergy, Purity] = *static_cast<AnalyticalConstraintParameters*>(params);
	// construct predictors
	const Kernels AllKernels(construct_all_parameters(x), TrainingSets, false, true, grad != nullptr);
	// calculate population and energy
	result[0] = AllKernels.calculate_population() - 1.0;
	result[1] = AllKernels.calculate_total_energy_average(Energies) - TotalEnergy;
	result[2] = AllKernels.calculate_purity() - Purity;
	if (grad != nullptr)
	{
		std::size_t iParam = 0;
		// population derivative
		{
			const ParameterVector& PplDeriv = construct_combined_parameters(construct_all_parameters_from_diagonal(AllKernels.population_derivative().data()));
			std::copy(PplDeriv.cbegin(), PplDeriv.cend(), grad + iParam);
			iParam += NumTotalParameters;
		}
		// energy derivative
		{
			const ParameterVector& EngDeriv = construct_combined_parameters(construct_all_parameters_from_diagonal(AllKernels.total_energy_derivative(Energies).data()));
			std::copy(EngDeriv.cbegin(), EngDeriv.cend(), grad + iParam);
			iParam += NumTotalParameters;
		}
		// purity derivative
		{
			const ParameterVector& PrtDeriv = AllKernels.purity_derivative();
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
	const QuantumVector<double>& Energies,
	const double TotalEnergy,
	const double Purity,
	nlopt::opt& minimizer,
	QuantumStorage<ParameterVector>& ParameterVectors
)
{
	// construct training set
	ParameterVector full_parameters;
	full_parameters.reserve(Kernel::NumTotalParameters * NumElements);
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES <= iPES; jPES++)
		{
			full_parameters.insert(
				full_parameters.cend(),
				ParameterVectors(iPES, jPES).cbegin(),
				ParameterVectors(iPES, jPES).cend()
			);
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
			const std::size_t ElementTotalParameters = iPES == jPES ? Kernel::NumTotalParameters : ComplexKernel::NumTotalParameters;
			ParameterVectors(iPES, jPES) = ParameterVector(full_parameters.cbegin() + iParam, full_parameters.cbegin() + iParam + ElementTotalParameters);
			spdlog::info(
				"{}Parameter of {}: {}.",
				indents<3>::apply(),
				get_element_name(iPES, jPES),
				Eigen::VectorXd::Map(ParameterVectors(iPES, jPES).data(), ElementTotalParameters).format(VectorFormatter)
			);
			iParam += ElementTotalParameters;
		}
	}
	const Kernels AllKernels(ParameterVectors, TrainingSets, false, true, false);
	spdlog::info(
		"{}Total error = {}; Population = {}; Energy = {}; Purity = {}; using {} steps.",
		indents<2>::apply(),
		err,
		AllKernels.calculate_population(),
		AllKernels.calculate_total_energy_average(Energies),
		AllKernels.calculate_purity(),
		minimizer.get_numevals()
	);
	return std::make_tuple(err, std::vector<std::size_t>(1, minimizer.get_numevals()));
}

/// @brief To check each of the averages (@<r@>, @<1@>, @<E@> and purity) is satisfactory or not
/// @param[in] AllKernels Kernels of all elements for prediction
/// @param[in] density The vector containing all known density matrix
/// @param[in] Energies The energy of each surfaces and the total energy
/// @param[in] TotalEnergy The energy of the whole partial Wigner-transformed density matrics
/// @param[in] Purity The purity of the partial Wigner transformed density matrix
/// @return Whether each of the averages (@<r@>, @<1@>, @<E@> and purity) is satisfactory or not
/// @details The comparison happens between integral and average for @<r@>,
/// and between integral and exact value for @<1@>, @<E@> and purity.
static Eigen::Vector4d check_averages(
	const Kernels& AllKernels,
	const AllPoints& density,
	const QuantumVector<double>& Energies,
	const double TotalEnergy,
	const double Purity
)
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
		const ClassicalPhaseVector r_ave = calculate_1st_order_average_one_surface(density(iPES));
		const ClassicalPhaseVector r_prm = calculate_1st_order_average_one_surface(AllKernels(iPES)); // prm = parameters
		spdlog::info(
			"{}On surface {}, Exact <r> = {}, Analytical integrated <r> = {}.",
			indents<3>::apply(),
			iPES,
			r_ave.format(VectorFormatter),
			r_prm.format(VectorFormatter)
		);
		result[0] += beyond_tolerance_error(r_prm, r_ave);
	}
	check_and_output(result[0], "<r>");
	// <1>
	const double ppl_prm = AllKernels.calculate_population();
	spdlog::info("{}Exact population = {}, analytical integrated population = {}.", indents<3>::apply(), 1.0, ppl_prm);
	result[1] += beyond_tolerance_error(ppl_prm, 1.0);
	check_and_output(result[1], "Population");
	// <E>
	const double eng_prm_ave = AllKernels.calculate_total_energy_average(Energies);
	spdlog::info("{}Exact energy = {}, analytical integrated energy = {}", indents<3>::apply(), TotalEnergy, eng_prm_ave);
	result[2] += beyond_tolerance_error(eng_prm_ave, TotalEnergy);
	check_and_output(result[2], "Energy");
	// purity
	const double prt_prm = AllKernels.calculate_purity();
	spdlog::info("{}Exact purity = {}, analytical integrated purity = {}.", indents<3>::apply(), Purity, prt_prm);
	result[3] += beyond_tolerance_error(prt_prm, Purity);
	check_and_output(result[3], "Purity");
	return result;
}

/// @details Using Gaussian Process Regression (GPR) to depict phase space distribution.
/// First optimize elementwise, then optimize the diagonal with normalization and energy conservation
Optimization::Result Optimization::optimize(const AllPoints& density, const AllPoints& extra_points)
{
	// calculate is small or not
	const QuantumMatrix<bool> IsSmall = is_very_small(density);
	// construct training sets
	const AllTrainingSets TrainingSets = construct_training_sets(density), ExtraTrainingSets = construct_training_sets(extra_points);
	// set bounds for current bounds, and calculate energy on each surfaces
	const QuantumStorage<Bounds> ParameterBounds = calculate_bounds(density);
	// check for the suitable noise
	set_optimizer_bounds(ParameterBounds, LocalMinimizers, DiagonalMinimizer, FullMinimizer, GlobalMinimizers);

	auto move_into_bounds = [&ParameterBounds](QuantumStorage<ParameterVector>& parameter_vectors)
	{
		for (std::size_t iPES = 0; iPES < NumPES; iPES++)
		{
			for (std::size_t jPES = 0; jPES <= iPES; jPES++)
			{
				const auto& [LowerBound, UpperBound] = ParameterBounds(iPES, jPES);
				for (std::size_t iParam = 0; iParam < (iPES == jPES ? Kernel::NumTotalParameters : ComplexKernel::NumTotalParameters); iParam++)
				{
					parameter_vectors(iPES, jPES)[iParam] = std::max(parameter_vectors(iPES, jPES)[iParam], LowerBound[iParam]);
					parameter_vectors(iPES, jPES)[iParam] = std::min(parameter_vectors(iPES, jPES)[iParam], UpperBound[iParam]);
				}
			}
		}
	};

	// construct a function for code reuse
	auto optimize_and_check =
		[this,
		 &density,
		 &IsSmall,
		 &TrainingSets,
		 &ExtraTrainingSets,
		 &move_into_bounds](
			QuantumStorage<ParameterVector>& parameter_vectors
		)
		-> std::tuple<Optimization::Result, Eigen::Vector4d>
	{
		const QuantumVector<double> Energies = calculate_total_energy_average_each_surface(density, mass);
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
		// initially, set the magnitude to be 1
		// and move the parameters into bounds
		for (std::size_t iPES = 0; iPES < NumPES; iPES++)
		{
			for (std::size_t jPES = 0; jPES <= iPES; jPES++)
			{
				parameter_vectors(iPES, jPES)[0] = InitialMagnitude;
			}
		}
		move_into_bounds(parameter_vectors);
		Optimization::Result result = optimize_elementwise(TrainingSets, ExtraTrainingSets, IsSmall, LocalMinimizers, parameter_vectors);
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
				DiagonalMinimizer,
				parameter_vectors
			);
			const auto& [TotalError, TotalStep] = optimize_full(
				TrainingSets,
				ExtraTrainingSets,
				Energies,
				TotalEnergy,
				Purity,
				FullMinimizer,
				parameter_vectors
			);
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
				DiagonalMinimizer,
				parameter_vectors
			);
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
					if (iPES == jPES)
					{
						parameter_vectors(iPES, jPES)[0] = Kernel(parameter_vectors(iPES, jPES), TrainingSets(iPES, jPES), false, false, false).get_magnitude();
						spdlog::info("{}Magnitude of {} is {}.", indents<2>::apply(), get_element_name(iPES, jPES), parameter_vectors(iPES, jPES)[0]);
					}
					else
					{
						parameter_vectors(iPES, jPES)[0] = ComplexKernel(parameter_vectors(iPES, jPES), TrainingSets(iPES, jPES), false, false, false).get_magnitude();
						spdlog::info("{}Magnitude of {} is {}.", indents<2>::apply(), get_element_name(iPES, jPES), parameter_vectors(iPES, jPES)[0]);
					}
				}
				else
				{
					spdlog::info("{}Magnitude of {} is not needed.", indents<2>::apply(), get_element_name(iPES, jPES));
				}
			}
		}
		spdlog::info("{}Error = {}", indents<2>::apply(), err);
		return std::make_tuple(result, check_averages(Kernels(parameter_vectors, TrainingSets, false, true, false), density, Energies, TotalEnergy, Purity));
	};

	auto compare_and_overwrite =
		[this](
			Optimization::Result& result,
			Eigen::Vector4d& check_result,
			const std::string_view& old_name,
			const Optimization::Result& result_new,
			const Eigen::Vector4d& check_new,
			const std::string_view& new_name,
			const QuantumStorage<ParameterVector>& param_vec_new
		) -> void
	{
		auto& [error, steps] = result;
		const auto& [error_new, steps_new] = result_new;
		const std::size_t BetterResults = (check_new.array() < check_result.array()).count();
		const std::size_t WorseResults = (check_new.array() > check_result.array()).count();
		if (BetterResults > WorseResults)
		{
			spdlog::info("{}{} is better because of better averages.", indents<1>::apply(), new_name);
			ParameterVectors = param_vec_new;
			error = error_new;
			for (std::size_t iElement = 0; iElement < steps_new.size(); iElement++)
			{
				steps[iElement] += steps_new[iElement];
			}
			check_result = check_new;
		}
		else if (BetterResults == WorseResults && error_new < error)
		{
			spdlog::info("{}{} is better because of smaller error.", indents<1>::apply(), new_name);
			ParameterVectors = param_vec_new;
			error = error_new;
			for (std::size_t iElement = 0; iElement < steps_new.size(); iElement++)
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
	QuantumStorage<ParameterVector> param_vec_initial(InitialKernelParameter, InitialComplexKernelParameter);
	const auto& [result_initial, check_initial] = optimize_and_check(param_vec_initial);
	compare_and_overwrite(
		result,
		check_result,
		"Local optimization with previous parameters",
		result_initial,
		check_initial,
		"Local optimization with initial parameters",
		param_vec_initial
	);
	if ((check_result.array() == 0.0).all())
	{
		spdlog::info("{}Local optimization succeeded.", indents<1>::apply());
		return result;
	}
	// 3. optimize globally
	spdlog::warn("{}Local optimization failed. Retry with global optimization.", indents<1>::apply());
	QuantumStorage<ParameterVector> param_vec_global(InitialKernelParameter, InitialComplexKernelParameter);
	move_into_bounds(param_vec_global);
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES <= iPES; jPES++)
		{
			param_vec_global(iPES, jPES) = local_parameter_to_global(param_vec_global(iPES, jPES));
		}
	}
	[[maybe_unused]] const auto& [error_global_elem, steps_global_elem] =
		optimize_elementwise(TrainingSets, ExtraTrainingSets, IsSmall, GlobalMinimizers, param_vec_global);
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES <= iPES; jPES++)
		{
			param_vec_global(iPES, jPES) = global_parameter_to_local(param_vec_global(iPES, jPES));
		}
	}
	auto [result_global, check_global] = optimize_and_check(param_vec_global);
	for (std::size_t iElement = 0; iElement < NumTriangularElements; iElement++)
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
		param_vec_global
	);
	if ((check_result.array() == 0.0).all())
	{
		spdlog::info("{}Optimization succeeded.", indents<1>::apply());
		return result;
	}
	// 4. end
	spdlog::warn("{}Optimization failed.", indents<1>::apply());
	return result;
}

QuantumStorage<ParameterVector> Optimization::get_lower_bounds(void) const
{
	QuantumStorage<ParameterVector> result;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES <= iPES; jPES++)
		{
			result(iPES, jPES) = LocalMinimizers(iPES, jPES).get_lower_bounds();
		}
	}
	return result;
}

QuantumStorage<ParameterVector> Optimization::get_upper_bounds(void) const
{
	QuantumStorage<ParameterVector> result;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES <= iPES; jPES++)
		{
			result(iPES, jPES) = LocalMinimizers(iPES, jPES).get_upper_bounds();
		}
	}
	return result;
}
