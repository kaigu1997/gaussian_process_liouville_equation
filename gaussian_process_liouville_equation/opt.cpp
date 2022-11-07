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
		Bounds result{ParameterVector(KernelBase::NumTotalParameters), ParameterVector(KernelBase::NumTotalParameters)};
		auto& [lb, ub] = result;
		std::size_t iParam = 0;
		// first, magnitude
		lb[iParam] = InitialMagnitude;
		ub[iParam] = InitialMagnitude;
		iParam++;
		// second, Gaussian
		for (const std::size_t iDim : std::ranges::iota_view{0ul, PhaseDim})
		{
			lb[iParam + iDim] = CharLengthLowerBound[iDim];
			ub[iParam + iDim] = CharLengthUpperBound[iDim];
		}
		iParam += PhaseDim;
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
		Bounds result{ParameterVector(ComplexKernelBase::NumTotalParameters), ParameterVector(ComplexKernelBase::NumTotalParameters)};
		auto& [lb, ub] = result;
		std::size_t iParam = 0;
		// complex kernel, similarly
		// first, magnitude
		lb[iParam] = InitialMagnitude;
		ub[iParam] = InitialMagnitude;
		iParam++;
		// second, kernels
		for ([[maybe_unused]] const std::size_t iKernel : std::ranges::iota_view{0ul, ComplexKernelBase::NumKernels})
		{
			// first, magnitude
			lb[iParam] = InitialMagnitude / 10.0;
			ub[iParam] = InitialMagnitude * 10.0;
			iParam++;
			// second, Gaussian
			for (const std::size_t iDim : std::ranges::iota_view{0ul, PhaseDim})
			{
				lb[iParam + iDim] = CharLengthLowerBound[iDim];
				ub[iParam + iDim] = CharLengthUpperBound[iDim];
			}
			iParam += PhaseDim;
		}
		// third, noise
		lb[iParam] = InitialNoise;
		ub[iParam] = InitialNoise;
		iParam++;
		return result;
	}();
	return ComplexKernelBounds;
}

/// @brief To transform local parameters to global parameters by ln
/// @param[in] param The parameters for local optimizer, i.e., the normal parameter
/// @return The parameters for global optimizer, i.e., the changed parameter
static inline ParameterVector local_parameter_to_global(const ParameterVector& param)
{
	assert(param.size() == KernelBase::NumTotalParameters || param.size() == ComplexKernelBase::NumTotalParameters);
	ParameterVector result = param;
	std::size_t iParam = 0;
	if (param.size() == ComplexKernelBase::NumTotalParameters)
	{
		// real, imaginary and noise to log
		// first, magnitude
		iParam++;
		// second, real and imaginary kernel
		for ([[maybe_unused]] const std::size_t iKernel : std::ranges::iota_view{0ul, ComplexKernelBase::NumKernels})
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
	else // if (param.size() == KernelBase::NumTotalParameters)
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
	assert(param.size() == KernelBase::NumTotalParameters || param.size() == ComplexKernelBase::NumTotalParameters);
	assert(grad.size() == param.size() || grad.empty());
	ParameterVector result = grad;
	std::size_t iParam = 0;
	if (result.size() == ComplexKernelBase::NumTotalParameters)
	{
		// real, imaginary and noise to log
		// first, magnitude
		iParam++;
		// second, real and imaginary kernel
		for ([[maybe_unused]] const std::size_t iKernel : std::ranges::iota_view{0ul, ComplexKernelBase::NumKernels})
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
	else if (result.size() == KernelBase::NumTotalParameters)
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
	assert(param.size() == KernelBase::NumTotalParameters || param.size() == ComplexKernelBase::NumTotalParameters);
	ParameterVector result = param;
	std::size_t iParam = 0;
	if (param.size() == ComplexKernelBase::NumTotalParameters)
	{
		// real, imaginary and noise to log
		// first, magnitude
		iParam++;
		// second, real and imaginary kernel
		for ([[maybe_unused]] const std::size_t iKernel : std::ranges::iota_view{0ul, ComplexKernelBase::NumKernels})
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
	else // if (param.size() == KernelBase::NumTotalParameters)
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
	for (const std::size_t iPES : std::ranges::iota_view{0ul, NumPES})
	{
		for (const std::size_t jPES : std::ranges::iota_view{0ul, iPES + 1})
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
	const nlopt::algorithm LocalDiagonalAlgorithm,
	const nlopt::algorithm LocalOffDiagonalAlgorithm,
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
			ParameterVector result(KernelBase::NumTotalParameters);
			std::size_t iParam = 0;
			// first, magnitude
			result[iParam] = InitialMagnitude;
			iParam++;
			// second, Gaussian
			for (const std::size_t iDim : std::ranges::iota_view{0ul, PhaseDim})
			{
				result[iParam + iDim] = rSigma[iDim];
			}
			iParam += PhaseDim;
			// thrid, noise
			result[iParam] = InitialNoise;
			iParam++;
			return result;
		}()
	),
	InitialComplexKernelParameter(
		[&rSigma = InitParams.get_sigma_r0()](void) -> ParameterVector
		{
			ParameterVector result(ComplexKernelBase::NumTotalParameters);
			std::size_t iParam = 0;
			// first, magnitude
			result[iParam] = InitialMagnitude;
			iParam++;
			// second, real and imaginary kernel
			for ([[maybe_unused]] const std::size_t iKernel : std::ranges::iota_view{0ul, ComplexKernelBase::NumKernels})
			{
				// first, magnitude
				result[iParam] = InitialMagnitude;
				iParam++;
				// second, Gaussian
				for (const std::size_t iDim : std::ranges::iota_view{0ul, PhaseDim})
				{
					result[iParam + iDim] = rSigma[iDim];
				}
				iParam += PhaseDim;
			}
			// finally, noise
			result[iParam] = InitialNoise;
			iParam++;
			return result;
		}()
	),
	LocalMinimizers(nlopt::opt(LocalDiagonalAlgorithm, KernelBase::NumTotalParameters), nlopt::opt(LocalOffDiagonalAlgorithm, ComplexKernelBase::NumTotalParameters)),
	DiagonalMinimizer(nlopt::algorithm::AUGLAG_EQ, KernelBase::NumTotalParameters * NumPES),
	FullMinimizer(nlopt::algorithm::AUGLAG_EQ, NumTotalParameters),
	GlobalMinimizers(nlopt::opt(GlobalAlgorithm, KernelBase::NumTotalParameters), nlopt::opt(GlobalAlgorithm, ComplexKernelBase::NumTotalParameters)),
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
	for (const std::size_t iPES : std::ranges::iota_view{0ul, NumPES})
	{
		for (const std::size_t jPES : std::ranges::iota_view{0ul, iPES + 1})
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
	set_optimizer_bounds(
		[](const ClassicalPhaseVector& rSize) -> QuantumStorage<Bounds>
		{
			static constexpr double GaussKerMinCharLength = 1.0 / 100.0; // Minimal characteristic length for Gaussian kernel
			const Bounds& KernelBounds = calculate_kernel_bounds(ClassicalPhaseVector::Ones() * GaussKerMinCharLength, rSize);
			const Bounds& ComplexKernelBounds = calculate_complex_kernel_bounds(ClassicalPhaseVector::Ones() * GaussKerMinCharLength, rSize);
			// then assignment
			QuantumStorage<Bounds> result;
			for (const std::size_t iPES : std::ranges::iota_view{0ul, NumPES})
			{
				for (const std::size_t jPES : std::ranges::iota_view{0ul, iPES + 1})
				{
					result(iPES, jPES) = iPES == jPES ? KernelBounds : ComplexKernelBounds;
				}
			}
			return result;
		}(InitParams.get_rmax() - InitParams.get_rmin()),
		LocalMinimizers,
		DiagonalMinimizer,
		FullMinimizer,
		GlobalMinimizers
	);
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
/// For the extra training set, difference is @f$ K(\vec{x}_i,X)K^{-1}\vec{y}-y_i @f$.
static double loose_function(const ParameterVector& x, ParameterVector& grad, void* params)
{
	// unpack the parameters
	const auto& [TrainingSet, ExtraTrainingSet] = *static_cast<ElementTrainingParameters*>(params);
	const auto& [ExtraTrainingFeature, ExtraTrainingLabel] = ExtraTrainingSet;
	// construct the kernel, then pass it to the error function
	double result = 0.0;
	if (x.size() == KernelBase::NumTotalParameters)
	{
		const TrainingKernel kernel(x, TrainingSet, true, false, !grad.empty());
		const PredictiveKernel ExtraKernel(ExtraTrainingFeature, kernel, !grad.empty(), ExtraTrainingLabel.real());
		result = kernel.get_error() + ExtraKernel.get_error();
		if (!grad.empty())
		{
			const KernelBase::ParameterArray<double> trn_deriv = kernel.get_error_derivative(), vld_deriv = ExtraKernel.get_error_derivative();
			for (const std::size_t iParam : std::ranges::iota_view{0ul, KernelBase::NumTotalParameters})
			{
				grad[iParam] = trn_deriv[iParam] + vld_deriv[iParam];
			}
		}
	}
	else // x.size() == ComplexKernelBase::NumTotalParameters
	{
		const TrainingComplexKernel kernel(x, TrainingSet, true, false, !grad.empty());
		const PredictiveComplexKernel ExtraKernel(ExtraTrainingFeature, kernel, !grad.empty(), ExtraTrainingLabel);
		result = kernel.get_error() + ExtraKernel.get_error();
		if (!grad.empty())
		{
			const ComplexKernelBase::ParameterArray<double> trn_deriv = kernel.get_error_derivative(), vld_deriv = ExtraKernel.get_error_derivative();
			for (const std::size_t iParam : std::ranges::iota_view{0ul, ComplexKernelBase::NumTotalParameters})
			{
				grad[iParam] = trn_deriv[iParam] + vld_deriv[iParam];
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
/// @param[inout] minimizers The minimizers to use
/// @param[inout] ParameterVectors The parameters for all elements of density matrix
/// @return The total error and the optimization steps of each element
static Optimization::Result optimize_elementwise(
	const AllTrainingSets& TrainingSets,
	const AllTrainingSets& ExtraTrainingSets,
	QuantumStorage<nlopt::opt>& minimizers,
	QuantumStorage<ParameterVector>& ParameterVectors
)
{
	const bool is_global = std::string_view(minimizers(0).get_algorithm_name()).find("global") != std::string_view::npos
		|| minimizers(0).get_algorithm() == nlopt::algorithm::GN_ESCH; // judge if it is global optimizer by checking its name
	double total_error = 0.0;
	std::vector<std::size_t> num_steps;
	num_steps.reserve(NumPES + NumOffDiagonalElements);
	// optimize element by element
	spdlog::info("{}Start elementwise optimization...", indents<2>::apply());
	for (const std::size_t iPES : std::ranges::iota_view{0ul, NumPES})
	{
		for (const std::size_t jPES : std::ranges::iota_view{0ul, iPES + 1})
		{
			if (std::get<0>(TrainingSets(iPES, jPES)).size() != 0)
			{
				// calculate error for non-zero case
				ElementTrainingParameters etp = std::tie(TrainingSets(iPES, jPES), ExtraTrainingSets(iPES, jPES));
				if (is_global)
				{
					minimizers(iPES, jPES).set_min_objective(loose_function_global_wrapper, static_cast<void*>(&etp));
				}
				else
				{
					minimizers(iPES, jPES).set_min_objective(loose_function, static_cast<void*>(&etp));
				}
				spdlog::info("{}Start {} optimization...", indents<3>::apply(), get_element_name(iPES, jPES));
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
					spdlog::error("{}Optimization of {} failed by {}", indents<3>::apply(), get_element_name(iPES, jPES), e.what());
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
					indents<3>::apply(),
					get_element_name(iPES, jPES),
					err,
					minimizers(iPES, jPES).get_numevals()
				);
				total_error += err;
				num_steps.push_back(minimizers(iPES, jPES).get_numevals());
			}
			else
			{
				spdlog::info("{}{} is 0 everywhere.", indents<3>::apply(), get_element_name(iPES, jPES));
				num_steps.push_back(0);
			}
		}
	}
	return std::make_tuple(total_error, num_steps, Optimization::OptimizationType::Default);
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
	for (const std::size_t iPES : std::ranges::iota_view{0ul, NumPES})
	{
		if (std::get<0>(TrainingSets(iPES)).size() != 0)
		{
			// calculate error for non-zero case
			const std::size_t BeginIndex = iPES * KernelBase::NumTotalParameters;
			const ParameterVector x_element(x.cbegin() + BeginIndex, x.cbegin() + BeginIndex + KernelBase::NumTotalParameters);
			ParameterVector grad_element(!grad.empty() ? KernelBase::NumTotalParameters : 0);
			ElementTrainingParameters etp = std::tie(TrainingSets(iPES), ExtraTrainingSets(iPES));
			err += loose_function(x_element, grad_element, static_cast<void*>(&etp));
			std::copy(std::execution::par_unseq, grad_element.cbegin(), grad_element.cend(), grad.begin() + BeginIndex);
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
static QuantumStorage<ParameterVector> construct_all_parameters_from_diagonal(const double* x)
{
	QuantumStorage<ParameterVector> result;
	for (const std::size_t iPES : std::ranges::iota_view{0ul, NumPES})
	{
		for (const std::size_t jPES : std::ranges::iota_view{0ul, iPES + 1})
		{
			result(iPES, jPES) = iPES == jPES
				? ParameterVector(x + iPES * KernelBase::NumTotalParameters, x + (iPES + 1) * KernelBase::NumTotalParameters)
				: ParameterVector(ComplexKernelBase::NumTotalParameters, 0.0);
		}
	}
	return result;
}

/// @brief To calculate the error on all constraints (population, energy, and maybe purity) by analytical integral of parameters
/// @param[in] NumConstraints The number of constraints, could be 2 (population + energy) or 3 (+ purity)
/// @param[out] result The error of each constraints
/// @param[in] NumParams The number of parameters, which should be @p NumPES * @p KernelBase::NumTotalParameters
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
	const TrainingKernels AllKernels(construct_all_parameters_from_diagonal(x), TrainingSets, false, true, grad != nullptr);
	// calculate population and energy, and purity sometimes
	result[0] = AllKernels.calculate_population() - 1.0;
	result[1] = AllKernels.calculate_total_energy_average(Energies) - TotalEnergy;
	if (NumConstraints == 3)
	{
		result[2] = AllKernels.calculate_purity() - Purity;
	}
	// gradient
	if (grad != nullptr)
	{
		std::size_t iParam = 0;
		// population derivative
		{
			const ParameterVector& PplDeriv = AllKernels.population_derivative();
			std::copy(std::execution::par_unseq, PplDeriv.cbegin(), PplDeriv.cend(), grad + iParam);
			iParam += NumPES * KernelBase::NumTotalParameters;
		}
		// energy derivative
		{
			const ParameterVector& EngDeriv = AllKernels.total_energy_derivative(Energies);
			std::copy(std::execution::par_unseq, EngDeriv.cbegin(), EngDeriv.cend(), grad + iParam);
			iParam += NumPES * KernelBase::NumTotalParameters;
		}
		// purity derivative
		if (NumConstraints == 3)
		{
			const ParameterVector& PrtDeriv = AllKernels.purity_derivative();
			std::size_t iPrtParam = 0;
			for (const std::size_t iPES : std::ranges::iota_view{0ul, NumPES})
			{
				for (const std::size_t jPES : std::ranges::iota_view{0ul, iPES + 1})
				{
					if (iPES == jPES)
					{
						std::copy(
							std::execution::par_unseq,
							PrtDeriv.cbegin() + iPrtParam,
							PrtDeriv.cbegin() + iPrtParam + KernelBase::NumTotalParameters,
							grad + iParam
						);
						iPrtParam += KernelBase::NumTotalParameters;
						iParam += KernelBase::NumTotalParameters;
					}
					else
					{
						iPrtParam += ComplexKernelBase::NumTotalParameters;
					}
				}
			}
		}
	}
	// prevent result from inf or nan
	for (const unsigned i : std::ranges::iota_view{0u, NumConstraints})
	{
		make_normal(result[i]);
	}
	if (grad != nullptr)
	{
		for (const unsigned i : std::ranges::iota_view{0u, NumConstraints * NumParams})
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
	diagonal_parameters.reserve(KernelBase::NumTotalParameters * NumPES);
	for (const std::size_t iPES : std::ranges::iota_view{0ul, NumPES})
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
	spdlog::info("{}Start diagonal elements optimization...", indents<2>::apply());
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
	for (const std::size_t iPES : std::ranges::iota_view{0ul, NumPES})
	{
		ParameterVectors(iPES) = ParameterVector(
			diagonal_parameters.cbegin() + iPES * KernelBase::NumTotalParameters,
			diagonal_parameters.cbegin() + (iPES + 1) * KernelBase::NumTotalParameters
		);
		spdlog::info(
			"{}Parameter of {}: {}.",
			indents<3>::apply(),
			get_element_name(iPES, iPES),
			Eigen::VectorXd::Map(ParameterVectors(iPES).data(), KernelBase::NumTotalParameters).format(VectorFormatter)
		);
	}
	const TrainingKernels AllKernels(ParameterVectors, TrainingSets, false, true, false);
	spdlog::info(
		"{}Diagonal error = {}; Population = {}; Energy = {}; Purity = {}; using {} steps",
		indents<2>::apply(),
		err,
		AllKernels.calculate_population(),
		AllKernels.calculate_total_energy_average(Energies),
		AllKernels.calculate_purity(),
		minimizer.get_numevals()
	);
	return std::make_tuple(err, std::vector<std::size_t>(1, minimizer.get_numevals()), Optimization::OptimizationType::Default);
}

/// @brief To construct parameters for all elements from parameters of all elements
/// @param[in] x The combined parameters for all elements
/// @return Parameters of all elements, leaving off-diagonal parameters blank
static QuantumStorage<ParameterVector> construct_all_parameters(const double* x)
{
	QuantumStorage<ParameterVector> result;
	std::size_t iParam = 0;
	for (const std::size_t iPES : std::ranges::iota_view{0ul, NumPES})
	{
		for (const std::size_t jPES : std::ranges::iota_view{0ul, iPES + 1})
		{
			const std::size_t Length = iPES == jPES ? KernelBase::NumTotalParameters : ComplexKernelBase::NumTotalParameters;
			result(iPES, jPES).resize(Length);
			std::copy(x + iParam, x + iParam + Length, result(iPES, jPES).begin());
			iParam += Length;
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
	for (const std::size_t iPES : std::ranges::iota_view{0ul, NumPES})
	{
		for (const std::size_t jPES : std::ranges::iota_view{0ul, iPES + 1})
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
	for (const std::size_t iPES : std::ranges::iota_view{0ul, NumPES})
	{
		for (const std::size_t jPES : std::ranges::iota_view{0ul, iPES + 1})
		{
			if (std::get<0>(TrainingSets(iPES, jPES)).size() != 0)
			{
				ElementTrainingParameters etp = std::tie(TrainingSets(iPES, jPES), ExtraTrainingSets(iPES, jPES));
				all_grads(iPES, jPES) = ParameterVector(!grad.empty() ? (iPES == jPES ? KernelBase::NumTotalParameters : ComplexKernelBase::NumTotalParameters) : 0);
				err += loose_function(AllParams(iPES, jPES), all_grads(iPES, jPES), static_cast<void*>(&etp));
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
	const auto& [TrainingSets, Energies, TotalEnergy, Purity] = *static_cast<AnalyticalConstraintParameters*>(params);
	// construct predictors
	const TrainingKernels AllKernels(construct_all_parameters(x), TrainingSets, false, true, grad != nullptr);
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
			std::copy(std::execution::par_unseq, PplDeriv.cbegin(), PplDeriv.cend(), grad + iParam);
			iParam += NumTotalParameters;
		}
		// energy derivative
		{
			const ParameterVector& EngDeriv = construct_combined_parameters(construct_all_parameters_from_diagonal(AllKernels.total_energy_derivative(Energies).data()));
			std::copy(std::execution::par_unseq, EngDeriv.cbegin(), EngDeriv.cend(), grad + iParam);
			iParam += NumTotalParameters;
		}
		// purity derivative
		{
			const ParameterVector& PrtDeriv = AllKernels.purity_derivative();
			std::copy(std::execution::par_unseq, PrtDeriv.cbegin(), PrtDeriv.cend(), grad + iParam);
			iParam += NumTotalParameters;
		}
	}
	// prevent result from inf or nan
	for (const unsigned i : std::ranges::iota_view{0u, NumConstraints})
	{
		make_normal(result[i]);
	}
	if (grad != nullptr)
	{
		for (const unsigned i : std::ranges::iota_view{0u, NumConstraints * NumParams})
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
	full_parameters.reserve(NumTotalParameters);
	for (const std::size_t iPES : std::ranges::iota_view{0ul, NumPES})
	{
		for (const std::size_t jPES : std::ranges::iota_view{0ul, iPES + 1})
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
	spdlog::info("{}Start all elements optimization...", indents<2>::apply());
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
	for (const std::size_t iPES : std::ranges::iota_view{0ul, NumPES})
	{
		for (const std::size_t jPES : std::ranges::iota_view{0ul, iPES + 1})
		{
			const std::size_t ElementTotalParameters = iPES == jPES ? KernelBase::NumTotalParameters : ComplexKernelBase::NumTotalParameters;
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
	const TrainingKernels AllKernels(ParameterVectors, TrainingSets, false, true, false);
	spdlog::info(
		"{}Total error = {}; Population = {}; Energy = {}; Purity = {}; using {} steps.",
		indents<2>::apply(),
		err,
		AllKernels.calculate_population(),
		AllKernels.calculate_total_energy_average(Energies),
		AllKernels.calculate_purity(),
		minimizer.get_numevals()
	);
	return std::make_tuple(err, std::vector<std::size_t>(1, minimizer.get_numevals()), Optimization::OptimizationType::Default);
}

/// @details Using Gaussian Process Regression (GPR) to depict phase space distribution. @n
/// First optimize elementwise, then optimize the diagonal with normalization and energy conservation
Optimization::Result Optimization::optimize(const AllPoints& density, const AllPoints& extra_points)
{
	// construct training sets
	const AllTrainingSets TrainingSets = construct_training_sets(density), ExtraTrainingSets = construct_training_sets(extra_points);
	// total energy
	const QuantumVector<double> Energies = calculate_total_energy_average_each_surface(density, mass);
	// set bounds for current bounds, and calculate energy on each surfaces
	const QuantumStorage<Bounds> ParameterBounds =
		[&density, &LocalMinimizers = LocalMinimizers]() -> QuantumStorage<Bounds>
	{
		QuantumStorage<Bounds> result;
		for (const std::size_t iPES : std::ranges::iota_view{0ul, NumPES})
		{
			for (const std::size_t jPES : std::ranges::iota_view{0ul, iPES + 1})
			{
				if (!density(iPES, jPES).empty())
				{
					const ClassicalPhaseVector StdDev = calculate_standard_deviation_one_surface(density(iPES, jPES));
					const ClassicalPhaseVector CharLengthLB = StdDev / std::sqrt(density(iPES, jPES).size()), CharLengthUB = 2.0 * StdDev;
					result(iPES, jPES) = iPES == jPES
						? calculate_kernel_bounds(CharLengthLB, CharLengthUB)
						: calculate_complex_kernel_bounds(CharLengthLB, CharLengthUB);
				}
				else
				{
					result(iPES, jPES)[0] = LocalMinimizers(iPES, jPES).get_lower_bounds();
					result(iPES, jPES)[1] = LocalMinimizers(iPES, jPES).get_upper_bounds();
				}
			}
		}
		return result;
	}();
	// check for the suitable noise
	set_optimizer_bounds(ParameterBounds, LocalMinimizers, DiagonalMinimizer, FullMinimizer, GlobalMinimizers);

	auto move_into_bounds = [&ParameterBounds](QuantumStorage<ParameterVector>& parameter_vectors)
	{
		for (const std::size_t iPES : std::ranges::iota_view{0ul, NumPES})
		{
			for (const std::size_t jPES : std::ranges::iota_view{0ul, iPES + 1})
			{
				const auto& [LowerBound, UpperBound] = ParameterBounds(iPES, jPES);
				for (const std::size_t iParam : std::ranges::iota_view{0ul, (iPES == jPES ? KernelBase::NumTotalParameters : ComplexKernelBase::NumTotalParameters)})
				{
					parameter_vectors(iPES, jPES)[iParam] = std::clamp(parameter_vectors(iPES, jPES)[iParam], LowerBound[iParam], UpperBound[iParam]);
				}
			}
		}
	};

	auto output_average = [&Energies](const TrainingKernels& kernels) -> void
	{
		for (const std::size_t iPES : std::ranges::iota_view{0ul, NumPES})
		{
			for (const std::size_t jPES : std::ranges::iota_view{0ul, iPES + 1})
			{
				if (iPES == jPES)
				{
					if (kernels(iPES).has_value())
					{
						spdlog::info(
							"{}On {}-th surface, population = {}, energy = {}, purity = {}",
							indents<2>::apply(),
							iPES,
							kernels(iPES)->get_population(),
							kernels(iPES)->get_population() * Energies[iPES],
							kernels(iPES)->get_purity()
						);
					}
				}
				else
				{
					if (kernels(iPES, jPES).has_value())
					{
						spdlog::info("{}For element ({}, {}), purity = {}", indents<2>::apply(), iPES, jPES, kernels(iPES, jPES)->get_purity());
					}
				}
			}
		}
	};

	// construct a function for code reuse
	auto do_optimize =
		[this,
		 &density,
		 &TrainingSets,
		 &ExtraTrainingSets,
		 &Energies,
		 &move_into_bounds,
		 &output_average](
			QuantumStorage<ParameterVector>& parameter_vectors,
			const OptimizationType OptType
		) -> Result
	{
		const bool OffDiagonalOptimization = std::any_of(
			density.get_offdiagonal_data().cbegin(),
			density.get_offdiagonal_data().cend(),
			[](const ElementPoints& points)
			{
				return !points.empty();
			}
		);
		// initially, set the magnitude to be 1
		// and move the parameters into bounds
		for (const std::size_t iPES : std::ranges::iota_view{0ul, NumPES})
		{
			for (const std::size_t jPES : std::ranges::iota_view{0ul, iPES + 1})
			{
				parameter_vectors(iPES, jPES)[0] = InitialMagnitude;
			}
		}
		move_into_bounds(parameter_vectors);
		Result result = optimize_elementwise(TrainingSets, ExtraTrainingSets, LocalMinimizers, parameter_vectors);
		output_average(TrainingKernels(parameter_vectors, TrainingSets, false, true, false));
		auto& [err, steps, type] = result;
		type = OptType;
		if (OffDiagonalOptimization)
		{
			// do not add purity condition for diagonal, so give a negative purity to indicate no necessity
			[[maybe_unused]] const auto& [DiagonalError, DiagonalStep, DiagonalOptType] = optimize_diagonal(
				TrainingSets,
				ExtraTrainingSets,
				Energies,
				TotalEnergy,
				NAN,
				DiagonalMinimizer,
				parameter_vectors
			);
			output_average(TrainingKernels(parameter_vectors, TrainingSets, false, true, false));
			[[maybe_unused]] const auto& [TotalError, TotalStep, TotalOptType] = optimize_full(
				TrainingSets,
				ExtraTrainingSets,
				Energies,
				TotalEnergy,
				Purity,
				FullMinimizer,
				parameter_vectors
			);
			output_average(TrainingKernels(parameter_vectors, TrainingSets, false, true, false));
			err = TotalError;
			steps.insert(steps.cend(), DiagonalStep.cbegin(), DiagonalStep.cend());
			steps.insert(steps.cend(), TotalStep.cbegin(), TotalStep.cend());
		}
		else
		{
			[[maybe_unused]] const auto& [DiagonalError, DiagonalStep, DiagonalType] = optimize_diagonal(
				TrainingSets,
				ExtraTrainingSets,
				Energies,
				TotalEnergy,
				Purity,
				DiagonalMinimizer,
				parameter_vectors
			);
			output_average(TrainingKernels(parameter_vectors, TrainingSets, false, true, false));
			err = DiagonalError;
			steps.insert(steps.cend(), DiagonalStep.cbegin(), DiagonalStep.cend());
			steps.push_back(0);
		}
		// afterwards, calculate the magnitude
		for (const std::size_t iPES : std::ranges::iota_view{0ul, NumPES})
		{
			for (const std::size_t jPES : std::ranges::iota_view{0ul, iPES + 1})
			{
				if (!density(iPES, jPES).empty())
				{
					parameter_vectors(iPES, jPES)[0] = iPES == jPES
						? TrainingKernel(parameter_vectors(iPES, jPES), TrainingSets(iPES, jPES), false, false, false).get_magnitude()
						: TrainingComplexKernel(parameter_vectors(iPES, jPES), TrainingSets(iPES, jPES), false, false, false).get_magnitude();
					spdlog::info("{}Magnitude of {} is {}.", indents<2>::apply(), get_element_name(iPES, jPES), parameter_vectors(iPES, jPES)[0]);
				}
				else
				{
					spdlog::info("{}Magnitude of {} is not needed.", indents<2>::apply(), get_element_name(iPES, jPES));
				}
			}
		}
		spdlog::info("{}Error = {}", indents<2>::apply(), err);
		return result;
	};

	auto check_averages =
		[&density,
		 &TrainingSets,
		 &Energies,
		 TotalEnergy = this->TotalEnergy,
		 Purity = this->Purity](
			const QuantumStorage<ParameterVector>& parameter_vectors
		) -> Eigen::Vector3d
	{
		const TrainingKernels AllKernels(parameter_vectors, TrainingSets, false, true, false);
		// to calculate the error that is above the tolerance or not
		// if within tolerance, return 0; otherwise, return relative error
		static auto beyond_tolerance_error = [](double calc, double ref) -> double
		{
			const double err = std::abs((calc / ref) - 1.0);
			if (err < AverageTolerance)
			{
				return 0.0;
			}
			else
			{
				return err;
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

		Eigen::Vector3d result = Eigen::Vector3d::Zero();
		// <r>
		for (const std::size_t iPES : std::ranges::iota_view{0ul, NumPES})
		{
			if (!density(iPES).empty())
			{
				const ClassicalPhaseVector r_ave = calculate_1st_order_average_one_surface(density(iPES));
				const ClassicalPhaseVector r_prm = calculate_1st_order_average_one_surface(AllKernels(iPES).value()); // prm = parameters
				spdlog::info(
					"{}On surface {}, Exact <r> = {}, Analytical integrated <r> = {}.",
					indents<3>::apply(),
					iPES,
					r_ave.format(VectorFormatter),
					r_prm.format(VectorFormatter)
				);
			}
		}
		// <1>
		const double ppl_prm = AllKernels.calculate_population();
		spdlog::info("{}Exact population = {}, analytical integrated population = {}.", indents<3>::apply(), 1.0, ppl_prm);
		result[0] += beyond_tolerance_error(ppl_prm, 1.0);
		check_and_output(result[0], "Population");
		// <E>
		const double eng_prm_ave = AllKernels.calculate_total_energy_average(Energies);
		spdlog::info("{}Exact energy = {}, analytical integrated energy = {}", indents<3>::apply(), TotalEnergy, eng_prm_ave);
		result[1] += beyond_tolerance_error(eng_prm_ave, TotalEnergy);
		check_and_output(result[1], "Energy");
		// purity
		const double prt_prm = AllKernels.calculate_purity();
		spdlog::info("{}Exact purity = {}, analytical integrated purity = {}.", indents<3>::apply(), Purity, prt_prm);
		result[2] += beyond_tolerance_error(prt_prm, Purity);
		check_and_output(result[2], "Purity");
		return result;
	};

	auto compare_and_overwrite =
		[&ParameterVectors = this->ParameterVectors](
			Result& result,
			Eigen::Vector3d& check_result,
			const std::string_view& old_name,
			const Result& result_new,
			const Eigen::Vector3d& check_new,
			const std::string_view& new_name,
			const QuantumStorage<ParameterVector>& param_vec_new
		) -> void
	{
		auto& [error, steps, type] = result;
		const auto& [error_new, steps_new, type_new] = result_new;
		// for each term, a better result means:
		// 1. The better one is smaller than the worse one
		// 2. The worse one is out of 2 * Tolerance
		const std::size_t BetterResults = (check_new.array() < check_result.array() && check_result.array() > 2.0 * AverageTolerance).count();
		const std::size_t WorseResults = (check_new.array() > check_result.array() && check_new.array() > 2.0 * AverageTolerance).count();
		if (BetterResults > WorseResults || (BetterResults == WorseResults && check_new.sum() < check_result.sum()))
		{
			spdlog::info("{}{} is better because of better averages.", indents<1>::apply(), new_name);
			ParameterVectors = param_vec_new;
			error = error_new;
			for (const std::size_t iElement : std::ranges::iota_view{0ul, steps_new.size()})
			{
				steps[iElement] += steps_new[iElement];
			}
			type = type_new;
			check_result = check_new;
		}
		else if (BetterResults == WorseResults && error_new < error)
		{
			spdlog::info("{}{} is better because of smaller error.", indents<1>::apply(), new_name);
			ParameterVectors = param_vec_new;
			error = error_new;
			for (const std::size_t iElement : std::ranges::iota_view{0ul, steps_new.size()})
			{
				steps[iElement] += steps_new[iElement];
			}
			type = type_new;
			check_result = check_new;
		}
		else
		{
			spdlog::info("{}{} is better.", indents<1>::apply(), old_name);
		}
	};

	// 1. optimize locally with parameters from last step
	spdlog::info("{}Local optimization with previous parameters.", indents<1>::apply());
	Result result = do_optimize(ParameterVectors, OptimizationType::LocalPrevious);
	Eigen::Vector3d check_result = check_averages(ParameterVectors);
	if ((check_result.array() == 0.0).all())
	{
		spdlog::info("{}Local optimization with previous parameters succeeded.", indents<1>::apply());
		return result;
	}
	// 2. optimize locally with initial parameter
	spdlog::warn("{}Local optimization with previous parameters failed. Retry with local optimization with initial parameters.", indents<1>::apply());
	QuantumStorage<ParameterVector> param_vec_initial(InitialKernelParameter, InitialComplexKernelParameter);
	const Result result_initial = do_optimize(param_vec_initial, OptimizationType::LocalInitial);
	const Eigen::Vector3d check_initial = check_averages(param_vec_initial);
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
	for (const std::size_t iPES : std::ranges::iota_view{0ul, NumPES})
	{
		for (const std::size_t jPES : std::ranges::iota_view{0ul, iPES + 1})
		{
			param_vec_global(iPES, jPES) = local_parameter_to_global(param_vec_global(iPES, jPES));
		}
	}
	[[maybe_unused]] const auto& [error_global_elm, steps_global_elm, opt_type_global_elm] =
		optimize_elementwise(TrainingSets, ExtraTrainingSets, GlobalMinimizers, param_vec_global);
	for (const std::size_t iPES : std::ranges::iota_view{0ul, NumPES})
	{
		for (const std::size_t jPES : std::ranges::iota_view{0ul, iPES + 1})
		{
			param_vec_global(iPES, jPES) = global_parameter_to_local(param_vec_global(iPES, jPES));
		}
	}
	output_average(TrainingKernels(param_vec_global, TrainingSets, false, true, false));
	Result result_global = do_optimize(param_vec_global, OptimizationType::Global);
	const Eigen::Vector3d check_global = check_averages(param_vec_global);
	for (const std::size_t iElement : std::ranges::iota_view{0ul, steps_global_elm.size()})
	{
		std::get<1>(result_global)[iElement] += steps_global_elm[iElement];
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
	for (const std::size_t iPES : std::ranges::iota_view{0ul, NumPES})
	{
		for (const std::size_t jPES : std::ranges::iota_view{0ul, iPES + 1})
		{
			result(iPES, jPES) = LocalMinimizers(iPES, jPES).get_lower_bounds();
		}
	}
	return result;
}

QuantumStorage<ParameterVector> Optimization::get_upper_bounds(void) const
{
	QuantumStorage<ParameterVector> result;
	for (const std::size_t iPES : std::ranges::iota_view{0ul, NumPES})
	{
		for (const std::size_t jPES : std::ranges::iota_view{0ul, iPES + 1})
		{
			result(iPES, jPES) = LocalMinimizers(iPES, jPES).get_upper_bounds();
		}
	}
	return result;
}
