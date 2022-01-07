/// @file opt.cpp
/// @brief Implementation of opt.h

#include "stdafx.h"

#include "opt.h"

#include "input.h"
#include "kernel.h"
#include "mc_predict.h"

/// The lower and upper bound
using Bounds = std::array<ParameterVector, 2>;
/// The type for the loose function; whether it is LOOCV or LMLL
using LooseFunction = std::function<double(const Kernel&, ParameterVector&)>;
/// The training sets of all elements
using AllTrainingSets = QuantumArray<ElementTrainingSet>;
/// The parameters used for population constraint, including the training sets
/// and the selected points for monte carlo integration
using PopulationConstraintParameter = std::tuple<AllTrainingSets, AllPoints>;
/// The parameters used for energy conservation, including the training sets,
/// the selected points for monte carlo integration,
/// the total energy and the mass of classical degree of freedom
using EnergyConstraintParameter = std::tuple<AllTrainingSets, AllPoints, double, ClassicalVector<double>>;
/// The parameters used for purity conservation, including the training sets,
/// the selected points for monte carlo integration and the exact initial purity
using PurityConstraintParameter = std::tuple<AllTrainingSets, AllPoints, double>;

/// @brief To get the element of density matrix
/// @param[in] DensityMatrix The density matrix
/// @param[in] iPES The index of the row of the density matrix element
/// @param[in] jPES The index of the column of the density matrix element
/// @return The element of density matrix
/// @details If the index corresponds to upper triangular elements, gives real part.
///
/// If the index corresponds to lower triangular elements, gives imaginary part.
///
/// If the index corresponds to diagonal elements, gives the original value.
static inline double get_density_matrix_element(const QuantumMatrix<std::complex<double>>& DensityMatrix, const int iPES, const int jPES)
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
	auto construct_element_training_set = [&density](const int iPES, const int jPES) -> ElementTrainingSet
	{
		const EigenVector<PhaseSpacePoint>& ElementDensity = density[iPES * NumPES + jPES];
		const int NumPoints = ElementDensity.size();
		// construct training feature (PhaseDim*N) and training labels (N*1)
		Eigen::MatrixXd feature = Eigen::MatrixXd::Zero(PhaseDim, NumPoints);
		Eigen::VectorXd label = Eigen::VectorXd::Zero(NumPoints);
		const std::vector<int> indices = get_indices(NumPoints);
		std::for_each(
			std::execution::par_unseq,
			indices.cbegin(),
			indices.cend(),
			[&ElementDensity, &feature, &label, iPES, jPES](int iPoint) -> void
			{
				const auto& [x, p, rho] = ElementDensity[iPoint];
				feature.col(iPoint) << x, p;
				label[iPoint] = get_density_matrix_element(rho, iPES, jPES);
			});
		return std::make_tuple(feature, label);
	};
	AllTrainingSets result;
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		for (int jPES = 0; jPES < NumPES; jPES++)
		{
			const int ElementIndex = iPES * NumPES + jPES;
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
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		for (int jPES = 0; jPES < NumPES; jPES++)
		{
			const int ElementIndex = iPES * NumPES + jPES;
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
	for (int iElement = 0; iElement < NumElements; iElement++)
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

static const double InitialNoise = 1.0;			 ///< The initial weight for noise
static const double InitialGaussianWeight = 1e2; ///< The initial weight for Gaussian kernel

/// @brief To set up the values for the initial parameter
/// @param[in] xSigma The variance of position
/// @param[in] pSigma The variance of momentum
/// @return Vector containing all parameters for one element
static ParameterVector set_initial_parameters(const ClassicalVector<double>& xSigma, const ClassicalVector<double>& pSigma)
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

/// @brief To set up the bounds for the parameter
/// @param[in] xSize The size of the box on x direction
/// @param[in] pSize The size of the box on p direction
/// @return Lower and upper bounds
static Bounds set_parameter_bounds(const ClassicalVector<double>& xSize, const ClassicalVector<double>& pSize)
{
	static const double GaussKerMinCharLength = 1.0 / 100.0; // Minimal characteristic length for Gaussian kernel
	Bounds result;
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

/// @brief To set up the bounds for the parameter
/// @param[in] density The selected points in phase space of one element of density matrices
/// @return Lower and upper bounds
static Bounds set_parameter_bounds(const EigenVector<PhaseSpacePoint>& density)
{
	assert(!density.empty());
	Bounds result;
	auto& [LowerBound, UpperBound] = result;
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
	// lower bound of characteristic length, which is half the minimum distance between points
	const int NumPoints = density.size();
	ClassicalPhaseVector min_dist = ClassicalPhaseVector::Ones() * std::numeric_limits<double>::max();
	for (int iPoint = 0; iPoint < NumPoints; iPoint++)
	{
		[[maybe_unused]] const auto& [x_i, p_i, rho_i] = density[iPoint];
		for (int jPoint = iPoint + 1; jPoint < NumPoints; jPoint++)
		{
			[[maybe_unused]] const auto& [x_j, p_j, rho_j] = density[jPoint];
			ClassicalPhaseVector diff;
			diff << (x_i - x_j).array().abs(), (p_i - p_j).array().abs();
			min_dist = min_dist.array().min(diff.array());
		}
	}
	min_dist /= 2.0;
	LowerBound.insert(LowerBound.cend(), min_dist.data(), min_dist.data() + PhaseDim);
	// upper bound of characteristic length, which is 10 (-5~+5) sigma
	const ClassicalPhaseVector stddev_10times = 10.0 * calculate_standard_deviation(density);
	UpperBound.insert(UpperBound.cend(), stddev_10times.data(), stddev_10times.data() + PhaseDim);
	return result;
}

/// @brief To set up bounds for optimizers
/// @param[inout] optimizers The local/global optimizers
/// @param[in] LowerBound The lower bounds for each element
/// @param[in] UpperBound The upper bounds for each element
/// @details The number of global optimizers must be exactly NumElements.
/// The number of local optimizers must be exactly 2 more than NumElements.
void set_optimizer_bounds(std::vector<nlopt::opt>& optimizers, const QuantumArray<Bounds>& bounds)
{
	assert(optimizers.size() == NumElements || optimizers.size() == NumElements + 2);
	const bool ExtraOptimizer = (optimizers.size() == NumElements + 2);
	ParameterVector diagonal_lower_bound, total_lower_bound, diagonal_upper_bound, total_upper_bound;
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		for (int jPES = 0; jPES < NumPES; jPES++)
		{
			const int ElementIndex = iPES * NumPES + jPES;
			const auto& [LowerBound, UpperBound] = bounds[ElementIndex];
			assert(LowerBound.size() == Kernel::NumTotalParameters && UpperBound.size() == Kernel::NumTotalParameters);
			optimizers[ElementIndex].set_lower_bounds(LowerBound);
			optimizers[ElementIndex].set_upper_bounds(UpperBound);
			if (ExtraOptimizer)
			{
				total_lower_bound.insert(total_lower_bound.cend(), LowerBound.cbegin(), LowerBound.cend());
				total_upper_bound.insert(total_upper_bound.cend(), UpperBound.cbegin(), UpperBound.cend());
				if (iPES == jPES)
				{
					diagonal_lower_bound.insert(diagonal_lower_bound.cend(), LowerBound.cbegin(), LowerBound.cend());
					diagonal_upper_bound.insert(diagonal_upper_bound.cend(), UpperBound.cbegin(), UpperBound.cend());
				}
			}
		}
	}
	if (ExtraOptimizer)
	{
		optimizers[NumElements].set_lower_bounds(diagonal_lower_bound);
		optimizers[NumElements].set_upper_bounds(diagonal_upper_bound);
		optimizers[NumElements + 1].set_lower_bounds(total_lower_bound);
		optimizers[NumElements + 1].set_upper_bounds(total_upper_bound);
	}
}

Optimization::Optimization(
	const Parameters& Params,
	const double InitialTotalEnergy,
	const double InitialPurity,
	const nlopt::algorithm LocalAlgorithm,
	const nlopt::algorithm ConstraintAlgorithm,
	const nlopt::algorithm GlobalAlgorithm,
	const nlopt::algorithm GlobalSubAlgorithm) :
	TotalEnergy(InitialTotalEnergy),
	Purity(InitialPurity),
	mass(Params.get_mass()),
	InitialParameter(set_initial_parameters(Params.get_sigma_x0(), Params.get_sigma_p0()))
{
	static const double InitStepSize = 0.5; // Initial step size in optimization
	static const double RelTol = 1e-5;		// Relative tolerance
	static const double AbsTol = 1e-15;		// Relative tolerance
	static const int MaxEval = 10000;		// Maximum evaluations for global optimizer

	// set up parameters and minimizers
	auto set_optimizer = [](nlopt::opt& optimizer) -> void
	{
		optimizer.set_xtol_rel(RelTol);
		optimizer.set_ftol_rel(RelTol);
		optimizer.set_xtol_abs(AbsTol);
		optimizer.set_ftol_abs(AbsTol);
		switch (optimizer.get_algorithm())
		{
		case nlopt::algorithm::LN_PRAXIS:
		case nlopt::algorithm::LN_COBYLA:
		case nlopt::algorithm::LN_NEWUOA:
		case nlopt::algorithm::LN_NEWUOA_BOUND:
		case nlopt::algorithm::LN_NELDERMEAD:
		case nlopt::algorithm::LN_SBPLX:
		case nlopt::algorithm::LN_AUGLAG:
		case nlopt::algorithm::LN_AUGLAG_EQ:
		case nlopt::algorithm::LN_BOBYQA:
			optimizer.set_initial_step(InitStepSize);
			break;
		default:
			break;
		}
	};
	// set up a local optimizer using given algorithm
	auto set_local_optimizer = [&set_optimizer](nlopt::opt& optimizer, const nlopt::algorithm Algorithm) -> void
	{
		nlopt::opt result(Algorithm, optimizer.get_dimension());
		set_optimizer(result);
		optimizer.set_local_optimizer(result);
	};

	// minimizer for each element
	for (int iElement = 0; iElement < NumElements; iElement++)
	{
		ParameterVectors[iElement] = InitialParameter;
		NLOptLocalMinimizers.push_back(nlopt::opt(LocalAlgorithm, Kernel::NumTotalParameters));
		set_optimizer(NLOptLocalMinimizers.back());
		NLOptGlobalMinimizers.push_back(nlopt::opt(GlobalAlgorithm, Kernel::NumTotalParameters));
		set_optimizer(NLOptGlobalMinimizers.back());
		NLOptGlobalMinimizers.back().set_maxeval(MaxEval);
		switch (NLOptGlobalMinimizers.back().get_algorithm())
		{
		case nlopt::algorithm::G_MLSL:
		case nlopt::algorithm::GN_MLSL:
		case nlopt::algorithm::GD_MLSL:
		case nlopt::algorithm::G_MLSL_LDS:
		case nlopt::algorithm::GN_MLSL_LDS:
		case nlopt::algorithm::GD_MLSL_LDS:
			set_local_optimizer(NLOptGlobalMinimizers.back(), GlobalSubAlgorithm);
			break;
		default:
			break;
		}
	}
	// minimizer for all diagonal elements with regularization (population + energy + purity)
	NLOptLocalMinimizers.push_back(nlopt::opt(nlopt::algorithm::AUGLAG_EQ, Kernel::NumTotalParameters * NumPES));
	set_optimizer(NLOptLocalMinimizers.back());
	set_local_optimizer(NLOptLocalMinimizers.back(), ConstraintAlgorithm);
	// minimizer for all off-diagonal elements with regularization (purity)
	// the dimension is set to be all elements, but the diagonal elements have no effect
	// this is for simpler code, while the memory cost is larger but acceptable
	if (NumPES != 1)
	{
		NLOptLocalMinimizers.push_back(nlopt::opt(nlopt::algorithm::AUGLAG_EQ, Kernel::NumTotalParameters * NumElements));
		set_optimizer(NLOptLocalMinimizers.back());
		set_local_optimizer(NLOptLocalMinimizers.back(), ConstraintAlgorithm);
	}

	// set up bounds
	QuantumArray<Bounds> bounds;
	bounds.fill(set_parameter_bounds(Params.get_xmax() - Params.get_xmin(), Params.get_pmax() - Params.get_pmin()));
	set_optimizer_bounds(NLOptLocalMinimizers, bounds);
	set_optimizer_bounds(NLOptGlobalMinimizers, bounds);
}

/// @brief The loose function of negative log marginal likelihood
/// @param[in] kernel The kernel to use
/// @param[out] grad The gradient at the given point
/// @return The @f$ -\mathrm{log}p(\mathbf{y}\|X,\theta) @f$
/// @details The result could be expanded as
///
/// @f[
/// -\mathrm{log}p(\mathbf{y}|X,\theta)=\frac{1}{2}\mathbf{y}^{\mathsf{T}}K^{-1}\mathbf{y}+\frac{1}{2}\mathrm{log}\|K\|+\frac{n}{2}\mathrm{log}(2\pi)
/// @f]
///
/// where the last term is a constant for a certain optimization, n is the size of the training set.
[[maybe_unused]] static double negative_log_marginal_likelihood(const Kernel& kernel, ParameterVector& grad)
{
	const Eigen::VectorXd& Label = kernel.get_label();
	const Eigen::VectorXd& InvLbl = kernel.get_inverse_label();
	if (!grad.empty())
	{
		const auto& NegInvDerivInvLbl = kernel.get_negative_inv_deriv_inv_lbl();
		const auto& InvDeriv = kernel.get_inv_deriv();
		for (std::size_t iParam = 0; iParam < Kernel::NumTotalParameters; iParam++)
		{
			grad[iParam] = (Label.dot(NegInvDerivInvLbl[iParam]) + InvDeriv[iParam].trace()) / 2.0;
		}
	}
	return (Label.dot(InvLbl) + kernel.get_log_determinant()) / 2.0;
}

/// @brief The loose function of the leave one out cross validation squared error
/// @param[in] kernel The kernel to use
/// @param[out] grad The gradient at the given point
/// @return The squared error of LOOCV
/// @details For each training set point, leave one out cross validation predicts
///
/// @f[
/// \mu_i=y_i-\frac{[K^{-1}\mathbf{y}]_i}{[K^{-1}]_{ii}}
/// @f]
///
/// and the summation over all possible i gives the squared error
[[maybe_unused]] static double leave_one_out_cross_validation(const Kernel& kernel, ParameterVector& grad)
{
	const auto& InvLbl = kernel.get_inverse_label().array();
	const auto& InvDiag = kernel.get_inverse().diagonal().array();
	if (!grad.empty())
	{
		const auto& NegInvDerivInvVec = kernel.get_negative_inv_deriv_inv();
		const auto& NegInvDerivInvLblVec = kernel.get_negative_inv_deriv_inv_lbl();
		for (std::size_t iParam = 0; iParam < Kernel::NumTotalParameters; iParam++)
		{
			const auto& NegInvDerivInv = NegInvDerivInvVec[iParam].diagonal().array();
			const auto& NegInvDerivInvLbl = NegInvDerivInvLblVec[iParam].array();
			grad[iParam] = 2.0 * ((InvLbl * NegInvDerivInvLbl * InvDiag - InvLbl.square() * NegInvDerivInv) / InvDiag.pow(3)).sum();
		}
	}
	return (InvLbl / InvDiag).square().sum();
}

/// @brief If the value is abnormal (inf, nan), set it to be the extreme
/// @param[inout] d The value to judge
static void make_normal(double& d)
{
	int fpc = std::fpclassify(d);
	switch (fpc)
	{
	case FP_NAN:
	case FP_INFINITE:
		d = std::numeric_limits<double>::max();
		break;
	case FP_SUBNORMAL:
		d = std::numeric_limits<double>::min();
		break;
	default: // FP_NORMAL and FP_ZERO
		break;
	}
}

/// @brief The function for nlopt optimizer to minimize
/// @param[in] x The input parameters, need to calculate the function value and gradient at this point
/// @param[out] grad The gradient at the given point.
/// @param[in] params Other parameters. Here it is the training set
/// @return The error
static double loose_function(const ParameterVector& x, ParameterVector& grad, void* params)
{
	static const LooseFunction& erf = leave_one_out_cross_validation; // the error function to use
	// unpack the parameters
	const auto& TrainingSet = *static_cast<ElementTrainingSet*>(params);
	const auto& [TrainingFeature, TrainingLabel] = TrainingSet;
	// construct the kernel, then pass it to the error function
	const Kernel k(x, TrainingFeature, TrainingLabel, false, !grad.empty());
	double result = erf(k, grad);
	make_normal(result);
	for (double& d : grad)
	{
		make_normal(d);
	}
	return result;
}

/// @brief To optimize parameters of each density matrix element based on the given density
/// @param[in] TrainingSets The training features and labels of all elements
/// @param[inout] minimizers The minimizers to use
/// @param[inout] ParameterVectors The parameters for all elements of density matrix
/// @return The total error and the optimization steps of each element
static Optimization::Result optimize_elementwise(
	const AllTrainingSets& TrainingSets,
	std::vector<nlopt::opt>& minimizers,
	QuantumArray<ParameterVector>& ParameterVectors)
{
	double total_error = 0.0;
	std::vector<int> num_steps(NumElements, 0);
	// optimize element by element
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		for (int jPES = 0; jPES < NumPES; jPES++)
		{
			const int ElementIndex = iPES * NumPES + jPES;
			ElementTrainingSet ets = TrainingSets[ElementIndex];
			if (std::get<0>(ets).size() != 0)
			{
				minimizers[ElementIndex].set_min_objective(loose_function, static_cast<void*>(&ets));
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
					spdlog::error("Optimization of rho[{}][{}] failed by {}", iPES, jPES, e.what());
				}
#endif
				total_error += err;
				num_steps[ElementIndex] = minimizers[ElementIndex].get_numevals();
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
	const auto& TrainingSets = *static_cast<AllTrainingSets*>(params);
	double err = 0.0;
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		const int ElementIndex = iPES * NumPES + iPES;
		[[maybe_unused]] const auto& [feature, label] = TrainingSets[ElementIndex];
		const int BeginIndex = iPES * Kernel::NumTotalParameters;
		const int EndIndex = BeginIndex + Kernel::NumTotalParameters;
		if (feature.size() != 0)
		{
			const ParameterVector x_element(x.cbegin() + BeginIndex, x.cbegin() + EndIndex);
			ParameterVector grad_element(!grad.empty() ? Kernel::NumTotalParameters : 0);
			err += loose_function(x_element, grad_element, static_cast<void*>(const_cast<ElementTrainingSet*>(&TrainingSets[ElementIndex])));
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
	return err;
};

/// @brief To calculate the error on population constraint
/// @param[in] x The input parameters, need to calculate the function value and gradient at this point
/// @param[out] grad The gradient at the given point
/// @param[in] params Other parameters. Here it is the diagonal training sets
/// @return @f$ \langle\rho\rangle - 1 @f$
static double population_constraint(const ParameterVector& x, ParameterVector& grad, void* params)
{
	const auto& [TrainingSets, mc_points] = *static_cast<PopulationConstraintParameter*>(params);
	// construct predictors
	QuantumArray<ParameterVector> all_params;
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		all_params[iPES * NumPES + iPES] = ParameterVector(
			x.cbegin() + iPES * Kernel::NumTotalParameters,
			x.cbegin() + (iPES + 1) * Kernel::NumTotalParameters);
	}
	const OptionalKernels Kernels = construct_predictors(all_params, TrainingSets, false, !grad.empty());
	// calculate population of each surfaces
	double result = calculate_population(Kernels, mc_points) - 1.0;
	// calculate_derivatives
	if (!grad.empty())
	{
		grad = population_derivative(Kernels, mc_points);
	}
	make_normal(result);
	for (double& d : grad)
	{
		make_normal(d);
	}
	return result;
};

/// @brief To calculate the error on energy conservation constraint
/// @param[in] x The input parameters, need to calculate the function value and gradient at this point
/// @param[out] grad The gradient at the given point
/// @param[in] params Other parameters. Here it is the diagonal training sets and energies
/// @return @f$ \langle E\rangle - E_0 @f$
static double energy_constraint(const ParameterVector& x, ParameterVector& grad, void* params)
{
	const auto& [TrainingSets, mc_points, TotalEnergy, mass] = *static_cast<EnergyConstraintParameter*>(params);
	// construct predictors
	QuantumArray<ParameterVector> all_params;
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		all_params[iPES * NumPES + iPES] = ParameterVector(
			x.cbegin() + iPES * Kernel::NumTotalParameters,
			x.cbegin() + (iPES + 1) * Kernel::NumTotalParameters);
	}
	const OptionalKernels Kernels = construct_predictors(all_params, TrainingSets, false, !grad.empty());
	// calculate population of each surfaces
	double result = calculate_total_energy_average(Kernels, mc_points, mass) - TotalEnergy;
	// calculate_derivatives
	if (!grad.empty())
	{
		grad = total_energy_derivative(Kernels, mc_points, mass);
	}
	make_normal(result);
	for (double& d : grad)
	{
		make_normal(d);
	}
	return result;
};

/// @brief To calculate the error on purity conservation constraint
/// @param[in] x The input parameters, need to calculate the function value and gradient at this point
/// @param[out] grad The gradient at the given point
/// @param[in] params Other parameters. Here it is the diagonal training sets and purity
/// @return @f$ \langle\rho^2\rangle - \langle\rho^2\rangle_{t=0} @f$
static double diagonal_purity_constraint(const ParameterVector& x, ParameterVector& grad, void* params)
{
	const auto& [TrainingSets, mc_points, Purity] = *static_cast<PurityConstraintParameter*>(params);
	// construct predictors
	QuantumArray<ParameterVector> all_params;
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		all_params[iPES * NumPES + iPES] = ParameterVector(
			x.cbegin() + iPES * Kernel::NumTotalParameters,
			x.cbegin() + (iPES + 1) * Kernel::NumTotalParameters);
	}
	const OptionalKernels Kernels = construct_predictors(all_params, TrainingSets, false, !grad.empty());
	// calculate population of each surfaces
	double result = calculate_purity(Kernels, mc_points) - Purity;
	// calculate_derivatives
	if (!grad.empty())
	{
		ParameterVector all_deriv = purity_derivative(Kernels, mc_points);
		for (int iPES = 0; iPES < NumPES; iPES++)
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
};

/// @brief To optimize parameters of diagonal elements based on the given density and constraint regularizations
/// @param[in] TrainingSets The training features and labels of all elements
/// @param[in] mc_points The selected points for calculating mc integration
/// @param[in] TotalEnergy The total energy of the partial Wigner transformed density matrix
/// @param[in] Purity The purity of the partial Wigner transformed density matrix
/// @param[in] mass Mass of classical degree of freedom
/// @param[inout] minimizer The minimizer to use
/// @param[inout] ParameterVectors The parameters for all elements of density matrix
/// @return The total error and the number of steps
static Optimization::Result optimize_diagonal(
	const AllTrainingSets& TrainingSets,
	const AllPoints& mc_points,
	const double TotalEnergy,
	const double Purity,
	const ClassicalVector<double>& mass,
	nlopt::opt& minimizer,
	QuantumArray<ParameterVector>& ParameterVectors)
{
	// construct training set and parameters
	AllTrainingSets ats = TrainingSets;
	ParameterVector DiagonalParameters;
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		for (int jPES = 0; jPES < NumPES; jPES++)
		{
			const int ElementIndex = iPES * NumPES + jPES;
			if (iPES == jPES)
			{
				// diagonal elements, save parameters
				DiagonalParameters.insert(
					DiagonalParameters.cend(),
					ParameterVectors[ElementIndex].cbegin(),
					ParameterVectors[ElementIndex].cend());
			}
			else
			{
				// off-diagonal elements, remove training set
				auto& [feature, label] = ats[ElementIndex];
				feature = Eigen::MatrixXd(0, 0);
				label = Eigen::VectorXd(0);
			}
		}
	}
	minimizer.set_min_objective(diagonal_loose, static_cast<void*>(&ats));
	// set up constraints
	PopulationConstraintParameter pplcp = std::make_tuple(ats, mc_points);
	minimizer.add_equality_constraint(population_constraint, static_cast<void*>(&pplcp));
	EnergyConstraintParameter ergcp = std::make_tuple(ats, mc_points, TotalEnergy, mass);
	minimizer.add_equality_constraint(energy_constraint, static_cast<void*>(&ergcp));
	PurityConstraintParameter prtcp = std::make_tuple(ats, mc_points, Purity); // definition must be out of if, or memory will be released
	if (Purity > 0)
	{
		minimizer.add_equality_constraint(diagonal_purity_constraint, static_cast<void*>(&prtcp));
	}
	// optimize
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
		spdlog::error("Diagonal elements optimization failed by {}", e.what());
	}
#endif
	minimizer.remove_equality_constraints();
	// set up parameters
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		const int ElementIndex = iPES * NumPES + iPES;
		if (!mc_points[ElementIndex].empty())
		{
			ParameterVectors[ElementIndex] = ParameterVector(
				DiagonalParameters.cbegin() + iPES * Kernel::NumTotalParameters,
				DiagonalParameters.cbegin() + (iPES + 1) * Kernel::NumTotalParameters);
		}
	}
	return std::make_tuple(err, std::vector<int>(1, minimizer.get_numevals()));
}

/// @brief To calculate the overall loose for off-diagonal elements
/// @param[in] x The input parameters, need to calculate the function value and gradient at this point
/// @param[out] grad The gradient at the given point
/// @param[in] params Other parameters. Here it is the off-diagonal training sets
/// @return The off-diagonal errors
static double offdiagonal_loose(const ParameterVector& x, ParameterVector& grad, void* params)
{
	double err = 0.0;
	// get the parameters
	const auto& TrainingSets = *static_cast<AllTrainingSets*>(params);
	for (int iElement = 0; iElement < NumElements; iElement++)
	{
		[[maybe_unused]] const auto& [feature, label] = TrainingSets[iElement];
		const int BeginIndex = iElement * Kernel::NumTotalParameters;
		const int EndIndex = BeginIndex + Kernel::NumTotalParameters;
		if (iElement / NumPES != iElement % NumPES && feature.size() != 0)
		{
			// for non-zero off-diagonal element, calculate its error and gradient if needed
			const ParameterVector x_element(x.cbegin() + BeginIndex, x.cbegin() + EndIndex);
			ParameterVector grad_element = ParameterVector(!grad.empty() ? Kernel::NumTotalParameters : 0);
			err += loose_function(x_element, grad_element, static_cast<void*>(const_cast<ElementTrainingSet*>(&TrainingSets[iElement])));
			if (!grad.empty())
			{
				std::copy(grad_element.cbegin(), grad_element.cend(), grad.begin() + BeginIndex);
			}
		}
		else if (!grad.empty())
		{
			// for any other case that gradient is needed, fill with 0
			std::fill(grad.begin() + BeginIndex, grad.begin() + EndIndex, 0.0);
		}
	}
	return err;
};

/// @brief To calculate the error on purity conservation constraint
/// @param[in] x The input parameters, need to calculate the function value and gradient at this point
/// @param[out] grad The gradient at the given point
/// @param[in] params Other parameters. Here it is the off-diagonal training sets and purity
/// @return @f$ \langle\rho^2\rangle - \langle\rho^2\rangle_{t=0} @f$
static double purity_constraint(const ParameterVector& x, ParameterVector& grad, void* params)
{
	const auto& [TrainingSets, mc_points, Purity] = *static_cast<PurityConstraintParameter*>(params);
	// construct kernels
	QuantumArray<ParameterVector> all_params;
	for (int iElement = 0; iElement < NumElements; iElement++)
	{
		all_params[iElement] = ParameterVector(
			x.cbegin() + iElement * Kernel::NumTotalParameters,
			x.cbegin() + (iElement + 1) * Kernel::NumTotalParameters);
	}
	const OptionalKernels Kernels = construct_predictors(all_params, TrainingSets, false, !grad.empty());
	// calculate population of each surfaces
	double result = calculate_purity(Kernels, mc_points) - Purity;
	// calculate_derivatives
	if (!grad.empty())
	{
		grad = purity_derivative(Kernels, mc_points);
	}
	make_normal(result);
	for (double& d : grad)
	{
		make_normal(d);
	}
	return result;
};

/// @brief To optimize parameters of off-diagonal element based on the given density and purity
/// @param[in] TrainingSets The training features and labels of all elements
/// @param[in] mc_points The selected points for calculating mc integration
/// @param[in] Purity The purity of the partial Wigner transformed density matrix, which should conserve
/// @param[inout] minimizer The minimizer to use
/// @param[inout] ParameterVectors The parameters for all elements of density matrix
/// @return The total error (MSE, log likelihood, etc, including regularizations) of all off-diagonal elements of density matrix
Optimization::Result optimize_offdiagonal(
	const AllTrainingSets& TrainingSets,
	const AllPoints& mc_points,
	const double Purity,
	nlopt::opt& minimizer,
	QuantumArray<ParameterVector>& ParameterVectors)
{
	// construct training set
	AllTrainingSets ats = TrainingSets;
	ParameterVector Parameters;
	for (int iElement = 0; iElement < NumElements; iElement++)
	{
		Parameters.insert(
			Parameters.cend(),
			ParameterVectors[iElement].cbegin(),
			ParameterVectors[iElement].cend());
	}
	minimizer.set_min_objective(offdiagonal_loose, static_cast<void*>(&ats));
	// setup constraint
	PurityConstraintParameter pcp = std::make_tuple(ats, mc_points, Purity);
	minimizer.add_equality_constraint(purity_constraint, &pcp);
	// optimize
	double err = 0.0;
	try
	{
		minimizer.optimize(Parameters, err);
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
	for (int iElement = 0; iElement < NumElements; iElement++)
	{
		if (iElement / NumElements != iElement % NumElements && !mc_points[iElement].empty())
		{
			ParameterVectors[iElement] = ParameterVector(
				Parameters.cbegin() + iElement * Kernel::NumTotalParameters,
				Parameters.cbegin() + (iElement + 1) * Kernel::NumTotalParameters);
		}
	}
	return std::make_tuple(err, std::vector<int>(1, minimizer.get_numevals()));
}

/// @brief To determine whether the optimization result is satisfy or not
/// @param[in] Kernels The kernels of all elements
/// @param[in] density The vector containing all known density matrix
/// @param[in] mc_points The selected points for calculating mc integration
/// @param[in] Energies The energy of each surfaces and the total energy
/// @param[in] Purity The purity of the partial Wigner transformed density matrix
/// @param[in] mass Mass of classical degree of freedom
/// @return Whether to redo the optimization or not
/// @details This functions compares the parameter average of <x> and <p> with
/// monte carlo average, and the total population, energy and purity. If any of
/// the things listed above deviates far, the function returns true.
static bool is_reoptimize(
	const OptionalKernels& Kernels,
	const AllPoints& density,
	const AllPoints& mc_points,
	const double TotalEnergy,
	const double Purity,
	const ClassicalVector<double>& mass)
{
	static const double AbsoluteTolerance = 0.2;
	static const double RelativeTolerance = 0.1;
	static auto is_tolerant = [](const double calc, const double ref) -> bool
	{
		static const double AverageTolerance = 2e-2;
		return std::abs((calc / ref) - 1.0) < AverageTolerance;
	};
	// first order average
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		const int ElementIndex = iPES * NumPES + iPES;
		if (Kernels[ElementIndex].has_value() && !density[ElementIndex].empty() && !mc_points[ElementIndex].empty())
		{
			const ClassicalPhaseVector r_ave = calculate_1st_order_average(density[ElementIndex]);
			const ClassicalPhaseVector r_mci = calculate_1st_order_average_one_surface(Kernels[ElementIndex].value(), mc_points); // mci = Monte Carlo Integral
			if (((r_mci - r_ave).array() / r_ave.array() > RelativeTolerance).any()
				&& ((r_mci - r_ave).array().abs() > AbsoluteTolerance).any())
			{
				return true;
			}
		}
	}
	// other averages
	if (!is_tolerant(calculate_population(Kernels, mc_points), 1.0)
		|| !is_tolerant(calculate_total_energy_average(Kernels, mc_points, mass), TotalEnergy)
		|| !is_tolerant(calculate_purity(Kernels, mc_points), Purity))
	{
		return true;
	}
	return false;
}

/// @details Using Gaussian Process Regression (GPR) to depict phase space distribution.
/// First optimize elementwise, then optimize the diagonal with normalization and energy conservation
Optimization::Result Optimization::optimize(
	const AllPoints& density,
	const AllPoints& mc_points)
{
	// construct training sets
	const AllTrainingSets TrainingSets = construct_training_sets(density);
	// set bounds for current bounds
	QuantumArray<Bounds> param_bounds;
	for (int iElement = 0; iElement < NumElements; iElement++)
	{
		if (!density[iElement].empty())
		{
			param_bounds[iElement] = set_parameter_bounds(density[iElement]);
		}
		else
		{
			param_bounds[iElement][0] = NLOptLocalMinimizers[iElement].get_lower_bounds();
			param_bounds[iElement][1] = NLOptLocalMinimizers[iElement].get_upper_bounds();
		}
	}
	set_optimizer_bounds(NLOptLocalMinimizers, param_bounds);
	set_optimizer_bounds(NLOptGlobalMinimizers, param_bounds);
	// any offdiagonal element is not small
	const bool OffDiagonalOptimization = [&density](void) -> bool
	{
		for (int iPES = 1; iPES < NumPES; iPES++)
		{
			for (int jPES = 0; jPES < iPES; jPES++)
			{
				if (!density[iPES * NumPES + jPES].empty())
				{
					return true;
				}
			}
		}
		return false;
	}();

	// construct a function for code reuse
	auto core_steps = [this, &TrainingSets, &mc_points, OffDiagonalOptimization](QuantumArray<ParameterVector>& parameter_vectors) -> Optimization::Result
	{
		Optimization::Result result = optimize_elementwise(TrainingSets, NLOptLocalMinimizers, parameter_vectors);
		auto& [err, steps] = result;
		if (OffDiagonalOptimization)
		{
			// do not add purity condition for diagonal, so give a negative purity to indicate no necessity
			const auto& [DiagonalError, DiagonalStep] = optimize_diagonal(
				TrainingSets,
				mc_points,
				TotalEnergy,
				-Purity,
				mass,
				NLOptLocalMinimizers[NumElements],
				parameter_vectors);
			const auto& [OffDiagonalError, OffDiagonalStep] = optimize_offdiagonal(
				TrainingSets,
				mc_points,
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
				mc_points,
				TotalEnergy,
				Purity,
				mass,
				NLOptLocalMinimizers[NumElements],
				parameter_vectors);
			err = DiagonalError;
			steps.insert(steps.cend(), DiagonalStep.cbegin(), DiagonalStep.cend());
			steps.push_back(0);
		}
		return result;
	};

	Optimization::Result result = core_steps(ParameterVectors);
	// next, check if the averages meet; otherwise, reopt with initial parameter
	if (is_reoptimize(construct_predictors(ParameterVectors, TrainingSets, false, false), density, mc_points, TotalEnergy, Purity, mass))
	{
		spdlog::warn("Local optimization with previous parameters failed. Retry with initial parameters.");
		QuantumArray<ParameterVector> param_vec;
		for (int iElement = 0; iElement < NumElements; iElement++)
		{
			param_vec[iElement] = InitialParameter;
		}
		Optimization::Result result_new = core_steps(param_vec);
		if (std::get<0>(result_new) < std::get<0>(result))
		{
			spdlog::info("Local optimization with initial parameters is better.");
			ParameterVectors = param_vec;
			result = result_new;
		}
		else
		{
			spdlog::info("Local optimization with previous parameters is better.");
		}
	}
	else
	{
		spdlog::info("Local optimization with previous parameters succeeded.");
		return result;
	}
	// if the initial parameter does not work either, go to global optimizers
	if (is_reoptimize(construct_predictors(ParameterVectors, TrainingSets, false, false), density, mc_points, TotalEnergy, Purity, mass))
	{
		spdlog::warn("Local optimization with initial parameters failed. Retry with global optimization.");
		QuantumArray<ParameterVector> param_vec;
		for (int iElement = 0; iElement < NumElements; iElement++)
		{
			param_vec[iElement] = InitialParameter;
		}
		optimize_elementwise(TrainingSets, NLOptGlobalMinimizers, param_vec);
		Optimization::Result result_new = core_steps(param_vec);
		if (std::get<0>(result_new) < std::get<0>(result))
		{
			spdlog::info("Local optimization is better.");
			ParameterVectors = param_vec;
			result = result_new;
		}
		else
		{
			spdlog::info("Global optimization is better.");
		}
	}
	else
	{
		spdlog::info("Local optimization with initial parameters succeeded.");
		return result;
	}
	// after global optimization, check if it works or not
	if (is_reoptimize(construct_predictors(ParameterVectors, TrainingSets, false, false), density, mc_points, TotalEnergy, Purity, mass))
	{
		spdlog::warn("Global optimization failed.");
	}
	else
	{
		spdlog::info("Global optimization succeeded.");
	}
	return result;
}