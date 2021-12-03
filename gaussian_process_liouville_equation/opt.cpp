/// @file opt.cpp
/// @brief Implementation of opt.h

#include "stdafx.h"

#include "opt.h"

#include "input.h"
#include "kernel.h"
#include "mc.h"
#include "mc_ave.h"

/// The type for the loose function; whether it is LOOCV or LMLL
using LooseFunction = std::function<double(const Kernel&, ParameterVector&)>;
/// The training set for diagonal elements and population constraint,
/// which is an array of training set for each diagonal element
using DiagonalTrainingSet = std::array<ElementTrainingSet, NumPES>;
/// The energy for each PES and the total energy (last term in array)
using AllEnergy = std::array<double, NumPES + 1>;
/// The parameters used for energy conservation, including the diagonal training set
/// and the energy for each PES and the total energy (last term in array)
using EnergyConstraintParam = std::tuple<DiagonalTrainingSet, AllEnergy>;
/// The parameters used for purity conservation,
/// including the elementwise training set and the initial purity
template <typename TrainingSet>
using PurityConstraintParam = std::tuple<TrainingSet, double>;
/// The training set for off-diagonal elements.
/// Notice that the diagonal element is also included for easier element accessing
using OffDiagonalTrainingSet = std::array<ElementTrainingSet, NumElements>;

/// @brief To construct the training set for single element
/// @param[in] density The vector containing all known density matrix
/// @param[in] ElementIndex The index of the element in density matrix
/// @return The training set of the corresponding element in density matrix
ElementTrainingSet construct_element_training_set(const EigenVector<PhaseSpacePoint>& density, const int ElementIndex)
{
	assert(density.size() % NumElements == 0);
	const int NumPoints = density.size() / NumElements;
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

Optimization::Optimization(
	const Parameters& Params,
	const nlopt::algorithm LocalAlgorithm,
	const nlopt::algorithm ConstraintAlgorithm,
	const nlopt::algorithm GlobalAlgorithm,
	const nlopt::algorithm GlobalSubAlgorithm) :
	NumPoints(Params.get_number_of_selected_points()),
	InitialParameter(Kernel::set_initial_parameters(Params.get_sigma_x0(), Params.get_sigma_p0()))
{
	static const double InitialStepSize = 0.5; // Initial step size in optimization
	static const double Tolerance = 1e-5;	   // Relative tolerance
	// set up bounds
	const auto& [LowerBound, UpperBound] = Kernel::set_parameter_bounds(Params.get_xmax() - Params.get_xmin(), Params.get_pmax() - Params.get_pmin());

	// set up parameters and minimizers
	ParameterVector LowerBounds = LowerBound, UpperBounds = UpperBound; // bounds for setting. It is changed latter
	auto set_optimizer = [&](nlopt::opt& optimizer) -> void
	{
		optimizer.set_lower_bounds(LowerBounds);
		optimizer.set_upper_bounds(UpperBounds);
		optimizer.set_xtol_rel(Tolerance);
		optimizer.set_ftol_rel(Tolerance);
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
			optimizer.set_initial_step(InitialStepSize);
			break;
		default:
			break;
		}
	};
	// set up a local optimizer using given algorithm
	auto set_local_optimizer = [&](nlopt::opt& optimizer, const nlopt::algorithm Algorithm) -> void
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
	for (int iPES = 1; iPES < NumPES; iPES++)
	{
		LowerBounds.insert(LowerBounds.cend(), LowerBound.cbegin(), LowerBound.cend());
		UpperBounds.insert(UpperBounds.cend(), UpperBound.cbegin(), UpperBound.cend());
	}
	NLOptLocalMinimizers.push_back(nlopt::opt(nlopt::algorithm::AUGLAG_EQ, Kernel::NumTotalParameters * NumPES));
	set_optimizer(NLOptLocalMinimizers.back());
	set_local_optimizer(NLOptLocalMinimizers.back(), ConstraintAlgorithm);
	// minimizer for all off-diagonal elements with regularization (purity)
	// the dimension is set to be all elements, but the diagonal elements have no effect
	// this is for simpler code, while the memory cost is larger but acceptable
	if (NumPES != 1)
	{
		for (int iElement = NumPES; iElement < NumElements; iElement++)
		{
			LowerBounds.insert(LowerBounds.cend(), LowerBound.cbegin(), LowerBound.cend());
			UpperBounds.insert(UpperBounds.cend(), UpperBound.cbegin(), UpperBound.cend());
		}
		NLOptLocalMinimizers.push_back(nlopt::opt(nlopt::algorithm::AUGLAG_EQ, Kernel::NumTotalParameters * NumElements));
		set_optimizer(NLOptLocalMinimizers.back());
		set_local_optimizer(NLOptLocalMinimizers.back(), ConstraintAlgorithm);
	}
}

/// @brief The loose function of negative log marginal likelihood
/// @param[in] kernel The kernel to use
/// @param[out] grad The gradient at the given point
/// @return The \f$ -\mathrm{log}p(\mathbf{y}\|X,\theta) \f$
/// @details The result could be expanded as
///
/// \f[
/// -\mathrm{log}p(\mathbf{y}|X,\theta)=\frac{1}{2}\mathbf{y}^{\mathsf{T}}K^{-1}\mathbf{y}+\frac{1}{2}\mathrm{log}\|K\|+\frac{n}{2}\mathrm{log}(2\pi)
/// \f]
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
/// \f[
/// \mu_i=y_i-\frac{[K^{-1}\mathbf{y}]_i}{[K^{-1}]_{ii}}
/// \f]
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
			grad[iParam] = 2.0 * ((InvLbl * NegInvDerivInvLbl * InvDiag - InvLbl.abs2() * NegInvDerivInv) / InvDiag.pow(3)).sum();
		}
	}
	return (InvLbl / InvDiag).abs2().sum();
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
	const Kernel k(x, TrainingFeature, TrainingFeature, TrainingLabel, false, !grad.empty());
	double result = erf(k, grad);
	// if the value is abnormal (inf, nan, etc), set it to be the maximum double
	if (!std::isfinite(result))
	{
		result = std::numeric_limits<double>::max();
	}
	for (double& d : grad)
	{
		if (!std::isfinite(d))
		{
			d = std::numeric_limits<double>::max();
		}
	}
	return result;
}

/// @brief To optimize parameters of each density matrix element based on the given density
/// @param[in] TrainingSets The training features and labels of all elements
/// @param[in] IsSmall The matrix that saves whether each element is small or not
/// @param[inout] minimizers The minimizers to use
/// @param[inout] ParameterVectors The parameters for all elements of density matrix
/// @return The total error and the optimization steps of each element
static Optimization::Result optimize_elementwise(
	const QuantumArray<ElementTrainingSet>& TrainingSets,
	const QuantumMatrix<bool>& IsSmall,
	std::vector<nlopt::opt>& minimizers,
	std::array<ParameterVector, NumElements>& ParameterVectors)
{
	double total_error = 0.0;
	std::vector<int> num_steps(NumElements, 0);
	// optimize element by element
	for (int iElement = 0; iElement < NumElements; iElement++)
	{
		// first, check if all points are very small or not
		if (!IsSmall(iElement / NumPES, iElement % NumPES))
		{
			ElementTrainingSet ets = TrainingSets[iElement];
			minimizers[iElement].set_min_objective(loose_function, static_cast<void*>(&ets));
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
				spdlog::error("Optimization of rho[{}][{}] failed by {}", iElement / NumPES, iElement % NumPES, e.what());
			}
#endif
			total_error += err;
			num_steps[iElement] = minimizers[iElement].get_numevals();
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
	const auto& TrainingSets = *static_cast<DiagonalTrainingSet*>(params);
	double err = 0.0;
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		[[maybe_unused]] const auto& [feature, label] = TrainingSets[iPES];
		const int BeginIndex = iPES * Kernel::NumTotalParameters;
		const int EndIndex = BeginIndex + Kernel::NumTotalParameters;
		if (feature.size() != 0)
		{
			const ParameterVector x_element(x.cbegin() + BeginIndex, x.cbegin() + EndIndex);
			ParameterVector grad_element = (!grad.empty() ? ParameterVector(Kernel::NumTotalParameters) : ParameterVector());
			err += loose_function(x_element, grad_element, static_cast<void*>(const_cast<ElementTrainingSet*>(&TrainingSets[iPES])));
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
/// @return \f$ \langle\rho\rangle - 1 \f$
static double population_constraint(const ParameterVector& x, ParameterVector& grad, void* params)
{
	const auto& TrainingSets = *static_cast<DiagonalTrainingSet*>(params);
	double ppl = 0.0;
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		const auto& [feature, label] = TrainingSets[iPES];
		const int BeginIndex = iPES * Kernel::NumTotalParameters;
		const int EndIndex = BeginIndex + Kernel::NumTotalParameters;
		if (feature.size() != 0)
		{
			const ParameterVector x_element(x.cbegin() + BeginIndex, x.cbegin() + EndIndex);
			// construct the kernel
			const Kernel k(x_element, feature, feature, label, true, !grad.empty());
			ppl += k.get_population();
			if (!grad.empty())
			{
				const ParameterVector& PplGrad = k.get_population_derivative();
				std::copy(PplGrad.cbegin(), PplGrad.cend(), grad.begin() + BeginIndex);
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
	return ppl - 1.0;
};

/// @brief To calculate the error on energy conservation constraint
/// @param[in] x The input parameters, need to calculate the function value and gradient at this point
/// @param[out] grad The gradient at the given point
/// @param[in] params Other parameters. Here it is the diagonal training sets and energies
/// @return \f$ \langle E\rangle - E_0 \f$
static double energy_constraint(const ParameterVector& x, ParameterVector& grad, void* params)
{
	const auto& [TrainingSets, Energies] = *static_cast<EnergyConstraintParam*>(params);
	double eng = 0.0;
	double ppl = 0.0;
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		const auto& [feature, label] = TrainingSets[iPES];
		const int BeginIndex = iPES * Kernel::NumTotalParameters;
		const int EndIndex = BeginIndex + Kernel::NumTotalParameters;
		if (feature.size() != 0)
		{
			const ParameterVector x_element(x.cbegin() + BeginIndex, x.cbegin() + EndIndex);
			// construct the kernel
			const Kernel k(x_element, feature, feature, label, true, !grad.empty());
			eng += k.get_population() * Energies[iPES];
			ppl += k.get_population();
			if (!grad.empty())
			{
				const ParameterVector& PplGrad = k.get_population_derivative();
				for (std::size_t iParam = 0; iParam < Kernel::NumTotalParameters; iParam++)
				{
					grad[iParam + BeginIndex] = PplGrad[iParam] * Energies[iPES];
				}
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
	return eng / ppl - Energies[NumPES];
};

/// @brief To calculate the error on purity conservation constraint
/// @param[in] x The input parameters, need to calculate the function value and gradient at this point
/// @param[out] grad The gradient at the given point
/// @param[in] params Other parameters. Here it is the diagonal training sets and purity
/// @return \f$ \langle\rho^2\rangle - \langle\rho^2\rangle_{t=0} \f$
static double diagonal_purity_constraint(const ParameterVector& x, ParameterVector& grad, void* params)
{
	const auto& [TrainingSets, Purity] = *static_cast<PurityConstraintParam<DiagonalTrainingSet>*>(params);
	double prt = 0.0;
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		const auto& [feature, label] = TrainingSets[iPES];
		const int BeginIndex = iPES * Kernel::NumTotalParameters;
		const int EndIndex = BeginIndex + Kernel::NumTotalParameters;
		if (feature.size() != 0)
		{
			const ParameterVector x_element(x.cbegin() + BeginIndex, x.cbegin() + EndIndex);
			// construct the kernel
			const Kernel k(x_element, feature, feature, label, true, !grad.empty());
			prt += k.get_purity();
			if (!grad.empty())
			{
				const ParameterVector& PrtGrad = k.get_purity_derivative();
				std::copy(PrtGrad.cbegin(), PrtGrad.cend(), grad.begin() + BeginIndex);
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
	return prt - Purity;
};

/// @brief To optimize parameters of diagonal elements based on the given density and constraint regularizations
/// @param[in] TrainingSets The training features and labels of all elements
/// @param[in] Energies The energy of each surfaces and the total energy
/// @param[in] Purity The purity of the partial Wigner transformed density matrix
/// @param[in] IsSmall The matrix that saves whether each element is small or not
/// @param[inout] minimizer The minimizer to use
/// @param[inout] ParameterVectors The parameters for all elements of density matrix
/// @return The total error and the number of steps
static Optimization::Result optimize_diagonal(
	const QuantumArray<ElementTrainingSet>& TrainingSets,
	const AllEnergy& Energies,
	const double Purity,
	const QuantumMatrix<bool>& IsSmall,
	nlopt::opt& minimizer,
	std::array<ParameterVector, NumElements>& ParameterVectors)
{
	// construct training set
	DiagonalTrainingSet dts;
	ParameterVector DiagonalParameters;
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		const int ElementIndex = iPES * (NumPES + 1);
		if (!IsSmall(iPES, iPES))
		{
			dts[iPES] = TrainingSets[ElementIndex];
		}
		DiagonalParameters.insert(
			DiagonalParameters.cend(),
			ParameterVectors[ElementIndex].cbegin(),
			ParameterVectors[ElementIndex].cend());
	}
	minimizer.set_min_objective(diagonal_loose, static_cast<void*>(&dts));
	// set up constraints
	minimizer.add_equality_constraint(population_constraint, static_cast<void*>(&dts));
	EnergyConstraintParam ecp = std::make_tuple(dts, Energies);
	minimizer.add_equality_constraint(energy_constraint, static_cast<void*>(&ecp));
	PurityConstraintParam<DiagonalTrainingSet> pcp = std::make_tuple(dts, Purity); // definition must be out of if, or memory will be released
	if (Purity > 0)
	{
		minimizer.add_equality_constraint(diagonal_purity_constraint, static_cast<void*>(&pcp));
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
	// set up kernels
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		if (!IsSmall(iPES, iPES))
		{
			const int ElementIndex = iPES * (NumPES + 1);
			const int ParamBeginIndex = iPES * Kernel::NumTotalParameters;
			const int ParamEndIndex = ParamBeginIndex + Kernel::NumTotalParameters;
			ParameterVector(DiagonalParameters.cbegin() + ParamBeginIndex, DiagonalParameters.cbegin() + ParamEndIndex).swap(ParameterVectors[ElementIndex]);
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
	const auto& TrainingSets = *static_cast<OffDiagonalTrainingSet*>(params);
	for (int iElement = 0; iElement < NumElements; iElement++)
	{
		[[maybe_unused]] const auto& [feature, label] = TrainingSets[iElement];
		const int BeginIndex = iElement * Kernel::NumTotalParameters;
		const int EndIndex = BeginIndex + Kernel::NumTotalParameters;
		if (iElement / NumPES != iElement % NumPES && feature.size() != 0)
		{
			// for non-zero off-diagonal element, calculate its error and gradient if needed
			const ParameterVector x_element(x.cbegin() + BeginIndex, x.cbegin() + EndIndex);
			ParameterVector grad_element = (!grad.empty() ? ParameterVector(Kernel::NumTotalParameters) : ParameterVector());
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
/// @return \f$ \langle\rho^2\rangle - \langle\rho^2\rangle_{t=0} \f$
static double purity_constraint(const ParameterVector& x, ParameterVector& grad, void* params)
{
	const auto& [TrainingSets, Purity] = *static_cast<PurityConstraintParam<OffDiagonalTrainingSet>*>(params);
	double prt = 0.0;
	for (int iElement = 0; iElement < NumElements; iElement++)
	{
		const auto& [feature, label] = TrainingSets[iElement];
		const int iPES = iElement / NumPES, jPES = iElement % NumPES;
		const int BeginIndex = iElement * Kernel::NumTotalParameters;
		const int EndIndex = BeginIndex + Kernel::NumTotalParameters;
		if (feature.size() != 0)
		{
			const ParameterVector x_element(x.cbegin() + BeginIndex, x.cbegin() + EndIndex);
			if (iPES != jPES)
			{
				// for non-zero off-diagonal element, calculate its error and gradient if needed
				const Kernel k(x_element, feature, feature, label, true, !grad.empty());
				prt += 2.0 * k.get_purity();
				if (!grad.empty())
				{
					const ParameterVector& PrtGrad = k.get_purity_derivative();
					for (std::size_t iParam = 0; iParam < Kernel::NumTotalParameters; iParam++)
					{
						grad[iParam + BeginIndex] = PrtGrad[iParam] * 2.0;
					}
				}
			}
			else
			{
				// for non-zero diagonal element, calculate its purity, and set gradient 0 if needed
				const Kernel k(x_element, feature, feature, label, true);
				prt += k.get_purity();
				if (!grad.empty())
				{
					std::fill(grad.begin() + BeginIndex, grad.begin() + EndIndex, 0.0);
				}
			}
		}
		else if (!grad.empty())
		{
			// for any other case that gradient is needed, fill with 0
			std::fill(grad.begin() + BeginIndex, grad.begin() + EndIndex, 0.0);
		}
	}
	return prt - Purity;
};

/// @brief To optimize parameters of off-diagonal element based on the given density and purity
/// @param[in] TrainingSets The training features and labels of all elements
/// @param[in] Purity The purity of the partial Wigner transformed density matrix, which should conserve
/// @param[in] IsSmall The matrix that saves whether each element is small or not
/// @param[inout] minimizer The minimizer to use
/// @param[inout] ParameterVectors The parameters for all elements of density matrix
/// @return The total error (MSE, log likelihood, etc, including regularizations) of all off-diagonal elements of density matrix
Optimization::Result optimize_offdiagonal(
	const QuantumArray<ElementTrainingSet>& TrainingSets,
	const double Purity,
	const QuantumMatrix<bool>& IsSmall,
	nlopt::opt& minimizer,
	std::array<ParameterVector, NumElements>& ParameterVectors)
{
	// construct training set
	OffDiagonalTrainingSet odts = TrainingSets;
	ParameterVector Parameters;
	for (int iElement = 0; iElement < NumElements; iElement++)
	{
		Parameters.insert(
			Parameters.cend(),
			ParameterVectors[iElement].cbegin(),
			ParameterVectors[iElement].cend());
	}
	minimizer.set_min_objective(offdiagonal_loose, static_cast<void*>(&odts));
	// setup constraint
	PurityConstraintParam<OffDiagonalTrainingSet> pcp = std::make_tuple(odts, Purity);
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
	// set up kernels
	for (int iElement = 0; iElement < NumElements; iElement++)
	{
		const int iPES = iElement / NumPES, jPES = iElement % NumPES;
		if (iPES != jPES && !IsSmall(iPES, jPES))
		{
			const int BeginIndex = iElement * Kernel::NumTotalParameters;
			const int EndIndex = BeginIndex + Kernel::NumTotalParameters;
			ParameterVector(Parameters.cbegin() + BeginIndex, Parameters.cbegin() + EndIndex).swap(ParameterVectors[iElement]);
		}
	}
	return std::make_tuple(err, std::vector<int>(1, minimizer.get_numevals()));
}

/// @brief To determine whether the optimization result is satisfy or not
/// @param[in] Optimizer It Gives the parameters
/// @param[in] TrainingSets The training features and labels of all elements
/// @param[in] density The vector containing all known density matrix
/// @param[in] Energies The energy of each surfaces and the total energy
/// @param[in] Purity The purity of the partial Wigner transformed density matrix
/// @param[in] IsSmall The matrix that saves whether each element is small or not
/// @return Whether to redo the optimization or not
/// @details This functions compares the parameter average of <x> and <p> with
/// monte carlo average, and the total population, energy and purity. If any of
/// the things listed above deviates far, the function returns true.
static bool is_reoptimize(
	const Optimization& Optimizer,
	const QuantumArray<ElementTrainingSet>& TrainingSets,
	const EigenVector<PhaseSpacePoint>& density,
	const AllEnergy& Energies,
	const double Purity,
	const QuantumMatrix<bool>& IsSmall)
{
	static const double AbsoluteTolerance = 0.2;
	static const double RelativeTolerance = 0.1;
	static auto is_tolerant = [](const double calc, const double ref) -> bool
	{
		static const double AverageTolerance = 1e-2;
		return std::abs((calc / ref) - 1.0) < AverageTolerance;
	};
	// first order average
	double ppl = 0.0;
	double eng = 0.0;
	double prt = 0.0;
	for (int iElement = 0; iElement < NumElements; iElement++)
	{
		const int iPES = iElement / NumPES, jPES = iElement % NumPES;
		if (!IsSmall(iPES, jPES))
		{
			// construct a kernel to calculate averages
			const auto& [feature, label] = TrainingSets[iElement];
			const Kernel k(Optimizer.get_parameter(iElement), feature, feature, label, true);
			if (iPES == jPES)
			{
				const ClassicalPhaseVector r_mc = calculate_1st_order_average(density, iPES);
				const ClassicalPhaseVector& r_prm = k.get_1st_order_average();
				if (((r_prm - r_mc).array() / r_mc.array() > RelativeTolerance).any()
					&& ((r_prm - r_mc).array().abs() > AbsoluteTolerance).any())
				{
					return true;
				}
				ppl += k.get_population();
				eng += k.get_population() * Energies[iPES];
				prt += k.get_purity();
			}
			else
			{
				prt += 2.0 * k.get_purity();
			}
		}
	}
	// averages
	if (!is_tolerant(ppl, 1.0) || !is_tolerant(eng, Energies[NumPES]) || !is_tolerant(prt, Purity))
	{
		return true;
	}
	return false;
}

/// @details Using Gaussian Process Regression (GPR) to depict phase space distribution.
/// First optimize elementwise, then optimize the diagonal with normalization and energy conservation
Optimization::Result Optimization::optimize(
	const EigenVector<PhaseSpacePoint>& density,
	const ClassicalVector<double>& mass,
	const double TotalEnergy,
	const double Purity,
	const QuantumMatrix<bool>& IsSmall)
{
	// construct training sets
	const QuantumArray<ElementTrainingSet> TrainingSets = [&](void) -> QuantumArray<ElementTrainingSet>
	{
		QuantumArray<ElementTrainingSet> result;
		for (int iElement = 0; iElement < NumElements; iElement++)
		{
			result[iElement] = construct_element_training_set(density, iElement);
		}
		return result;
	}();
	// calculate energies
	const AllEnergy Energies = [&](void) -> AllEnergy
	{
		AllEnergy result = {0.0};
		for (int iPES = 0; iPES < NumPES; iPES++)
		{
			if (!IsSmall(iPES, iPES))
			{
				result[iPES] = calculate_total_energy_average(density, iPES, mass);
			}
		}
		result[NumPES] = TotalEnergy;
		return result;
	}();

	// any offdiagonal element is not small
	const bool OffDiagonalOptimization = [&](void) -> bool
	{
		for (int iElement = 0; iElement < NumElements; iElement++)
		{
			const int iPES = iElement / NumPES, jPES = iElement % NumPES;
			if (iPES != jPES && !IsSmall(iPES, jPES))
			{
				return true;
			}
		}
		return false;
	}();

	// construct a function for code reuse
	auto core_steps = [&](void) -> Optimization::Result
	{
		Optimization::Result result = optimize_elementwise(TrainingSets, IsSmall, NLOptLocalMinimizers, ParameterVectors);
		auto& [err, steps] = result;
		if (OffDiagonalOptimization)
		{
			// do not add purity condition for diagonal, so give a negative purity to indicate no necessity
			const auto& [DiagonalError, DiagonalStep] = optimize_diagonal(TrainingSets, Energies, -Purity, IsSmall, NLOptLocalMinimizers[NumElements], ParameterVectors);
			const auto& [OffDiagonalError, OffDiagonalStep] = optimize_offdiagonal(TrainingSets, Purity, IsSmall, NLOptLocalMinimizers[NumElements + 1], ParameterVectors);
			err = DiagonalError + OffDiagonalError;
			steps.insert(steps.cend(), DiagonalStep.cbegin(), DiagonalStep.cend());
			steps.insert(steps.cend(), OffDiagonalStep.cbegin(), OffDiagonalStep.cend());
		}
		else
		{
			const auto& [DiagonalError, DiagonalStep] = optimize_diagonal(TrainingSets, Energies, Purity, IsSmall, NLOptLocalMinimizers[NumElements], ParameterVectors);
			err = DiagonalError;
			steps.insert(steps.cend(), DiagonalStep.cbegin(), DiagonalStep.cend());
			steps.push_back(0);
		}
		return result;
	};

	std::tuple<double, std::vector<int>> result = core_steps();
	// next, check if the averages meet; otherwise, reopt with initial parameter
	if (is_reoptimize(*this, TrainingSets, density, Energies, Purity, IsSmall))
	{
		spdlog::warn("Local optimization with previous parameters failed. Retry with initial parameters.");
		for (int iElement = 0; iElement < NumElements; iElement++)
		{
			ParameterVectors[iElement] = InitialParameter;
		}
		result = core_steps();
	}
	else
	{
		spdlog::info("Local optimization with previous parameters succeeded.");
		return result;
	}
	// if the initial parameter does not work either, go to global optimizers
	if (is_reoptimize(*this, TrainingSets, density, Energies, Purity, IsSmall))
	{
		spdlog::warn("Local optimization with initial parameters failed. Retry with global optimization.");
		for (int iElement = 0; iElement < NumElements; iElement++)
		{
			ParameterVectors[iElement] = InitialParameter;
		}
		optimize_elementwise(TrainingSets, IsSmall, NLOptGlobalMinimizers, ParameterVectors);
		result = core_steps();
	}
	else
	{
		spdlog::info("Local optimization with initial parameters succeeded.");
		return result;
	}
	// after global optimization, check if it works or not
	if (is_reoptimize(*this, TrainingSets, density, Energies, Purity, IsSmall))
	{
		spdlog::warn("Global optimization failed.");
	}
	else
	{
		spdlog::info("Global optimization succeeded.");
	}
	return result;
}
