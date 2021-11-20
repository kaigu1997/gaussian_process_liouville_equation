/// @file opt.cpp
/// @brief Implementation of opt.h

#include "stdafx.h"

#include "opt.h"

#include "kernel.h"
#include "mc.h"
#include "pes.h"

/// The type for the loose function; whether it is LOOCV or LMLL
using LooseFunction = std::function<double(const Kernel&, ParameterVector&)>;
/// The training set for parameter optimization of one element of density matrix
/// passed as parameter to the optimization function.
/// First is feature, second is label
using ElementTrainingSet = std::tuple<Eigen::MatrixXd, Eigen::VectorXd>;
/// The training set for diagonal elements and population constraint,
/// which is an array of training set for each diagonal element
using DiagonalTrainingSet = std::array<ElementTrainingSet, NumDiagonalElements>;
/// The parameters used for energy conservation, including the diagonal training set
/// and the energy for each PES and the total energy (last term in array)
using EnergyConstraintParam = std::tuple<DiagonalTrainingSet, std::array<double, NumDiagonalElements + 1>>;
/// The parameters used for purity conservation,
/// including the elementwise training set and the initial purity
template <typename TrainingSet>
using PurityConstraintParam = std::tuple<TrainingSet, double>;
/// The training set for off-diagonal elements.
/// Notice that the diagonal element is also included for easier element accessing
using OffDiagonalTrainingSet = std::array<ElementTrainingSet, NumElements>;

const double Optimization::Tolerance = 1e-5;

Optimization::Optimization(
	const Parameters& Params,
	const nlopt::algorithm LocalAlgorithm,
	const nlopt::algorithm GlobalAlgorithm) :
	NumPoints(Params.get_number_of_selected_points()),
	InitialParameter(Kernel::set_initial_parameters(Params.get_sigma_x0(), Params.get_sigma_p0()))
{
	static const double InitialStepSize = 0.5; // Initial step size in optimization
	static const int PointsPerDimension = 10;  // The number of points to search per parameter
	// set up bounds
	const auto& [LowerBound, UpperBound] = Kernel::set_parameter_bounds(Params.get_xmax() - Params.get_xmin(), Params.get_pmax() - Params.get_pmin());

	// set up parameters and minimizers
	ParameterVector LowerBounds = LowerBound, UpperBounds = UpperBound;
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
			NLOptGlobalMinimizers.back().set_local_optimizer(NLOptLocalMinimizers.back());
			[[fallthrough]];
		case nlopt::algorithm::GN_CRS2_LM:
		case nlopt::algorithm::GN_ISRES:
			NLOptGlobalMinimizers.back().set_population(PointsPerDimension * Kernel::NumTotalParameters);
			break;
		default:
			break;
		}
	}
	// minimizer for all diagonal elements with regularization (population + energy + purity)
	auto set_local_optimizer = [&](nlopt::opt& optimizer, const nlopt::algorithm& Algorithm) -> void
	{
		nlopt::opt result(Algorithm, optimizer.get_dimension());
		set_optimizer(result);
		optimizer.set_local_optimizer(result);
	};
	for (int iPES = 1; iPES < NumDiagonalElements; iPES++)
	{
		LowerBounds.insert(LowerBounds.cend(), LowerBound.cbegin(), LowerBound.cend());
		UpperBounds.insert(UpperBounds.cend(), UpperBound.cbegin(), UpperBound.cend());
	}
	NLOptLocalMinimizers.push_back(nlopt::opt(nlopt::algorithm::AUGLAG_EQ, Kernel::NumTotalParameters * NumDiagonalElements));
	set_optimizer(NLOptLocalMinimizers.back());
	set_local_optimizer(NLOptLocalMinimizers.back(), LocalAlgorithm);
	// minimizer for all off-diagonal elements with regularization (purity)
	// the dimension is set to be all elements, but the diagonal elements have no effect
	// this is for simpler code, while the memory cost is small
	if (NumPES != 1)
	{
		for (int iElement = NumDiagonalElements; iElement < NumElements; iElement++)
		{
			LowerBounds.insert(LowerBounds.cend(), LowerBound.cbegin(), LowerBound.cend());
			UpperBounds.insert(UpperBounds.cend(), UpperBound.cbegin(), UpperBound.cend());
		}
		NLOptLocalMinimizers.push_back(nlopt::opt(nlopt::algorithm::AUGLAG_EQ, Kernel::NumTotalParameters * NumElements));
		set_optimizer(NLOptLocalMinimizers.back());
		set_local_optimizer(NLOptLocalMinimizers.back(), LocalAlgorithm);
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
	if (grad.empty() == false)
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
	if (grad.empty() == false)
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
	const Kernel k(x, TrainingFeature, TrainingFeature, TrainingLabel, false, grad.empty() == false);
	return erf(k, grad);
}

/// @brief To construct the training set for single element
/// @param[in] density The vector containing all known density matrix
/// @param[in] ElementIndex The index of the element in density matrix
/// @param[in] NumPoints The number of points selected for parameter optimization
/// @return The training set of the corresponding element in density matrix
static ElementTrainingSet construct_element_training_set(
	const EigenVector<PhaseSpacePoint>& density,
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
	const EigenVector<PhaseSpacePoint>& density,
	const QuantumMatrix<bool>& IsSmall,
	std::vector<nlopt::opt>& minimizers)
{
	static const LooseFunction& lf = leave_one_out_cross_validation;
	std::vector<int> result(NumElements, 0);
	// optimize element by element
	for (int iElement = 0; iElement < NumElements; iElement++)
	{
		// first, check if all points are very small or not
		if (IsSmall(iElement / NumPES, iElement % NumPES) == false)
		{
			ElementTrainingSet ets = construct_element_training_set(density, iElement, NumPoints);
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
				std::cerr << e.what() << " at " << print_time << std::endl;
			}
#endif
			result[iElement] = minimizers[iElement].get_numevals();
			const auto& [feature, label] = ets;
			TrainingFeatures[iElement] = feature;
			Kernels[iElement] = std::make_shared<const Kernel>(ParameterVectors[iElement], feature, feature, label, true);
		}
	}
	return result;
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
			ParameterVector grad_element = (grad.empty() == false ? ParameterVector(Kernel::NumTotalParameters) : ParameterVector());
			err += loose_function(x_element, grad_element, static_cast<void*>(const_cast<ElementTrainingSet*>(&TrainingSets[iPES])));
			if (grad.empty() == false)
			{
				std::copy(grad_element.cbegin(), grad_element.cend(), grad.begin() + BeginIndex);
			}
		}
		else
		{
			// element is small while gradient is needed, set gradient 0
			if (grad.empty() == false)
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
			const Kernel k(x_element, feature, feature, label, true, grad.empty() == false);
			ppl += k.get_population();
			if (grad.empty() == false)
			{
				const ParameterVector& PplGrad = k.get_population_derivative();
				std::copy(PplGrad.cbegin(), PplGrad.cend(), grad.begin() + BeginIndex);
			}
		}
		else
		{
			// element is small while gradient is needed, set gradient 0
			if (grad.empty() == false)
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
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		const auto& [feature, label] = TrainingSets[iPES];
		const int BeginIndex = iPES * Kernel::NumTotalParameters;
		const int EndIndex = BeginIndex + Kernel::NumTotalParameters;
		if (feature.size() != 0)
		{
			const ParameterVector x_element(x.cbegin() + BeginIndex, x.cbegin() + EndIndex);
			// construct the kernel
			const Kernel k(x_element, feature, feature, label, true, grad.empty() == false);
			eng += k.get_population() * Energies[iPES];
			if (grad.empty() == false)
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
			if (grad.empty() == false)
			{
				std::fill(grad.begin() + BeginIndex, grad.begin() + EndIndex, 0.0);
			}
		}
	}
	return eng - Energies[NumPES];
};

/// @brief To calculate the error on purity conservation constraint
/// @param[in] x The input parameters, need to calculate the function value and gradient at this point
/// @param[out] grad The gradient at the given point
/// @param[in] params Other parameters. Here it is the diagonal training sets and energies
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
			const Kernel k(x_element, feature, feature, label, true, grad.empty() == false);
			prt += k.get_purity();
			if (grad.empty() == false)
			{
				const ParameterVector& PrtGrad = k.get_purity_derivative();
				std::copy(PrtGrad.cbegin(), PrtGrad.cend(), grad.begin() + BeginIndex);
			}
		}
		else
		{
			// element is small while gradient is needed, set gradient 0
			if (grad.empty() == false)
			{
				std::fill(grad.begin() + BeginIndex, grad.begin() + EndIndex, 0.0);
			}
		}
	}
	return prt - Purity;
};

std::tuple<double, int> Optimization::optimize_diagonal(
	const EigenVector<PhaseSpacePoint>& density,
	const ClassicalVector<double>& mass,
	const double TotalEnergy,
	const double Purity,
	const QuantumMatrix<bool>& IsSmall,
	nlopt::opt& minimizer)
{
	// construct training set
	DiagonalTrainingSet dts;
	ParameterVector DiagonalParameters;
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		const int ElementIndex = iPES * (NumPES + 1);
		if (IsSmall(iPES, iPES) == false)
		{
			dts[iPES] = construct_element_training_set(density, ElementIndex, NumPoints);
		}
		DiagonalParameters.insert(
			DiagonalParameters.cend(),
			ParameterVectors[ElementIndex].cbegin(),
			ParameterVectors[ElementIndex].cend());
	}
	minimizer.set_min_objective(diagonal_loose, static_cast<void*>(&dts));
	// set up constraints
	minimizer.add_equality_constraint(population_constraint, static_cast<void*>(&dts), Tolerance);
	EnergyConstraintParam ecp;
	auto& [dts_ecp, eng] = ecp;
	dts_ecp = dts;
	eng[NumPES] = TotalEnergy;
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		if (IsSmall(iPES, iPES) == false)
		{
			eng[iPES] = calculate_kinetic_average(density, iPES, mass) + calculate_potential_average(density, iPES);
		}
		else
		{
			eng[iPES] = 0.0;
		}
	}
	minimizer.add_equality_constraint(energy_constraint, static_cast<void*>(&ecp), TotalEnergy * Tolerance);
	PurityConstraintParam<DiagonalTrainingSet> pcp; // definition must be out of if, or memory will be released
	if (Purity > 0)
	{
		auto& [dts_pcp, prt] = pcp;
		dts_pcp = dts;
		prt = Purity;
		minimizer.add_equality_constraint(diagonal_purity_constraint, static_cast<void*>(&pcp), Purity * Tolerance);
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
		std::cerr << e.what() << " at " << print_time << std::endl;
	}
#endif
	minimizer.remove_equality_constraints();
	// set up kernels
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		if (IsSmall(iPES, iPES) == false)
		{
			const auto& [feature, label] = dts[iPES];
			const int ElementIndex = iPES * (NumPES + 1);
			const int ParamBeginIndex = iPES * Kernel::NumTotalParameters;
			const int ParamEndIndex = ParamBeginIndex + Kernel::NumTotalParameters;
			ParameterVector(DiagonalParameters.cbegin() + ParamBeginIndex, DiagonalParameters.cbegin() + ParamEndIndex).swap(ParameterVectors[ElementIndex]);
			Kernels[ElementIndex] = std::make_shared<const Kernel>(ParameterVectors[ElementIndex], feature, feature, label, true);
		}
	}
	return std::make_tuple(err, minimizer.get_numevals());
}

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
			ParameterVector grad_element = (grad.empty() == false ? ParameterVector(Kernel::NumTotalParameters) : ParameterVector());
			err += loose_function(x_element, grad_element, static_cast<void*>(const_cast<ElementTrainingSet*>(&TrainingSets[iElement])));
			if (grad.empty() == false)
			{
				std::copy(grad_element.cbegin(), grad_element.cend(), grad.begin() + BeginIndex);
			}
		}
		else if (grad.empty() == false)
		{
			// for any other case that gradient is needed, fill with 0
			std::fill(grad.begin() + BeginIndex, grad.begin() + EndIndex, 0.0);
		}
	}
	return err;
};

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
				const Kernel k(x_element, feature, feature, label, true, grad.empty() == false);
				prt += 2.0 * k.get_purity();
				if (grad.empty() == false)
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
				if (grad.empty() == false)
				{
					std::fill(grad.begin() + BeginIndex, grad.begin() + EndIndex, 0.0);
				}
			}
		}
		else if (grad.empty() == false)
		{
			// for any other case that gradient is needed, fill with 0
			std::fill(grad.begin() + BeginIndex, grad.begin() + EndIndex, 0.0);
		}
	}
	return prt - Purity;
};

std::tuple<double, int> Optimization::optimize_offdiagonal(
	const EigenVector<PhaseSpacePoint>& density,
	const double Purity,
	const QuantumMatrix<bool>& IsSmall,
	nlopt::opt& minimizer)
{
	// construct training set
	OffDiagonalTrainingSet odts;
	ParameterVector Parameters;
	for (int iElement = 0, iOffDiagElement = 0; iElement < NumElements; iElement++)
	{
		if (IsSmall(iElement / NumPES, iElement % NumPES) == false)
		{
			odts[iElement] = construct_element_training_set(density, iElement, NumPoints);
		}
		Parameters.insert(
			Parameters.cend(),
			ParameterVectors[iElement].cbegin(),
			ParameterVectors[iElement].cend());
		iOffDiagElement++;
	}
	minimizer.set_min_objective(offdiagonal_loose, static_cast<void*>(&odts));
	// setup constraint
	PurityConstraintParam<OffDiagonalTrainingSet> pcp;
	auto& [ts, prt] = pcp;
	ts = odts;
	prt = Purity;
	minimizer.add_equality_constraint(purity_constraint, &pcp, Purity * Tolerance);
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
		std::cerr << e.what() << " at " << print_time << std::endl;
	}
#endif
	minimizer.remove_equality_constraints();
	// set up kernels
	for (int iElement = 0; iElement < NumElements; iElement++)
	{
		const int iPES = iElement / NumPES, jPES = iElement % NumPES;
		if (iPES != jPES && IsSmall(iPES, jPES) == false)
		{
			const auto& [feature, label] = odts[iElement];
			const int BeginIndex = iElement * Kernel::NumTotalParameters;
			const int EndIndex = BeginIndex + Kernel::NumTotalParameters;
			ParameterVector(Parameters.cbegin() + BeginIndex, Parameters.cbegin() + EndIndex).swap(ParameterVectors[iElement]);
			Kernels[iElement] = std::make_shared<const Kernel>(ParameterVectors[iElement], feature, feature, label, true);
		}
	}
	return std::make_tuple(err, minimizer.get_numevals());
}

/// @details This function calculates the average position and momentum of each surfaces
/// by parameters and by MC average and compares them to judge the reoptimization.
bool Optimization::is_reoptimize(const EigenVector<PhaseSpacePoint>& density, const QuantumMatrix<bool>& IsSmall)
{
	static const double AbsoluteTolerance = 0.2;
	static const double RelativeTolerance = 0.1;
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		if (IsSmall(iPES, iPES) == false)
		{
			const int ElementIndex = iPES * (NumPES + 1);
			const ClassicalPhaseVector r_mc = ::calculate_1st_order_average(density, iPES);
			const auto& r_prm = Kernels[ElementIndex]->get_first_order_average() / Kernels[ElementIndex]->get_population();
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
	const EigenVector<PhaseSpacePoint>& density,
	const ClassicalVector<double>& mass,
	const double TotalEnergy,
	const double Purity,
	const QuantumMatrix<bool>& IsSmall)
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
		steps = optimize_elementwise(density, IsSmall, NLOptLocalMinimizers);
		if (OffDiagonalOptimization == true)
		{
			// do not add purity condition for diagonal, so give a negative purity to indicate no necessity
			const auto& [DiagonalError, DiagonalStep] = optimize_diagonal(density, mass, TotalEnergy, -Purity, IsSmall, NLOptLocalMinimizers[NumElements]);
			const auto& [OffDiagonalError, OffDiagonalStep] = optimize_offdiagonal(density, Purity, IsSmall, NLOptLocalMinimizers[NumElements + 1]);
			err = DiagonalError + OffDiagonalError;
			steps.push_back(DiagonalStep);
			steps.push_back(OffDiagonalStep);
		}
		else
		{
			const auto& [DiagonalError, DiagonalStep] = optimize_diagonal(density, mass, TotalEnergy, Purity, IsSmall, NLOptLocalMinimizers[NumElements]);
			err = DiagonalError;
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
double Optimization::normalize(EigenVector<PhaseSpacePoint>& density, const QuantumMatrix<bool>& IsSmall) const
{
	double population = 0.0;
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		if (IsSmall(iPES, iPES) == false)
		{
			const int ElementIndex = iPES * (NumPES + 1);
			population += Kernels[ElementIndex]->get_population();
		}
	}
	for (auto& [x, p, rho] : density)
	{
		rho /= population;
	}
	return population;
}

void Optimization::update_training_set(const EigenVector<PhaseSpacePoint>& density, const QuantumMatrix<bool>& IsSmall)
{
	for (int iElement = 0; iElement < NumElements; iElement++)
	{
		// first, check if all points are very small or not
		if (IsSmall(iElement / NumPES, iElement % NumPES) == false)
		{
			const int BeginIndex = iElement * NumPoints;
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
			Kernels[iElement] = std::make_shared<const Kernel>(ParameterVectors[iElement], feature, feature, label, true);
		}
	}
}

/// @details Using Gaussian Process Regression to predict by
///
/// \f[
/// E[p(\mathbf{f}_*|X_*,X,\mathbf{y})]=K(X_*,X)[K(X,X)+\sigma_n^2I]^{-1}\mathbf{y}
/// \f]
///
/// where \f$ X_* \f$ is the test features, \f$ X \f$ is the training features,
/// and \f$ \mathbf{y} \f$ is the training labels.
///
/// Warning: must call optimize() first before callings of this function!
double Optimization::predict_element(
	const QuantumMatrix<bool>& IsSmall,
	const ClassicalVector<double>& x,
	const ClassicalVector<double>& p,
	const int ElementIndex) const
{
	if (IsSmall(ElementIndex / NumPES, ElementIndex % NumPES) == false)
	{
		// not small, have something to do
		Eigen::MatrixXd test_feat(PhaseDim, 1);
		test_feat << x, p;
		Kernel k(ParameterVectors[ElementIndex], test_feat, TrainingFeatures[ElementIndex]);
		return (k.get_kernel() * Kernels[ElementIndex]->get_inverse_label()).value();
	}
	else
	{
		return 0.;
	}
}

QuantumMatrix<std::complex<double>> Optimization::predict_matrix(
	const QuantumMatrix<bool>& IsSmall,
	const ClassicalVector<double>& x,
	const ClassicalVector<double>& p) const
{
	using namespace std::literals::complex_literals;
	assert(x.size() == p.size());
	QuantumMatrix<std::complex<double>> result = QuantumMatrix<std::complex<double>>::Zero();
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
	const QuantumMatrix<bool>& IsSmall,
	const EigenVector<ClassicalVector<double>>& x,
	const EigenVector<ClassicalVector<double>>& p,
	const int ElementIndex) const
{
	assert(x.size() == p.size());
	const int NumPoints = x.size();
	if (IsSmall(ElementIndex / NumPES, ElementIndex % NumPES) == false)
	{
		// not small, have something to do
		Eigen::MatrixXd test_feat(PhaseDim, NumPoints);
		for (int i = 0; i < NumPoints; i++)
		{
			test_feat.col(i) << x[i], p[i];
		}
		Kernel k(ParameterVectors[ElementIndex], test_feat, TrainingFeatures[ElementIndex]);
		return k.get_kernel() * Kernels[ElementIndex]->get_inverse_label();
	}
	else
	{
		return Eigen::VectorXd::Zero(NumPoints);
	}
}

Eigen::VectorXd Optimization::print_element(const QuantumMatrix<bool>& IsSmall, const Eigen::MatrixXd& PhaseGrids, const int ElementIndex) const
{
	if (IsSmall(ElementIndex / NumPES, ElementIndex % NumPES) == false)
	{
		Kernel k(ParameterVectors[ElementIndex], PhaseGrids, TrainingFeatures[ElementIndex]);
		return k.get_kernel() * Kernels[ElementIndex]->get_inverse_label();
	}
	else
	{
		return Eigen::VectorXd::Zero(PhaseGrids.cols());
	}
}

/// @details The global purity is the weighted sum of purity of all elements, with weight 1 for diagonal and 2 for off-diagonal
double Optimization::calculate_purity(const QuantumMatrix<bool>& IsSmall) const
{
	double result = 0.0;
	for (int iElement = 0; iElement < NumElements; iElement++)
	{
		if (IsSmall(iElement / NumPES, iElement % NumPES) == false)
		{
			if (iElement / NumPES == iElement % NumPES)
			{
				result += Kernels[iElement]->get_purity();
			}
			else
			{
				result += 2.0 * Kernels[iElement]->get_purity();
			}
		}
	}
	return result;
}

ClassicalPhaseVector calculate_1st_order_average(const EigenVector<PhaseSpacePoint>& density, const int PESIndex)
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

double calculate_kinetic_average(const EigenVector<PhaseSpacePoint>& density, const int PESIndex, const ClassicalVector<double>& mass)
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

double calculate_potential_average(const EigenVector<PhaseSpacePoint>& density, const int PESIndex)
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
