/// @file opt.cpp
/// @brief Implementation of opt.h

#include "stdafx.h"

#include "opt.h"

#include "mc.h"
#include "pes.h"

/// The smart pointer to feature matrix
using FeaturePointer = std::shared_ptr<shogun::Features>;
/// The training set for hyperparameter optimization of one element of density matrix
/// passed as parameter to the optimization function.
/// First is feature, second is label, last is the kernels to use
using ElementTrainingSet = std::tuple<Eigen::MatrixXd, Eigen::VectorXd, KernelTypeList>;
/// The training set for optimization of all diagonal elements
/// altogether with regularization of population and energy.
/// First is separate training set for each diagonal elements, then the energy for each PES and the total energy (last term)
using DiagonalTrainingSet = std::tuple<std::array<ElementTrainingSet, NumPES>, std::array<double, NumPES + 1>>;

/// @brief Calculate the overall number of hyperparameters the optimization will use
/// @param[in] TypesOfKernels The vector containing all the kernel type used in optimization
/// @return The overall number of hyperparameters (including the magnitude of each kernel)
static int number_of_overall_hyperparameters(const KernelTypeList& TypesOfKernels)
{
	int sum = 0;
	for (shogun::EKernelType type : TypesOfKernels)
	{
		sum++; // weight
		switch (type)
		{
		case shogun::EKernelType::K_DIAG:
			break; // no extra hyperparameter for diagonal kernel
		case shogun::EKernelType::K_GAUSSIANARD:
			sum += PhaseDim;
			break;
		default:
			std::cerr << "UNKNOWN KERNEL!\n";
			break;
		}
	}
	return sum;
}

/// @brief Construct kernels from hyperparameters and their types, with features of training set
/// @param[in] TypeOfKernel The vector containing type of all kernels that will be used
/// @param[in] Hyperparameters The hyperparameters of all kernels (magnitude and other)
/// @return A vector of all kernels, with parameters set, but without any feature
static KernelList generate_kernels(const KernelTypeList& TypesOfKernels, const ParameterVector& Hyperparameters)
{
	KernelList result;
	int iParam = 0;
	for (shogun::EKernelType type : TypesOfKernels)
	{
		const double weight = Hyperparameters[iParam++];
		switch (type)
		{
		case shogun::EKernelType::K_DIAG:
			result.push_back(std::make_pair(weight, std::make_shared<shogun::DiagKernel>()));
			break;
		case shogun::EKernelType::K_GAUSSIANARD:
		{
			std::shared_ptr<shogun::GaussianARDKernel> gauss_ard_kernel_ptr = std::make_shared<shogun::GaussianARDKernel>();
			Eigen::VectorXd characteristic(PhaseDim);
			for (int j = 0; j < PhaseDim; j++)
			{
				characteristic[j] = Hyperparameters[iParam++];
			}
			gauss_ard_kernel_ptr->set_vector_weights(characteristic);
			result.push_back(std::make_pair(weight, gauss_ard_kernel_ptr));
			break;
		}
		default:
			break;
		}
	}
	return result;
}

/// @brief Calculate the kernel matrix, \f$ K(X_1,X_2) \f$
/// @param[in] LeftFeature The feature on the left, \f$ X_1 \f$
/// @param[in] RightFeature The feature on the right, \f$ X_2 \f$
/// @param[inout] Kernels The vector containing all kernels that will be used, with parameters set and features not set
/// @param[in] IsTraining Whether the features are all training set or not
/// @return The kernel matrix
/// @details This function calculates the kernel matrix with noise using the SHOGUN library,
///
/// \f[
/// k(\mathbf{x}_1,\mathbf{x}_2)=\sigma_f^2\mathrm{exp}\left(-\frac{(\mathbf{x}_1-\mathbf{x}_2)^\top M (\mathbf{x}_1-\mathbf{x}_2)}{2}\right)+\sigma_n^2\delta(\mathbf{x}_1-\mathbf{x}_2)
/// \f]
///
/// where \f$ M \f$ is the characteristic matrix in parameter list.
/// Regarded as a real-symmetric matrix - only lower-triangular part are used，
/// its diagonal term is the characteristic length of each dimention and
/// off-diagonal term is the correlation between different dimensions
///
/// When there are more than one feature, the kernel matrix follows \f$ K_{ij}=k(X_{1_{\cdot i}}, X_{2_{\cdot j}}) \f$.
static Eigen::MatrixXd get_kernel_matrix(
	const Eigen::MatrixXd& LeftFeature,
	const Eigen::MatrixXd& RightFeature,
	KernelList& Kernels,
	const bool IsTraining = false)
{
	assert(LeftFeature.rows() == PhaseDim && RightFeature.rows() == PhaseDim);
	std::shared_ptr<shogun::DenseFeatures<double>> left_feature = std::make_shared<shogun::DenseFeatures<double>>(const_cast<Eigen::MatrixXd&>(LeftFeature));
	std::shared_ptr<shogun::DenseFeatures<double>> right_feature = std::make_shared<shogun::DenseFeatures<double>>(const_cast<Eigen::MatrixXd&>(RightFeature));
	Eigen::MatrixXd result = Eigen::MatrixXd::Zero(LeftFeature.cols(), RightFeature.cols());
	for (auto& [weight, kernel_ptr] : Kernels)
	{
		if (IsTraining == false && kernel_ptr->get_kernel_type() == shogun::EKernelType::K_DIAG)
		{
			// in case when it is not training feature, the diagonal kernel (working as noise) is not needed
			continue;
		}
		else
		{
			kernel_ptr->init(left_feature, right_feature);
			result += weight * weight * Eigen::Map<Eigen::MatrixXd>(kernel_ptr->get_kernel_matrix());
		}
	}
	return result;
}

/// @brief The function for nlopt optimizer to minimize, return the negative log likelihood
/// @param[in] x The input hyperparameters, need to calculate the function value and gradient at this point
/// @param[out] grad The gradient at the given point. It will not be used
/// @param[in] params Other parameters. Here it is combination of kernel types and selected training set
/// @return The -ln(marginal likelihood + phasedim/2*ln(2*pi))
static double negative_log_marginal_likelihood(const ParameterVector& x, ParameterVector& grad, void* params)
{
	// get the parameters
	const auto& [TrainingFeature, TrainingLabel, TypesOfKernels] = *static_cast<ElementTrainingSet*>(params);
	// get kernel
	KernelList Kernels = generate_kernels(TypesOfKernels, x);
	const Eigen::MatrixXd& KernelMatrix = get_kernel_matrix(TrainingFeature, TrainingFeature, Kernels, true);
	// calculate
	const Eigen::LLT<Eigen::MatrixXd> DecompOfKernel(KernelMatrix);
	const Eigen::MatrixXd& L = DecompOfKernel.matrixL();
	const Eigen::VectorXd& KInvLbl = DecompOfKernel.solve(TrainingLabel); // K^{-1}*y
	return (TrainingLabel.adjoint() * KInvLbl).value() / 2.0 + L.diagonal().array().abs().log().sum();
}

/// @brief The function for nlopt optimizer to minimize, return the LOOCV MSE
/// @param[in] x The input hyperparameters, need to calculate the function value and gradient at this point
/// @param[out] grad The gradient at the given point. It will not be used
/// @param[in] params Other parameters. Here it is combination of kernel types and selected training set
/// @return The squared error of LOOCV
static double leave_one_out_cross_validation(const ParameterVector& x, ParameterVector& grad, void* params)
{
	// get the parameters
	const auto& [TrainingFeature, TrainingLabel, TypesOfKernels] = *static_cast<ElementTrainingSet*>(params);
	// get kernel
	KernelList Kernels = generate_kernels(TypesOfKernels, x);
	const Eigen::MatrixXd& KernelMatrix = get_kernel_matrix(TrainingFeature, TrainingFeature, Kernels, true);
	// prediction: mu_i=y_i-[K^{-1}*y]_i/[K^{-1}]_{ii}
	const Eigen::LLT<Eigen::MatrixXd> DecompOfKernel(KernelMatrix);
	const Eigen::MatrixXd& KInv = DecompOfKernel.solve(Eigen::MatrixXd::Identity(KernelMatrix.rows(), KernelMatrix.cols())); // K^{-1}
	const Eigen::VectorXd& KInvLbl = DecompOfKernel.solve(TrainingLabel);													 // K^{-1}*y
	return (KInvLbl.array() / KInv.diagonal().array()).abs2().sum();
}

/// @brief To calculate the population of one diagonal element by analytic integration of hyperparameters
/// @param[in] Kernel The kernels with hyperparameters set for the given element
/// @param[in] KInvLbl The inverse of kernel matrix of training features times the training labels
/// @return Population of the given element
static double calculate_population(const KernelList& Kernel, const Eigen::VectorXd& KInvLbl)
{
	static const double GlobalFactor = std::pow(2.0 * M_PI, Dim);
	double ppl = 0.0;
	for (const auto& [weight, kernel] : Kernel)
	{
		// select the gaussian ARD kernel
		if (kernel->get_kernel_type() == shogun::EKernelType::K_GAUSSIANARD)
		{
			const Eigen::Matrix<double, PhaseDim, 1> Characteristic = Eigen::Map<Eigen::MatrixXd>(std::dynamic_pointer_cast<shogun::GaussianARDKernel>(kernel)->get_weights());
			ppl += GlobalFactor * weight * weight / Characteristic.prod() * KInvLbl.sum();
		}
	}
	return ppl;
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

/// @brief To calculate the average position and momentum of one diagonal element by analytic integration of hyperparameters
/// @param[in] Features The training features (phase space coordinates) of the given density matrix element
/// @param[in] Kernel The kernels with hyperparameters set for the given elements of density matrix
/// @param[in] KInvLbl The inverse of kernel matrix of training features times the training labels
/// @return Average position and momentum
static ClassicalPhaseVector calculate_1st_order_average(const Eigen::MatrixXd& Features, const KernelList& Kernel, const Eigen::VectorXd& KInvLbl)
{
	static const double GlobalFactor = std::pow(2.0 * M_PI, Dim);
	ClassicalPhaseVector r = ClassicalPhaseVector::Zero();
	for (const auto& [weight, kernel] : Kernel)
	{
		// select the gaussian ARD kernel
		if (kernel->get_kernel_type() == shogun::EKernelType::K_GAUSSIANARD)
		{
			const Eigen::Matrix<double, PhaseDim, 1> Characteristic = Eigen::Map<Eigen::MatrixXd>(std::dynamic_pointer_cast<shogun::GaussianARDKernel>(kernel)->get_weights());
			r += GlobalFactor * weight * weight / Characteristic.prod() * Features * KInvLbl;
		}
	}
	return r;
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

/// @brief The function for nlopt optimizer to minimize, return the LOOCV MSE + Regularization of Normalization
/// @param[in] x The input hyperparameters of all diagonal elements, need to calculate the function value and gradient at this point
/// @param[out] grad The gradient at the given point. It will not be used
/// @param[in] params Other parameters. Here it is combination of kernel types and selected training set
/// @return The squared error of LOOCV and the magnified squared error of normalization
static double diagonal_loocv_with_normalization(const ParameterVector& x, ParameterVector& grad, void* params)
{
	static const double NormRegCoe = 1e3; // enlargement coefficient of normalization regularization
	static const double EngRegCoe = 1e3;  // enlargement coefficient of energy regularization
	double err = 0.0;
	double ppl = 0.0;
	double eng = 0.0;
	// get the parameters
	const auto& [TrainingSets, Energies] = *static_cast<DiagonalTrainingSet*>(params);
	const int NumHyperparameters = x.size() / NumPES;
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		const auto& [features, labels, TypesOfKernels] = TrainingSets[iPES];
		if (TypesOfKernels.size() != 0)
		{
			const int ParamBeginIndex = iPES * NumHyperparameters;
			const ParameterVector param(x.cbegin() + ParamBeginIndex, x.cbegin() + ParamBeginIndex + NumHyperparameters);
			// get loocv error
			err += leave_one_out_cross_validation(param, grad, const_cast<ElementTrainingSet*>(&TrainingSets[iPES]));
			// get population
			KernelList Kernels = generate_kernels(TypesOfKernels, param);
			const Eigen::VectorXd& KInvLbl = get_kernel_matrix(features, features, Kernels, true).llt().solve(labels);
			const double ppl_i = calculate_population(Kernels, KInvLbl);
			ppl += ppl_i;
			eng += ppl_i * Energies[iPES];
		}
	}
	return err + NormRegCoe * std::pow(1.0 - ppl, 2) + EngRegCoe * std::pow(Energies[NumPES] - eng, 2);
}

const double Optimization::DiagMagMin = 1e-4;	  ///< Minimal value of the magnitude of the diagonal kernel
const double Optimization::DiagMagMax = 1;		  ///< Maximal value of the magnitude of the diagonal kernel
const double Optimization::Tolerance = 1e-5;	  ///< Absolute tolerance of independent variable (x) in optimization
const double Optimization::InitialStepSize = 0.5; ///< Initial step size in optimization

Optimization::Optimization(
	const Parameters& Params,
	const KernelTypeList& KernelTypes,
	const nlopt::algorithm Algorithm):
	TypesOfKernels(KernelTypes),
	NumHyperparameters(number_of_overall_hyperparameters(KernelTypes)),
	NumPoints(Params.get_number_of_selected_points())
{
	// set up bounds and hyperparameters
	ParameterVector LowerBound(NumHyperparameters, std::numeric_limits<double>::lowest());
	ParameterVector UpperBound(NumHyperparameters, std::numeric_limits<double>::max());
	Hyperparameters[0].resize(NumHyperparameters, 0.0);
	const ClassicalDoubleVector xSize = Params.get_xmax() - Params.get_xmin();
	const ClassicalDoubleVector pSize = Params.get_pmax() - Params.get_pmin();
	const ClassicalDoubleVector& xSigma = Params.get_sigma_x0();
	const ClassicalDoubleVector& pSigma = Params.get_sigma_p0();
	int iParam = 0;
	for (shogun::EKernelType type : KernelTypes)
	{
		switch (type)
		{
		case shogun::EKernelType::K_DIAG:
			LowerBound[iParam] = DiagMagMin;
			UpperBound[iParam] = DiagMagMax;
			Hyperparameters[0][iParam] = DiagMagMin;
			iParam++;
			break;
		case shogun::EKernelType::K_GAUSSIANARD:
			LowerBound[iParam] = std::numeric_limits<double>::min();
			UpperBound[iParam] = std::numeric_limits<double>::max();
			Hyperparameters[0][iParam] = 1.0;
			iParam++;
			// dealing with x
			for (int iDim = 0; iDim < Dim; iDim++)
			{
				LowerBound[iParam] = 1.0 / xSize[iDim];
				Hyperparameters[0][iParam] = 1.0 / xSigma[iDim];
				iParam++;
			}
			// dealing with p
			for (int iDim = 0; iDim < Dim; iDim++)
			{
				LowerBound[iParam] = 1.0 / pSize[iDim];
				Hyperparameters[0][iParam] = 1.0 / pSigma[iDim];
				iParam++;
			}
			break;
		default:
			break;
		}
	}
	// set up other hyperparameters
	for (int iElement = 1; iElement < NumElements; iElement++)
	{
		Hyperparameters[iElement] = Hyperparameters[0];
	}
	// set up minimizers
	for (int iElement = 0; iElement < NumElements; iElement++)
	{
		NLOptMinimizers.push_back(nlopt::opt(Algorithm, NumHyperparameters));
		NLOptMinimizers.rbegin()->set_lower_bounds(LowerBound);
		NLOptMinimizers.rbegin()->set_upper_bounds(UpperBound);
		NLOptMinimizers.rbegin()->set_xtol_abs(Tolerance);
		NLOptMinimizers.rbegin()->set_ftol_rel(Tolerance);
		NLOptMinimizers.rbegin()->set_initial_step(InitialStepSize);
	}
	// last minimizer for all diagonal elements with population regularization
	ParameterVector DiagLowerBound = LowerBound;
	ParameterVector DiagUpperBound = UpperBound;
	for (int iPES = 1; iPES < NumPES; iPES++)
	{
		DiagLowerBound.insert(DiagLowerBound.cend(), LowerBound.cbegin(), LowerBound.cend());
		DiagUpperBound.insert(DiagUpperBound.cend(), UpperBound.cbegin(), UpperBound.cend());
	}
	NLOptMinimizers.push_back(nlopt::opt(Algorithm, NumHyperparameters * NumPES));
	NLOptMinimizers.rbegin()->set_lower_bounds(DiagLowerBound);
	NLOptMinimizers.rbegin()->set_upper_bounds(DiagUpperBound);
	NLOptMinimizers.rbegin()->set_xtol_abs(Tolerance);
	NLOptMinimizers.rbegin()->set_ftol_rel(Tolerance);
	NLOptMinimizers.rbegin()->set_initial_step(InitialStepSize);
}

double Optimization::optimize_elementwise(const EvolvingDensity& density, const QuantumBoolMatrix& IsSmall)
{
	static const nlopt::vfunc minimizing_function = leave_one_out_cross_validation;
	// first, optimize element by element, and prepare for diagonal elements
	double sum_err = 0.0;
	for (int iElement = 0; iElement < NumElements; iElement++)
	{
		// first, check if all points are very small or not
		if (IsSmall(iElement / NumPES, iElement % NumPES) == false)
		{
			// select points for optimization
			const int BeginIndex = iElement * NumPoints;
			// construct training feature (PhaseDim*N) and training labels (N*1)
			ElementTrainingSet ts;
			auto& [feature, label, kernels] = ts;
			feature.resize(PhaseDim, NumPoints);
			label.resize(NumPoints);
			for (int iPoint = 0; iPoint < NumPoints; iPoint++)
			{
				const auto& [x, p, rho] = density[BeginIndex + iPoint];
				feature.col(iPoint) << x, p;
				label[iPoint] = get_density_matrix_element(rho, iElement);
			}
			kernels = TypesOfKernels;
			NLOptMinimizers[iElement].set_min_objective(minimizing_function, &ts);
			// set variable for saving hyperparameters and function value (marginal likelihood)
			double err = 0.0;
			try
			{
				NLOptMinimizers[iElement].optimize(Hyperparameters[iElement], err);
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
			TrainingFeatures[iElement] = feature;
			Kernels[iElement] = generate_kernels(TypesOfKernels, Hyperparameters[iElement]);
			KInvLbls[iElement] = get_kernel_matrix(feature, feature, Kernels[iElement], true).llt().solve(label);
			if (iElement % (NumPES + 1) != 0)
			{
				// off-diagonal element
				// add up
				sum_err += err;
			}
		}
	}
	return sum_err;
}

double Optimization::optimize_diagonal(
	const EvolvingDensity& density,
	const ClassicalDoubleVector& mass,
	const double TotalEnergy,
	const QuantumBoolMatrix& IsSmall)
{
	static const nlopt::vfunc minimizing_function = diagonal_loocv_with_normalization;
	// construct training set
	DiagonalTrainingSet dts;
	auto& [diagonal_element_ts, energies] = dts;
	energies[NumPES] = TotalEnergy;
	ParameterVector DiagonalHyperparameters;
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		const int ElementIndex = iPES * (NumPES + 1);
		if (IsSmall(iPES, iPES) == false)
		{
			const int BeginIndex = ElementIndex * NumPoints;
			energies[iPES] = calculate_kinetic_average(density, iPES, mass) + calculate_potential_average(density, iPES);
			auto& [feature, label, kernels] = diagonal_element_ts[iPES];
			feature.resize(PhaseDim, NumPoints);
			label.resize(NumPoints);
			for (int iPoint = 0; iPoint < NumPoints; iPoint++)
			{
				const auto& [x, p, rho] = density[BeginIndex + iPoint];
				feature.col(iPoint) << x, p;
				label[iPoint] = get_density_matrix_element(rho, ElementIndex);
			}
			kernels = TypesOfKernels;
		}
		DiagonalHyperparameters.insert(
			DiagonalHyperparameters.cend(),
			Hyperparameters[ElementIndex].cbegin(),
			Hyperparameters[ElementIndex].cend());
	}
	NLOptMinimizers[NumElements].set_min_objective(minimizing_function, &dts);
	ParameterVector grad;
	// set variable for saving hyperparameters and function value (marginal likelihood)
	double err = 0.0;
	try
	{
		NLOptMinimizers[NumElements].optimize(DiagonalHyperparameters, err);
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
			const auto& [feature, label, kernels] = diagonal_element_ts[iPES];
			const int ElementIndex = iPES * (NumPES + 1);
			const int ParamIndex = iPES * NumHyperparameters;
			const int LabelIndex = iPES * NumPoints;
			ParameterVector(DiagonalHyperparameters.cbegin() + ParamIndex, DiagonalHyperparameters.cbegin() + ParamIndex + NumHyperparameters).swap(Hyperparameters[ElementIndex]);
			TrainingFeatures[ElementIndex] = feature;
			Kernels[ElementIndex] = generate_kernels(TypesOfKernels, Hyperparameters[ElementIndex]);
			KInvLbls[ElementIndex] = get_kernel_matrix(TrainingFeatures[ElementIndex], TrainingFeatures[ElementIndex], Kernels[ElementIndex], true).llt().solve(label);
		}
	}
	return err;
}

/// @details Using Gaussian Process Regression (GPR) to depict phase space distribution.
/// First optimize elementwise, then optimize the diagonal with normalization and energy conservation
double Optimization::optimize(
	const EvolvingDensity& density,
	const ClassicalDoubleVector& mass,
	const double TotalEnergy,
	const QuantumBoolMatrix& IsSmall)
{
	double err = 0.0;
	err += optimize_elementwise(density, IsSmall);
	err += optimize_diagonal(density, mass, TotalEnergy, IsSmall);
	return err;
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
			population += calculate_population(Kernels[iPES * (NumPES + 1)], KInvLbls[iPES * (NumPES + 1)]);
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
			KInvLbls[iElement] = get_kernel_matrix(feature, feature, Kernels[iElement], true).llt().solve(label);
			TrainingFeatures[iElement] = feature;
		}
	}
}

/// @details Using Gaussian Process Regression to predict by
///
/// \f[
/// E[p(\mathbf{f}_*|X_*,X,\mathbf{y})]=K(X_*,X)[K(X,X)+\sigma_n^2I]^{-1}\mathbf{y}
/// \f]
///
/// where \f$ X_* \f$ is the test features, \f$ X \f$ is the training features, and  \f$ \mathbf{y} \f$ is the training labels.
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
		KernelList Kernel = generate_kernels(TypesOfKernels, Hyperparameters[ElementIndex]);
		return (get_kernel_matrix(test_feat, TrainingFeatures[ElementIndex], Kernel) * KInvLbls[ElementIndex]).value();
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
		KernelList Kernel = generate_kernels(TypesOfKernels, Hyperparameters[ElementIndex]);
		return get_kernel_matrix(test_feat, TrainingFeatures[ElementIndex], Kernel) * KInvLbls[ElementIndex];
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
		KernelList Kernel = generate_kernels(TypesOfKernels, Hyperparameters[ElementIndex]);
		return get_kernel_matrix(PhaseGrids, TrainingFeatures[ElementIndex], Kernel) * KInvLbls[ElementIndex];
	}
	else
	{
		return Eigen::VectorXd::Zero(PhaseGrids.cols());
	}
}

/// @details To calculate averages,
///
/// \f[
/// <\rho>_{ii}=(2\pi)^{\frac{d}{2}}(\sigma_f^{ii})^2\left|\Lambda^{ii}\right|^{-1}(1\ 1\ \dots\ 1)\qty(K^{ii})^{-1}\vb{y}^{ii}
///
/// <r>_{ii}=(2\pi)^{\frac{d}{2}}(\sigma_f^{ii})^2\left|\Lambda^{ii}\right|^{-1}(r_0\ r_1\ \dots\ r_n)\qty(K^{ii})^{-1}\vb{y}^{ii}
///
/// <p^2>_{ii}=(2\pi)^{\frac{d}{2}}(\sigma_f^{ii})^2\left|\Lambda^{ii}\right|^{-1}(p_0^2+l_p^2\ p_1^2+l_p^2\ \dots\ p_n^2+l_p^2)\qty(K^{ii})^{-1}\vb{y}^{ii}
/// \f]
///
/// where the \f$ \Lambda \f$ matrix is the lower-triangular matrix used in Gaussian ARD kernel, \f$ r = x,p\f$,
/// \f$ l_p \f$ is the characteristic length of momentum.
///
/// For potential, a numerical integration is needed.
/// 
/// To integrate with Monte Carlo, \f$ I=\int f(x)dx \f$, one select a weight function \f$ w(x) \f$,
/// so that the integration becomes \f$ I=\int \frac{f(x)}{w(x)} w(x)dx \f$.
/// 
/// To do that, one samples \f$ N \f$ points with the weight function, then \f$ I\approx\frac{1}{N}\sum\frac{f(x_i)}{w(x_i)} \f$.
/// 
/// To calculate average of \f$ f=x,p,V,T \f$, one integrate \f$ <f>=\int f(x)w(x)dx \f$, or \f$ <f>\approx\frac{1}{N}\sum f(x_i) \f$
Averages calculate_average(
	const Optimization& optimizer,
	const EvolvingDensity& density,
	const ClassicalDoubleVector& mass,
	const QuantumBoolMatrix& IsSmall,
	const int PESIndex)
{
	assert(PESIndex >= 0 && PESIndex < NumPES);
	if (IsSmall(PESIndex, PESIndex) == false)
	{
		// not small, calculate by exact hyperparameter integration
		const int ElementIndex = PESIndex * (NumPES + 1);
		const Eigen::MatrixXd& Features = optimizer.TrainingFeatures[ElementIndex]; // get feature matrix, PhaseDim-by-NumPoints
		const KernelList& Kernel = optimizer.Kernels[ElementIndex];
		const Eigen::VectorXd& KInvLbl = optimizer.KInvLbls[ElementIndex];
		return std::make_tuple(
			calculate_population(Kernel, KInvLbl),
			calculate_1st_order_average(Features, Kernel, KInvLbl),
			calculate_1st_order_average(density, PESIndex),
			calculate_kinetic_average(density, PESIndex, mass),
			calculate_potential_average(density, PESIndex));
	}
	else
	{
		return std::make_tuple(0.0, ClassicalPhaseVector::Zero(), ClassicalPhaseVector::Zero(), 0.0, 0.0);
	}
}
