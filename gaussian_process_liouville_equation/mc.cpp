/// @file mc.cpp
/// @brief Implementation of mc.h

#include "stdafx.h"

#include "mc.h"

#include "pes.h"

/// The smart pointer to feature matrix
using FeaturePointer = std::shared_ptr<shogun::Features>;
/// The training and validation set for initial optimization
/// passed as parameter to the optimization function.
/// Training feature & label, validation feature & label, and kernels to use
using MSETrainingSet = std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::VectorXd, KernelTypeList>;
/// The training set for full hyperparameter optimization
/// passed as parameter to the optimization function.
/// First is feature, second is label, last is the kernels to use
using FullMargllTrainingSet = std::tuple<Eigen::MatrixXd, Eigen::VectorXd, KernelTypeList>;
/// An array of Eigen matrices, used for the derivatives of the kernel matrix
using MatrixVector = std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>>;

/// @brief To get the element of density matrix
/// @param[in] DensityMatrix The density matrix
/// @param[in] ElementIndex The index of the element in density matrix
/// @return The element of density matrix
/// @details If the index corresponds to upper triangular elements, gives real part.
///
/// If the index corresponds to lower triangular elements, gives imaginary part.
///
/// If the index corresponds to diagonal elements, gives the original value.
static inline double get_density_matrix_element(const QuantumComplexMatrix& DensityMatrix, const int ElementIndex)
{
	const int iPES = ElementIndex / NumPES, jPES = ElementIndex % NumPES;
	if (iPES <= jPES)
	{
		return DensityMatrix(iPES, jPES).real();
	}
	else
	{
		return DensityMatrix(iPES, jPES).imag();
	}
}

QuantumBoolMatrix is_very_small(const EvolvingDensity& density)
{
	// value below this are regarded as 0
	static const double epsilon = 1e-2;
	QuantumBoolMatrix result;
	for (int iElement = 0; iElement < NumElements; iElement++)
	{
		result(iElement / NumPES, iElement % NumPES) = std::all_of(std::execution::par_unseq, density.begin(), density.end(), [=](const PhaseSpacePoint& psp) -> bool { return get_density_matrix_element(std::get<2>(psp), iElement) < epsilon; });
	}
	return result;
}

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

/// @brief The function for nlopt optimizer to minimize, return the MSE of validation set
/// @param[in] x The input hyperparameters, need to calculate the function value and gradient at this point
/// @param[out] grad The gradient at the given point. It will not be used
/// @param[in] params Other parameters. Here it is combination of kernel types, training set and validation set
/// @return The squared error of validation set
static double mean_squared_error(const ParameterVector& x, ParameterVector& grad, void* params)
{
	// get parameters
	const auto& [TrainingFeature, TrainingLabel, ValidationFeature, ValidationLabel, TypesOfKernels] = *static_cast<MSETrainingSet*>(params);
	// get the kernel
	KernelList Kernels = generate_kernels(TypesOfKernels, x);
	const Eigen::MatrixXd& KernelMatrix = get_kernel_matrix(TrainingFeature, TrainingFeature, Kernels, true);
	// get prediction
	const Eigen::VectorXd& PredictLabel = get_kernel_matrix(ValidationFeature, TrainingFeature, Kernels) * KernelMatrix.llt().solve(TrainingLabel);
	// calculate MSE
	return (PredictLabel - ValidationLabel).array().abs2().sum();
}

/// @brief The function for nlopt optimizer to minimize, return the negative log likelihood
/// @param[in] x The input hyperparameters, need to calculate the function value and gradient at this point
/// @param[out] grad The gradient at the given point. It will not be used
/// @param[in] params Other parameters. Here it is combination of kernel types and selected training set
/// @return The -ln(marginal likelihood + phasedim/2*ln(2*pi))
static double negative_log_marginal_likelihood(const ParameterVector& x, ParameterVector& grad, void* params)
{
	// get the parameters
	const auto& [TrainingFeature, TrainingLabel, TypesOfKernels] = *static_cast<FullMargllTrainingSet*>(params);
	// get kernel
	KernelList Kernels = generate_kernels(TypesOfKernels, x);
	const Eigen::MatrixXd& KernelMatrix = get_kernel_matrix(TrainingFeature, TrainingFeature, Kernels, true);
	// calculate
	const Eigen::LLT<Eigen::MatrixXd> DecompOfKernel(KernelMatrix);
	const Eigen::MatrixXd& L = DecompOfKernel.matrixL();
	const Eigen::VectorXd& KInvLbl = DecompOfKernel.solve(TrainingLabel); // K^{-1}*y
	return (TrainingLabel.adjoint() * KInvLbl).value() / 2.0 + L.diagonal().array().abs().log().sum();
}

const double Optimization::DiagMagMin = 1e-4;		  ///< Minimal value of the magnitude of the diagonal kernel
const double Optimization::DiagMagMax = 1;			  ///< Maximal value of the magnitude of the diagonal kernel
const double Optimization::AbsoluteTolerance = 1e-10; ///< Absolute tolerance of independent variable (x) in optimization
const double Optimization::InitialStepSize = 0.5;	  ///< Initial step size in optimization

Optimization::Optimization(
	const Parameters& params,
	const KernelTypeList& KernelTypes,
	const nlopt::algorithm Algorithm):
	TypesOfKernels(KernelTypes),
	NumHyperparameters(number_of_overall_hyperparameters(KernelTypes)),
	NumPoint(params.get_number_of_selected_points())
{
	// set up bounds and hyperparameters
	ParameterVector LowerBound(NumHyperparameters, std::numeric_limits<double>::lowest());
	ParameterVector UpperBound(NumHyperparameters, std::numeric_limits<double>::max());
	Hyperparameters[0].resize(NumHyperparameters, 0.0);
	const ClassicalDoubleVector xSize = params.get_xmax() - params.get_xmin();
	const ClassicalDoubleVector pSize = params.get_pmax() - params.get_pmin();
	const ClassicalDoubleVector& xSigma = params.get_sigma_x0();
	const ClassicalDoubleVector& pSigma = params.get_sigma_p0();
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
		NLOptMinimizers.rbegin()->set_xtol_abs(AbsoluteTolerance);
		NLOptMinimizers.rbegin()->set_initial_step(InitialStepSize);
	}
}

/// @details To reconstruct the given distribution using Gaussian process regression by optimizing weighted mean squared error
double Optimization::initial_optimize(
	const EvolvingDensity& density,
	const Parameters& params,
	const DistributionFunction& distribution,
	const QuantumBoolMatrix& IsSmall)
{
	assert(density.size() == NumPoint && IsSmall.diagonal().any() == true);
	const nlopt::vfunc minimizing_function = mean_squared_error;
	// construct validation set
	const ClassicalDoubleVector& x0 = params.get_x0(); // save initial positions
	const ClassicalDoubleVector& p0 = params.get_p0(); // save initial momentum
	EvolvingDensity validation_set;
	for (int i = 0; i < NumPoint; i++)
	{
		validation_set.push_back(std::make_tuple(x0, p0, distribution(x0, p0))); // put the initial point into density
	}
	monte_carlo_selection(params, distribution, IsSmall, validation_set);
	double mse = 0.0;
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		// only one of them is non-zero
		if (IsSmall(iPES, iPES) == false)
		{
			const int ElementIndex = iPES * (NumPES + 1);
			// construct training and validation set
			Eigen::MatrixXd training_feature(PhaseDim, NumPoint), validation_feature(PhaseDim, NumPoint);
			Eigen::VectorXd training_label(NumPoint), validation_label(NumPoint);
			for (int iPoint = 0; iPoint < NumPoint; iPoint++)
			{
				const auto& [x_train, p_train, rho_train] = density[iPoint];
				training_label[iPoint] = get_density_matrix_element(rho_train, ElementIndex);
				training_feature.col(iPoint) << x_train, p_train;
				const auto& [x_valid, p_valid, rho_valid] = validation_set[iPoint];
				validation_label[iPoint] = get_density_matrix_element(rho_valid, ElementIndex);
				validation_feature.col(iPoint) << x_valid, p_valid;
			}
			MSETrainingSet ts = std::make_tuple(training_feature, training_label, validation_feature, validation_label, TypesOfKernels);
			NLOptMinimizers[ElementIndex].set_min_objective(minimizing_function, &ts);
			// set variable for saving hyperparameters and function value (marginal likelihood)
			try
			{
				NLOptMinimizers[ElementIndex].optimize(Hyperparameters[ElementIndex], mse);
			}
			catch (...)
			{
			}
			// set up kernels
			Kernels[ElementIndex] = generate_kernels(TypesOfKernels, Hyperparameters[ElementIndex]);
			KInvLbls[ElementIndex] = get_kernel_matrix(training_feature, training_feature, Kernels[ElementIndex], true).llt().solve(training_label);
			TrainingFeatures[ElementIndex] = training_feature;
			break; // only 1 diagonal element is used
		}
	}
	return mse;
	// as MSE is minimized, no need to normalize
}

/// @details Using Gaussian Process Regression (GPR) to depict phase space distribution, element by element.
double Optimization::optimize_full_and_normalize(EvolvingDensity& density, const QuantumBoolMatrix& IsSmall)
{
	auto weight_func = [](const double x) -> double {
		return x * x;
	};
	const nlopt::vfunc minimizing_function = negative_log_marginal_likelihood;
	double sum_marg_ll = 0.0;
	// get a copy for sort
	EvolvingDensity density_copy = density;
	for (int iElement = 0; iElement < NumElements; iElement++)
	{
		// first, check if all points are very small or not
		if (IsSmall(iElement / NumPES, iElement % NumPES) == false)
		{
			// select points for optimization
			std::sort(
				std::execution::par_unseq,
				density_copy.begin(),
				density_copy.end(),
				[&](const PhaseSpacePoint& psp1, const PhaseSpacePoint& psp2) -> bool { return weight_func(get_density_matrix_element(std::get<2>(psp1), iElement)) > weight_func(get_density_matrix_element(std::get<2>(psp2), iElement)); });
			// reconstruct training feature (PhaseDim*N) and training labels (N*1)
			Eigen::MatrixXd feature(PhaseDim, NumPoint);
			Eigen::VectorXd label(NumPoint);
			for (int iPoint = 0; iPoint < NumPoint; iPoint++)
			{
				const auto& [x, p, rho] = density_copy[iPoint];
				label[iPoint] = get_density_matrix_element(rho, iElement);
				feature.col(iPoint) << x, p;
			}
			FullMargllTrainingSet ts = std::make_tuple(feature, label, TypesOfKernels);
			NLOptMinimizers[iElement].set_min_objective(minimizing_function, &ts);
			// set variable for saving hyperparameters and function value (marginal likelihood)
			double marg_ll = 0.0;
			try
			{
				NLOptMinimizers[iElement].optimize(Hyperparameters[iElement], marg_ll);
			}
			catch (...)
			{
			}
			// add up
			sum_marg_ll += marg_ll;
			// set up kernels
			Kernels[iElement] = generate_kernels(TypesOfKernels, Hyperparameters[iElement]);
			KInvLbls[iElement] = get_kernel_matrix(feature, feature, Kernels[iElement], true).llt().solve(label);
			TrainingFeatures[iElement] = feature;
		}
	}
	// after optimization, normalization
	double population = 0.0;
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		if (IsSmall(iPES, iPES) == false)
		{
			const int ElementIndex = iPES * (NumPES + 1);
			for (const auto& [weight, kernel] : Kernels[ElementIndex])
			{
				// select the gaussian ARD kernel
				if (kernel->get_kernel_type() == shogun::EKernelType::K_GAUSSIANARD)
				{
					const Eigen::Matrix<double, PhaseDim, 1> Characteristic = Eigen::Map<Eigen::MatrixXd>(std::dynamic_pointer_cast<shogun::GaussianARDKernel>(kernel)->get_weights());
					population += weight * weight / Characteristic.prod() * KInvLbls[ElementIndex].sum();
				}
			}
		}
	}
	population *= std::pow(2.0 * M_PI, Dim);
	for (auto& [x, p, rho] : density)
	{
		rho /= population;
	}
	for (int iElement = 0; iElement < NumElements; iElement++)
	{
		if (IsSmall(iElement / NumPES, iElement % NumPES) == false)
		{
			KInvLbls[iElement] /= population;
		}
	}
	return sum_marg_ll;
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
		return (get_kernel_matrix(test_feat, TrainingFeatures[ElementIndex], Kernels[ElementIndex]) * KInvLbls[ElementIndex]).value();
	}
	else
	{
		return 0;
	}
}
Eigen::VectorXd Optimization::predict_elements(
	const QuantumBoolMatrix& IsSmall,
	const ClassicalVectors& x,
	const ClassicalVectors& p,
	const int ElementIndex) const
{
	assert(x.size() == p.size());
	const int NPoint = x.size();
	if (IsSmall(ElementIndex / NumPES, ElementIndex % NumPES) == false)
	{
		// not small, have something to do
		// generate feature
		Eigen::MatrixXd test_feat(PhaseDim, NPoint);
		for (int i = 0; i < NPoint; i++)
		{
			test_feat.col(i) << x[i], p[i];
		}
		// predict
		return get_kernel_matrix(test_feat, TrainingFeatures[ElementIndex], Kernels[ElementIndex]) * KInvLbls[ElementIndex];
	}
	else
	{
		return Eigen::VectorXd::Zero(NPoint);
	}
}

Eigen::VectorXd Optimization::print_element(const QuantumBoolMatrix& IsSmall, const Eigen::MatrixXd& PhaseGrids, const int ElementIndex) const
{
	if (IsSmall(ElementIndex / NumPES, ElementIndex % NumPES) == false)
	{
		return get_kernel_matrix(PhaseGrids, TrainingFeatures[ElementIndex], Kernels[ElementIndex]) * KInvLbls[ElementIndex];
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
Averages Optimization::calculate_average(const ClassicalDoubleVector& mass, const QuantumBoolMatrix& IsSmall, const int PESIndex) const
{
	static const double NumericIntegrationInitialStepsize = 1e-2;		   // initial step size for numeric integration below
	static const double epsilon = std::exp(-12.5) / std::sqrt(2.0 * M_PI); // value below this is 0; gaussian function value at 5 sigma
	boost::numeric::odeint::bulirsch_stoer<double> stepper;				   // the stepper for numerical integration
	double ppl = 0.0, T = 0.0, V = 0.0;
	ClassicalDoubleVector x = ClassicalDoubleVector::Zero(), p = ClassicalDoubleVector::Zero();
	if (IsSmall(PESIndex, PESIndex) == false)
	{
		// not small, calculate by integration
		const int ElementIndex = PESIndex * (NumPES + 1);
		const Eigen::VectorXd& KInvLbl = KInvLbls[ElementIndex];
		for (const auto& [weight, kernel] : Kernels[ElementIndex])
		{
			// select the gaussian ARD kernel
			if (kernel->get_kernel_type() == shogun::EKernelType::K_GAUSSIANARD)
			{
				const Eigen::MatrixXd& Features = TrainingFeatures[ElementIndex]; // get feature matrix, PhaseDim-by-NPoint
				const int NPoint = Features.cols();
				const Eigen::Matrix<double, PhaseDim, 1> Characteristic = Eigen::Map<Eigen::MatrixXd>(std::dynamic_pointer_cast<shogun::GaussianARDKernel>(kernel)->get_weights());
				const double GlobalWeight = std::pow(2.0 * M_PI, Dim) * weight * weight / Characteristic.prod();
				// population
				ppl += GlobalWeight * KInvLbl.sum();
				// x and p
				const auto& r = GlobalWeight * Features * KInvLbl;
				x += r.block<Dim, 1>(0, 0);
				p += r.block<Dim, 1>(Dim, 0);
				// T
				T += GlobalWeight * ((Features.block(Dim, 0, Dim, NPoint).array().abs2().matrix() * KInvLbl).array() / mass.array()).sum() / 2.0;
				// V, int_R{f(x)dx}=int_0^1{dt[f((1-t)/t)+f((t-1)/t)]t^2} by x=(1-t)/t, f(x)=V(x)*rho(x)
				auto potential_times_population = [&, weight = weight](const ClassicalDoubleVector& x0) -> double {
					// rho(x)=int{dp rho(x,p)}=sigma_f^2*(2*pi)^(Dim/4)*\prod{l_p}*(s1,s2,...)*K^{-1}*y
					// si = exp(-\sum_j{(x0j-xij)^2/l_{xj}^2}/2.0)
					const auto& CharX = Characteristic.block<Dim, 1>(0, 0);
					return weight * weight * std::pow(2.0 * M_PI, Dim / 2.0) / CharX.prod() * (((Features.block(0, 0, Dim, NPoint).colwise() - x0).array().colwise() * CharX.array()).abs2().colwise().sum() / -2.0).exp().matrix().dot(KInvLbl) * adiabatic_potential(x0)[PESIndex];
				};
				std::vector<std::function<double(const Eigen::VectorXd&)>> integrate_n_dim(Dim);
				integrate_n_dim[0] = [&](const Eigen::VectorXd& x0) -> double {
					return potential_times_population(x0);
				};
				// loop until the last dimension, where the vector parameter will be length 1
				for (int iDim = 1; iDim < Dim; iDim++)
				{
					// integrate over the last dimension, from -inf to inf
					integrate_n_dim[iDim] = [&, iDim](const Eigen::VectorXd& x0) -> double {
						double result = 0.0;
						boost::numeric::odeint::integrate_adaptive(
							stepper,
							[&](const double& /*x*/, double& dxdt, const double t) {
								if (std::abs(t) < epsilon)
								{
									// this dimension is +-inf, very far away, population will be 0 there
									dxdt = 0.0;
								}
								else
								{
									const double x_val = (1.0 - t) / t;
									Eigen::VectorXd x_vec_plus(Dim + 1 - iDim), x_vec_minus(Dim + 1 - iDim);
									x_vec_plus << x0, x_val;
									x_vec_minus << x0, -x_val;
									dxdt = (integrate_n_dim[iDim - 1](x_vec_plus) + integrate_n_dim[iDim - 1](x_vec_minus)) / (t * t);
								}
							},
							result,
							0.0,
							1.0,
							NumericIntegrationInitialStepsize);
						return result;
					};
				}
				boost::numeric::odeint::integrate_adaptive(
					stepper,
					[&](const double& /*x*/, double& dxdt, const double t) {
						if (std::abs(t) < epsilon)
						{
							dxdt = 0.0;
						}
						else
						{
							const double x_val = (1.0 - t) / t;
							Eigen::VectorXd x_vec_plus(1), x_vec_minus(1);
							x_vec_plus[0] = x_val;
							x_vec_minus[0] = -x_val;
							dxdt = (integrate_n_dim[Dim - 1](x_vec_plus) + integrate_n_dim[Dim - 1](x_vec_minus)) / (t * t);
						}
					},
					V,
					0.0,
					1.0,
					NumericIntegrationInitialStepsize);
			}
		}
	}
	return std::make_tuple(ppl, x, p, V, T);
}

/// @details Under the adiabatic basis, only rho[0][0] is non-zero initially,
/// i.e. all the population lies on the ground state. For rho[0][0],
///
/// \f[
/// P(\bf{x},\bf{p})=\prod_i{P(x_i,p_i)}=\prod_i{\left(\frac{1}{2\pi\sigma_{x_i}\sigma_{p_i}}\exp\left[-\frac{(x_i-x_{i0})^2}{2\sigma_{x_i}^2}-\frac{(p_i-p_{i0})^2}{2\sigma_{p_i}^2}\right]\right)}
/// \f]
///
/// where i goes over all direction in the classical DOF
QuantumComplexMatrix initial_distribution(
	const Parameters& params,
	const ClassicalDoubleVector& x,
	const ClassicalDoubleVector& p)
{
	QuantumComplexMatrix result = QuantumComplexMatrix::Zero(NumPES, NumPES);
	const ClassicalDoubleVector& x0 = params.get_x0();
	const ClassicalDoubleVector& p0 = params.get_p0();
	const ClassicalDoubleVector& SigmaX0 = params.get_sigma_x0();
	const ClassicalDoubleVector& SigmaP0 = params.get_sigma_p0();
	result(0, 0) = std::exp(-(((x - x0).array() / SigmaX0.array()).abs2().sum() + ((p - p0).array() / SigmaP0.array()).abs2().sum()) / 2.0)
		/ (std::pow(2.0 * M_PI, Dim) * SigmaX0.prod() * SigmaP0.prod());
	return result;
}


/// @details Assuming that the distribution function containing all the required information. If
/// it is a distribution from some parameters, those parameters should have already been
/// bound; if it is some learning method, the parameters should have already been optimized.
/// This MC function gives a random displacement in the phase space each time, then simply
/// input the phase space coordinate and hope to get the predicted density matrix, calculate
/// its weight by the weight function, and finally do the MC selection based on the weight.
void monte_carlo_selection(
	const Parameters& params,
	const DistributionFunction& distribution,
	const QuantumBoolMatrix& IsSmall,
	EvolvingDensity& density)
{
	// TODO: mc techniques: begin from maximum of points, stepsize, number of MC steps
	static std::mt19937 engine(std::chrono::system_clock::now().time_since_epoch().count()); // Random number generator, using merseen twister with seed from time
	static const int MinNOMC = 100 * Dim * 2;												 // The minimum number of monte carlo steps that will be done for each phase point selection
	static int NOMC = MinNOMC;																 // The number of monte carlo steps used
	static const int NGrids = 20;															 // The number of grids on each phase direction
	// alias names
	static const ClassicalDoubleVector& dxmax = (params.get_xmax() - params.get_xmin()) / NGrids;
	static const ClassicalDoubleVector& dpmax = (params.get_pmax() - params.get_pmin()) / NGrids;
	static const ClassicalDoubleVector& mass = params.get_mass();
	static const double dt = params.get_dt();
	static const int NumPoints = params.get_number_of_selected_points();
	auto weight_func = [](const double x) -> double {
		return x * x;
	};
	// get maximum, then evolve the points adiabatically
	std::array<EvolvingDensity, NumElements> Maximums;
	for (int iElement = 0; iElement < NumElements; iElement++)
	{
		const int iPES = iElement / NumPES, jPES = iElement % NumPES;
		if (IsSmall(iPES, jPES) == false)
		{
			std::sort(
				std::execution::par_unseq,
				density.begin(),
				density.end(),
				[&](const PhaseSpacePoint& psp1, const PhaseSpacePoint& psp2) -> bool { return weight_func(get_density_matrix_element(std::get<2>(psp1), iElement)) > weight_func(get_density_matrix_element(std::get<2>(psp2), iElement)); });
			for (auto [x, p, rho] : density)
			{
				const Tensor3d f = adiabatic_force(x);
				x = x.array() + p.array() / mass.array() * dt;
				for (int iDim = 0; iDim < Dim; iDim++)
				{
					p[iDim] -= (f[iDim](iPES, iPES) + f[iDim](jPES, jPES)) / 2.0 * dt;
				}
				rho = distribution(x, p);
				Maximums[iElement].push_back(std::make_tuple(x, p, rho));
			}
		}
	}
	density.clear();

	// moving random number generator and MC random number generator
	std::vector<std::uniform_real_distribution<double>> x_displacement;
	std::vector<std::uniform_real_distribution<double>> p_displacement;
	for (int iDim = 0; iDim < Dim; iDim++)
	{
		x_displacement.push_back(std::uniform_real_distribution<double>(-dxmax[iDim], dxmax[iDim]));
		p_displacement.push_back(std::uniform_real_distribution<double>(-dpmax[iDim], dpmax[iDim]));
	}
	std::uniform_real_distribution<double> mc_selection(0.0, 1.0); // for whether pick or not

	// Monte Carlo selection
	for (int iElement = 0; iElement < NumElements; iElement++)
	{
		if (IsSmall(iElement / NumPES, iElement % NumPES) == false)
		{
			// begin from an old point
			for (auto& [x_old, p_old, rho_old] : Maximums[iElement])
			{
				double weight_old = weight_func(get_density_matrix_element(rho_old, iElement));
				// do MC
				// the number of MC steps (NOMC) is adaptive
				int acc = 0;
				for (int iIter = 0;; iIter++)
				{
					// calculate new weight there
					ClassicalDoubleVector x_new = x_old, p_new = p_old;
					for (int iDim = 0; iDim < Dim; iDim++)
					{
						x_new[iDim] += x_displacement[iDim](engine);
						p_new[iDim] += p_displacement[iDim](engine);
					}
					const QuantumComplexMatrix& rho_new = distribution(x_new, p_new);
					const double weight_new = weight_func(get_density_matrix_element(rho_new, iElement));
					if (weight_new > weight_old || weight_new / weight_old > mc_selection(engine))
					{
						// new weight is larger than the random number, accept
						// otherwise, reject
						x_old = x_new;
						p_old = p_new;
						rho_old = rho_new;
						weight_old = weight_new;
						acc++;
					}
					if (iIter == NOMC)
					{
						if (acc * 1. / NOMC < 0.4)
						{
							// good acceptance, finish MC
							if (NOMC > MinNOMC)
							{
								NOMC /= 2;
							}
							break;
						}
						else
						{
							// accpetance too high, double the step
							NOMC *= 2;
						}
					}
				}
				// after MC, the new point is at the original place
			}
			// after all points updated, insert into the EvolvingDensity
			density.insert(density.end(), Maximums[iElement].cbegin(), Maximums[iElement].cend());
		}
	}
}
