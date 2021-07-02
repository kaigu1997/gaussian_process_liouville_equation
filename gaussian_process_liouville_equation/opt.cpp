/// @file opt.cpp
/// @brief Implementation of opt.h

#include "stdafx.h"

#include "opt.h"

#include "mc.h"
#include "pes.h"

/// The smart pointer to feature matrix
using FeaturePointer = std::shared_ptr<shogun::Features>;
/// The training set for full hyperparameter optimization
/// passed as parameter to the optimization function.
/// First is feature, second is label, last is the kernels to use
using TrainingSet = std::tuple<Eigen::MatrixXd, Eigen::VectorXd, KernelTypeList>;

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
	const auto& [TrainingFeature, TrainingLabel, TypesOfKernels] = *static_cast<TrainingSet*>(params);
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
	const auto& [TrainingFeature, TrainingLabel, TypesOfKernels] = *static_cast<TrainingSet*>(params);
	// get kernel
	KernelList Kernels = generate_kernels(TypesOfKernels, x);
	const Eigen::MatrixXd& KernelMatrix = get_kernel_matrix(TrainingFeature, TrainingFeature, Kernels, true);
	// prediction: mu_i=y_i-[K^{-1}*y]_i/[K^{-1}]_{ii}
	const Eigen::LLT<Eigen::MatrixXd> DecompOfKernel(KernelMatrix);
	const Eigen::MatrixXd& KInv = DecompOfKernel.solve(Eigen::MatrixXd::Identity(KernelMatrix.rows(), KernelMatrix.cols())); // K^{-1}
	const Eigen::VectorXd& KInvLbl = DecompOfKernel.solve(TrainingLabel);													 // K^{-1}*y
	return (KInvLbl.array() / KInv.diagonal().array()).abs2().sum();
}

const double Optimization::DiagMagMin = 1e-4;		  ///< Minimal value of the magnitude of the diagonal kernel
const double Optimization::DiagMagMax = 1;			  ///< Maximal value of the magnitude of the diagonal kernel
const double Optimization::AbsoluteTolerance = 1e-10; ///< Absolute tolerance of independent variable (x) in optimization
const double Optimization::InitialStepSize = 0.5;	  ///< Initial step size in optimization

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
		NLOptMinimizers.rbegin()->set_xtol_abs(AbsoluteTolerance);
		NLOptMinimizers.rbegin()->set_initial_step(InitialStepSize);
	}
}

/// @details Using Gaussian Process Regression (GPR) to depict phase space distribution, element by element.
double Optimization::optimize(const EvolvingDensity& density, const QuantumBoolMatrix& IsSmall)
{
	const nlopt::vfunc minimizing_function = leave_one_out_cross_validation;
	double sum_err = 0.0;
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
			TrainingSet ts = std::make_tuple(feature, label, TypesOfKernels);
			NLOptMinimizers[iElement].set_min_objective(minimizing_function, &ts);
			// set variable for saving hyperparameters and function value (marginal likelihood)
			double err = 0.0;
			try
			{
				NLOptMinimizers[iElement].optimize(Hyperparameters[iElement], err);
			}
			catch (...)
			{
			}
			// add up
			sum_err += err;
			// set up kernels
			Kernels[iElement] = generate_kernels(TypesOfKernels, Hyperparameters[iElement]);
			KInvLbls[iElement] = get_kernel_matrix(feature, feature, Kernels[iElement], true).llt().solve(label);
			TrainingFeatures[iElement] = feature;
		}
	}
	// after optimization, normalization
	return sum_err;
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
		// not using the saved kernels because of multi-threading, and this is const member function
		KernelList Kernel = generate_kernels(TypesOfKernels, Hyperparameters[ElementIndex]);
		return get_kernel_matrix(test_feat, TrainingFeatures[ElementIndex], Kernel) * KInvLbls[ElementIndex];
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
