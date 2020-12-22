/// @file gpr.cpp
/// @brief Definition of Gaussian Process Regression functions

#include "gpr.h"

#include "io.h"

/// A vector containing weights and pointers to kernels
using KernelList = std::vector<std::pair<double, std::shared_ptr<shogun::Kernel>>>;
/// Training set of one element of phase space distribution, with the kernel to use
using ElementTrainingSet = std::tuple<Eigen::MatrixXd, Eigen::VectorXd, KernelTypeList>;
/// An array of Eigen matrices, used for the derivatives of the kernel matrix
using MatrixVector = std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>>;

/// Subsystem Diabatic Hamiltonian, being the potential of the bath
/// @param[in] x The bath DOF position
/// @return The potential matrix in diabatic basis, which is real symmetric
static QuantumMatrixDouble diabatic_potential(const double x)
{
	// parameters of Tully's 2nd model, Dual Avoided Crossing (DAC)
	static const double DAC_A = 0.10;  ///< A in DAC model
	static const double DAC_B = 0.28;  ///< B in DAC model
	static const double DAC_C = 0.015; ///< C in DAC model
	static const double DAC_D = 0.06;  ///< D in DAC model
	static const double DAC_E = 0.05;  ///< E in DAC model
	QuantumMatrixDouble Potential;
	Potential(0, 0) = 0.0;
	Potential(0, 1) = Potential(1, 0) = DAC_C * std::exp(-DAC_D * x * x);
	Potential(1, 1) = DAC_E - DAC_A * std::exp(-DAC_B * x * x);
	return Potential;
}

/// @brief The eigenvalues of the diabatic potential, or the diagonalized diabatic potential matrix
/// @param[in] x The bath DOF position
/// @return The adiabatic potential matrix at this position, which is real diagonal
static QuantumMatrixDouble adiabatic_potential(const double x)
{
	Eigen::SelfAdjointEigenSolver<QuantumMatrixDouble> solver(diabatic_potential(x));
	return solver.eigenvalues().asDiagonal();
}

QuantumVectorDouble calculate_population_from_grid(const SuperMatrix& PhaseSpaceDistribution, const double dx, const double dp)
{
	QuantumVectorDouble result;
	for (int i = 0; i < NumPES; i++)
	{
		result[i] = PhaseSpaceDistribution[i][i].sum();
	}
	return result * dx * dp;
}

QuantumVectorDouble calculate_potential_energy_from_grid(const SuperMatrix& AdiabaticDistribution, const Eigen::VectorXd& x, const Eigen::VectorXd& p)
{
	const int nx = x.size(), np = p.size();
	const double dx = (x[nx - 1] - x[0]) / nx, dp = (p[np - 1] - p[0]) / np;
	QuantumVectorDouble result = QuantumVectorDouble::Zero();
	for (int i = 0; i < nx; i++)
	{
		const QuantumMatrixDouble& AdiabaticPotential = adiabatic_potential(x[i]);
		for (int j = 0; j < NumPES; j++)
		{
			result[j] += AdiabaticDistribution[j][j].row(i).sum() * AdiabaticPotential(j, j);
		}
	}
	return result * dx * dp;
}

QuantumVectorDouble calculate_kinetic_energy_from_grid(const SuperMatrix& AdiabaticDistribution, const Eigen::VectorXd& x, const Eigen::VectorXd& p)
{
	const int nx = x.size(), np = p.size();
	const double dx = (x[nx - 1] - x[0]) / nx, dp = (p[np - 1] - p[0]) / np;
	QuantumVectorDouble result = QuantumVectorDouble::Zero();
	for (int i = 0; i < np; i++)
	{
		const double KineticEnergy = p[i] * p[i] / 2.0 / Mass;
		for (int j = 0; j < NumPES; j++)
		{
			result[j] += AdiabaticDistribution[j][j].col(i).sum() * KineticEnergy;
		}
	}
	return result * dx * dp;
}

/// @brief Calculate the overall number of hyperparameters the optimization will use
/// @param[in] TypesOfKernels The vector containing all the kernel type used in optimization
/// @return The overall number of hyperparameters (including the magnitude of each kernel)
static int number_of_overall_hyperparameters(const KernelTypeList& TypesOfKernels)
{
	const int size = TypesOfKernels.size();
	int sum = 0;
	for (int i = 0; i < size; i++)
	{
		sum++; // weight
		switch (TypesOfKernels[i])
		{
		case shogun::EKernelType::K_DIAG:
			break; // no extra hyperparameter for diagonal kernel
		case shogun::EKernelType::K_GAUSSIANARD:
#ifndef NOCROSS
			sum += PhaseDim * (PhaseDim + 1) / 2; // extra hyperparameter from relevance and characteristic lengths
#else
			sum += PhaseDim;
#endif
			break;
		default:
			std::cerr << "UNKNOWN KERNEL!\n";
			break;
		}
	}
	return sum;
}

std::vector<ParameterVector> set_initial_value(
	const KernelTypeList& TypesOfKernels,
	const SuperMatrix& data,
	const Eigen::VectorXd& x,
	const Eigen::VectorXd& p)
{
	const std::pair<int, int> MaxIndex = [&]() -> std::pair<int, int> {
		int idx_x, idx_p;
		double max = 0.0;
		for (int i = 0; i < data[0][0].rows(); i++)
		{
			for (int j = 0; j < data[0][0].cols(); j++)
			{
				if (data[0][0](i, j) > max)
				{
					idx_x = i;
					idx_p = j;
					max = data[0][0](i, j);
				}
			}
		}
		return std::make_pair(idx_x, idx_p);
	}();
	const double sigma_p = p[MaxIndex.second] / 20.0;
	const double sigma_x = 0.5 / sigma_p;
	static const double DiagMagMin = 1e-8;	// Minimal value of the magnitude of the diagonal kernel
	static const double DiagMagMax = 1e-5;	// Maximal value of the magnitude of the diagonal kernel
	static const double GaussMagMin = 1e-4; // Minimal value of the magintude of the gaussian kernel
	static const double GaussMagMax = 1.0;	// Maximal value of the magnitude of the gaussian kernel
	const int NoVar = number_of_overall_hyperparameters(TypesOfKernels);
	static std::vector<double> lower_bound(NoVar * NumPES * NumPES, std::numeric_limits<double>::lowest());
	static std::vector<double> upper_bound(NoVar * NumPES * NumPES, std::numeric_limits<double>::max());
	static std::vector<double> hyperparameters(NoVar * NumPES * NumPES, 0);
	for (int i = 0; i < NumPES * NumPES; i++)
	{
		for (int j = 0, iparam = 0; j < TypesOfKernels.size(); j++)
		{
			switch (TypesOfKernels[j])
			{
			case shogun::EKernelType::K_DIAG:
				lower_bound[iparam + i * NoVar] = DiagMagMin;
				upper_bound[iparam + i * NoVar] = DiagMagMax;
				hyperparameters[iparam + i * NoVar] = DiagMagMin;
				iparam++;
				break;
			case shogun::EKernelType::K_GAUSSIANARD:
				lower_bound[iparam + i * NoVar] = GaussMagMin;
				upper_bound[iparam + i * NoVar] = GaussMagMax;
				hyperparameters[iparam + i * NoVar] = GaussMagMax;
				iparam++;
				for (int l = 0; l < PhaseDim; l++)
				{
					if (l % 2 == 0)
					{
						lower_bound[iparam + i * NoVar] = 1.0 / (x.maxCoeff() - x.minCoeff());
						hyperparameters[iparam + i * NoVar] = 1.0 / sigma_x;
					}
					else
					{
						lower_bound[iparam + i * NoVar] = 1.0 / (p.maxCoeff() - p.minCoeff());
						hyperparameters[iparam + i * NoVar] = 1.0 / sigma_p;
					}
					iparam++;
#ifndef NOCROSS
					for (int k = l + 1; k < PhaseDim; k++)
					{
						iparam++;
					}
#endif
				}
				break;
			default:
				std::cerr << "UNKNOWN KERNEL!\n";
				break;
			}
		}
	}
	std::vector<ParameterVector> result;
	result.push_back(lower_bound);
	result.push_back(upper_bound);
	result.push_back(hyperparameters);
	return result;
}

QuantumMatrixBool is_very_small_everywhere(const SuperMatrix& data)
{
	// value below this are regarded as 0
	static const double epsilon = 1e-2;
	QuantumMatrixBool result = QuantumMatrixBool::Constant(true);
	for (int i = 0; i < NumPES; i++)
	{
		for (int j = 0; j < NumPES; j++)
		{
			result(i, j) = (data[i][j].maxCoeff() < epsilon && data[i][j].minCoeff() > -epsilon);
		}
	}
	return result;
}

/// @details This function choose points independently for each element based on whether they are small.
/// If all very small, then select points randomly without weight.
/// Otherwise, select based on the weights at the point by MC procedure.
FullTrainingSet generate_training_set(
	const SuperMatrix& data,
	const QuantumMatrixBool& IsSmall,
	const Eigen::VectorXd& x,
	const Eigen::VectorXd& p)
{
	static std::mt19937_64 generator(std::chrono::system_clock::now().time_since_epoch().count()); // random number generator, 64 bits Mersenne twister engine
	const int nx = data[0][0].rows(), np = data[0][0].cols(), n = data[0][0].size();
	PointSet chosen;
	SuperMatrix training_feature;
	VectorMatrix training_label;
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		for (int jPES = 0; jPES < NumPES; jPES++)
		{
			chosen.clear();
			// chose the point
			if (IsSmall(iPES, jPES) == true)
			{
				// all very small, select points randomly
				std::uniform_int_distribution<int> x_dis(0, nx - 1), p_dis(0, np - 1);
				while (chosen.size() < NPoint)
				{
					chosen.insert(std::make_pair(x_dis(generator), p_dis(generator)));
				}
			}
			else
			{
				// not small. weighted choose from the absolute value on the grid
				auto abs_sum = [](double a, double b) -> double {
					return std::abs(a) + std::abs(b);
				};
				const double weight = std::accumulate(data[iPES][jPES].data(), data[iPES][jPES].data() + n, 0.0, abs_sum);
				std::uniform_real_distribution<double> urd(0.0, weight);
				while (chosen.size() < NPoint)
				{
					double acc_weight = urd(generator);
					for (int iGrid = 0; iGrid < nx; iGrid++)
					{
						for (int jGrid = 0; jGrid < np; jGrid++)
						{
							acc_weight -= std::abs(data[iPES][jPES](iGrid, jGrid));
							if (acc_weight < 0)
							{
								chosen.insert(std::make_pair(iGrid, jGrid));
								goto next;
							}
						}
					}
				next:;
				}
				/*
				Eigen::MatrixXd weight_matrix = data[iPES][jPES].array().abs();
				while (chosen.size() < NPoint)
				{
					int idx_x, idx_p;
					weight_matrix.maxCoeff(&idx_x, &idx_p);
					weight_matrix(idx_x, idx_p) = 0;
					chosen.insert(std::make_pair(idx_x, idx_p));
				}
				*/
			}

			// the get the labels & features
			training_label[iPES][jPES].resize(NPoint);
			training_feature[iPES][jPES].resize(PhaseDim, NPoint);
			int Index = 0;
			for (PointSet::iterator iter = chosen.begin(); iter != chosen.end(); ++iter, Index++)
			{
				training_feature[iPES][jPES](0, Index) = x[iter->first];
				training_feature[iPES][jPES](1, Index) = p[iter->second];
				training_label[iPES][jPES][Index] = data[iPES][jPES](iter->first, iter->second);
			}
		}
	}
	return std::make_pair(training_feature, training_label);
}

/// @brief Construct kernels from hyperparameters and their types
/// @param[in] TypeOfKernel The vector containing type of all kernels that will be used
/// @param[in] Hyperparameters The hyperparameters of all kernels (magnitude and other)
/// @return A vector of all kernels, with parameters set, but without any feature
static KernelList generate_kernels(const KernelTypeList& TypesOfKernels, const ParameterVector& Hyperparameters)
{
	const int NKernel = TypesOfKernels.size();
	KernelList result;
	for (int iKernel = 0, iParam = 0; iKernel < NKernel; iKernel++)
	{
		const double weight = Hyperparameters[iParam++];
		switch (TypesOfKernels[iKernel])
		{
		case shogun::EKernelType::K_DIAG:
			result.push_back(std::make_pair(weight, std::make_shared<shogun::DiagKernel>()));
			break;
		case shogun::EKernelType::K_GAUSSIANARD:
		{
			std::shared_ptr<shogun::GaussianARDKernel> gauss_ard_kernel_ptr = std::make_shared<shogun::GaussianARDKernel>();
			Eigen::MatrixXd characteristic = Eigen::MatrixXd::Zero(PhaseDim, PhaseDim);
#ifndef NOCROSS
			// rowwise parameters
			for (int k = 0; k < PhaseDim; k++)
			{
				for (int j = k; j < PhaseDim; j++)
				{
					characteristic(j, k) = Hyperparameters[iParam++];
				}
			}
#else
			for (int j = 0; j < PhaseDim; j++)
			{
				characteristic(j, j) = Hyperparameters[iParam++];
			}
#endif
			gauss_ard_kernel_ptr->set_matrix_weights(characteristic);
			result.push_back(std::make_pair(weight, gauss_ard_kernel_ptr));
			break;
		}
		default:
			std::cerr << "UNKNOWN KERNEL!\n";
			break;
		}
	}
	return result;
}

/// @brief Deep copy of Eigen matrix to shogun matrix
/// @param[in] mat Eigen matrix
/// @return shogun matrix of same content
static inline shogun::SGMatrix<double> generate_shogun_matrix(const Eigen::MatrixXd& mat)
{
	shogun::SGMatrix<double> result(mat.rows(), mat.cols());
	std::copy(mat.data(), mat.data() + mat.size(), result.data());
	return result;
}

/// @brief Calculate the kernel matrix, \f$ K(X_1,X_2) \f$
/// @param[in] LeftFeature The feature on the left, \f$ X_1 \f$
/// @param[in] RightFeature The feature on the right, \f$ X_2 \f$
/// @param[inout] Kernels The vector containing all kernels that will be used, with parameters set and features not set
/// @param[in] IsTraining Whether the features are all training set or not
/// @return The kernel matrix
/// @details This function calculates the kernel matrix with noise using the shogun library,
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
	// construct the feature
	std::shared_ptr<shogun::DenseFeatures<double>> left_feature = std::make_shared<shogun::DenseFeatures<double>>(generate_shogun_matrix(LeftFeature));
	std::shared_ptr<shogun::DenseFeatures<double>> right_feature = std::make_shared<shogun::DenseFeatures<double>>(generate_shogun_matrix(RightFeature));
	const int NKernel = Kernels.size();
	Eigen::MatrixXd result = Eigen::MatrixXd::Zero(LeftFeature.cols(), RightFeature.cols());
	for (int i = 0; i < NKernel; i++)
	{
		const double weight = Kernels[i].first;
		std::shared_ptr<shogun::Kernel>& kernel_ptr = Kernels[i].second;
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

/// @brief Calculate the derivative of kernel matrix over hyperparameters
/// @param[in] Feature The left and right feature of the kernel
/// @param[inout] Kernels The vector containing all kernels that will be used, with parameters set and features not set
/// @return A vector of matrices being the derivative of kernel matrix over hyperparameters, in the same order as Hyperparameters
/// @details This function calculate the derivative of kernel matrix over each hyperparameter,
/// and each gives a matrix,  so that the overall result is a vector of matrices.
///
/// For general kernels, the derivative over the square root of its weight gives
/// the square root times the kernel without any weight. For special cases
/// (like the Gaussian kernel) the derivative are calculated correspondingly.
static MatrixVector kernel_derivative_over_hyperparameters(const Eigen::MatrixXd& Feature, KernelList& Kernels)
{
	assert(Feature.rows() == PhaseDim);
	// construct the feature
	std::shared_ptr<shogun::DenseFeatures<double>> feature = std::make_shared<shogun::DenseFeatures<double>>(generate_shogun_matrix(Feature));
	const int NKernel = Kernels.size();
	MatrixVector result;
	for (int i = 0; i < NKernel; i++)
	{
		const double weight = Kernels[i].first;
		std::shared_ptr<shogun::Kernel>& kernel_ptr = Kernels[i].second;
		switch (kernel_ptr->get_kernel_type())
		{
		case shogun::EKernelType::K_DIAG:
		{
			kernel_ptr->init(feature, feature);
			// calculate derivative over weight
			result.push_back(weight * Eigen::Map<Eigen::MatrixXd>(kernel_ptr->get_kernel_matrix()));
			break;
		}
		case shogun::EKernelType::K_GAUSSIANARD:
		{
			kernel_ptr->init(feature, feature);
			// calculate derivative over weight
			result.push_back(weight * Eigen::Map<Eigen::MatrixXd>(kernel_ptr->get_kernel_matrix()));
			// calculate derivative over the characteristic matrix elements
			const std::pair<std::string, std::shared_ptr<shogun::AnyParameter>> Param = std::make_pair("log_weights", std::make_shared<shogun::AnyParameter>());
			const Eigen::Map<Eigen::MatrixXd>& Characteristic = std::dynamic_pointer_cast<shogun::GaussianARDKernel>(kernel_ptr)->get_weights();
#ifndef NOCROSS
			for (int k = 0, index = 0; k < PhaseDim; k++)
			{
				for (int j = k; j < PhaseDim; j++)
				{
					const Eigen::Map<Eigen::MatrixXd>& Deriv = kernel_ptr->get_parameter_gradient(Param, index);
					if (j == k)
					{
						result.push_back(weight * weight / Characteristic(j, k) * Deriv);
					}
					else
					{
						result.push_back(weight * weight * Deriv);
					}
					index++;
				}
			}
#else
			for (int j = 0; j < PhaseDim; j++)
			{
				const Eigen::Map<Eigen::MatrixXd>& Deriv = kernel_ptr->get_parameter_gradient(Param, j);
				result.push_back(weight * weight / Characteristic(j, j) * Deriv);
			}
#endif
			break;
		}
		default:
			std::cerr << "UNKNOWN KERNEL!\n";
			break;
		}
	}
	return result;
}

/// @brief Calculate the negative marginal likelihood, \f$ -\mathrm{ln}p(\mathbf{y}|X,\mathbf{\theta}) \f$
/// @param[in] x The vector containing all the hyperparameters
/// @param[out] grad The gradient of each hyperparameter
/// @param[in] params The pointer to a training set, including feature and label
/// @return The negative marginal likelihood
/// @details The formula of negative marginal likelihood is
///
/// \f[
/// -\mathrm{ln}{p(\mathbf{y}|X,\mathbf{\theta})}=\frac{1}{2}\mathbf{y}^\top K_y^{-1}\mathbf{y}+\frac{1}{2}\mathbf{ln}|K_y|+\frac{n}{2}\mathbf{ln}2\pi
/// \f]
///
/// where \f$ K_y \f$ is the kernel matrix of training set (may contain noise), \f$ \mathbf{y} \f$ is the training label,
/// and the last term \f$ \frac{n}{2}\mathbf{ln}2\pi \f$ is a constant during optimizing of hyperparameter. Overall, the return
/// of this function is \f$ \mathbf{y}^\top K_y^{-1}\mathbf{y}/2+\mathbf{ln}|K_y|/2 \f$, neglecting the last term. With the helf of
/// Cholesky decomposition (the kernel matrix is always positive-definite), the inverse and half log determinate could be done easily.
///
/// The formula of the derivative over hyperparameter is
///
/// \f{eqnarray*}{
/// -\frac{\partial}{\partial\theta}\mathrm{ln}{p(\mathbf{y}|X,\mathbf{\theta})}
/// 	&=&	-\frac{1}{2}\mathbf{y}^\top K_y^{-1}\frac{\partial K_y}{\partial\theta}K_y^{-1}\mathbf{y}
/// 		+\frac{1}{2}\mathrm{tr}\left(K_y^{-1}\frac{\partial K_y}{\partial\theta}\right) \\
/// 	&=& \frac{1}{2}\mathrm{tr}\left[\left(K_y^{-1}-\mathbf{b}\mathbf{b}^\top\right)\frac{\partial K_y}{\partial\theta}\right]
/// \f}
///
/// where \f$ \theta \f$ indicates one of the hyperparameters, \f$ \mathbf{b}=K_y^{-1}\mathbf{y} \f$ is a column vector. The negative
/// sign in front of the whole expression is due to this function being the negative log marginal likelihood.
///
/// The parameters of this function is quiet weird compared with other functions, as the requirement of nlopt interface.
static double negative_log_marginal_likelihood(const ParameterVector& x, ParameterVector& grad, void* params)
{
	// receive the parameter
	const ElementTrainingSet& Training = *static_cast<ElementTrainingSet*>(params);
	// get the parameters
	const Eigen::MatrixXd& TrainingFeature = std::get<0>(Training);
	const Eigen::VectorXd& TrainingLabel = std::get<1>(Training);
	const KernelTypeList& TypesOfKernels = std::get<2>(Training);
	KernelList Kernels = generate_kernels(TypesOfKernels, x);
	// get kernel and the derivatives of kernel over hyperparameters
	const Eigen::MatrixXd& KernelMatrix = get_kernel_matrix(TrainingFeature, TrainingFeature, Kernels, true);
	// calculate
	Eigen::LLT<Eigen::MatrixXd> DecompOfKernel(KernelMatrix);
	const Eigen::MatrixXd& L = DecompOfKernel.matrixL();
	const Eigen::MatrixXd& KInv = DecompOfKernel.solve(Eigen::MatrixXd::Identity(TrainingFeature.cols(), TrainingFeature.cols())); // inverse of kernel
	const Eigen::VectorXd& KInvLbl = DecompOfKernel.solve(TrainingLabel);														   // K^{-1}*y
	const double result = (TrainingLabel.adjoint() * KInvLbl).value() / 2.0 + L.diagonal().array().abs().log().sum();
	// print current result and combination
	std::clog << indent(2) << result << '\n';
	print_kernel(std::clog, TypesOfKernels, x, 3);
	if (grad.empty() == false)
	{
		// need gradient information, calculated here
		const MatrixVector& KernelDerivative = kernel_derivative_over_hyperparameters(TrainingFeature, Kernels);
		for (int i = 0; i < x.size(); i++)
		{
			grad[i] = ((KInv - KInvLbl * KInvLbl.adjoint()) * KernelDerivative[i]).trace() / 2.0;
		}
		// print current gradient if uses gradient
		print_kernel(std::clog, TypesOfKernels, grad, 4);
	}
	std::clog << std::endl;
	return result;
}

/// @details This function using the NLOPT to optimize the hyperparameters and return them.
ParameterVector optimize(
	const SuperMatrix& TrainingFeatures,
	const VectorMatrix& TrainingLabels,
	const KernelTypeList& TypesOfKernels,
	const QuantumMatrixBool& IsSmall,
	const nlopt::algorithm& ALGO,
	const ParameterVector& LowerBound,
	const ParameterVector& UpperBound,
	const ParameterVector& InitialHyperparameters)
{
	const int NKernel = TypesOfKernels.size();
	// constructor the NLOPT minimizer using Nelder-Mead simplex algorithm
	static const int NoVar = number_of_overall_hyperparameters(TypesOfKernels); // the number of variables to optimize
	double sum_marg_ll = 0.0;													// summation of all the likelihood
	ParameterVector result;
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		for (int jPES = 0; jPES < NumPES; jPES++)
		{
			// the position where the component of this element in hyperparameter/bounds begins
			const int BeginIndex = (iPES * NumPES + jPES) * NoVar;
			// the hyperparameters used in this element
			ParameterVector hyperparameters(InitialHyperparameters.cbegin() + BeginIndex, InitialHyperparameters.cbegin() + BeginIndex + NoVar);
			// the marginal likelihood of this element
			double marg_ll = 0.0;
			// check. if not small, need optimizing
			std::clog << indent(1);
			if (iPES == jPES)
			{
				std::clog << "rho[" << iPES << "][" << jPES << ']';
			}
			else if (iPES < jPES)
			{
				std::clog << "Re(rho[" << iPES << "][" << jPES << "])";
			}
			else // if (i > j)
			{
				std::clog << "Im(rho[" << iPES << "][" << jPES << "])";
			}
			std::clog << ": \n";
			if (IsSmall(iPES, jPES) == true)
			{
				// everywhere is almost 0, so no optimization needed
				std::clog << indent(2) << "0 everywhere\n";
			}
			else
			{
				// need optimization
				nlopt::opt minimizer(ALGO, NoVar);
				// set minimizer function
				const nlopt::vfunc minimizing_function = negative_log_marginal_likelihood;
				ElementTrainingSet this_element_training_set = std::make_tuple(TrainingFeatures[iPES][jPES], TrainingLabels[iPES][jPES], TypesOfKernels);
				minimizer.set_min_objective(minimizing_function, &this_element_training_set);
				// set bounds for noise and characteristic length, as well as for initial values
				minimizer.set_lower_bounds(ParameterVector(LowerBound.cbegin() + BeginIndex, LowerBound.cbegin() + BeginIndex + NoVar));
				minimizer.set_upper_bounds(ParameterVector(UpperBound.cbegin() + BeginIndex, UpperBound.cbegin() + BeginIndex + NoVar));
				// set stop criteria
				static const double AbsTol = 1e-10; // tolerance in minimization
				minimizer.set_xtol_abs(AbsTol);
				// set initial step size
				static const double InitStepSize = 0.5; // initial step size
				minimizer.set_initial_step(InitStepSize);

				// optimize
				std::clog << indent(1) << "Begin Optimization" << std::endl;
				try
				{
					nlopt::result result = minimizer.optimize(hyperparameters, marg_ll);
					std::clog << indent(1);
					switch (result)
					{
					case nlopt::result::SUCCESS:
						std::clog << "Successfully stopped";
						break;
					case nlopt::result::STOPVAL_REACHED:
						std::clog << "Stopping value reached";
						break;
					case nlopt::result::FTOL_REACHED:
						std::clog << "Function value tolerance reached";
						break;
					case nlopt::result::XTOL_REACHED:
						std::clog << "Step size tolerance reached";
						break;
					case nlopt::result::MAXEVAL_REACHED:
						std::clog << "Maximum evaluation time reached";
						break;
					case nlopt::result::MAXTIME_REACHED:
						std::clog << "Maximum cpu time reached";
						break;
					}
					std::clog << '\n';
				}
				catch (const std::exception& e)
				{
					std::cerr << indent(1) << "NLOPT optimization failed for " << e.what() << '\n';
				}
				std::clog << indent(1) << "Best Combination is\n";
				ParameterVector grad;
				marg_ll = minimizing_function(hyperparameters, grad, &this_element_training_set);
			}
			// insert the optimized hyperparameter, and add the marginal likelihood of this element
			result.insert(result.cend(), hyperparameters.cbegin(), hyperparameters.cend());
			sum_marg_ll += marg_ll;
		}
	}

	result.push_back(sum_marg_ll);
	return result;
}

/// @details This function follows the formula
///
/// \f[
/// E[p(\mathbf{f}_*|X_*,X,\mathbf{y})]=K(X_*,X)[K(X,X)+\sigma_n^2I]^{-1}\mathbf{y}
/// \f]
///
/// where \f$ X_* \f$ is the test features, \f$ X \f$ is the training features, and  \f$ \mathbf{y} \f$ is the training labels.
///
/// Besides, it needs normalization: the trace should be 1 and the energy should conserve.
void predict_phase(
	SuperMatrix& PredictedDistribution,
	const SuperMatrix& TrainingFeatures,
	const VectorMatrix& TrainingLabels,
	const KernelTypeList& TypesOfKernels,
	const QuantumMatrixBool& IsSmall,
	const Eigen::VectorXd& x,
	const Eigen::VectorXd& p,
	const ParameterVector& Hyperparameters)
{
	const int nx = x.size(), np = p.size();
	const int NoVar = number_of_overall_hyperparameters(TypesOfKernels);
	// predict one by one, with some pre-calculation needed for normalization
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		for (int jPES = 0; jPES < NumPES; jPES++)
		{
			if (IsSmall(iPES, jPES) == true)
			{
				// very small, 0 everywhere
				for (int iGrid = 0; iGrid < nx; iGrid++)
				{
					for (int jGrid = 0; jGrid < np; jGrid++)
					{
						PredictedDistribution[iPES][jPES].setZero();
					}
				}
			}
			else
			{
				// the position where the component of this element in hyperparameter/bounds begins
				const int BeginIndex = (iPES * NumPES + jPES) * NoVar;
				// get hyperparameter and label for this element of PWTDM
				const ParameterVector& HyperparameterHere = ParameterVector(Hyperparameters.cbegin() + BeginIndex, Hyperparameters.cbegin() + BeginIndex + NoVar);
				const Eigen::MatrixXd& TrainingFeatureOfThisElement = TrainingFeatures[iPES][jPES];
				const Eigen::VectorXd& TrainingLabelOfThisElement = TrainingLabels[iPES][jPES];
				KernelList KernelsOfThisElement = generate_kernels(TypesOfKernels, HyperparameterHere);
				const Eigen::MatrixXd& KernelMatrix = get_kernel_matrix(TrainingFeatureOfThisElement, TrainingFeatureOfThisElement, KernelsOfThisElement, true);
				const Eigen::VectorXd& KInvLbl = KernelMatrix.llt().solve(TrainingLabelOfThisElement);
				Eigen::VectorXd coord(PhaseDim);
				for (int iGrid = 0; iGrid < nx; iGrid++)
				{
					for (int jGrid = 0; jGrid < np; jGrid++)
					{
						coord[0] = x[iGrid];
						coord[1] = p[jGrid];
						PredictedDistribution[iPES][jPES](iGrid, jGrid) = (get_kernel_matrix(coord, TrainingFeatureOfThisElement, KernelsOfThisElement) * KInvLbl).value();
					}
				}
			}
		}
	}
}

/// @details To calculate \f$ <\rho_{ii}> \f$,
///
/// \f[
/// <\rho_{ii}>=(2\pi)^{\frac{d}{2}}(\sigma_f^{ii})^2\left|\Lambda^{ii}\right|(1\ 1\ \dots\ 1)\qty(K^{ii})^{-1}\vb{y}^{ii}
/// \f]
///
/// where the \f$ \Lambda \f$ matrix is the lower-triangular matrix used in Gaussian ARD kernel
QuantumVectorDouble calculate_population_from_gpr(
	const SuperMatrix& TrainingFeatures,
	const VectorMatrix& TrainingLabels,
	const KernelTypeList& TypesOfKernels,
	const QuantumMatrixBool& IsSmall,
	const ParameterVector& Hyperparameters)
{
	const int NKernel = TypesOfKernels.size();
	const int NoVar = number_of_overall_hyperparameters(TypesOfKernels);
	QuantumVectorDouble result = QuantumVectorDouble::Zero();
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		if (IsSmall(iPES, iPES) == false)
		{
			const int BeginIndex = (iPES * NumPES + iPES) * NoVar;
			// get K^{-1}*y
			const ParameterVector& HyperparameterHere = ParameterVector(Hyperparameters.cbegin() + BeginIndex, Hyperparameters.cbegin() + BeginIndex + NoVar);
			KernelList KernelsOfThisElement = generate_kernels(TypesOfKernels, HyperparameterHere);
			const Eigen::MatrixXd& TrainingFeatureOfThisElement = TrainingFeatures[iPES][iPES];
			const Eigen::VectorXd& TrainingLabelOfThisElement = TrainingLabels[iPES][iPES];
			const Eigen::MatrixXd& KernelMatrix = get_kernel_matrix(TrainingFeatureOfThisElement, TrainingFeatureOfThisElement, KernelsOfThisElement, true);
			const Eigen::VectorXd& KInvLbl = KernelMatrix.llt().solve(TrainingLabelOfThisElement);
			// get coefficient, (2*pi)^{D/2}\sigma_f^{ii})^2|\Lambda^{ii}|
			double coe = 0.0;
			for (int i = 0; i < NKernel; i++)
			{
				const double weight = KernelsOfThisElement[i].first;
				std::shared_ptr<shogun::Kernel>& kernel_ptr = KernelsOfThisElement[i].second;
				switch (kernel_ptr->get_kernel_type())
				{
				case shogun::EKernelType::K_DIAG:
					break;
				case shogun::EKernelType::K_GAUSSIANARD:
				{
					const Eigen::Map<Eigen::MatrixXd>& Characteristic = std::dynamic_pointer_cast<shogun::GaussianARDKernel>(kernel_ptr)->get_weights();
					coe += std::pow(2.0 * pi, Dim) * weight * weight / Characteristic.diagonal().prod();
					break;
				}
				default:
					std::cerr << "UNKNOWN KERNEL!\n";
					break;
				}
			}
			result[iPES] = coe * KInvLbl.sum();
		}
	}
	return result;
}

/// @details Potential energy could be calculated by numerical integration
QuantumVectorDouble calculate_potential_energy_from_gpr(
	const SuperMatrix& TrainingFeatures,
	const VectorMatrix& TrainingLabels,
	const KernelTypeList& TypesOfKernels,
	const QuantumMatrixBool& IsSmall,
	const ParameterVector& Hyperparameters)
{
	static const double numerical_integration_initial_stepsize = 1e-2;
	static const double epsilon = std::exp(-12.5) / std::sqrt(2 * pi); // value below this is 0
	const int NKernel = TypesOfKernels.size();
	const int NoVar = number_of_overall_hyperparameters(TypesOfKernels);
	QuantumVectorDouble result = QuantumVectorDouble::Zero();
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		if (IsSmall(iPES, iPES) == false)
		{
			const int BeginIndex = (iPES * NumPES + iPES) * NoVar;
			// get K^{-1}*y
			const ParameterVector& HyperparameterHere = ParameterVector(Hyperparameters.cbegin() + BeginIndex, Hyperparameters.cbegin() + BeginIndex + NoVar);
			KernelList KernelsOfThisElement = generate_kernels(TypesOfKernels, HyperparameterHere);
			const Eigen::MatrixXd& TrainingFeatureOfThisElement = TrainingFeatures[iPES][iPES];
			const Eigen::VectorXd& TrainingLabelOfThisElement = TrainingLabels[iPES][iPES];
			const Eigen::MatrixXd& KernelMatrix = get_kernel_matrix(TrainingFeatureOfThisElement, TrainingFeatureOfThisElement, KernelsOfThisElement, true);
			const Eigen::VectorXd& KInvLbl = KernelMatrix.llt().solve(TrainingLabelOfThisElement);
			// calculate potential energy
			boost::numeric::odeint::bulirsch_stoer<double> stepper; // stepper for adaptive step control
			auto potential_times_population = [&](double x0) -> double {
				double population_given_x = 0.0;
				// analytical integration
				for (int i = 0; i < NKernel; i++)
				{
					const double weight = KernelsOfThisElement[i].first;
					std::shared_ptr<shogun::Kernel> kernel_ptr = KernelsOfThisElement[i].second;
					if (kernel_ptr->get_kernel_type() == shogun::EKernelType::K_GAUSSIANARD)
					{
#ifndef NOCROSS
						// vec_i=sigma_f^2*sqrt(2*pi/(A01^2+A11^2))exp(-A00^2(x-xi)^2/2-A00^2A01^2(x-xi)^2/2/(A01^2+A11^2))
						// rho(x)=vec*K^{-1}*y
						const Eigen::Map<Eigen::MatrixXd>& Characteristic = std::dynamic_pointer_cast<shogun::GaussianARDKernel>(kernel_ptr)->get_weights();
						const Eigen::MatrixXd Char2 = Characteristic.array().abs2();
						const Eigen::VectorXd IntegrateOverP = weight * weight * std::sqrt(2.0 * pi / (Char2(0, 1) + Char2(1, 1)))
															   * (-(x0 - TrainingFeatureOfThisElement.row(0).array()).abs2() * (Char2(0, 0) / 2.0 * (1.0 + Char2(0, 1) / (Char2(0, 1) + Char2(1, 1))))).exp();
#else
						// vec_i=sigma_f^2*sqrt(2*pi)*lp*exp(-(x-xi)^2/2/lx^2)
						// rho(x)=vec*K^{-1}*y
						const Eigen::VectorXd Characteristic = Eigen::Map<Eigen::MatrixXd>(std::dynamic_pointer_cast<shogun::GaussianARDKernel>(kernel_ptr)->get_weights()).diagonal().array();
						const Eigen::VectorXd IntegrateOverP = weight * weight * std::sqrt(2.0 * pi) / Characteristic[1] * (-(x0 - TrainingFeatureOfThisElement.row(0).array()).abs2() * (Characteristic[0] * Characteristic[0] / 2.0)).exp();
#endif
						population_given_x = IntegrateOverP.dot(KInvLbl);
						break;
					}
				}
				return adiabatic_potential(x0)(iPES, iPES) * population_given_x;
			};

			// int_R{f(x)dx}=int_0^1{dt[f((1-t)/t)+f((t-1)/t)]t^2} by x=(1-t)/t
			boost::numeric::odeint::integrate_adaptive(
				stepper,
				[&](const double& /*f*/, double& fx, double t) -> void {
					if (std::abs(t) < epsilon)
					{
						fx = 0.0;
					}
					else
					{
						const double x = (1.0 - t) / t;
						fx = (potential_times_population(x) + potential_times_population(-x)) / (t * t);
					}
				},
				result[iPES],
				0.0,
				1.0,
				numerical_integration_initial_stepsize);
		}
	}
	return result;
}

/// @details For potential energy, calculate from integration.
/// For kinetic energy, the integral could be done analytically:
///
/// \f{eqnarray*}{
/// <p^2\rho_{ii}>&=&(2\pi)^{\frac{d}{2}}(\sigma_f^{ii})^2\left|\Lambda^{ii}\right|[(1\ 1\ \dots\ 1)(\Lambda^{ii}(\Lambda^{ii})^\top)^{-1}_{pp} \\
/// &&+(p_1^2\ p_2^2\ \dots\ p_n^2)]\qty(K^{ii})^{-1}\vb{y}^{ii}
/// \f}
///
/// where the \f$ \Lambda \f$ matrix is the lower-triangular matrix used in Gaussian ARD kernel,
/// and the p in row vector is the momentum of the training set.
QuantumVectorDouble calculate_kinetic_energy_from_gpr(
	const SuperMatrix& TrainingFeatures,
	const VectorMatrix& TrainingLabels,
	const KernelTypeList& TypesOfKernels,
	const QuantumMatrixBool& IsSmall,
	const ParameterVector& Hyperparameters)
{
	const int NKernel = TypesOfKernels.size();
	const int NoVar = number_of_overall_hyperparameters(TypesOfKernels);
	QuantumVectorDouble result = QuantumVectorDouble::Zero();
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		if (IsSmall(iPES, iPES) == false)
		{
			const int BeginIndex = (iPES * NumPES + iPES) * NoVar;
			// get K^{-1}*y
			const ParameterVector& HyperparameterHere = ParameterVector(Hyperparameters.cbegin() + BeginIndex, Hyperparameters.cbegin() + BeginIndex + NoVar);
			KernelList KernelsOfThisElement = generate_kernels(TypesOfKernels, HyperparameterHere);
			const Eigen::MatrixXd& TrainingFeatureOfThisElement = TrainingFeatures[iPES][iPES];
			const Eigen::VectorXd& TrainingLabelOfThisElement = TrainingLabels[iPES][iPES];
			const Eigen::MatrixXd& KernelMatrix = get_kernel_matrix(TrainingFeatureOfThisElement, TrainingFeatureOfThisElement, KernelsOfThisElement, true);
			const Eigen::VectorXd& KInvLbl = KernelMatrix.llt().solve(TrainingLabelOfThisElement);

			// calculate kinetic energy
			Eigen::VectorXd row_vector(TrainingFeatures[iPES][iPES].cols());
			for (int iTrain = 0; iTrain < TrainingFeatures[iPES][iPES].cols(); iTrain++)
			{
				row_vector[iTrain] = TrainingFeatures[iPES][iPES](1, iTrain) * TrainingFeatures[iPES][iPES](1, iTrain);
			}
			double coe = 0.0;
			for (int i = 0; i < NKernel; i++)
			{
				const double weight = KernelsOfThisElement[i].first;
				std::shared_ptr<shogun::Kernel>& kernel_ptr = KernelsOfThisElement[i].second;
				switch (kernel_ptr->get_kernel_type())
				{
				case shogun::EKernelType::K_DIAG:
					break;
				case shogun::EKernelType::K_GAUSSIANARD:
				{
					const Eigen::Map<Eigen::MatrixXd>& Characteristic = std::dynamic_pointer_cast<shogun::GaussianARDKernel>(kernel_ptr)->get_weights();
					coe += std::pow(2.0 * pi, Dim) * weight * weight / Characteristic.diagonal().prod();
#ifndef NOCROSS
					row_vector.array() += Characteristic.transpose().triangularView<Eigen::Upper>().solve(Characteristic.triangularView<Eigen::Lower>().solve(Eigen::MatrixXd::Identity(PhaseDim, PhaseDim)))(1, 1);
#else
					row_vector.array() += std::pow(Characteristic(1, 1), -2);
#endif
					break;
				}
				default:
					std::cerr << "UNKNOWN KERNEL!\n";
					break;
				}
			}
			result[iPES] += coe * row_vector.dot(KInvLbl) / 2.0 / Mass;
		}
	}
	return result;
}

void obey_conservation(
	SuperMatrix& PredictedDistribution,
	const SuperMatrix& TrainingFeatures,
	VectorMatrix& TrainingLabels,
	const KernelTypeList& TypesOfKernels,
	const QuantumMatrixBool& IsSmall,
	const double InitialTotalEnergy,
	const Eigen::VectorXd& x,
	const Eigen::VectorXd& p,
	const ParameterVector& Hyperparameters)
{
	const int NoVar = number_of_overall_hyperparameters(TypesOfKernels);
	const int nx = x.size(), np = p.size();
	// what are the non-zero diagonal elements
	std::vector<int> NonZeroDiagonalElements;
	for (int i = 0; i < NumPES; i++)
	{
		if (IsSmall(i, i) == false)
		{
			NonZeroDiagonalElements.push_back(i);
		}
	}

	const QuantumVectorDouble& Population = calculate_population_from_gpr(TrainingFeatures, TrainingLabels, TypesOfKernels, IsSmall, Hyperparameters);
	const QuantumVectorDouble& Energy = calculate_potential_energy_from_gpr(TrainingFeatures, TrainingLabels, TypesOfKernels, IsSmall, Hyperparameters) + calculate_kinetic_energy_from_gpr(TrainingFeatures, TrainingLabels, TypesOfKernels, IsSmall, Hyperparameters);

	// normalize
	if (NonZeroDiagonalElements.size() == 1)
	{
		// if there is only one, only do normalization, without energy conservation
		const int PESIndex = NonZeroDiagonalElements[0];
		const double NormalizationFactor = 1.0 / Population[PESIndex];
		for (int iGrid = 0; iGrid < nx; iGrid++)
		{
			for (int jGrid = 0; jGrid < np; jGrid++)
			{
				PredictedDistribution[PESIndex][PESIndex](iGrid, jGrid) *= NormalizationFactor;
			}
		}
		for (int iPoint = 0; iPoint < TrainingLabels[PESIndex][PESIndex].size(); iPoint++)
		{
			TrainingLabels[PESIndex][PESIndex][iPoint] *= NormalizationFactor;
		}
	}
	else
	{
		const int NoNonZeroDiagonalElements = NonZeroDiagonalElements.size();
		// more than 1, first half using an altogether coefficient,
		// the vector to be solved
		Eigen::Vector2d Conservations;
		// row 0: population conservation; row 1: energy conservation
		Conservations << 1.0, InitialTotalEnergy;
		Eigen::Matrix2d Coefficients = Eigen::Matrix2d::Zero();
		for (int i = 0; i < NoNonZeroDiagonalElements; i++)
		{
			const int PESIndex = NonZeroDiagonalElements[i];
			// first half, add to column 0; last half, add to column 1
			Coefficients(0, i < NoNonZeroDiagonalElements / 2 ? 0 : 1) += Population[PESIndex];
			Coefficients(1, i < NoNonZeroDiagonalElements / 2 ? 0 : 1) += Energy[PESIndex];
		}
		const Eigen::Vector2d& ConservationFactor = Coefficients.fullPivLu().solve(Conservations);
		for (int i = 0; i < NoNonZeroDiagonalElements; i++)
		{
			const int PESIndex = NonZeroDiagonalElements[i];
			// first half, using factor[0]
			// last half, using factor[1]
			for (int iGrid = 0; iGrid < nx; iGrid++)
			{
				for (int jGrid = 0; jGrid < np; jGrid++)
				{
					PredictedDistribution[PESIndex][PESIndex](iGrid, jGrid) *= ConservationFactor[i < NoNonZeroDiagonalElements / 2 ? 0 : 1];
				}
			}
			for (int iPoint = 0; iPoint < TrainingLabels[PESIndex][PESIndex].size(); iPoint++)
			{
				TrainingLabels[PESIndex][PESIndex][iPoint] *= ConservationFactor[i < NoNonZeroDiagonalElements / 2 ? 0 : 1];
			}
		}
	}
}

QuantumMatrixDouble mean_squared_error(const SuperMatrix& lhs, const SuperMatrix& rhs)
{
	QuantumMatrixDouble result = QuantumMatrixDouble::Zero();
	for (int i = 0; i < NumPES; i++)
	{
		for (int j = 0; j < NumPES; j++)
		{
			result(i, j) = (lhs[i][j] - rhs[i][j]).array().abs2().sum();
		}
	}
	return result;
}
