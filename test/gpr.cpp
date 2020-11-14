/// @file gpr.cpp
/// @brief Definition of Gaussian Process Regression functions

#include "gpr.h"

#include "io.h"

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

double calculate_energy_from_grid(const SuperMatrix& AdiabaticDistribution, const Eigen::VectorXd& x, const Eigen::VectorXd& p)
{
	const int nx = x.size(), np = p.size();
	const double dx = (x[nx - 1] - x[0]) / nx, dp = (p[np - 1] - p[0]) / np;
	double result = 0.0;
	// add potential energy
	for (int i = 0; i < nx; i++)
	{
		const QuantumMatrixDouble& AdiabaticPotential = adiabatic_potential(x[i]);
		for (int j = 0; j < NumPES; j++)
		{
			result += AdiabaticDistribution[j][j].row(i).sum() * AdiabaticPotential(j, j);
		}
	}
	// add kinetic energy
	for (int i = 0; i < np; i++)
	{
		const double KineticEnergy = p[i] * p[i] / 2.0 / Mass;
		for (int j = 0; j < NumPES; j++)
		{
			result += AdiabaticDistribution[j][j].col(i).sum() * KineticEnergy;
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
			sum += 0; // no extra hyperparameter for diagonal kernel
			break;
		case shogun::EKernelType::K_GAUSSIANARD:
			sum += PhaseDim * (PhaseDim + 1) / 2; // extra hyperparameter from relevance and characteristic lengths
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
					for (int k = l + 1; k < PhaseDim; k++)
					{
						iparam++;
					}
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

/// @brief Calculate the weight at a given point
/// @param[in] data The PWTDM at the given point
/// @return The weight at the given (RowIndex, ColIndex) point
static double weight_function(const QuantumMatrixDouble& data)
{
	// construct the PWTDM at the given point
	assert(data.rows() == NumPES && data.cols() == NumPES);
	double result = 0.0;
	for (int i = 0; i < NumPES; i++)
	{
		result += std::abs(data(i, i));
		for (int j = 0; j < i; j++)
		{
			result += std::abs(std::complex<double>(data(j, i), data(i, j)));
		}
	}
	// sum up the absolute value of the lower-triangular.
	return result;
}

/// @brief Choose npoint points based on their weight using MC. If the point has already chosen, redo
/// @param[in] data The gridded whole PWTDM
/// @return A set containing the pointer to the coordinate table (x[first]=xi, p[second]=pi)
/// @details This function check if the values in the input matrix are all small or not.
/// If all very small, then select points randomly without weight.
/// Otherwise, select based on the weights at the point by MC procedure.
static PointSet choose_point(const SuperMatrix& data)
{
	static std::mt19937_64 generator(std::chrono::system_clock::now().time_since_epoch().count()); // random number generator, 64 bits Mersenne twister engine
	PointSet result;
	const int nx = data[0][0].rows(), np = data[0][0].cols(), n = data[0][0].size();
	// construct the weight matrix
	Eigen::MatrixXd weight_matrix(nx, np);
	for (int iGrid = 0; iGrid < nx; iGrid++)
	{
		for (int jGrid = 0; jGrid < np; jGrid++)
		{
			QuantumMatrixDouble DensityOnGrid;
			for (int iPES = 0; iPES < NumPES; iPES++)
			{
				for (int jPES = 0; jPES < NumPES; jPES++)
				{
					DensityOnGrid(iPES, jPES) = data[iPES][jPES](iGrid, jGrid);
				}
			}
			weight_matrix(iGrid, jGrid) = weight_function(DensityOnGrid);
		}
	}
	const double weight = std::accumulate(weight_matrix.data(), weight_matrix.data() + n, 0.0);
	std::uniform_real_distribution<double> urd(0.0, weight);
	while (result.size() < NPoint)
	{
		double acc_weight = urd(generator);
		for (int j = 0; j < nx; j++)
		{
			for (int k = 0; k < np; k++)
			{
				acc_weight -= weight_matrix(j, k);
				if (acc_weight < 0)
				{
					result.insert(std::make_pair(j, k));
					goto next;
				}
			}
		}
	next:;
	}
	/*
	while (result.size() < NPoint)
	{
		int idx_x, idx_p;
		weight_matrix.maxCoeff(&idx_x, &idx_p);
		weight_matrix(idx_x, idx_p) = 0;
		result.insert(std::make_pair(idx_x, idx_p));
	}
*/
	return result;
}

FullTrainingSet generate_training_set(
	const SuperMatrix& data,
	const Eigen::VectorXd& x,
	const Eigen::VectorXd& p)
{
	const PointSet& chosen = choose_point(data);
	Eigen::MatrixXd training_feature(PhaseDim, chosen.size());
	VectorMatrix training_label;
	for (int i = 0; i < NumPES; i++)
	{
		for (int j = 0; j < NumPES; j++)
		{
			training_label[i][j].resize(chosen.size());
		}
	}
	int Index = 0;
	for (PointSet::iterator iter = chosen.begin(); iter != chosen.end(); ++iter, Index++)
	{
		training_feature(0, Index) = x[iter->first];
		training_feature(1, Index) = p[iter->second];
		for (int i = 0; i < NumPES; i++)
		{
			for (int j = 0; j < NumPES; j++)
			{
				training_label[i][j][Index] = data[i][j](iter->first, iter->second);
			}
		}
	}
	return std::make_pair(training_feature, training_label);
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

/// @brief Deep copy of Eigen matrix to shogun matrix
/// @param[in] mat Eigen matrix
/// @return shogun matrix of same content
static shogun::SGMatrix<double> generate_shogun_matrix(const Eigen::MatrixXd& mat)
{
	shogun::SGMatrix<double> result(mat.rows(), mat.cols());
	std::copy(mat.data(), mat.data() + mat.size(), result.data());
	return result;
}

/// @brief Calculate the kernel matrix, \f$ K(X_1,X_2) \f$
/// @param[in] LeftFeature The feature on the left, \f$ X_1 \f$
/// @param[in] RightFeature The feature on the right, \f$ X_2 \f$
/// @param[in] TypeOfKernel The vector containing type of all kernels that will be used
/// @param[in] Hyperparameters The hyperparameters of all kernels (magnitude and other)
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
	const KernelTypeList& TypesOfKernels,
	const ParameterVector& Hyperparameters,
	const bool IsTraining = false)
{
	assert(LeftFeature.rows() == PhaseDim && RightFeature.rows() == PhaseDim);
	assert(Hyperparameters.size() == number_of_overall_hyperparameters(TypesOfKernels));
	// construct the feature
	shogun::Some<shogun::CDenseFeatures<double>> left_feature = shogun::some<shogun::CDenseFeatures<double>>(generate_shogun_matrix(LeftFeature));
	shogun::Some<shogun::CDenseFeatures<double>> right_feature = shogun::some<shogun::CDenseFeatures<double>>(generate_shogun_matrix(RightFeature));
	const int NKernel = TypesOfKernels.size();
	Eigen::MatrixXd result = Eigen::MatrixXd::Zero(LeftFeature.cols(), RightFeature.cols());
	for (int i = 0, iparam = 0; i < NKernel; i++)
	{
		const double weight = Hyperparameters[iparam++];
		std::shared_ptr<shogun::CKernel> kernel_ptr;
		switch (TypesOfKernels[i])
		{
		case shogun::EKernelType::K_DIAG:
			if (IsTraining == true)
			{
				kernel_ptr = std::make_shared<shogun::CDiagKernel>();
			}
			else
			{
				kernel_ptr = std::make_shared<shogun::CConstKernel>(0.0);
			}
			break;
		case shogun::EKernelType::K_GAUSSIANARD:
		{
			Eigen::MatrixXd characteristic = Eigen::MatrixXd::Zero(PhaseDim, PhaseDim);
			for (int k = 0; k < PhaseDim; k++)
			{
				for (int j = k; j < PhaseDim; j++)
				{
					characteristic(j, k) = Hyperparameters[iparam++];
				}
			}
			std::shared_ptr<shogun::CGaussianARDKernel> gauss_ard_kernel_ptr = std::make_shared<shogun::CGaussianARDKernel>();
			gauss_ard_kernel_ptr->set_matrix_weights(characteristic);
			kernel_ptr = gauss_ard_kernel_ptr;
			break;
		}
		default:
			std::cerr << "UNKNOWN KERNEL!\n";
			break;
		}
		kernel_ptr->init(left_feature, right_feature);
		result += weight * weight * static_cast<shogun::SGMatrix<double>::EigenMatrixXtMap>(kernel_ptr->get_kernel_matrix());
	}
	return result;
}

/// @brief Calculate the derivative of kernel matrix over hyperparameters
/// @param[in] Feature The left and right feature of the kernel
/// @param[in] TypeOfKernel The vector containing type of all kernels that will be used
/// @param[in] Hyperparameters The hyperparameters of all kernels (magnitude and other)
/// @return A vector of matrices being the derivative of kernel matrix over hyperparameters, in the same order as Hyperparameters
/// @details This function calculate the derivative of kernel matrix over each hyperparameter,
/// and each gives a matrix,  so that the overall result is a vector of matrices.
///
/// For general kernels, the derivative over the square root of its weight gives
/// the square root times the kernel without any weight. For special cases
/// (like the Gaussian kernel) the derivative are calculated correspondingly.
static MatrixVector kernel_derivative_over_hyperparameters(
	const Eigen::MatrixXd& Feature,
	const KernelTypeList& TypesOfKernels,
	const ParameterVector& Hyperparameters)
{
	assert(Feature.rows() == PhaseDim);
	assert(Hyperparameters.size() == number_of_overall_hyperparameters(TypesOfKernels));
	// construct the feature
	shogun::Some<shogun::CDenseFeatures<double>> feature = shogun::some<shogun::CDenseFeatures<double>>(generate_shogun_matrix(Feature));
	const int NKernel = TypesOfKernels.size();
	MatrixVector result;
	for (int i = 0, iparam = 0; i < NKernel; i++)
	{
		const double weight = Hyperparameters[iparam++];
		switch (TypesOfKernels[i])
		{
		case shogun::EKernelType::K_DIAG:
		{
			std::shared_ptr<shogun::CDiagKernel> diag_kernel_ptr = std::make_shared<shogun::CDiagKernel>();
			diag_kernel_ptr->init(feature, feature);
			// calculate derivative over weight
			result.push_back(weight * shogun::SGMatrix<double>::EigenMatrixXtMap(diag_kernel_ptr->get_kernel_matrix()));
			break;
		}
		case shogun::EKernelType::K_GAUSSIANARD:
		{
			Eigen::MatrixXd characteristic = Eigen::MatrixXd::Zero(PhaseDim, PhaseDim);
			for (int k = 0; k < PhaseDim; k++)
			{
				for (int j = k; j < PhaseDim; j++)
				{
					characteristic(j, k) = Hyperparameters[iparam++];
				}
			}
			std::shared_ptr<shogun::CGaussianARDKernel> gauss_ard_kernel_ptr = std::make_shared<shogun::CGaussianARDKernel>();
			gauss_ard_kernel_ptr->set_matrix_weights(characteristic);
			gauss_ard_kernel_ptr->init(feature, feature);
			// calculate derivative over weight
			result.push_back(weight * shogun::SGMatrix<double>::EigenMatrixXtMap(gauss_ard_kernel_ptr->get_kernel_matrix()));
			// calculate derivative over the characteristic matrix elements
			const std::shared_ptr<shogun::TSGDataType> tsgdt_ptr = std::make_shared<shogun::TSGDataType>(shogun::EContainerType::CT_SGMATRIX, shogun::EStructType::ST_NONE, shogun::EPrimitiveType::PT_FLOAT64);
			const std::shared_ptr<shogun::TParameter> tp_ptr = std::make_shared<shogun::TParameter>(tsgdt_ptr.get(), nullptr, "log_weights", nullptr);
			for (int k = 0, index = 0; k < PhaseDim; k++)
			{
				for (int j = k; j < PhaseDim; j++)
				{
					const Eigen::Map<Eigen::MatrixXd>& Deriv = gauss_ard_kernel_ptr->get_parameter_gradient(tp_ptr.get(), index);
					if (j == k)
					{
						result.push_back(weight * weight / characteristic(j, k) * Deriv);
					}
					else
					{
						result.push_back(weight * weight * Deriv);
					}
					index++;
				}
			}
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
	const int NoVar = number_of_overall_hyperparameters(TypesOfKernels);
	// get kernel and the derivatives of kernel over hyperparameters
	const Eigen::MatrixXd& KernelMatrix = get_kernel_matrix(TrainingFeature, TrainingFeature, TypesOfKernels, x, true);
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
		const MatrixVector& KernelDerivative = kernel_derivative_over_hyperparameters(TrainingFeature, TypesOfKernels, x);
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
	const Eigen::MatrixXd& TrainingFeatures,
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
				ElementTrainingSet this_element_training_set
					= std::make_tuple(TrainingFeatures, TrainingLabels[iPES][jPES], TypesOfKernels);
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
	const Eigen::MatrixXd& TrainingFeatures,
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
				const Eigen::VectorXd& TrainingLabelOfThisElement = TrainingLabels[iPES][jPES];
				const Eigen::MatrixXd& KernelMatrix = get_kernel_matrix(TrainingFeatures, TrainingFeatures, TypesOfKernels, HyperparameterHere, true);
				const Eigen::VectorXd& Coe = KernelMatrix.llt().solve(TrainingLabelOfThisElement);
				Eigen::VectorXd coord(PhaseDim);
				for (int iGrid = 0; iGrid < nx; iGrid++)
				{
					for (int jGrid = 0; jGrid < np; jGrid++)
					{
						coord[0] = x[iGrid];
						coord[1] = p[jGrid];
						PredictedDistribution[iPES][jPES](iGrid, jGrid) = (get_kernel_matrix(coord, TrainingFeatures, TypesOfKernels, HyperparameterHere) * Coe).value();
					}
				}
			}
		}
	}
}

/// @brief Calculate the normalization factor, or \f$ <\rho_{ii}> \f$
/// @param[in] TypesOfKernels The vector containing all the kernel type used in optimization
/// @param[in] Hyperparameters The hyperparameters of the given diagonal element in phase space distribution
/// @param[in] KInvLbl The inverse of the kernel matrix times the labels
/// @return \f$ <\rho_{ii}> \f$
/// @details To calculate \f$ <\rho_{ii}> \f$,
///
/// \f[
/// <\rho_{ii}>=(2\pi)^{\frac{d}{2}}(\sigma_f^{ii})^2\left|\Lambda^{ii}\right|(1\ 1\ \dots\ 1)\qty(K^{ii})^{-1}\vb{y}^{ii}
/// \f]
///
/// where the \f$ \Lambda \f$ matrix is the lower-triangular matrix used in Gaussian ARD kernel
static double normalize_coefficient(const KernelTypeList& TypesOfKernels, const ParameterVector& Hyperparameters, const Eigen::VectorXd& KInvLbl)
{
	const int NKernel = TypesOfKernels.size();
	double coe = 0.0;
	for (int i = 0, iparam = 0; i < NKernel; i++)
	{
		const double weight = Hyperparameters[iparam++];
		std::shared_ptr<shogun::CKernel> kernel_ptr;
		switch (TypesOfKernels[i])
		{
		case shogun::EKernelType::K_DIAG:
			break;
		case shogun::EKernelType::K_GAUSSIANARD:
		{
			Eigen::MatrixXd characteristic = Eigen::MatrixXd::Zero(PhaseDim, PhaseDim);
			for (int k = 0; k < PhaseDim; k++)
			{
				for (int j = k; j < PhaseDim; j++)
				{
					characteristic(j, k) = Hyperparameters[iparam++];
				}
			}
			coe += std::pow(2.0 * pi, Dim) * weight * weight * characteristic.determinant();
			break;
		}
		default:
			std::cerr << "UNKNOWN KERNEL!\n";
			break;
		}
	}
	return coe * KInvLbl.sum();
}

/// @brief Calculate energy of the Gaussian process predicted phase space distribution
/// @param[in] TrainingFeatures The features of the training set
/// @param[in] TypesOfKernels The types of all kernels used in Gaussian Process
/// @param[in] x The coordinates of x grids
/// @param[in] p The coordinates of p grids
/// @param[in] Hyperparameters Optimized hyperparameters of this element used in kernel
/// @param[in] KInvLbl The inverse of the kernel matrix times the labels
/// @param[in] PredictedDistribution The gridded whole PWTDM
/// @param[in] PESIndex The index of the diagonal element
/// @return The total energy of the element, potential energy from grids, and kinetic energy from analytical integration
/// @details For potential energy, just add the energy of grids all together.
/// For kinetic energy, the integral could be done analytically:
///
/// \f{eqnarray*}{
/// <p^2\rho_{ii}>&=&(2\pi)^{\frac{d}{2}}(\sigma_f^{ii})^2\left|\Lambda^{ii}\right|[(1\ 1\ \dots\ 1)(\Lambda^{ii}(\Lambda^{ii})^\top)^{-1}_{pp} \\
/// &&+(p_1^2\ p_2^2\ \dots\ p_n^2)]\qty(K^{ii})^{-1}\vb{y}^{ii}
/// \f}
///
/// where the \f$ \Lambda \f$ matrix is the lower-triangular matrix used in Gaussian ARD kernel,
/// and the p in row vector is the momentum of the training set.
static double energy_on_pes_from_gpr(
	const Eigen::MatrixXd& TrainingFeatures,
	const KernelTypeList& TypesOfKernels,
	const Eigen::VectorXd& x,
	const Eigen::VectorXd& p,
	const ParameterVector& Hyperparameters,
	const Eigen::VectorXd& KInvLbl,
	const SuperMatrix& PredictedDistribution,
	const int PESIndex)
{
	double result = 0.0;
	// first, calculate the potential energy
	const int nx = x.size(), np = p.size();
	const double dx = (x[nx - 1] - x[0]) / nx, dp = (p[np - 1] - p[0]) / np;
	for (int i = 0; i < nx; i++)
	{
		result += PredictedDistribution[PESIndex][PESIndex].row(i).sum() * adiabatic_potential(x[i])(PESIndex, PESIndex);
	}
	result *= dx * dp;
	// then, calculate the kinetic energy
	const int NKernel = TypesOfKernels.size();
	Eigen::VectorXd row_vector(TrainingFeatures.cols());
	for (int i = 0; i < TrainingFeatures.cols(); i++)
	{
		row_vector[i] = TrainingFeatures(1, i) * TrainingFeatures(1, i);
	}
	double coe = 0.0;
	for (int i = 0, iparam = 0; i < NKernel; i++)
	{
		const double weight = Hyperparameters[iparam++];
		std::shared_ptr<shogun::CKernel> kernel_ptr;
		switch (TypesOfKernels[i])
		{
		case shogun::EKernelType::K_DIAG:
			break;
		case shogun::EKernelType::K_GAUSSIANARD:
		{
			Eigen::MatrixXd characteristic = Eigen::MatrixXd::Zero(PhaseDim, PhaseDim);
			for (int k = 0; k < PhaseDim; k++)
			{
				for (int j = k; j < PhaseDim; j++)
				{
					characteristic(j, k) = Hyperparameters[iparam++];
				}
			}
			coe += std::pow(2.0 * pi, Dim) * weight * weight * characteristic.determinant();
			row_vector.array() += characteristic.transpose().fullPivLu().solve(characteristic.fullPivLu().solve(Eigen::MatrixXd::Identity(PhaseDim, PhaseDim)))(1, 1);
			break;
		}
		default:
			std::cerr << "UNKNOWN KERNEL!\n";
			break;
		}
	}
	result += coe * row_vector.dot(KInvLbl) / 2.0 / Mass;
	return result;
}

void obey_conservation(
	SuperMatrix& PredictedDistribution,
	const Eigen::MatrixXd& TrainingFeatures,
	const VectorMatrix& TrainingLabels,
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
	std::vector<Eigen::VectorXd, Eigen::aligned_allocator<Eigen::VectorXd>> KInvLbls;
	for (int i = 0; i < NumPES; i++)
	{
		if (IsSmall(i, i) == false)
		{
			NonZeroDiagonalElements.push_back(i);
			// the position where the component of this element in hyperparameter/bounds begins
			const int BeginIndex = (i * NumPES + i) * NoVar;
			// get hyperparameter and label for this element of PWTDM
			const ParameterVector& HyperparameterHere = ParameterVector(Hyperparameters.cbegin() + BeginIndex * NoVar, Hyperparameters.cbegin() + BeginIndex + NoVar);
			const Eigen::VectorXd& TrainingLabelOfThisElement = TrainingLabels[i][i];
			const Eigen::MatrixXd& KernelMatrix = get_kernel_matrix(TrainingFeatures, TrainingFeatures, TypesOfKernels, HyperparameterHere, true);
			const Eigen::VectorXd& Coe = KernelMatrix.llt().solve(TrainingLabelOfThisElement);
			KInvLbls.push_back(Coe);
		}
	}

	// normalize
	if (NonZeroDiagonalElements.size() == 1)
	{
		// if there is only one, only do normalization, without energy conservation
		const int PESIndex = NonZeroDiagonalElements[0];
		const int BeginIndex = (PESIndex * NumPES + PESIndex) * NoVar;
		const ParameterVector& HyperparameterHere = ParameterVector(Hyperparameters.cbegin() + BeginIndex, Hyperparameters.cbegin() + BeginIndex + NoVar);
		const double NormalizationFactor = normalize_coefficient(TypesOfKernels, HyperparameterHere, KInvLbls[0]);
		for (int iGrid = 0; iGrid < nx; iGrid++)
		{
			for (int jGrid = 0; jGrid < np; jGrid++)
			{
				PredictedDistribution[PESIndex][PESIndex](iGrid, jGrid) /= NormalizationFactor;
			}
		}
	}
	else
	{
		const int NoNonZeroDiagonalElements = NonZeroDiagonalElements.size();
		// more than 1, first half using an altogether coefficient,
		// the vector to be solved
		Eigen::Vector2d Conservations;
		Conservations << 1.0, InitialTotalEnergy;
		Eigen::Matrix2d Coefficients = Eigen::Matrix2d::Zero();
		for (int i = 0; i < NoNonZeroDiagonalElements; i++)
		{
			const int PESIndex = NonZeroDiagonalElements[i];
			const int BeginIndex = (PESIndex * NumPES + PESIndex) * NoVar;
			const ParameterVector& HyperparameterHere = ParameterVector(Hyperparameters.cbegin() + BeginIndex, Hyperparameters.cbegin() + BeginIndex + NoVar);
			if (i < NoNonZeroDiagonalElements / 2)
			{
				// first half, add to column 0
				Coefficients(0, 0) += normalize_coefficient(TypesOfKernels, HyperparameterHere, KInvLbls[i]);
				Coefficients(1, 0) += energy_on_pes_from_gpr(TrainingFeatures, TypesOfKernels, x, p, HyperparameterHere, KInvLbls[i], PredictedDistribution, PESIndex);
			}
			else
			{
				// last half, add to column 1
				Coefficients(0, 1) += normalize_coefficient(TypesOfKernels, HyperparameterHere, KInvLbls[i]);
				Coefficients(1, 1) += energy_on_pes_from_gpr(TrainingFeatures, TypesOfKernels, x, p, HyperparameterHere, KInvLbls[i], PredictedDistribution, PESIndex);
			}
		}
		const Eigen::Vector2d& ConservationFactor = Coefficients.fullPivLu().solve(Conservations);
		for (int i = 0; i < NoNonZeroDiagonalElements; i++)
		{
			const int PESIndex = NonZeroDiagonalElements[i];
			if (i < NoNonZeroDiagonalElements / 2)
			{
				// first half, using factor[0]
				for (int iGrid = 0; iGrid < nx; iGrid++)
				{
					for (int jGrid = 0; jGrid < np; jGrid++)
					{
						PredictedDistribution[PESIndex][PESIndex](iGrid, jGrid) *= ConservationFactor[0];
					}
				}
			}
			else
			{
				// last half, using factor[1]
				for (int iGrid = 0; iGrid < nx; iGrid++)
				{
					for (int jGrid = 0; jGrid < np; jGrid++)
					{
						PredictedDistribution[PESIndex][PESIndex](iGrid, jGrid) *= ConservationFactor[1];
					}
				}
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

double calculate_population_from_grid(const SuperMatrix& PhaseSpaceDistribution, const double dx, const double dp)
{
	double result = 0.0;
	for (int i = 0; i < NumPES; i++)
	{
		result += PhaseSpaceDistribution[i][i].sum();
	}
	return result * dx * dp;
}
