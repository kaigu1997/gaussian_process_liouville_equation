/// @file gpr.cpp
/// @brief Definition of Gaussian Process Regression functions

#include "gpr.h"

#include "io.h"

/// Judge if all grids of certain density matrix element are very small or not
using SmallJudgeResult = Eigen::Matrix<bool, NumPES, NumPES>;
/// All the labels, or the PWTDM at the given point
using VectorMatrix = Eigen::Matrix<Eigen::Matrix<double, NumPES, NumPES>, Eigen::Dynamic, 1>;
/// Passed to optimization
using TrainingSet = std::tuple<Eigen::MatrixXd, VectorMatrix, KernelTypeList, SuperMatrix>;
/// An array of Eigen matrices, used for the derivatives of the kernel matrix
using MatrixVector = std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>>;

/// @brief Judge if all points has very small weight
/// @param[in] data The gridded phase space distribution
/// @return If abs of all value are below a certain limit, return true, else return false
static SmallJudgeResult is_very_small(const SuperMatrix& data)
{
	// value below this are regarded as 0
	static const double epsilon = 1e-2;
	const int Rows = data.rows(), Cols = data.cols();
	SmallJudgeResult result = SmallJudgeResult::Constant(true);
	for (int iGrid = 0; iGrid < Rows; iGrid++)
	{
		for (int jGrid = 0; jGrid < Cols; jGrid++)
		{
			for (int iPES = 0; iPES < NumPES; iPES++)
			{
				for (int jPES = 0; jPES < NumPES; jPES++)
				{
					if (std::abs(data(iGrid, jGrid)(iPES, jPES)) > epsilon)
					{
						result(iPES, jPES) = false;
					}
				}
			}
		}
	}
	return result;
}

/// @brief Calculate the weight at a given point
/// @param[in] data The PWTDM at the given point
/// @return The weight at the given (RowIndex, ColIndex) point
static double weight_function(const Eigen::MatrixXd& data)
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
	const int nx = data.rows(), np = data.cols(), n = data.size();
	// construct the weight matrix
	Eigen::MatrixXd weight_matrix(nx, np);
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < np; j++)
		{
			weight_matrix(i, j) = weight_function(data(i, j));
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

/// @brief Set the initial value of hyperparameters and its upper/lower bounds
/// @param[in] TypesOfKernels The vector containing all the kernel type used in optimization
/// @param[in] data The gridded whole PWTDM
/// @param[in] x The coordinates of x grids
/// @param[in] p The coordinates of p grids
/// @return A vector of vector, {lower_bound, upper_bound, hyperparameter}
std::vector<ParameterVector> set_initial_value(
	const KernelTypeList& TypesOfKernels,
	const SuperMatrix& data,
	const Eigen::VectorXd& x,
	const Eigen::VectorXd& p)
{
	const std::pair<int, int> MaxIndex = [&]() -> std::pair<int, int>
	{
		int idx_x, idx_p;
		double max = 0.0;
		for (int i = 0; i < data.rows(); i++)
		{
			for (int j = 0; j < data.cols(); j++)
			{
				if (data(i, j)(0, 0) > max)
				{
					idx_x = i;
					idx_p = j;
					max = data(i, j)(0, 0);
				}
			}
		}
		return std::make_pair(idx_x, idx_p);
	}();
	const double sigma_p = p[MaxIndex.second] / 20.0;
	const double sigma_x = 0.5 / sigma_p;
	static const double DiagMagMin = 1e-8;	 // Minimal value of the magnitude of the diagonal kernel
	static const double DiagMagMax = 1e-5;	 // Maximal value of the magnitude of the diagonal kernel
	static const double GaussMagMin = 1e-4;	 // Minimal value of the magintude of the gaussian kernel
	static const double GaussMagMax = 1.0;	 // Maximal value of the magnitude of the gaussian kernel
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
	const std::vector<double>& Hyperparameters,
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
	const std::vector<double>& Hyperparameters)
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

/// @brief get the training label (the gridded population values) of a certain element of the PWTDM
/// @param[in] FullTrainLabel The labels of the PWTDM of the selected points
/// @param[in] RowIndex The index of the row of the PWTDM
/// @param[in] ColIndex The index of the column of the PWTDM
/// @return The labels of the given element in PWTDM of all the selected points 
Eigen::VectorXd get_training_label(const VectorMatrix& FullTrainLabel, const int RowIndex, const int ColIndex)
{
	assert(RowIndex >= 0 && RowIndex < NumPES && ColIndex >= 0 && ColIndex < NumPES);
	Eigen::VectorXd result(FullTrainLabel.size());
	for (int i = 0; i < result.size(); i++)
	{
		result[i] = FullTrainLabel[i](RowIndex, ColIndex);
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
	const TrainingSet& Training = *static_cast<TrainingSet*>(params);
	// get the parameters
	const Eigen::MatrixXd& TrainingFeature = std::get<0>(Training);
	const VectorMatrix& FullTrainingLabel = std::get<1>(Training);
	const KernelTypeList& TypesOfKernels = std::get<2>(Training);
	const SmallJudgeResult& IsSmall = is_very_small(std::get<3>(Training));
	const int NoVar = number_of_overall_hyperparameters(TypesOfKernels);
	double sum_margll = 0.0;
	for (int i = 0; i < NumPES; i++)
	{
		for (int j = 0; j < NumPES; j++)
		{
			std::clog << indent(3);
			if (i == j)
			{
				std::clog << "rho[" << i << "][" << j << ']';
			}
			else if (i < j)
			{
				std::clog << "Re(rho[" << i << "][" << j << "])";
			}
			else // if (i > j)
			{
				std::clog << "Im(rho[" << i << "][" << j << "])";
			}
			std::clog << ": ";
			if (IsSmall(i, j) == true)
			{
				// very small, no fitting, 0 margll, 0 grad
				if (grad.empty() == false)
				{
					std::fill(grad.begin() + (i * NumPES + j) * NoVar, grad.begin() + (i * NumPES + j + 1) * NoVar, 0);
				}
				std::clog << "0 everywhere\n";
			}
			else
			{
				// have gradient, need to do all the calculations
				// get hyperparameter and label for this element of PWTDM
				ParameterVector hyperparameter_here(NoVar);
				std::copy(x.cbegin() + (i * NumPES + j) * NoVar, x.cbegin() + (i * NumPES + j + 1) * NoVar, hyperparameter_here.begin());
				const Eigen::VectorXd& TrainLabel = get_training_label(FullTrainingLabel, i, j);
				// get kernel and the derivatives of kernel over hyperparameters
				const Eigen::MatrixXd& KernelMatrix = get_kernel_matrix(TrainingFeature, TrainingFeature, TypesOfKernels, hyperparameter_here, true);
				// calculate
				Eigen::LLT<Eigen::MatrixXd> DecompOfKernel(KernelMatrix);
				const Eigen::MatrixXd& L = DecompOfKernel.matrixL();
				const Eigen::MatrixXd& KInv = DecompOfKernel.solve(Eigen::MatrixXd::Identity(TrainingFeature.cols(), TrainingFeature.cols())); // inverse of kernel
				const Eigen::VectorXd& KInvLbl = DecompOfKernel.solve(TrainLabel); // K^{-1}*y
				const double result = (TrainLabel.adjoint() * KInvLbl).value() / 2.0 + L.diagonal().array().abs().log().sum();
				std::clog << result << '\n';
				sum_margll += result;
				// print current combination
				print_kernel(std::clog, TypesOfKernels, hyperparameter_here, 4);
				if (grad.empty() == false)
				{
					ParameterVector gradient_here(NoVar);
					// need gradient information, calculated here
					const MatrixVector& KernelDerivative = kernel_derivative_over_hyperparameters(TrainingFeature, TypesOfKernels, hyperparameter_here);
					for (int k = 0; k < hyperparameter_here.size(); k++)
					{
						gradient_here[k] = ((KInv - KInvLbl * KInvLbl.adjoint()) * KernelDerivative[k]).trace() / 2.0;
					}
					// print current gradient if uses gradient
					print_kernel(std::clog, TypesOfKernels, gradient_here, 5);
					std::copy(gradient_here.cbegin(), gradient_here.cend(), grad.begin() + (i * NumPES + j) * NoVar);
				}
			}
		}
	}
	std::clog << std::endl;
	return sum_margll;
}

/// @brief To opmitize the hyperparameters: 3 in gaussian kernel, and 2 weights
/// @param[in] Training The training set, containing the features and labels of the training set and the types of all kernels
/// @param[in] xSize The size of the box of x direction
/// @param[in] pSize The size of the box of p direction
/// @param[in] ALGO The optimization algorithm, which is the nlopt::algorithm enum type
/// @param[in] FirstRun Given if it is the first time doing optimization; if not, do not set hyperparameters and bounds
/// @return The vector containing all hyperparameters, and the last element is the final function value
/// @details This function using the NLOPT to optimize the hyperparameters and return them.
/// Currently the optimization method is Nelder-Mead Simplex algorithm.
static ParameterVector optimize(
	const TrainingSet& Training,
	const Eigen::VectorXd& x,
	const Eigen::VectorXd& p,
	const nlopt::algorithm& ALGO)
{
	const KernelTypeList& TypesOfKernels = std::get<2>(Training);
	const int NKernel = TypesOfKernels.size();
	// constructor the NLOPT minimizer using Nelder-Mead simplex algorithm
	static const int NoVar = number_of_overall_hyperparameters(TypesOfKernels); // the number of variables to optimize
	nlopt::opt minimizer(nlopt::algorithm::AUGLAG_EQ, NoVar * NumPES * NumPES);
	// set minimizer function
	const nlopt::vfunc minimizing_function = negative_log_marginal_likelihood;
	minimizer.set_min_objective(minimizing_function, const_cast<TrainingSet*>(&Training));
	// set bounds for noise and characteristic length, as well as for initial values
	static const std::vector<ParameterVector>& initials = set_initial_value(TypesOfKernels, std::get<3>(Training), x, p);
	static const ParameterVector& lower_bound = initials[0];
	static const ParameterVector& upper_bound = initials[1];
	static ParameterVector hyperparameters = initials[2];
	minimizer.set_lower_bounds(lower_bound);
	minimizer.set_upper_bounds(upper_bound);
	// set stop criteria
	static const double AbsTol = 1e-10; // tolerance in minimization
	minimizer.set_xtol_abs(AbsTol);
	// set initial step size
	static const double InitStepSize = 0.5; // initial step size
	minimizer.set_initial_step(InitStepSize);
	// set local minimizer
	nlopt::opt local_minimizer(ALGO, NoVar * NumPES * NumPES);
	local_minimizer.set_xtol_abs(AbsTol);
	minimizer.set_local_optimizer(local_minimizer);

	// optimize
	std::clog << indent(2) << "Begin Optimization" << std::endl;
	double final_value = 0.0;
	try
	{
		nlopt::result result = minimizer.optimize(hyperparameters, final_value);
		std::clog << indent(2);
		switch (result)
		{
		case nlopt::result::SUCCESS:
			std::clog << "Successfully stopped" << std::endl;
			break;
		case nlopt::result::STOPVAL_REACHED:
			std::clog << "Stopping value reached" << std::endl;
			break;
		case nlopt::result::FTOL_REACHED:
			std::clog << "Function value tolerance reached" << std::endl;
			break;
		case nlopt::result::XTOL_REACHED:
			std::clog << "Step size tolerance reached" << std::endl;
			break;
		case nlopt::result::MAXEVAL_REACHED:
			std::clog << "Maximum evaluation time reached" << std::endl;
			break;
		case nlopt::result::MAXTIME_REACHED:
			std::clog << "Maximum cpu time reached" << std::endl;
			break;
		}
	}
	catch (const std::exception& e)
	{
		std::cerr << indent(2) << "NLOPT optimization failed for " << e.what() << '\n';
	}
	std::clog << indent(2) << "Best Combination is\n";
	std::vector<double> grad;
	final_value = minimizing_function(hyperparameters, grad, const_cast<TrainingSet*>(&Training));

	ParameterVector result = hyperparameters;
	result.push_back(final_value);
	return result;
}

/// @brief Predict labels of test set
/// @param[in] TrainFeature Features of training set
/// @param[in] TrainLabel Labels of training set
/// @param[in] x The coordinates of x grids
/// @param[in] p The coordinates of p grids
/// @param[in] Hyperparameters Optimized hyperparameters used in kernel
/// @return Predicted Labels of test set
/// @details This function follows the formula
///
/// \f[
/// E[p(\mathbf{f}_*|X_*,X,\mathbf{y})]=K(X_*,X)[K(X,X)+\sigma_n^2I]^{-1}\mathbf{y}
/// \f]
///
/// where \f$ X_* \f$ is the test features, \f$ X \f$ is the training features, and  \f$ \mathbf{y} \f$ is the training labels.
static SuperMatrix predict_phase(
	const TrainingSet& Training,
	const Eigen::VectorXd& x,
	const Eigen::VectorXd& p,
	const std::vector<double>& Hyperparameters)
{
	const Eigen::MatrixXd& TrainingFeature = std::get<0>(Training);
	const VectorMatrix& FullTrainingLabel = std::get<1>(Training);
	const KernelTypeList& TypesOfKernels = std::get<2>(Training);
	const SmallJudgeResult& IsSmall = is_very_small(std::get<3>(Training));
	const int nx = x.size(), np = p.size();
	SuperMatrix result(nx ,np);
	const int NoVar = number_of_overall_hyperparameters(TypesOfKernels);
	// predict one by one
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
						result(iGrid, jGrid)(iPES, jPES) = 0;
					}
				}
			}
			else
			{
				// get hyperparameter and label for this element of PWTDM
				ParameterVector hyperparameter_here(NoVar);
				std::copy(Hyperparameters.cbegin() + (iPES * NumPES + jPES) * NoVar, Hyperparameters.cbegin() + (iPES * NumPES + jPES + 1) * NoVar, hyperparameter_here.begin());
				const Eigen::VectorXd& TrainingLabel = get_training_label(FullTrainingLabel, iPES, jPES);
				const Eigen::MatrixXd& KernelMatrix = get_kernel_matrix(TrainingFeature, TrainingFeature, TypesOfKernels, hyperparameter_here, true);
				const Eigen::VectorXd& Coe = KernelMatrix.llt().solve(TrainingLabel);
				Eigen::VectorXd coord(PhaseDim);
				for (int iGrid = 0; iGrid < nx; iGrid++)
				{
					for (int jGrid = 0; jGrid < np; jGrid++)
					{
						coord[0] = x[iGrid];
						coord[1] = p[jGrid];
						result(iGrid, jGrid)(iPES, jPES) = (get_kernel_matrix(coord, TrainingFeature, TypesOfKernels, hyperparameter_here) * Coe).value();
					}
				}
			}
		}
	}
	return result;
}

QuantumMatrixD mean_squared_error(const SuperMatrix& lhs, const SuperMatrix& rhs)
{
	assert(lhs.rows() == rhs.rows() && lhs.cols() == rhs.cols());
	QuantumMatrixD result = QuantumMatrixD::Zero();
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		for (int jPES = 0; jPES < NumPES; jPES++)
		{
			for (int iGrid = 0; iGrid < lhs.rows(); iGrid++)
			{
				for (int jGrid = 0; jGrid < lhs.cols(); jGrid++)
				{
					const double diff = lhs(iGrid, jGrid)(iPES, jPES) - rhs(iGrid, jGrid)(iPES, jPES);
					result(iPES, jPES) += diff * diff;
				}
			}
		}
	}
	return result;
}

FittingResult fit(const SuperMatrix& data, const Eigen::VectorXd& x, const Eigen::VectorXd& p)
{
	// the number of MC chosen
	static const int NRound = 1; // 0;
	const int nx = x.size(), np = p.size(), n = data(0, 0).size();
	double minMSE = DBL_MAX, neg_log_marg_ll = 0;
	PointSet finally_chosen;
	SuperMatrix finally_predict;
	for (int i = 0; i < NRound; i++)
	{
		std::clog << indent(1) << "Round " << i << '\n';
		// choose the points
		const PointSet chosen = choose_point(data);
		// select the points
		Eigen::MatrixXd training_feature(2, NPoint);
		VectorMatrix training_label(NPoint);
		for (int i = 0, itrain = 0; i < nx; i++)
		{
			for (int j = 0; j < np; j++)
			{
				if (chosen.find(std::make_pair(i, j)) != chosen.end())
				{
					// in the training set
					training_feature(0, itrain) = x(i);
					training_feature(1, itrain) = p(j);
					training_label[itrain] = data(i, j);
					itrain++;
				}
			}
		}

		// set the kernel to use
		const KernelTypeList TypesOfKernels = { shogun::EKernelType::K_DIAG, shogun::EKernelType::K_GAUSSIANARD };
		const TrainingSet& Training = std::make_tuple(training_feature, training_label, TypesOfKernels, data);
		// optimize. Derivative-free algorithm first, then derivative method
		optimize(Training, x, p, nlopt::algorithm::LN_NELDERMEAD);
		const std::vector<double>& Hyperparameters = optimize(Training, x, p, nlopt::algorithm::LD_TNEWTON_PRECOND_RESTART);

		// predict
		const SuperMatrix& PredictPhase = predict_phase(Training, x, p, Hyperparameters);
		const double this_term_mse = mean_squared_error(data, PredictPhase).sum();
		if (this_term_mse < minMSE)
		{
			// this time a smaller min MSE is gotten, using this value
			minMSE = this_term_mse;
			finally_chosen = chosen;
			finally_predict = PredictPhase;
			neg_log_marg_ll = Hyperparameters[Hyperparameters.size() - 1];
		}
	}
	return std::make_tuple(finally_predict, finally_chosen, neg_log_marg_ll);
}
