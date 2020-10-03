/// @file gpr.cpp
/// @brief Definition of Gaussian Process Regression functions

#include "gpr.h"

#include "io.h"

/// Passed to optimization for get negative_log_marginal_likelihood()
using TrainingSet = std::tuple<Eigen::MatrixXd, Eigen::VectorXd, KernelTypeList>;
/// An array of Eigen matrices, used for the derivatives of the kernel matrix
using MatrixVector = std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>>;

/// @brief Judge if all points has very small weight
/// @param[in] data The gridded phase space distribution
/// @return If abs of all value are below epsilon, return true, else return false
static bool is_very_small(const Eigen::MatrixXd& data)
{
	// value below this are regarded as 0; x=5sigma for normal distribution
	static const double epsilon = std::exp(-12.5) / std::sqrt(2 * std::acos(-1.0));
	const int Row = data.rows(), Col = data.cols();
	for (int i = 0; i < Row; i++)
	{
		for (int j = 0; j < Col; j++)
		{
			if (std::abs(data(i, j)) > epsilon)
			{
				return false;
			}
		}
	}
	return true;
}

/// @brief Choose npoint points based on their weight using MC. If the point has already chosen, redo
/// @param[in] data The gridded phase space distribution
/// @param[in] x The coordinates of x grids
/// @param[in] p The coordinates of p grids
/// @return A set containing the pointer to the coordinate table (first[i][0]=xi, first[i][1]=pi)
///
/// This function check if the values in the input matrix are all small or not.
/// If all very small, then select points randomly without weight.
/// Otherwise, select based on the weights at the point by MC procedure.
static PointSet choose_point(const Eigen::MatrixXd& data, const Eigen::VectorXd& x, const Eigen::VectorXd& p)
{
	static std::mt19937_64 generator(std::chrono::system_clock::now().time_since_epoch().count()); // random number generator, 64 bits Mersenne twister engine
	PointSet result;
	const int nx = data.rows(), np = data.cols(), n = data.size();
	if (is_very_small(data) == true)
	{
		// if not stopped before, then all points are very small
		// uniformly choose points
		std::uniform_int_distribution<int> x_dis(0, nx - 1), p_dis(0, np - 1);
		while (result.size() < NPoint)
		{
			const int ranx = x_dis(generator), ranp = p_dis(generator);
			result.insert(std::make_pair(ranx, ranp));
		}
	}
	else
	{
		const double weight = std::accumulate(data.data(), data.data() + n, 0.0, [](double a, double b) -> double { return abs(a) + abs(b); });
		std::uniform_real_distribution<double> urd(0.0, weight);
		while (result.size() < NPoint)
		{
			double acc_weight = urd(generator);
			for (int j = 0; j < nx; j++)
			{
				for (int k = 0; k < np; k++)
				{
					acc_weight -= abs(data(j, k));
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
		Eigen::MatrixXd abs_data = data.array().abs();
		while (result.size() < NPoint)
		{
			int idx_x, idx_p;
			abs_data.maxCoeff(&idx_x, &idx_p);
			abs_data(idx_x, idx_p) = 0;
			result.insert(std::make_pair(idx_x, idx_p));
		}
*/
	}
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
///
/// This function calculates the kernel matrix with noise using the shogun library,
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
	assert(Hyperparameters.size() == number_of_overall_hyperparameters(TypesOfKernels) || Hyperparameters.size() == number_of_overall_hyperparameters(TypesOfKernels) + 1);
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
///
/// This function calculate the derivative of kernel matrix over each hyperparameter,
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
		std::shared_ptr<shogun::CKernel> kernel_ptr;
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
						result.push_back(weight * weight  * Deriv);
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
///
/// The formula of negative marginal likelihood is
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
static double negative_log_marginal_likelihood(const std::vector<double>& x, std::vector<double>& grad, void* params)
{
	// receive the parameter
	const TrainingSet& Training = *static_cast<TrainingSet*>(params);
	// get the kernel matrix, x being the hyperparameters
	const Eigen::MatrixXd& Feature = std::get<0>(Training);
	const KernelTypeList& TypesOfKernels = std::get<2>(Training);
	// print current combination
	print_kernel(std::clog, TypesOfKernels, x, 3);
	// get kernel and the derivatives of kernel over hyperparameters
	const Eigen::MatrixXd& KernelMatrix = get_kernel_matrix(Feature, Feature, TypesOfKernels, x, true);
	// calculate
	Eigen::LLT<Eigen::MatrixXd> DecompOfKernel(KernelMatrix);
	const Eigen::MatrixXd& L = DecompOfKernel.matrixL();
	const Eigen::MatrixXd& KInv = DecompOfKernel.solve(Eigen::MatrixXd::Identity(Feature.cols(), Feature.cols())); // inverse of kernel
	const Eigen::VectorXd& TrainLabel = std::get<1>(Training);
	const Eigen::VectorXd& KInvLbl = DecompOfKernel.solve(TrainLabel); // K^{-1}*y
	if (grad.empty() == false)
	{
		// need gradient information, calculated here
		const MatrixVector& KernelDerivative = kernel_derivative_over_hyperparameters(Feature, TypesOfKernels, x);
		for (int i = 0; i < x.size(); i++)
		{
			grad[i] = ((KInv - KInvLbl * KInvLbl.adjoint()) * KernelDerivative[i]).trace() / 2.0;
		}
		// print current gradient if uses gradient
		print_kernel(std::clog, TypesOfKernels, grad, 4);
	}
	const double result = (TrainLabel.adjoint() * KInvLbl).value() / 2.0 + L.diagonal().array().abs().log().sum();
	std::clog << "\t\t\t\t" << result << std::endl;
	return result;
}

/// @brief To opmitize the hyperparameters: 3 in gaussian kernel, and 2 weights
/// @param[in] Training The training set, containing the features and labels of the training set and the types of all kernels
/// @param[in] xSize The size of the box of x direction
/// @param[in] pSize The size of the box of p direction
/// @param[in] ALGO The optimization algorithm, which is the nlopt::algorithm enum type
/// @param[in] FirstRun Given if it is the first time doing optimization; if not, do not set hyperparameters and bounds
/// @return The vector containing all hyperparameters, and the last element is the final function value
///
/// This function using the NLOPT to optimize the hyperparameters and return them.
/// Currently the optimization method is Nelder-Mead Simplex algorithm.
static std::vector<double> optimize(
	const TrainingSet& Training,
	const double xSize,
	const double pSize,
	const nlopt::algorithm& ALGO,
	const bool FirstRun = false)
{
	const KernelTypeList& TypesOfKernels = std::get<2>(Training);
	const int NKernel = TypesOfKernels.size();
	// constructor the NLOPT minimizer using Nelder-Mead simplex algorithm
	static const int NoVar = number_of_overall_hyperparameters(TypesOfKernels); // the number of variables to optimize
	nlopt::opt minimizer(nlopt::algorithm::AUGLAG_EQ, NoVar);
	// set minimizer function
	minimizer.set_min_objective(negative_log_marginal_likelihood, const_cast<TrainingSet*>(&Training));
	// set bounds for noise and characteristic length, as well as for initial values
	static const double DiagMagMin = 1e-8;	 // Minimal value of the magnitude of the diagonal kernel
	static const double DiagMagMax = 1e-6;	 // Maximal value of the magnitude of the diagonal kernel
	static const double GaussMagMin = 1e-4;	 // Minimal value of the magintude of the gaussian kernel
	static const double GaussMagMax = 1.0;	 // Maximal value of the magnitude of the gaussian kernel
	static const double CharLenMin = 1e-4;	 // Minimal value of the characteristic length in Gaussian kernel
	static std::vector<double> lower_bound(NoVar, std::numeric_limits<double>::lowest());
	static std::vector<double> upper_bound(NoVar, std::numeric_limits<double>::max());
	static std::vector<double> hyperparameters(NoVar);
	while (hyperparameters.size() > NoVar)
	{
		hyperparameters.pop_back();
	}
	while (hyperparameters.size() < NoVar)
	{
		hyperparameters.push_back(0);
	}
	if (FirstRun == true)
	{
		for (int i = 0, iparam = 0; i < NKernel; i++)
		{
			switch (TypesOfKernels[i])
			{
			case shogun::EKernelType::K_DIAG:
				lower_bound[iparam] = DiagMagMin;
				upper_bound[iparam] = DiagMagMax;
				hyperparameters[iparam] = DiagMagMin;
				iparam++;
				break;
			case shogun::EKernelType::K_GAUSSIANARD:
				lower_bound[iparam] = GaussMagMin;
				upper_bound[iparam] = GaussMagMax;
				hyperparameters[iparam] = GaussMagMax;
				iparam++;
				for (int j = 0; j < PhaseDim; j++)
				{
					lower_bound[iparam] = CharLenMin;
					if (j == 0)
					{
						upper_bound[iparam] = xSize;
					}
					else
					{
						upper_bound[iparam] = pSize;
					}
					hyperparameters[iparam] = 1.0;
					iparam++;
					for (int i = j + 1; i < PhaseDim; i++)
					{
						hyperparameters[iparam] = 0.0;
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
	minimizer.set_lower_bounds(lower_bound);
	minimizer.set_upper_bounds(upper_bound);
	// set stop criteria
	static const double AbsTol = 1e-10; // tolerance in minimization
	minimizer.set_xtol_abs(AbsTol);
	// set initial step size
	static const double InitStepSize = 0.5; // initial step size
	minimizer.set_initial_step(InitStepSize);
	// set local minimizer
	nlopt::opt local_minimizer(ALGO, NoVar);
	local_minimizer.set_xtol_abs(AbsTol);
	minimizer.set_local_optimizer(local_minimizer);

	// optimize
	std::clog << "\t\tBegin Optimization" << std::endl;
	double final_value = 0.0;
	try
	{
		nlopt::result result = minimizer.optimize(hyperparameters, final_value);
		std::clog << "\t\t";
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
	}
	catch (const std::exception& e)
	{
		std::cerr << "\t\tNLOPT optimization failed for " << e.what() << '\n';
	}
	std::clog << "\n\t\tBest Combination is\n";
	std::vector<double> grad;
	final_value = negative_log_marginal_likelihood(hyperparameters, grad, const_cast<TrainingSet*>(&Training));
	hyperparameters.push_back(final_value);

	return hyperparameters;
}

/// @brief Predict labels of test set
/// @param[in] TrainFeature Features of training set
/// @param[in] TrainLabel Labels of training set
/// @param[in] x The coordinates of x grids
/// @param[in] p The coordinates of p grids
/// @param[in] Hyperparameters Optimized hyperparameters used in kernel
/// @return Predicted Labels of test set
///
/// This function follows the formula
///
/// \f[
/// E[p(\mathbf{f}_*|X_*,X,\mathbf{y})]=K(X_*,X)[K(X,X)+\sigma_n^2I]^{-1}\mathbf{y}
/// \f]
///
/// where \f$ X_* \f$ is the test features, \f$ X \f$ is the training features, and  \f$ \mathbf{y} \f$ is the training labels.
static Eigen::MatrixXd predict_phase(
	const TrainingSet& Training,
	const Eigen::VectorXd& x,
	const Eigen::VectorXd& p,
	const std::vector<double>& Hyperparameters)
{
	const Eigen::MatrixXd& TrainingFeature = std::get<0>(Training);
	const KernelTypeList& TypesOfKernels = std::get<2>(Training);
	const Eigen::MatrixXd& KernelMatrix = get_kernel_matrix(TrainingFeature, TrainingFeature, TypesOfKernels, Hyperparameters, true);
	const Eigen::VectorXd& TrainingLabel = std::get<1>(Training);
	const Eigen::VectorXd& Coe = KernelMatrix.llt().solve(TrainingLabel);
	const int nx = x.size(), np = p.size();
	Eigen::VectorXd coord(2);
	Eigen::MatrixXd result(nx, np);
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < np; j++)
		{
			coord[0] = x[i];
			coord[1] = p[j];
			result(i, j) = (get_kernel_matrix(coord, TrainingFeature, TypesOfKernels, Hyperparameters) * Coe).value();
		}
	}
	return result;
}

FittingResult fit(const Eigen::MatrixXd& data, const Eigen::VectorXd& x, const Eigen::VectorXd& p)
{
	// the number of MC chosen
	static const int NRound = 1; // 0;
	const int nx = x.size(), np = p.size(), n = data.size();
	if (is_very_small(data) == true)
	{
		return std::make_tuple(Eigen::MatrixXd::Zero(nx, np), choose_point(data, x, p), NAN);
	}
	else
	{
		double minMSE = DBL_MAX, neg_log_marg_ll = 0;
		PointSet finally_chosen;
		Eigen::MatrixXd finally_predict(nx, np);
		for (int i = 0; i < NRound; i++)
		{
			std::clog << "\tRound " << i << '\n';
			// choose the points
			const PointSet chosen = choose_point(data, x, p);
			// select the points
			Eigen::MatrixXd training_feature(2, NPoint);
			Eigen::VectorXd training_label(NPoint);
			for (int i = 0, itrain = 0; i < nx; i++)
			{
				for (int j = 0; j < np; j++)
				{
					if (chosen.find(std::make_pair(i, j)) != chosen.end())
					{
						// in the training set
						training_feature(0, itrain) = x(i);
						training_feature(1, itrain) = p(j);
						training_label(itrain) = data(i, j);
						itrain++;
					}
				}
			}

			// set the kernel to use
			const KernelTypeList TypesOfKernels = { shogun::EKernelType::K_DIAG, shogun::EKernelType::K_GAUSSIANARD };
			const TrainingSet& Training = std::make_tuple(training_feature, training_label, TypesOfKernels);
			// optimize. Derivative-free algorithm first, then derivative method
			optimize(
				Training,
				x[nx - 1] - x[0],
				p[nx - 1] - p[0],
				nlopt::algorithm::LN_NELDERMEAD,
				true);
			const std::vector<double>& Hyperparameters
				= optimize(
					Training,
					x[nx - 1] - x[0],
					p[nx - 1] - p[0],
					nlopt::algorithm::LD_TNEWTON_PRECOND_RESTART);

			// predict
			const Eigen::MatrixXd& PredictPhase = predict_phase(Training, x, p, Hyperparameters);
			const double this_term_mse = (data - PredictPhase).array().square().sum();
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
}
