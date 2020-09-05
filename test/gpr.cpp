/// @file gpr.cpp
/// @brief definition of Gaussian Process Regression functions

#include "gpr.h"

#include "io.h"

/// passed to gsl optimization for get negative_log_marginal_likelihood()
using TrainingSet = pair<MatrixXd, VectorXd>;

/// @brief judge if all points has very small weight
/// @param[in] data the gridded phase space distribution
/// @return if abs of all value are below epsilon, return true, else return false
static bool is_very_small(const MatrixXd& data)
{
	// value below this are regarded as 0; x=5sigma for normal distribution
	static const double epsilon = exp(-12.5) / sqrt(2 * acos(-1.0));
	for (int i = 0; i < data.rows(); i++)
	{
		for (int j = 0; j < data.cols(); j++)
		{
			if (abs(data(i, j)) > epsilon)
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
static set<pair<int, int>> choose_point(const MatrixXd& data, const VectorXd& x, const VectorXd& p)
{
	static mt19937_64 generator(chrono::system_clock::now().time_since_epoch().count()); ///< random number generator, 64 bits Mersenne twister engine
	set<pair<int, int>> result;
	const int nx = data.rows(), np = data.cols(), n = data.size();
	if (is_very_small(data) == true)
	{
		// if not stopped before, then all points are very small
		// uniformly choose points
		uniform_int_distribution<int> x_dis(0, nx - 1), p_dis(0, np - 1);
		while (result.size() < NPoint)
		{
			const int ranx = x_dis(generator), ranp = p_dis(generator);
			result.insert(make_pair(ranx, ranp));
		}
	}
	else
	{
		const double weight = accumulate(data.data(), data.data() + n, 0.0, [](double a, double b) -> double { return abs(a) + abs(b); });
		uniform_real_distribution<double> urd(0.0, weight);
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
						result.insert(make_pair(j, k));
						goto next;
					}
				}
			}
		next:;
		}
	}
	return result;
}

/// @brief The Gaussian kernel function, \f$ K(X_1,X_2) \f$
/// @param[in] LeftFeature The feature on the left, \f$ X_1 \f$
/// @param[in] RightFeature The feature on the right, \f$ X_2 \f$
/// @param[in] Character The characteristic matrix, see explanation below
/// @param[in] weight The weight of Gaussian part
/// @param[in] both_training Whether the two features are both training features or not (with at least one test feature)
/// @param[in] noise The noise added when both features are training features
/// @return The kernel matrix
///
/// This function calculates the Gaussian kernel with noise on training features,
///
/// /f[
/// k(\mathbf{x}_1,\mathbf{x}_2)=\sigma_f^2\mathrm{exp}\left(-\frac{(\mathbf{x}_1-\mathbf{x}_2)^\top M (\mathbf{x}_1-\mathbf{x}_2)}{2}\right)+\sigma_n^2\delta(\mathbf{x}_1-\mathbf{x}_2)
/// /f]
///
/// where \f$ M \f$ is the characteristic matrix in parameter list.
/// Regarded as a real-symmetric matrix - only lower-triangular part are used，
/// its diagonal term is the characteristic length of each dimention and
/// off-diagonal term is the correlation between different dimensions
///
/// Noise part \f$ \sigma_n^2 \f$ is added when both features are training.
///
/// When there are more than one feature, the kernel matrix follows \f$ K_{ij}=k(X_{1_{\cdot i}}, X_{2_{\cdot j}}) \f$.
static MatrixXd gaussian_kernel(
	const MatrixXd& LeftFeature,
	const MatrixXd& RightFeature,
	const MatrixXd& Character,
	const double weight,
	const bool both_training = false,
	const double noise = 0.0)
{
	const int Rows = LeftFeature.cols(), Cols = RightFeature.cols();
	MatrixXd kernel = MatrixXd::Zero(Rows, Cols);
	for (int i = 0; i < Rows; i++)
	{
		for (int j = 0; j < Cols; j++)
		{
			const VectorXd& Diff = LeftFeature.col(i) - RightFeature.col(j);
			kernel(i, j) = exp(-(Diff.adjoint() * Character * Character.adjoint() * Diff).value() / 2.0);
		}
	}
	kernel *= weight * weight;
	if (both_training == true && LeftFeature.data() == RightFeature.data())
	{
		kernel += noise * noise * MatrixXd::Identity(Rows, Cols);
	}
	return kernel;
}


// some minimal for hyperparameters; below that would lead to non-sense
const double NoiseMin = 1e-10;	  ///< Minimal value of the noise
const double MagnitudeMin = 1e-4; ///< Minimal value of the magnitude of the normal kernel
const double CharLenMin = 1e-4;	  ///< Minimal value of the characteristic length in Gaussian kernel

/// @brief Calculate the negative marginal likelihood, \f$ -\mathrm{ln}p(\mathbf{y}|X,\mathbf{\theta}) \f$
/// @param[in] x The gsl vector containing all the hyperparameters
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
/// of this function is \mathbf{y}^\top K_y^{-1}\mathbf{y}+\mathbf{ln}|K_y|, neglecting constant coefficient and add-on.
///
/// The parameters of this function is quiet weird compared with other functions, as the requirement of gsl interface.
static double negative_log_marginal_likelihood(const gsl_vector* x, void* params)
{
	const TrainingSet* ts_ptr = static_cast<TrainingSet*>(params);
	const MatrixXd& TrainFeature = ts_ptr->first;
	const VectorXd& TrainLabel = ts_ptr->second;
	MatrixXd character(2, 2);
	character(0, 0) = abs(gsl_vector_get(x, 0)) < CharLenMin ? CharLenMin : gsl_vector_get(x, 0);
	character(0, 1) = 0;
	character(1, 0) = gsl_vector_get(x, 1);
	character(1, 1) = abs(gsl_vector_get(x, 2)) < CharLenMin ? CharLenMin : gsl_vector_get(x, 2);
	const MatrixXd& KernelMatrix = gaussian_kernel(
		TrainFeature,
		TrainFeature,
		character,
		abs(gsl_vector_get(x, 3)) < MagnitudeMin ? MagnitudeMin : gsl_vector_get(x, 3),
		true,
		abs(gsl_vector_get(x, 4)) < NoiseMin ? NoiseMin : gsl_vector_get(x, 4));
	LLT<MatrixXd> LLTOfKernel(KernelMatrix);
	const MatrixXd& L = LLTOfKernel.matrixL();
	return (TrainLabel.adjoint() * LLTOfKernel.solve(TrainLabel)).value() / 2.0 + L.diagonal().array().abs().log().sum();
}

/// @brief To opmitize the hyperparameters: 3 in gaussian kernel, and 2 weights
/// @param[in] TrainFeature The features of the training set
/// @param[in] TrainLabel The corresponding labels of the training set
/// @return The vector containing optimized hyperparameters.
///
/// This function using the gsl multidimensional optimization to optimize the hyperparameters and return them.
static vector<double> optimize(const MatrixXd& TrainFeature, const VectorXd& TrainLabel)
{
	// initial step size
	static const double InitStepSize = 0.5;
	// the maximum number of iteration
	static const int NumIter = 1000;
	// tolerance in minimization
	static const double AbsTol = 1e-10;
	// the number of variables to optimize
	static const int NumVar = 5;
	// init the minizer
	gsl_vector* x = gsl_vector_alloc(NumVar);
	gsl_vector_set(x, 0, 1.5);
	gsl_vector_set(x, 1, 0.0);
	gsl_vector_set(x, 2, 1.5);
	gsl_vector_set(x, 3, 1.0);
	gsl_vector_set(x, 4, 0.0 + InitStepSize);
	gsl_vector* step_size = gsl_vector_alloc(NumVar);
	gsl_vector_set_all(step_size, InitStepSize);
	gsl_multimin_function my_func;
	my_func.n = NumVar;
	my_func.f = negative_log_marginal_likelihood;
	TrainingSet ts_temp = make_pair(TrainFeature, TrainLabel);
	my_func.params = &ts_temp;
	gsl_multimin_fminimizer* minimizer = gsl_multimin_fminimizer_alloc(gsl_multimin_fminimizer_nmsimplex2rand, NumVar);
	gsl_multimin_fminimizer_set(minimizer, &my_func, x, step_size);
	cout << "\t\tInitial hyperparameters:\n";
	print_kernel(cout, x, 3);
	cout << ' ' << gsl_multimin_fminimizer_minimum(minimizer) << '\n';

	// iteration
	int status = GSL_CONTINUE;
	for (int i = 1; i <= NumIter && status == GSL_CONTINUE; i++)
	{
		status = gsl_multimin_fminimizer_iterate(minimizer);
		if (status)
		{
			break;
		}
		// output
		print_kernel(cout, gsl_multimin_fminimizer_x(minimizer), 3);
		cout << ' ' << gsl_multimin_fminimizer_minimum(minimizer) << '\n';
		// test convergence
		double size = gsl_multimin_fminimizer_size(minimizer);
		status = gsl_multimin_test_size(size, AbsTol);
		if (status == GSL_SUCCESS)
		{
			cout << "\t\tFinish optimize with " << i << " iterations\n";
		}
	}
	gsl_vector* x_temp = gsl_multimin_fminimizer_x(minimizer);
	print_kernel(cout, x_temp, 3);
	cout << ' ' << gsl_multimin_fminimizer_minimum(minimizer) << '\n';

	// set hyperparameter, free memory
	vector<double> hyperparameters(NumVar + 1);
	hyperparameters[0] = abs(gsl_vector_get(x, 0)) < CharLenMin ? CharLenMin : gsl_vector_get(x, 0);
	hyperparameters[1] = gsl_vector_get(x, 1);
	hyperparameters[2] = abs(gsl_vector_get(x, 2)) < CharLenMin ? CharLenMin : gsl_vector_get(x, 2);
	hyperparameters[3] = abs(gsl_vector_get(x, 3)) < MagnitudeMin ? MagnitudeMin : gsl_vector_get(x, 3);
	hyperparameters[4] = abs(gsl_vector_get(x, 4)) < NoiseMin ? NoiseMin : gsl_vector_get(x, 4);
	hyperparameters[NumVar] = gsl_multimin_fminimizer_minimum(minimizer);
	gsl_vector_free(x);
	gsl_vector_free(step_size);
	gsl_multimin_fminimizer_free(minimizer);
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
/// \mathcal{E}[p(\mathbf{f}_*|X_*,X,\mathbf{y})]=K(X_*,X)[K(X,X)+\sigma_n^2I]^{-1}\mathbf{y}
/// \f]
///
/// where \f$ X_* \f$ is the test features, \f$ X \f$ is the training features, and  \f$ \mathbf{y} \f$ is the training labels.
static MatrixXd predict_phase(
	const MatrixXd& TrainFeature,
	const VectorXd& TrainLabel,
	const VectorXd& x,
	const VectorXd& p,
	const vector<double>& Hyperparameters)
{
	MatrixXd character(2, 2);
	character(0, 0) = Hyperparameters[0];
	character(0, 1) = 0;
	character(1, 0) = Hyperparameters[1];
	character(1, 1) = Hyperparameters[2];
	const double& weight = Hyperparameters[3];
	const MatrixXd& KernelMatrix = gaussian_kernel(
		TrainFeature,
		TrainFeature,
		character,
		weight,
		true,
		Hyperparameters[4]);
	const VectorXd& Coe = KernelMatrix.llt().solve(TrainLabel);
	const int nx = x.size(), np = p.size();
	VectorXd coord(2);
	MatrixXd result(nx, np);
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < np; j++)
		{
			coord[0] = x[i];
			coord[1] = p[j];
			result(i, j) = (gaussian_kernel(coord, TrainFeature, character, weight) * Coe).value();
		}
	}
	return result;
}

void fit(
	const MatrixXd& data,
	const VectorXd& x,
	const VectorXd& p,
	ostream& sim,
	ostream& choose,
	ostream& log)
{
	// the number of MC chosen
	static const int NRound = 1; // 0;
	const int nx = x.size(), np = p.size(), n = data.size();
	if (is_very_small(data) == true)
	{
		// in case all points are very small, assuming they are all zero
		for (int i = 0; i < nx * np; i++)
		{
			sim << ' ' << 0;
		}
		sim << '\n';
		log << ' ' << accumulate(data.data(), data.data() + n, 0.0, [](double a, double b) -> double { return a + b * b; }) / n << ' ' << NAN;
		// choose the points
		const set<pair<int, int>> chosen = choose_point(data, x, p);
		// output the chosen points
		print_point(choose, chosen, x, p);
	}
	else
	{
		double minMSE = DBL_MAX, neg_log_marg_ll = 0;
		set<pair<int, int>> finally_chosen;
		MatrixXd finally_predict(nx, np);
		for (int i = 0; i < NRound; i++)
		{
			cout << "\tRound " << i << '\n';
			// choose the points
			const set<pair<int, int>> chosen = choose_point(data, x, p);
			// select the points
			MatrixXd training_feature(2, NPoint);
			VectorXd training_label(NPoint);
			for (int i = 0, itrain = 0; i < nx; i++)
			{
				for (int j = 0; j < np; j++)
				{
					if (chosen.find(make_pair(i, j)) != chosen.end())
					{
						// in the training set
						training_feature(0, itrain) = x(i);
						training_feature(1, itrain) = p(j);
						training_label(itrain) = data(i, j);
						itrain++;
					}
				}
			}

			// optimize
			const vector<double>& Hyperparameters = optimize(training_feature, training_label);

			// predict
			const MatrixXd& PredictPhase = predict_phase(training_feature, training_label, x, p, Hyperparameters);
			const double this_term_mse = (data - PredictPhase).array().square().sum();
			if (this_term_mse < minMSE)
			{
				// this time a smaller min MSE is gotten, using this value
				minMSE = this_term_mse;
				neg_log_marg_ll = Hyperparameters[Hyperparameters.size() - 1];
				finally_chosen = chosen;
				finally_predict = PredictPhase;
			}
		}
		print_point(choose, finally_chosen, x, p);
		log << ' ' << minMSE << ' ' << neg_log_marg_ll;
		for (int i = 0; i < nx; i++)
		{
			sim << ' ' << finally_predict.row(i);
		}
		sim << '\n';
	}
}
