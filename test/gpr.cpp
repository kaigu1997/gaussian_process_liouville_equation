/// @file gpr.cpp
/// @brief definition of Gaussian Process Regression functions

#include "io.h"
#include "gpr.h"

const double epsilon = exp(-12.5) / sqrt(2 * acos(-1.0)); ///< value below this are regarded as 0; x=5sigma for normal distribution

/// judge if all points has very small weight
/// @param[in] data the gridded phase space distribution
/// @return if abs of all value are below epsilon, return true, else return false
static bool is_very_small(const MatrixXd& data)
{
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

static mt19937_64 generator(chrono::system_clock::now().time_since_epoch().count()); ///< random number generator, 64 bits Mersenne twister engine

/// choose npoint points based on their weight using MC. If the point has already chosen, redo
/// @param[in] data the gridded phase space distribution
/// @param[in] nx the number of x grids
/// @param[in] np the number of p grids
/// @param[in] x the coordinates of x grids
/// @param[in] p the coordinates of p grids
/// @return a pair, first being the pointer to the coordinate table (first[i][0]=xi, first[i][1]=pi), second being the vector containing the phase space distribution at the point (second[i]=P(xi,pi))
static set<pair<int, int>> choose_point(const MatrixXd& data, const VectorXd& x, const VectorXd& p)
{
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
		const double weight = accumulate(data.data(), data.data() + n, 0.0, [](double a, double b)->double { return abs(a) + abs(b); });
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

/// to set hyperparameter into the combined kernel
/// @param[in] x the hyperparameters
/// @param[in] kernel_ptr the pointer to the combined kernel
static void set_hyperparameter(const gsl_vector* x, CCombinedKernel* kernel_ptr)
{
	// get parameter
	SGMatrix<double> gaussian_kernel_weights(2, 2);
	gaussian_kernel_weights(0, 0) = gsl_vector_get(x, 0);
	gaussian_kernel_weights(0, 1) = 0;
	gaussian_kernel_weights(1, 0) = gsl_vector_get(x, 1);
	gaussian_kernel_weights(1, 1) = gsl_vector_get(x, 2);
	const double diag_weight = gsl_vector_get(x, 3);
	// set into kernel
	CFeatures* lhs = kernel_ptr->get_lhs();
	CFeatures* rhs = kernel_ptr->get_rhs();
	kernel_ptr->remove_lhs_and_rhs();
	for (int i = 0; i < kernel_ptr->get_num_kernels(); i++)
	{
		CKernel* subkernel_ptr = kernel_ptr->get_kernel(i);
		switch (subkernel_ptr->get_kernel_type())
		{
		case K_DIAG:
			static_cast<CDiagKernel*>(subkernel_ptr)->set_combined_kernel_weight(diag_weight);
			break;
		case K_GAUSSIANARD:
		{
			CGaussianARDKernel* gaussian_ard_kernel_ptr = static_cast<CGaussianARDKernel*>(subkernel_ptr);
			gaussian_ard_kernel_ptr->set_matrix_weights(gaussian_kernel_weights);
			gaussian_ard_kernel_ptr->set_combined_kernel_weight(1 - diag_weight);
		}
		break;
		default:
			cerr << "UNIDENTIFIED KERNEL\n";
			exit(EXIT_FAILURE);
		}
	}
	kernel_ptr->init(lhs, rhs);
}

/// the function for gsl multidimentional minimization
/// @param x a vector containing input variables
/// @param params parameters used in function
/// @return the function value based on the variables and parameters
static double func(const gsl_vector* x, void* params)
{
	CGaussianProcessRegression* gpr_ptr = static_cast<CGaussianProcessRegression*>(params);
	CInference* inference_ptr = gpr_ptr->get_inference_method();
	set_hyperparameter(x, static_cast<CCombinedKernel*>(inference_ptr->get_kernel()));
	gpr_ptr->train();
	return inference_ptr->get_negative_log_marginal_likelihood();
}

const int NumIter = 1000; ///< the maximum number of iteration
const double AbsTol = 1e-5; ///< tolerance in minimization

/// to opmitize the hyperparameters: 3 in gaussian kernel, and 1 weight for diagonal
/// @param[in] gpr_ptr the Gaussian Process Regression pointer
static void optimize(CGaussianProcessRegression* gpr_ptr)
{
	const int NumVar = 4;
	// init the minizer
	gsl_vector* x = gsl_vector_alloc(NumVar);
	gsl_vector_set(x, 0, 1);
	gsl_vector_set(x, 1, 0);
	gsl_vector_set(x, 2, 1);
	gsl_vector_set(x, 3, 0);
	gsl_vector* step_size = gsl_vector_alloc(NumVar);
	gsl_vector_set_all(step_size, 0.05);
	gsl_multimin_function my_func;
	my_func.n = NumVar;
	my_func.f = func;
	my_func.params = gpr_ptr;
	gsl_multimin_fminimizer* minimizer = gsl_multimin_fminimizer_alloc(gsl_multimin_fminimizer_nmsimplex2, NumVar);
	gsl_multimin_fminimizer_set(minimizer, &my_func, x, step_size);

	// iteration
	int status = GSL_CONTINUE;
	for (int i = 1; i <= NumIter && status == GSL_CONTINUE; i++)
	{
		status = gsl_multimin_fminimizer_iterate(minimizer);
		if (status)
		{
			break;
		}
		double size = gsl_multimin_fminimizer_size(minimizer);
		status = gsl_multimin_test_size(size, AbsTol);
		if (status == GSL_SUCCESS)
		{
			cout << "\t\tFinish optimize with " << i << " iterations\n";
		}
	}

	// set hyperparameter
	set_hyperparameter(minimizer->x, static_cast<CCombinedKernel*>(gpr_ptr->get_inference_method()->get_kernel()));

	// free memory
	gsl_vector_free(x);
	gsl_vector_free(step_size);
	gsl_multimin_fminimizer_free(minimizer);
}

const int NRound = 10; ///< the number of MC chosen

void fit
(
	const MatrixXd& data,
	const VectorXd& x,
	const VectorXd& p,
	ostream& sim,
	ostream& choose,
	ostream& log
)
{
	const int nx = x.size(), np = p.size(), n = data.size();
	if (is_very_small(data) == true)
	{
		// in case all points are very small, assuming they are all zero
		for (int i = 0; i < nx * np; i++)
		{
			sim << ' ' << 0;
		}
		sim << '\n';
		log << ' ' << accumulate(data.data(), data.data() + n, 0.0, [](double a, double b)->double { return a + b * b; }) / n << ' ' << NAN;
		// choose the points
		const set<pair<int, int>> chosen = choose_point(data, x, p);
		// output the chosen points
		output_point(chosen, x, p, choose);
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
			const int NTest = nx * np - NPoint;
			MatrixXd test_feature(2, NTest);
			VectorXd test_label(NTest);
			for (int i = 0, itrain = 0, itest = 0; i < nx; i++)
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
					else
					{
						// in the test set
						test_feature(0, itest) = x(i);
						test_feature(1, itest) = p(j);
						test_label(itest) = data(i, j);
						itest++;
					}
				}
			}
			Some<CDenseFeatures<double>> train_feature_ptr = some<CDenseFeatures<double>>(training_feature);
			Some<CDenseFeatures<double>> test_feature_ptr = some<CDenseFeatures<double>>(test_feature);
			Some<CRegressionLabels> train_label_ptr = some<CRegressionLabels>(training_label);
			Some<CRegressionLabels> test_label_ptr = some<CRegressionLabels>(test_label);

			// create combined kernel: k(x,x')=c1*exp(-|x-x'|^2/t^2)+c2*delta(x-x')
			// use gaussianARD kernel first and optimize it
			Some<CGaussianARDKernel> kernel_gaussianARD_ptr = some<CGaussianARDKernel>();
			kernel_gaussianARD_ptr->set_matrix_weights(SGMatrix<double>::create_identity_matrix(2, 1.0));
			Some<CDiagKernel> kernel_diag_ptr = some<CDiagKernel>();
			Some<CCombinedKernel> kernel_combined_ptr = some<CCombinedKernel>();
			kernel_combined_ptr->append_kernel(kernel_gaussianARD_ptr);
			kernel_combined_ptr->append_kernel(kernel_diag_ptr);
			kernel_combined_ptr->init(train_feature_ptr, train_feature_ptr);
			cout << "\t\tBefore training:\n";
			print_kernel((CKernel*)(kernel_combined_ptr.get()), 2);

			// create mean function: 0 mean
			Some<CZeroMean> mean_ptr = some<CZeroMean>();
			// likelihood: gaussian likelihood
			Some<CGaussianLikelihood> likelihood_ptr = some<CGaussianLikelihood>();
			// exact inference
			Some<CExactInferenceMethod> inference_ptr = some<CExactInferenceMethod>(kernel_combined_ptr, train_feature_ptr, mean_ptr, train_label_ptr, likelihood_ptr);
			// gp regression
			Some<CGaussianProcessRegression> gpr_ptr = some<CGaussianProcessRegression>(inference_ptr);

			// optimize
			optimize(gpr_ptr);
			// training the model
			gpr_ptr->train();
			// print kernel weights
			cout << "\t\tAfter training:\n";
			print_kernel((CKernel*)(kernel_combined_ptr.get()), 2);

			// predict
			Some<CRegressionLabels> predict_label_ptr(gpr_ptr->apply_regression(test_feature_ptr));
			Some<CMeanSquaredError> eval = some<CMeanSquaredError>();
			const double this_term_mse = eval->evaluate(predict_label_ptr, test_label_ptr);
			if (this_term_mse < minMSE)
			{
				// this time a smaller min MSE is gotten, using this value
				minMSE = this_term_mse;
				neg_log_marg_ll = inference_ptr->get_negative_log_marginal_likelihood();
				finally_chosen = chosen;
				for (int i = 0, itest = 0; i < nx; i++)
				{
					for (int j = 0; j < np; j++)
					{
						if (chosen.find(make_pair(i, j)) != chosen.end())
						{
							finally_predict(i, j) = data(i, j);
						}
						else
						{
							finally_predict(i, j) = predict_label_ptr->get_label(itest);
							itest++;
						}
					}
				}
			}
		}
		output_point(finally_chosen, x, p, choose);
		log << ' ' << minMSE << ' ' << neg_log_marg_ll;
		for (int i = 0; i < finally_predict.rows(); i++)
		{
			sim << ' ' << finally_predict.row(i);
		}
		sim << '\n';
	}
}
