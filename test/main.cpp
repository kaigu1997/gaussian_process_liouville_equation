/// @file main.cpp
/// @brief Test of GPR to rebuild the phase space distribution
///
/// This file is the combination of all functions (including the main function)
/// to rebuild the phase space distribution. 

#include <algorithm>
#include <chrono>
#include <cmath>
#include <climits>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <mkl.h>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

#pragma warning(push, 0)
#pragma warning(disable: 3346 654)
#include <shogun/base/init.h>
#include <shogun/base/some.h>
#include <shogun/classifier/LDA.h>
#include <shogun/evaluation/Evaluation.h>
#include <shogun/evaluation/GradientCriterion.h>
#include <shogun/evaluation/GradientEvaluation.h>
#include <shogun/evaluation/MeanSquaredError.h>
#include <shogun/features/MatrixFeatures.h>
#include <shogun/kernel/CombinedKernel.h>
#include <shogun/kernel/DiagKernel.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/kernel/GaussianARDKernel.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/machine/gp/GaussianLikelihood.h>
#include <shogun/machine/gp/ExactInferenceMethod.h>
#include <shogun/machine/gp/ZeroMean.h>
#include <shogun/modelselection/GradientModelSelection.h>
#include <shogun/regression/GaussianProcessRegression.h>
#include <shogun/statistical_testing/internals/Kernel.h>
#include <Eigen/Eigen>
#pragma warning(pop)

using namespace std;
using namespace Eigen;
using namespace shogun;

const int NPoint = 100; ///< the size of training set

/// to read the coordinate (of x, p and t) from a file
/// @param[in] filename the name of the input file
/// @return the vector containing the data
VectorXd read_coord(const string& filename)
{
	vector<double> v;
	double tmp;
	ifstream f(filename);
	while (f >> tmp)
	{
		v.push_back(tmp);
	}
	VectorXd coord(v.size());
	copy(v.cbegin(), v.cend(), coord.data());
	return coord;
}

static mt19937_64 generator(chrono::system_clock::now().time_since_epoch().count()); ///< random number generator, 64 bits Mersenne twister engine

const double epsilon = exp(-12.5) / sqrt(2 * acos(-1.0)); ///< value below this are regarded as 0; x=5sigma for normal distribution

/// choose npoint points based on their weight using MC. If the point has already chosen, redo
/// @param[in] data the gridded phase space distribution
/// @param[in] nx the number of x grids
/// @param[in] np the number of p grids
/// @param[in] x the coordinates of x grids
/// @param[in] p the coordinates of p grids
/// @return a pair, first being the pointer to the coordinate table (first[i][0]=xi, first[i][1]=pi), second being the vector containing the phase space distribution at the point (second[i]=P(xi,pi))
set<pair<int, int>> choose_point(const MatrixXd& data, const VectorXd& x, const VectorXd& p)
{
	set<pair<int, int>> result;
	const int nx = data.rows(), np = data.cols(), n = data.size();
	const double weight = accumulate(data.data(), data.data() + n, 0.0, [](double a, double b)->double { return abs(a) + abs(b); });
	if (weight < n * epsilon)
	{
		// if the weight is very small, check if all points are very small
		for (int i = 0; i < nx; i++)
		{
			for (int j = 0; j < np; j++)
			{
				if (abs(data(i, j)) > epsilon)
				{
					goto normal_choose;
				}
			}
		}
		// if not stopped before, then all points are very small
		// uniformly choose points
		uniform_int_distribution<int> x_dis(0, nx - 1), p_dis(0, np - 1);
		while (result.size() < NPoint)
		{
			const int ranx = x_dis(generator), ranp = p_dis(generator);
			result.insert(make_pair(ranx, ranp));
		}
		result.insert(make_pair(INT_MAX, INT_MAX)); // indicate that they are small
		return result;
	}
normal_choose:
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
	return result;
}

/// output the chosen point to an output stream
/// @param[in] point the chosen points
/// @param[in] x the gridded position coordinates
/// @param[in] p the gridded momentum coordinates
/// @param[inout] out the output stream
void output_point
(
	const set<pair<int, int>>& point,
	const VectorXd& x,
	const VectorXd& p,
	ostream& out
)
{
	set<pair<int,int>>::const_iterator iter = point.cbegin();
	for (int i = 0; i < NPoint; i++, ++iter)
	{
		out << ' ' << x(iter->first) << ' ' << p(iter->second);
	}
	out << '\n';
}

/// calculate the difference between GPR and exact value
/// @param[in] data the exact value
/// @param[in] x the gridded position coordinates
/// @param[in] p the gridded momentum coordinates
/// @param[inout] sim the output stream for simulated phase space distribution
/// @param[inout] choose the output stream for the chosen point
/// @param[inout] log the output stream for log info (mse, -log(marg ll))
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
	// choose the points
	const set<pair<int, int>> chosen = choose_point(data, x, p);
	// output the chosen points
	output_point(chosen, x, p, choose);
	if (chosen.size() == NPoint + 1 && chosen.crbegin()->first == INT_MAX && chosen.crbegin()->second == INT_MAX)
	{
		// in case all points are very small, assuming they are all zero
		for (int i = 0; i < nx * np; i++)
		{
			sim << ' ' << 0;
		}
		sim << '\n';
		log << ' ' << accumulate(data.data(), data.data() + n, 0.0, [](double a, double b)->double { return a + b * b; }) / n << ' ' << NAN;
	}
	else
	{
		// otherwise, select the points
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
		/*auto kernel_gaussianARD = some<CGaussianARDKernel>();
		VectorXd gaussian_weights(2);
		gaussian_weights[0] = 1.0;
		gaussian_weights[1] = 1.0;
		kernel_gaussianARD->set_vector_weights(gaussian_weights);
		auto kernel_diag = some<CDiagKernel>();
		auto kernel_ptr = some<CCombinedKernel>();
		kernel_ptr->append_kernel(kernel_gaussianARD.get());
		kernel_ptr->append_kernel(kernel_diag.get());
		kernel_ptr->init(train_feature_ptr.get(), train_feature_ptr.get());*/
		Some<CGaussianKernel> kernel_ptr = some<CGaussianKernel>(train_feature_ptr, train_feature_ptr, 1.0);
		// create mean function: 0 mean
		Some<CZeroMean> mean_ptr = some<CZeroMean>();
		// likelihood: gaussian likelihood
		Some<CGaussianLikelihood> likelihood_ptr = some<CGaussianLikelihood>();
		// exact inference
		Some<CExactInferenceMethod> inference_ptr = some<CExactInferenceMethod>(kernel_ptr, train_feature_ptr, mean_ptr, train_label_ptr, likelihood_ptr);
		// gp regression
		Some<CGaussianProcessRegression> gpr_ptr = some<CGaussianProcessRegression>(inference_ptr);

		// training the model
		gpr_ptr->train();

		// optimize
		// the criterion: gradient
		Some<CGradientCriterion> grad_criterion_ptr = some<CGradientCriterion>();
		// the evaluation: also gradient
		Some<CGradientEvaluation> grad_eval_ptr = some<CGradientEvaluation>(gpr_ptr, train_feature_ptr, train_label_ptr, grad_criterion_ptr);
		grad_eval_ptr->set_function(inference_ptr);
		// model selection: gradient
		Some<CGradientModelSelection> grad_model_sel_ptr = some<CGradientModelSelection>(grad_eval_ptr);
		// get the best hyperparameter, theta
		Some<CParameterCombination> best_hyperparam_ptr = some<CParameterCombination>(grad_model_sel_ptr->select_model());
		// use this to the gpr machine
		best_hyperparam_ptr->apply_to_machine(gpr_ptr);
		gpr_ptr->train();

		// predict
		Some<CRegressionLabels> predict_label_ptr(gpr_ptr->apply_regression(test_feature_ptr));
		for (int i = 0, itest = 0; i < nx; i++)
		{
			for (int j = 0; j < np; j++)
			{
				sim << ' ';
				if (chosen.find(make_pair(i, j)) != chosen.end())
				{
					sim << data(i, j);
				}
				else
				{
					sim << predict_label_ptr->get_label(itest);
					itest++;
				}
			}
		}
		Some<CMeanSquaredError> eval = some<CMeanSquaredError>();
		log << ' ' << eval->evaluate(predict_label_ptr, test_label_ptr)
			<< ' ' << inference_ptr->get_negative_log_marginal_likelihood();
	}
}

int main()
{
	clog.sync_with_stdio(false);
	init_shogun_with_defaults();
	// read input
	VectorXd x = read_coord("x.txt");
	const int nx = x.size();
	VectorXd p = read_coord("p.txt");
	const int np = p.size();
	VectorXd t = read_coord("t.txt");
	const int nt = t.size();

	ifstream phase("phase.txt"); // phase space distribution for input
	ofstream sim("sim.txt"); // simulated phase space distribution
	ofstream choose("choose.txt"); // chosen point for simulation
	ofstream log("log.txt"); // time, MSE and -log(Marginal likelihood)
	for (int ti = 0; ti < nt; ti++)
	{
		// then read the rho00 and rho11
		double tmp;
		MatrixXd rho0(nx, np), rho1(nx, np), rho_re(nx, np), rho_im(nx, np);
		// rho00
		for (int i = 0; i < nx; i++)
		{
			for (int j = 0; j < np; j++)
			{
				phase >> rho0(i, j) >> tmp;
			}
		}
		// rho01 and rho10
		for (int i = 0; i < nx; i++)
		{
			for (int j = 0; j < np; j++)
			{
				phase >> rho_re(i, j) >> rho_im(i, j);
			}
		}
		for (int i = 0; i < nx; i++)
		{
			for (int j = 0; j < np; j++)
			{
				phase >> tmp;
				rho_re(i, j) = (rho_re(i, j) + tmp) / 2.0;
				phase >> tmp;
				rho_im(i, j) = (rho_im(i, j) - tmp) / 2.0;
			}
		}
		// rho11
		for (int i = 0; i < nx; i++)
		{
			for (int j = 0; j < np; j++)
			{
				phase >> rho1(i, j) >> tmp;
			}
		}
		log << t(ti);
		fit(rho0, x, p, sim, choose, log);
		fit(rho1, x, p, sim, choose, log);
		fit(rho_re, x, p, sim, choose, log);
		fit(rho_im, x, p, sim, choose, log);
		log << endl;
		sim << '\n';
		choose << '\n';
	}
	phase.close();
	sim.close();
	log.close();

	exit_shogun();
	return 0;
}
