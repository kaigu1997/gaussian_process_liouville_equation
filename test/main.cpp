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
const int NRound = 10; ///< the number of MC chosen

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

/// judge if all points has very small weight
/// @param[in] data the gridded phase space distribution
/// @return if abs of all value are below epsilon, return true, else return false
bool is_very_small(const MatrixXd& data)
{
	for (int i = 0; i < data.rows(); i++)
	{
		for (int j = 0; j < data.cols(); j++)
		{
			if (abs(data(i,j)) > epsilon)
			{
				return false;
			}
		}
	}
	return true;
}

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

/// print kernel information. If a combined kernel, print its inner
/// @param[in] kernel_ptr a raw pointer to a CKernel object to be printed
/// @param[in] layer the layer of the kernel
void print_kernel(CKernel* kernel_ptr, int layer)
{
	for (int i = 0; i < layer; i++)
	{
		cout << '\t';
	}
	cout << kernel_ptr << " - " << kernel_ptr->get_name() << " weight = " << kernel_ptr->get_combined_kernel_weight() << '\n';
	switch (kernel_ptr->get_kernel_type())
	{
		case K_GAUSSIAN:
		{
			for (int i = 0; i <= layer; i++)
			{
				cout << '\t';
			}
			cout << "width = " << dynamic_cast<CGaussianKernel*>(kernel_ptr)->get_width() << '\n';
		}
		break;
		case K_COMBINED:
		{
			CCombinedKernel* combined_kernel_ptr = dynamic_cast<CCombinedKernel*>(kernel_ptr);
			for (int i = 0; i < combined_kernel_ptr->get_num_kernels(); i++)
			{
				print_kernel(combined_kernel_ptr->get_kernel(i), layer + 1);
			}
		}
		break;
		case K_DIAG:
		{
			for (int i = 0; i <= layer; i++)
			{
				cout << '\t';
			}
			cout << "diag = " << dynamic_cast<CDiagKernel*>(kernel_ptr)->kernel(0, 0) << '\n';
		}
		break;
		case K_GAUSSIANARD:
		{
			SGMatrix<double> weight = dynamic_cast<CGaussianARDKernel*>(kernel_ptr)->get_weights();
			for (int i = 0; i <= layer; i++)
			{
				cout << '\t';
			}
			cout << "weight = [";
			for (int i = 0; i < weight.num_rows; i++)
			{
				if (i != 0)
				{
					cout << ", ";
				}
				cout << '[';
				for (int j = 0; j < weight.num_cols; j++)
				{
					if (j != 0)
					{
						cout << ", ";
					}
					cout << weight(i, j);
				}
				cout << ']';
			}
			cout << "]\n";
		}
		break;
		default:
		{
			for (int i = 0; i <= layer; i++)
			{
				cout << '\t';
			}
			cout << "UNIDENTIFIED KERNEL\n";
		}
	}
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
			// use gaussianARD kernel first and optimize it
			Some<CGaussianARDKernel> kernel_gaussianARD_ptr = some<CGaussianARDKernel>();
			kernel_gaussianARD_ptr->set_matrix_weights(SGMatrix<double>::create_identity_matrix(2, 1.0));
			kernel_gaussianARD_ptr->init(train_feature_ptr, train_feature_ptr);
			cout << "\t\tBefore training:\n";
			print_kernel((CKernel*)(kernel_gaussianARD_ptr.get()), 2);

			// create mean function: 0 mean
			Some<CZeroMean> mean_ptr = some<CZeroMean>();
			// likelihood: gaussian likelihood
			Some<CGaussianLikelihood> likelihood_ptr = some<CGaussianLikelihood>();
			// exact inference
			Some<CExactInferenceMethod> inference_ptr = some<CExactInferenceMethod>(kernel_gaussianARD_ptr, train_feature_ptr, mean_ptr, train_label_ptr, likelihood_ptr);
			// gp regression
			Some<CGaussianProcessRegression> gpr_ptr = some<CGaussianProcessRegression>(inference_ptr);

			// optimize
			// the criterion: gradient
			Some<CGradientCriterion> grad_criterion_ptr = some<CGradientCriterion>();
			// the evaluation: also gradient
			Some<CGradientEvaluation> grad_eval_ptr = some<CGradientEvaluation>(gpr_ptr, train_feature_ptr, train_label_ptr, grad_criterion_ptr);
			grad_eval_ptr->set_function(inference_ptr);
			// model selection: gradient
			Some<CGradientModelSelection> grad_model_sel_ptr = some<CGradientModelSelection>(grad_eval_ptr);
			// get the best hyperparameter, theta
			Some<CParameterCombination> best_hyperparam_gaussianARD_ptr = some<CParameterCombination>(grad_model_sel_ptr->select_model());
			// use this to the gpr machine
			best_hyperparam_gaussianARD_ptr->apply_to_machine(gpr_ptr);
			// print kernel weights
			cout << "\t\tAfter training:\n";
			print_kernel((CKernel*)(kernel_gaussianARD_ptr.get()), 2);

			// reset diag kernel
			Some<CDiagKernel> kernel_diag_ptr = some<CDiagKernel>();
			Some<CCombinedKernel> kernel_combined_ptr = some<CCombinedKernel>();
			kernel_combined_ptr->append_kernel(kernel_gaussianARD_ptr);
			kernel_combined_ptr->append_kernel(kernel_diag_ptr);
			kernel_combined_ptr->init(train_feature_ptr, train_feature_ptr);
			kernel_combined_ptr->enable_subkernel_weight_learning();
			cout << "\t\tBefore training:\n";
			print_kernel((CKernel*)(kernel_combined_ptr.get()), 2);
			inference_ptr->set_kernel(kernel_combined_ptr);
			Some<CParameterCombination> best_hyperparam_combined_ptr = some<CParameterCombination>(grad_model_sel_ptr->select_model());
			// use this to the gpr machine
			best_hyperparam_combined_ptr->apply_to_machine(gpr_ptr);
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
							finally_predict(i,j) = data(i, j);
						}
						else
						{
							finally_predict(i,j) = predict_label_ptr->get_label(itest);
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

int main()
{
	cout.sync_with_stdio(true);
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
		cout << "\nT = "<< t(ti) << "\nrho[0][0]:\n";
		fit(rho0, x, p, sim, choose, log);
		cout << "\nRe(rho[0][1]):\n";
		fit(rho_re, x, p, sim, choose, log);
		cout << "\nIm(rho[0][1]):\n";
		fit(rho_im, x, p, sim, choose, log);
		cout << "\nrho[1][1]:\n";
		fit(rho1, x, p, sim, choose, log);
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
