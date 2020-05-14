/// @file main.cpp
/// @brief Test of GPR to rebuild the phase space distribution
///
/// This file is the combination of all functions (including the main function)
/// to rebuild the phase space distribution. 

#include <fstream>
#include <iostream>
#include <string>

#include <algorithm>
#include <numeric>
#include <set>
#include <utility>
#include <vector>

#include <cmath>
#include <mkl.h>
#ifndef EIGEN_USE_MKL_ALL
	#define EIGEN_USE_MKL_ALL
#endif
#include <Eigen/Eigen>
#include <gsl/gsl_multimin.h>

#include "rand.h"
#include "gpr.h"

using namespace std;
using namespace Eigen;

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

/// read the phase space distribution of the diagonal elements of the density matrix at a given moment
/// @param[in] filename the filename of the file containing the gridded phase space distribution 
/// @param[in] ti the chosen time moment
/// @param[in] nx the number of x-direction grids
/// @param[in] np the number of p-direction grids
/// @return the phase space distribution of diagonal elements at the give moment
pair<MatrixXd, MatrixXd> read_rho
(
	const string& filename,
	const int ti,
	const int nx,
	const int np
)
{
	ifstream phase(filename);
	string buffer;
	double tmp;
	MatrixXd rho00(nx, np), rho11(nx, np);
	// read previous lines
	for (int i = 0; i < ti * 5; i++)
	{
		getline(phase, buffer);
	}
	// read rho[0][0](t)
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < np; j++)
		{
			phase >> rho00(i,j) >> tmp;
		}
	}
	getline(phase, buffer); // rho[0][1]
	getline(phase, buffer); // rho[1][0]
	// read rho[1][1](t)
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < np; j++)
		{
			phase >> rho11(i, j) >> tmp;
		}
	}
	return make_pair(rho00, rho11);
}


/// calculate the standard deviation
/// @param[in] data the exact value
/// @param[in] x the gridded position coordinates
/// @param[in] p the gridded momentum coordinates
/// @return the two standard deviation, first for x, second for p
pair<double, double> standard_deviation(const MatrixXd& data, const VectorXd& x, const VectorXd& p)
{
	const int nx = x.size(), np = p.size();
	const double total_weight = accumulate(data.data(), data.data() + nx * np, 0.0);
	double x_ave = 0, x2_ave = 0, p_ave = 0, p2_ave = 0;
	for (int i = 0; i < nx; i++)
	{
		x_ave += x(i) * data.row(i).sum();
		x2_ave += pow(x(i), 2) * data.row(i).sum();
	}
	for (int i = 0; i < np; i++)
	{
		p_ave += p(i) * data.col(i).sum();
		p2_ave += pow(x(i), 2) * data.col(i).sum();
	}
	return make_pair(x2_ave / total_weight - pow(x_ave / total_weight, 2), p2_ave / total_weight - pow(p_ave / total_weight, 2));
}

/// calculate the difference between GPR and exact value
/// @param[in] data the exact value
/// @param[in] x the gridded position coordinates
/// @param[in] p the gridded momentum coordinates
/// @return the accumulated square error on each grids
double error(const MatrixXd& data, const VectorXd& x, const VectorXd& p)
{
	cout << "Begin initialization..." << endl;
	static const int MaxIter = 1000;
	const int nx = x.size(), np = p.size();
	// choose the points
	const set<pair<int, int>> chosen = choose_point(data, x, p);
	DataSet training;
	MatrixX2d test(nx * np - NPoint, 2);
	for (int i = 0, itrain = 0, itest = 0; i < nx; i++)
	{
		for (int j = 0; j < np; j++)
		{
			if (chosen.find(make_pair(i, j)) != chosen.end())
			{
				// in the training set
				training.first(itrain, 0) = x(i);
				training.first(itrain, 1) = p(j);
				training.second(itrain) = data(i, j);
				itrain++;
			}
			else
			{
				// in the test set
				test(itest, 0) = x(i);
				test(itest, 1) = p(j);
				itest++;
			}
		}
	}
	// initialize multidimensional minimizer, the order is: length_x^2, length_p^2, sigma_f^2, sigma_n^2
	gsl_multimin_function_fdf error_function;
	error_function.n = 4;
	error_function.f = log_marginal_likelihood;
	error_function.df = derivatives;
	error_function.fdf = my_fdf;
	error_function.params = reinterpret_cast<void*>(&training);
	// initial characteristic length-scale is set to be the variance
	gsl_vector* hyperparameter = gsl_vector_alloc(4);
	const pair<double, double> charateristic_length_scale = standard_deviation(data, x, p);
	gsl_vector_set(hyperparameter, 0, charateristic_length_scale.first);
	gsl_vector_set(hyperparameter, 1, charateristic_length_scale.second);
	gsl_vector_set(hyperparameter, 2, 1);
	gsl_vector_set(hyperparameter, 3, 0);
	gsl_multimin_fdfminimizer* minimizer = gsl_multimin_fdfminimizer_alloc(gsl_multimin_fdfminimizer_conjugate_fr, 4);
	gsl_multimin_fdfminimizer_set(minimizer, &error_function, hyperparameter, 0.01, 0.1);

	cout << "Finish initialization. Begin iteration..." << endl;
	int status = GSL_CONTINUE;
	// do the optimization
	for (int i = 0; i < MaxIter && status == GSL_CONTINUE; i++)
	{
		status = gsl_multimin_fdfminimizer_iterate(minimizer);
		if (status != 0)
		{
			break;
		}
		status = gsl_multimin_test_gradient(minimizer->gradient, 1e-3);
		if (status == GSL_SUCCESS)
		{
			cout << "Finish Optimization Without Reaching Maximum Iterations\n";
			break;
		}
	}

	cout << "Finish iteration. Begin prediction..." << endl;
	// free the memory and generate the distribution
	gsl_multimin_fdfminimizer_free(minimizer);
	const double length_x2 = gsl_vector_get(hyperparameter, 0);
	const double length_p2 = gsl_vector_get(hyperparameter, 1);
	const double sigma_f2 = gsl_vector_get(hyperparameter, 2);
	const double sigma_n2 = gsl_vector_get(hyperparameter, 3);
	const MatrixXd kernel_inverse = rbf_se
	(
		training.first,
		training.first,
		length_x2,
		length_p2,
		sigma_f2,
		sigma_n2
	).inverse();
	const MatrixXd k_test_training = rbf_se
	(
		test,
		training.first,
		length_x2,
		length_p2,
		sigma_f2,
		sigma_n2
	);
	const VectorXd mu = k_test_training * kernel_inverse * training.second;
	const MatrixXd covariance = rbf_se(test, test, length_x2, length_p2, sigma_f2, sigma_n2)
		- k_test_training * kernel_inverse * k_test_training.adjoint();
	// generate a vector of gaussian-type random variables, #nx*np-npoint
	const VectorXd regression_result = mu + covariance.llt().matrixL() * gaussian_random_generate(nx * np - NPoint);

	/// calculate the error
	double err_sum = 0;
	for (int i = 0, itest = 0; i < nx; i++)
	{
		for (int j = 0; j < np; j++)
		{
			if (chosen.find(make_pair(i, j)) == chosen.end())
			{
				err_sum += pow(data(i, j) - regression_result(itest), 2);
				itest++;
			}
		}
	}
	return err_sum;
}

int main()
{
	cout.sync_with_stdio(false);
	// read input
	VectorXd x = read_coord("x.txt");
	const int nx = x.size();
	VectorXd p = read_coord("p.txt");
	const int np = p.size();
	VectorXd t = read_coord("t.txt");
	const int nt = t.size();
	// choose a random time to depict
	const int ti = choose_time(nt);
	// then read the rho00 and rho11
	const pair<MatrixXd, MatrixXd> rho = read_rho("phase.txt", ti, nx, np);

	cout << "Ground state:" << endl;
	const double e1 = error(rho.first, x, p);
	cout << "Squared Error = " << e1 << endl;
	cout << "Excited state:" << endl;
	const double e2 = error(rho.second, x, p);
	cout << "Squared Error = " << ' ' << e2 << endl;

	return 0;
}
