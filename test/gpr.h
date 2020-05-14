/// @file gpr.h
/// This file includes the functions related to Gaussian process regression (GPR),
/// including the kernel function (squared exponential), the log marginal likelihood
/// and its derivative over all hyperparameters.
#ifndef GPR_H
#define GPR_H

#include <mkl.h>
#ifndef EIGEN_USE_MKL_ALL
	#define EIGEN_USE_MKL_ALL
#endif
#include <Eigen/Eigen>
#include <gsl/gsl_vector.h>

#include "rand.h"

using namespace std;
using namespace Eigen;

using DataSet = pair<Matrix<double, NPoint, 2>, Matrix<double, NPoint, 1>>;

/// the square exponential kernel function, with different characteristic length-scale for different dimension
/// @param[in] X1 the first input, a n*2 matrix
/// @param[in] X2 the second input, a m*2 matrix
/// @param[in] length_x2 the squared charateristic length-scale of x direction
/// @param[in] length_p2 the squared charateristic length-scale of p direction
/// @param[in] sigma_f2 the pre-exponential factor, or the signal variance
/// @param[in] sigma_n2 the noise variance
/// @return the kernel matrix
MatrixXd rbf_se
(
	const MatrixX2d& X1,
	const MatrixX2d& X2,
	const double length_x2,
	const double length_p2,
	const double sigma_f2,
	const double sigma_n2
);

/// @brief calculate the log marginal likelihood as the error function to optimize
/// @param[in] v the input variables, or the hyperparameters, including the characteristic length-scales length_x^2 and length_p^2, signal variance sigma_f^2, and noise varianve sigma_n^2
/// @param[in] params the training set, including X and y
/// @return the function value at the given hyperparameters and training sets
double log_marginal_likelihood(const gsl_vector* v, void* params);

/// @brief calculate the derivatives of the log marginal likelihood over each hyperparameters
/// @param[in] v the input variables, or the hyperparameters, including the characteristic length-scales length_x^2 and length_p^2, signal variance sigma_f^2, and noise varianve sigma_n^2
/// @param[in] params the training set, including X and y
/// @param[out] df the derivative values at the given hyperparameters and training sets
void derivatives(const gsl_vector* v, void* params, gsl_vector* df);

/// the interface of f and derivative together
/// @param[in] v the input variables, or the hyperparameters
/// @param[in] params the parameters, or the training set
/// @param[out] f the pointer to save the value of the function
/// @param[out] df the vector to save the value of the derivatives
void my_fdf
(
	const gsl_vector* v,
	void* params,
	double* f,
	gsl_vector* df
);

#endif // !GPR_H
