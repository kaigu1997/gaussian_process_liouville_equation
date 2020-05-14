/// @file gpr.cpp
/// This file is the implementation of the corresponding header.

#include <cmath>
#include <mkl.h>
#define EIGEM_USE_MKL_ALL
#include <Eigen/Eigen>
#include <gsl/gsl_vector.h>

#include "gpr.h"

using namespace std;
using namespace Eigen;

/// k_ij=sigma_f2*exp(-(xi-xj)^2/2/length_x2-(pi-pj)^2/2/length_p2)+sigma_n^2*delta_ij
MatrixXd rbf_se
(
	const MatrixX2d& X1,
	const MatrixX2d& X2,
	const double length_x2,
	const double length_p2,
	const double sigma_f2,
	const double sigma_n2
)
{
	const int n1 = X1.rows(), n2 = X2.rows();
	MatrixXd kernel(n1, n2);
	for (int i = 0; i < n1; i++)
	{
		const double& xi = X1(i, 0);
		const double& pi = X1(i, 1);
		for (int j = 0; j < n2; j++)
		{
			const double& xj = X2(j, 0);
			const double& pj = X2(j, 1);
			kernel(i, j) = sigma_f2 * exp(-(pow((xi - xj), 2) / length_x2 + pow((pi - pj), 2) / length_p2) / 2.0);
		}
	}
	return kernel + decltype(kernel)::Identity(n1, n2) * sigma_n2;
}

/// the real return is log p(y|X)+n/2*log(2*pi),
/// or -y^T*K^{-1}*y/2-log|K|/2, where the last
/// term is a constant and has no effect on minimum 
double log_marginal_likelihood(const gsl_vector* v, void* params)
{
	const double& length_x2 = gsl_vector_get(v, 0);
	const double& length_p2 = gsl_vector_get(v, 1);
	const double& sigma_f2 = gsl_vector_get(v, 2);
	const double& sigma_n2 = gsl_vector_get(v, 3);
	const DataSet data = *reinterpret_cast<DataSet*>(params);
	const auto& y = data.second;
	const auto kernel = rbf_se(data.first, data.first, length_x2, length_p2, sigma_f2, sigma_n2);
	return -y.adjoint().dot(kernel.inverse() * y) / 2.0 - log(kernel.determinant()) / 2.0;
}

/// calculate the derivative of kernel matrix over the characteristic length-scale
///
/// for the rbf se kernel, k_ij=sigma_f2*exp(-(xi-xj)^2/2/length_x2-(pi-pj)^2/2/length_p2)+sigma_n^2*delta_ij
///
/// so d(k_ij)/dlength_r2=sigma_f2*exp(-(xi-xj)^2/2/length_x2-(pi-pj)^2/2/length_p2)*(ri-rj)^2/2/length_r2^2
/// @param[in] coord the coordinates, or the input of the training set
/// @param[in] length_x2 the squared charateristic length-scale of x direction
/// @param[in] length_p2 the squared charateristic length-scale of p direction
/// @param[in] sigma_f2 the pre-exponential factor, or the signal variance
/// @param[in] which_length whether it is the x or p direction, 0 for x and 1 for p
/// @return the derivative of kernel matrix over the characteristic length-scale, which is also a matrix
static Matrix<double, NPoint, NPoint> kernel_derivative_length2
(
	const Matrix<double, NPoint, 2>& coord,
	const double length_x2,
	const double length_p2,
	const double sigma_f2,
	const int which_length
)
{
	Matrix<double, NPoint, NPoint> kernel_derivative_length2;
	for (int i = 0; i < NPoint; i++)
	{
		const double& xi = coord(i, 0);
		const double& pi = coord(i, 1);
		for (int j = 0; j < NPoint; j++)
		{
			const double& xj = coord(j, 0);
			const double& pj = coord(j, 1);
			kernel_derivative_length2(i, j) = sigma_f2 * exp(-(pow((xi - xj), 2) / length_x2 + pow((pi - pj), 2) / length_p2) / 2.0);
			if (which_length == 0)
			{
				kernel_derivative_length2(i, j) *= pow((xi - xj) / length_x2, 2) / 2.0;
			}
			else
			{
				kernel_derivative_length2(i, j) *= pow((pi - pj) / length_p2, 2) / 2.0;
			}
		}
	}
	return kernel_derivative_length2;
}

/// for any theta, dlogp/dtheta = tr((alpha*alpha^T-K^{-1})*dK/dtheta), where alpha=K^{-1}*y
void derivatives(const gsl_vector* v, void* params, gsl_vector* df)
{
	const double& length_x2 = gsl_vector_get(v, 0);
	const double& length_p2 = gsl_vector_get(v, 1);
	const double& sigma_f2 = gsl_vector_get(v, 2);
	const double& sigma_n2 = gsl_vector_get(v, 3);
	const DataSet data = *reinterpret_cast<DataSet*>(params);
	const auto& X = data.first;
	const auto& y = data.second;
	const auto kernel = rbf_se(X, X, length_x2, length_p2, sigma_f2, sigma_n2);
	const auto kernel_inverse = kernel.inverse();
	// for lengths, dK/dlength_r^2=sigma_f^2*exp(-(xi-xj)^2/2/length_x^2-(pi-pj)^2/2/length_p^2)*(ri-rj)^2/2/length_r^4
	// for sigma_f^2, dK/dsigma_f^2=exp(-(xi-xj)^2/2/length_x^2-(pi-pj)^2/2/length_p^2)
	// for sigma_n^2, dK/dsigna_n^2=I
	const auto aalpha = kernel_inverse * y;
	const auto first_term = aalpha * aalpha.adjoint() - kernel_inverse;
	gsl_vector_set(df, 0, (first_term * kernel_derivative_length2(X, length_x2, length_p2, sigma_f2, 0)).trace());
	gsl_vector_set(df, 1, (first_term * kernel_derivative_length2(X, length_x2, length_p2, sigma_f2, 1)).trace());
	gsl_vector_set(df, 2, (first_term * rbf_se(X, X, length_x2, length_p2, 1, 0)).trace());
	gsl_vector_set(df, 3, (first_term * Matrix<double, NPoint, NPoint>::Identity()).trace());
}

void my_fdf
(
	const gsl_vector* v,
	void* params,
	double* f,
	gsl_vector* df
)
{
	*f = log_marginal_likelihood(v, params);
	derivatives(v, params, df);
}