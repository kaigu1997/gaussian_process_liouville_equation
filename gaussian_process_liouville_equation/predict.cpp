/// @file predict.cpp
/// @brief Implementation of predict.h

#include "stdafx.h"

#include "predict.h"

#include "kernel.h"
#include "pes.h"

/// @details The reduction is not combined in return because of STUPID clang-format.
ClassicalPhaseVector calculate_1st_order_average(const EigenVector<PhaseSpacePoint>& density)
{
	assert(!density.empty());
	const ClassicalPhaseVector result =
		std::transform_reduce(
			std::execution::par_unseq,
			density.cbegin(),
			density.cend(),
			ClassicalPhaseVector(ClassicalPhaseVector::Zero()),
			std::plus<ClassicalPhaseVector>(),
			[](const PhaseSpacePoint& psp) -> ClassicalPhaseVector
			{
				[[maybe_unused]] const auto& [x, p, rho] = psp;
				ClassicalPhaseVector result;
				result << x, p;
				return result;
			});
	return result / density.size();
}

ClassicalPhaseVector calculate_standard_deviation(const EigenVector<PhaseSpacePoint>& density)
{
	assert(!density.empty());
	const ClassicalPhaseVector sum_square_r =
		std::transform_reduce(
			std::execution::par_unseq,
			density.cbegin(),
			density.cend(),
			ClassicalPhaseVector(ClassicalPhaseVector::Zero()),
			std::plus<ClassicalPhaseVector>(),
			[](const PhaseSpacePoint& psp) -> ClassicalPhaseVector
			{
				[[maybe_unused]] const auto& [x, p, rho] = psp;
				ClassicalPhaseVector result;
				result << x, p;
				return result.array().square();
			});
	return (sum_square_r.array() / density.size() - calculate_1st_order_average(density).array().square()).sqrt();
}

double calculate_total_energy_average(
	const EigenVector<PhaseSpacePoint>& density,
	const int PESIndex,
	const ClassicalVector<double>& mass)
{
	assert(!density.empty());
	const double result =
		std::transform_reduce(
			std::execution::par_unseq,
			density.cbegin(),
			density.cend(),
			0.0,
			std::plus<double>(),
			[&mass, PESIndex](const PhaseSpacePoint& psp) -> double
			{
				[[maybe_unused]] const auto& [x, p, rho] = psp;
				return p.dot((p.array() / mass.array()).matrix()) / 2.0 + adiabatic_potential(x)[PESIndex];
			});
	return result / density.size();
}

Eigen::MatrixXd construct_training_feature(const EigenVector<ClassicalVector<double>>& x, const EigenVector<ClassicalVector<double>>& p)
{
	assert(x.size() == p.size());
	const int size = x.size();
	Eigen::MatrixXd result(PhaseDim, size);
	const std::vector<int> indices = get_indices(size);
	std::for_each(
		std::execution::par_unseq,
		indices.cbegin(),
		indices.cend(),
		[&result, &x, &p](int i) -> void
		{
			result.col(i) << x[i], p[i];
		});
	return result;
}

/// @details Using Gaussian Process Regression to predict by
///
/// @f[
/// \mathbb{E}[p(\mathbf{f}_*|X_*,X,\mathbf{y})]=K(X_*,X)[K(X,X)+\sigma_n^2I]^{-1}\mathbf{y}
/// @f]
///
/// where @f$ X_* @f$ is the test features, @f$ X @f$ is the training features,
/// and @f$ \mathbf{y} @f$ is the training labels.
Eigen::VectorXd predict_elements(const Kernel& kernel, const Eigen::MatrixXd& PhaseGrids)
{
	assert(kernel.is_same_feature() && PhaseGrids.rows() == PhaseDim);
	const Kernel k_new_old(kernel.get_parameters(), PhaseGrids, kernel.get_left_feature(), false);
	return k_new_old.get_kernel() * kernel.get_inverse_label();
}

/**
 * @details The derivative is given by
 *
 * @f{eqnarray*}{
 * \frac{\partial}{\partial\theta}\mathbb{E}[p(\mathbf{f}_*|X_*,X,\mathbf{y})]
 * &=&\frac{\partial K(X_*,X)}{\partial\theta}[K(X,X)+\sigma_n^2I]^{-1}\mathbf{y} \\
 * &-&K(X_*,X)[K(X,X)+\sigma_n^2I]^{-1}\frac{\partial[K(X,X)+\sigma_n^2I]}{\partial\theta}[K(X,X)+\sigma_n^2I]^{-1}\mathbf{y}
 * @f}
 *
 * where @f$ \theta @f$ stands for the parameters.
 */
ParameterVector prediction_derivative(const Kernel& kernel, const ClassicalVector<double>& x, const ClassicalVector<double>& p)
{
	assert(kernel.is_same_feature() && kernel.is_derivative_calculated());
	const Kernel k_new_old(kernel.get_parameters(), construct_training_feature({x}, {p}), kernel.get_left_feature(), true);
	ParameterVector result(Kernel::NumTotalParameters);
	for (std::size_t iParam = 0; iParam < Kernel::NumTotalParameters; iParam++)
	{
		result[iParam] = (k_new_old.get_derivatives()[iParam] * kernel.get_inverse_label()).value()
			+ (k_new_old.get_kernel() * kernel.get_negative_inv_deriv_inv_lbl()[iParam]).value();
	}
	return result;
}

/// @details The variance is given by
///
/// @f[
/// \mathrm{Var}=K(X_*,X_*)-K(X_*,X)[K(X,X)+\sigma_n^2I]^{-1}K(X,X_*)
/// @f]
///
/// and if the prediction is smaller than variance, the prediction will be set as 0.
///
/// The variance matrix is generally too big and not necessary, so it will be calculated elementwise.
Eigen::VectorXd predict_variances(const Kernel& kernel, const Eigen::MatrixXd& PhaseGrids)
{
	assert(kernel.is_same_feature() && PhaseGrids.rows() == PhaseDim);
	const Kernel k_new_old(kernel.get_parameters(), PhaseGrids, kernel.get_left_feature(), false);
	auto calc_result = [&kernel, &k_new_old, &PhaseGrids](const int iCol) -> double
	{
		// construct two features so that K(X*,X*) does not contain noise
		const Eigen::MatrixXd feature = PhaseGrids.col(iCol);
		const Kernel k_new_new(kernel.get_parameters(), feature, feature, false);
		const auto& k_new_old_row_i = k_new_old.get_kernel().row(iCol);
		return (k_new_new.get_kernel() - k_new_old_row_i * kernel.get_inverse() * k_new_old_row_i.transpose()).value();
	};
	const unsigned NumPoints = PhaseGrids.cols();
	Eigen::VectorXd result(NumPoints);
	const std::vector<int> indices = get_indices(NumPoints);
	std::for_each(
		std::execution::par_unseq,
		indices.cbegin(),
		indices.cend(),
		[&result, &calc_result](int iCol) -> void
		{
			result[iCol] = calc_result(iCol);
		});
	return result;
}

QuantumMatrix<std::complex<double>> predict_matrix(
	const OptionalKernels& Kernels,
	const ClassicalVector<double>& x,
	const ClassicalVector<double>& p)
{
	using namespace std::literals::complex_literals;
	const Eigen::MatrixXd PhaseCoord = construct_training_feature({x}, {p});
	auto predict = [&Kernels, &PhaseCoord](const int iPES, const int jPES) -> double
	{
		const int ElementIndex = iPES * NumPES + jPES;
		if (Kernels[ElementIndex].has_value())
		{
			return predict_elements(Kernels[ElementIndex].value(), PhaseCoord).value();
		}
		else
		{
			return 0.0;
		}
	};
	QuantumMatrix<std::complex<double>> result = QuantumMatrix<std::complex<double>>::Zero();
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		for (int jPES = 0; jPES < NumPES; jPES++)
		{
			const double ElementValue = predict(iPES, jPES);
			if (iPES <= jPES)
			{
				result(jPES, iPES) += ElementValue;
			}
			else
			{
				result(iPES, jPES) += 1.0i * ElementValue;
			}
		}
	}
	return result.selfadjointView<Eigen::Lower>();
}

/// @details The global purity is the weighted sum of purity of all elements, with weight 1 for diagonal and 2 for off-diagonal
double calculate_purity(const OptionalKernels& Kernels)
{
	double result = 0.0;
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		for (int jPES = 0; jPES < NumPES; jPES++)
		{
			const int ElementIndex = iPES * NumPES + jPES;
			if (Kernels[ElementIndex].has_value())
			{
				if (iPES == jPES)
				{
					result += Kernels[ElementIndex]->get_purity();
				}
				else
				{
					result += 2.0 * Kernels[ElementIndex]->get_purity();
				}
			}
		}
	}
	return result;
}
