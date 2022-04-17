/// @file predict.cpp
/// @brief Implementation of predict.h

#include "stdafx.h"

#include "predict.h"

#include "kernel.h"
#include "pes.h"

/// @details The reduction is not combined in return because of STUPID clang-format.
ClassicalPhaseVector calculate_1st_order_average_one_surface(const ElementPoints& density)
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
				[[maybe_unused]] const auto& [r, rho] = psp;
				return r;
			});
	return result / density.size();
}

ClassicalPhaseVector calculate_standard_deviation_one_surface(const ElementPoints& density)
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
				[[maybe_unused]] const auto& [r, rho] = psp;
				return r.array().square();
			});
	return (sum_square_r.array() / density.size() - calculate_1st_order_average_one_surface(density).array().square()).sqrt();
}

double calculate_total_energy_average_one_surface(
	const ElementPoints& density,
	const ClassicalVector<double>& mass,
	const std::size_t PESIndex)
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
				[[maybe_unused]] const auto& [r, rho] = psp;
				const auto [x, p] = split_phase_coordinate(r);
				return p.dot((p.array() / mass.array()).matrix()) / 2.0 + adiabatic_potential(x)[PESIndex];
			});
	return result / density.size();
}

QuantumVector<double> calculate_total_energy_average_each_surface(const AllPoints& density, const ClassicalVector<double>& mass)
{
	QuantumVector<double> result = QuantumVector<double>::Zero();
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		const std::size_t ElementIndex = iPES * NumPES + iPES;
		if (!density[ElementIndex].empty())
		{
			result[iPES] = calculate_total_energy_average_one_surface(density[ElementIndex], mass, iPES);
		}
	}
	return result;
}

/// @details Using Gaussian Process Regression to predict by @n
/// @f[
/// \mathbb{E}[p(\mathbf{f}_*|X_*,X,\mathbf{y})]=K(X_*,X)[K(X,X)+\sigma_n^2I]^{-1}\mathbf{y}
/// @f] @n
/// where @f$ X_* @f$ is the test features, @f$ X @f$ is the training features,
/// and @f$ \mathbf{y} @f$ is the training labels.
Eigen::VectorXd predict_elements(const Kernel& kernel, const Eigen::MatrixXd& PhaseGrids)
{
	assert(kernel.is_same_feature() && PhaseGrids.rows() == PhaseDim);
	const Kernel k_new_old(kernel.get_parameters(), PhaseGrids, kernel.get_left_feature(), false);
	return k_new_old.get_kernel() * kernel.get_inverse_label();
}

/**
 * @details The derivative is given by @n
 * @f{eqnarray*}{
 * \frac{\partial}{\partial\theta}\mathbb{E}[p(\mathbf{f}_*|X_*,X,\mathbf{y})]
 * &=&\frac{\partial K(X_*,X)}{\partial\theta}[K(X,X)+\sigma_n^2I]^{-1}\mathbf{y} \\
 * &-&K(X_*,X)[K(X,X)+\sigma_n^2I]^{-1}\frac{\partial[K(X,X)+\sigma_n^2I]}{\partial\theta}[K(X,X)+\sigma_n^2I]^{-1}\mathbf{y}
 * @f} @n
 * where @f$ \theta @f$ stands for the parameters.
 */
ElementParameter prediction_derivative(const Kernel& kernel, const ClassicalPhaseVector& r)
{
	assert(kernel.is_same_feature() && kernel.is_derivative_calculated());
	const Kernel k_new_old(kernel.get_parameters(), r, kernel.get_left_feature(), true);
	ElementParameter result = ElementParameter::Zero();
	for (std::size_t iParam = 0; iParam < Kernel::NumTotalParameters; iParam++)
	{
		result[iParam] = (k_new_old.get_derivatives()[iParam] * kernel.get_inverse_label()).value()
			+ (k_new_old.get_kernel() * kernel.get_negative_inv_deriv_inv_lbl()[iParam]).value();
	}
	return result;
}

/// @details The variance is given by @n
/// @f[
/// \mathrm{Var}=K(X_*,X_*)-K(X_*,X)[K(X,X)+\sigma_n^2I]^{-1}K(X,X_*)
/// @f] @n
/// and if the prediction is smaller than variance, the prediction will be set as 0. @n
/// The variance matrix is generally too big and not necessary, so it will be calculated elementwise.
Eigen::VectorXd predict_variances(const Kernel& kernel, const Eigen::MatrixXd& PhaseGrids)
{
	assert(kernel.is_same_feature() && PhaseGrids.rows() == PhaseDim);
	const Kernel k_new_old(kernel.get_parameters(), PhaseGrids, kernel.get_left_feature(), false);
	auto calc_result = [&kernel, &k_new_old, &PhaseGrids](const std::size_t iCol) -> double
	{
		// construct two features so that K(X*,X*) does not contain noise
		const Eigen::MatrixXd feature = PhaseGrids.col(iCol);
		const Kernel k_new_new(kernel.get_parameters(), feature, feature, false);
		const auto& k_new_old_row_i = k_new_old.get_kernel().row(iCol);
		return (k_new_new.get_kernel() - k_new_old_row_i * kernel.get_inverse() * k_new_old_row_i.transpose()).value();
	};
	const std::size_t NumPoints = PhaseGrids.cols();
	Eigen::VectorXd result(NumPoints);
	const std::vector<std::size_t> indices = get_indices(NumPoints);
	std::for_each(
		std::execution::par_unseq,
		indices.cbegin(),
		indices.cend(),
		[&result, &calc_result](std::size_t iCol) -> void
		{
			result[iCol] = calc_result(iCol);
		});
	return result;
}

/**
 * @details This function uses a connection procedure. @n
 * When @f$ \bar{y}^2<\mathrm{Var}(y) @f$, predicts 0; @n
 * when @f$ \bar{y}^2>t^2\mathrm{Var}(y) @f$, predicts @f$ \bar{y} @f$, where @f$ t>1 @f$ is a parameter; @n
 * Otherwise, use a cubic function to connect them. This cubic function @f$ f @f$ satisfies
 * @f{eqnarray*}{
 * f(\bar{y})&=&0\\
 * f(t(\bar{y}))&=&1\\
 * f^{\prime}(\bar{y})&=&0\\
 * f^{\prime}(t\bar{y})&=&0
 * @f} @n
 * In other words, its value and 1st order derivative is continuous with the other parts. @n
 * The prediction will then be @f$ f\bar{y} @f$.
 */
Eigen::VectorXd predict_elements_with_variance_comparison(const Kernel& kernel, const Eigen::MatrixXd& PhaseGrids)
{
	static constexpr double ConnectingPoint = 2.0;
	static_assert(ConnectingPoint > 1.0);
	assert(PhaseGrids.rows() == PhaseDim);
	const Eigen::VectorXd prediction = predict_elements(kernel, PhaseGrids), variance = predict_variances(kernel, PhaseGrids);
	const std::size_t NumPoints = PhaseGrids.cols();
	Eigen::VectorXd result(NumPoints);
	const std::vector<std::size_t> indices = get_indices(NumPoints);
	std::for_each(
		std::execution::par_unseq,
		indices.cbegin(),
		indices.cend(),
		[&prediction, &variance, &result](std::size_t i) -> void
		{
			if (power<2>(prediction[i]) >= power<2>(ConnectingPoint) * variance[i])
			{
				result[i] = prediction[i];
			}
			else if (power<2>(prediction[i]) <= variance[i])
			{
				result[i] = 0.0;
			}
			else
			{
				const double AbsoluteRelativePrediction = std::abs(prediction[i]) / std::sqrt(variance[i]);
				result[i] = prediction[i] * (3.0 * ConnectingPoint - 2.0 * AbsoluteRelativePrediction - 1.0)
					* power<2>(AbsoluteRelativePrediction - 1) / power<3>(ConnectingPoint - 1);
			}
		});
	return result;
}

QuantumMatrix<std::complex<double>> predict_matrix(const OptionalKernels& Kernels, const ClassicalPhaseVector& r)
{
	using namespace std::literals::complex_literals;
	auto predict = [&Kernels, &r](const std::size_t iPES, const std::size_t jPES) -> double
	{
		const std::size_t ElementIndex = iPES * NumPES + jPES;
		if (Kernels[ElementIndex].has_value())
		{
			return predict_elements(Kernels[ElementIndex].value(), r).value();
		}
		else
		{
			return 0.0;
		}
	};
	QuantumMatrix<std::complex<double>> result = QuantumMatrix<std::complex<double>>::Zero();
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES < NumPES; jPES++)
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

QuantumMatrix<std::complex<double>> predict_matrix_with_variance_comparison(const OptionalKernels& Kernels, const ClassicalPhaseVector& r)
{
	using namespace std::literals::complex_literals;
	auto predict = [&Kernels, &r](const std::size_t iPES, const std::size_t jPES) -> double
	{
		const std::size_t ElementIndex = iPES * NumPES + jPES;
		if (Kernels[ElementIndex].has_value())
		{
			return predict_elements_with_variance_comparison(Kernels[ElementIndex].value(), r).value();
		}
		else
		{
			return 0.0;
		}
	};
	QuantumMatrix<std::complex<double>> result = QuantumMatrix<std::complex<double>>::Zero();
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES < NumPES; jPES++)
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

double calculate_population(const OptionalKernels& Kernels)
{
	double result = 0.0;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		const std::size_t ElementIndex = iPES * NumPES + iPES;
		if (Kernels[ElementIndex].has_value())
		{
			result += Kernels[ElementIndex]->get_population();
		}
	}
	return result;
}

ClassicalPhaseVector calculate_1st_order_average(const OptionalKernels& Kernels)
{
	ClassicalPhaseVector result = ClassicalPhaseVector::Zero();
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		const std::size_t ElementIndex = iPES * NumPES + iPES;
		if (Kernels[ElementIndex].has_value())
		{
			result += Kernels[ElementIndex]->get_1st_order_average();
		}
	}
	return result;
}

/// @details This function uses the population on each surface by analytical integral
/// with energy calculated by averaging the energy of density
double calculate_total_energy_average(const OptionalKernels& Kernels, const QuantumVector<double>& Energies)
{
	double result = 0.0;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		const std::size_t ElementIndex = iPES * NumPES + iPES;
		if (Kernels[ElementIndex].has_value())
		{
			result += Kernels[ElementIndex]->get_population() * Energies[iPES];
		}
	}
	return result;
}

/// @details The global purity is the weighted sum of purity of all elements, with weight 1 for diagonal and 2 for off-diagonal
double calculate_purity(const OptionalKernels& Kernels)
{
	double result = 0.0;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES < NumPES; jPES++)
		{
			const std::size_t ElementIndex = iPES * NumPES + jPES;
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

ParameterVector population_derivative(const OptionalKernels& Kernels)
{
	ParameterVector result;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		const std::size_t ElementIndex = iPES * NumPES + iPES;
		if (Kernels[ElementIndex].has_value())
		{
			const ParameterVector& grad = Kernels[ElementIndex]->get_population_derivative();
			result.insert(result.cend(), grad.cbegin(), grad.cend());
		}
		else
		{
			result.insert(result.cend(), Kernel::NumTotalParameters, 0.0);
		}
	}
	return result;
}

ParameterVector total_energy_derivative(const OptionalKernels& Kernels, const QuantumVector<double>& Energies)
{
	ParameterVector result;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		const std::size_t ElementIndex = iPES * NumPES + iPES;
		if (Kernels[ElementIndex].has_value())
		{
			ParameterVector grad = Kernels[ElementIndex]->get_population_derivative();
			for (double& d : grad)
			{
				d *= Energies[iPES];
			}
			result.insert(result.cend(), grad.cbegin(), grad.cend());
		}
		else
		{
			result.insert(result.cend(), Kernel::NumTotalParameters, 0.0);
		}
	}
	return result;
}

ParameterVector purity_derivative(const OptionalKernels& Kernels)
{
	ParameterVector result;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES < NumPES; jPES++)
		{
			const std::size_t ElementIndex = iPES * NumPES + jPES;
			if (Kernels[ElementIndex].has_value())
			{
				ParameterVector grad = Kernels[ElementIndex]->get_purity_derivative();
				if (iPES != jPES)
				{
					for (double& d : grad)
					{
						d *= 2;
					}
				}
				result.insert(result.cend(), grad.cbegin(), grad.cend());
			}
			else
			{
				result.insert(result.cend(), Kernel::NumTotalParameters, 0.0);
			}
		}
	}
	return result;
}
