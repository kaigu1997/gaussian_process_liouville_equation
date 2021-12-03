/// @file predict.cpp
/// @brief Implementation of predict.h

#include "stdafx.h"

#include "predict.h"

#include "kernel.h"
#include "mc_ave.h"
#include "opt.h"

Eigen::VectorXd construct_training_feature(const EigenVector<ClassicalVector<double>>& x, const EigenVector<ClassicalVector<double>>& p)
{
	assert(x.size() == p.size());
	Eigen::VectorXd result(PhaseDim, x.size());
	for (std::size_t i = 0; i < x.size(); i++)
	{
		result.col(i) << x[i], p[i];
	}
	return result;
}

/// @details Using Gaussian Process Regression to predict by
///
/// \f[
/// \mathbb{E}[p(\mathbf{f}_*|X_*,X,\mathbf{y})]=K(X_*,X)[K(X,X)+\sigma_n^2I]^{-1}\mathbf{y}
/// \f]
///
/// where \f$ X_* \f$ is the test features, \f$ X \f$ is the training features,
/// and \f$ \mathbf{y} \f$ is the training labels.
///
/// The prediction result will be compared with variance, given by
///
/// \f[
/// \mathrm{Var}=K(X_*,X_*)-K(X_*,X)[K(X,X)+\sigma_n^2I]^{-1}K(X,X_*)
/// \f]
///
/// and if the prediction is smaller than variance, the prediction will be set as 0.
///
/// The variance matrix is generally too big and not necessary, so it will be calculated elementwise.
Eigen::VectorXd Prediction::predict_elements(const Eigen::MatrixXd& PhaseGrids) const
{
	const Kernel k_new_old(Prms, PhaseGrids, TrainingFeatures);
	const int NumPoints = PhaseGrids.cols();
	const auto ave = k_new_old.get_kernel() * Krn.get_inverse_label();
	const Eigen::VectorXd var = [&](void) -> Eigen::VectorXd
	{
		Eigen::VectorXd result(NumPoints);
		auto calc_result = [&](const int iCol) -> double
		{
			const auto feature = PhaseGrids.col(iCol);
			const Kernel k_i(Prms, feature, feature);
			return (k_i.get_kernel() - k_new_old.get_kernel().row(iCol) * Krn.get_inverse() * k_new_old.get_kernel().row(iCol).transpose()).value();
		};
		if (NumPoints > omp_get_num_threads())
		{
#pragma omp parallel for
			for (int iCol = 0; iCol < NumPoints; iCol++)
			{
				result[iCol] = calc_result(iCol);
			}
		}
		else
		{
			for (int iCol = 0; iCol < NumPoints; iCol++)
			{
				result[iCol] = calc_result(iCol);
			}
		}
		return result;
	}();
	return ave.array() * (ave.array().abs2() > var.array()).cast<double>();
}

void construct_predictors(
	Predictions& predictors,
	const Optimization& Optimizer,
	const QuantumMatrix<bool>& IsSmall,
	const EigenVector<PhaseSpacePoint>& density)
{
	for (int iElement = 0; iElement < NumElements; iElement++)
	{
		if (!IsSmall(iElement / NumPES, iElement % NumPES))
		{
			const auto& [feature, label] = construct_element_training_set(density, iElement);
			predictors[iElement].emplace(Optimizer.get_parameter(iElement), feature, label);
		}
		else
		{
			predictors[iElement] = std::nullopt;
		}
	}
}

QuantumMatrix<std::complex<double>> predict_matrix(
	const Predictions& Predictors,
	const ClassicalVector<double>& x,
	const ClassicalVector<double>& p)
{
	using namespace std::literals::complex_literals;
	assert(x.size() == p.size());
	auto predict = [&](const int ElementIndex) -> double
	{
		if (Predictors[ElementIndex].has_value())
		{
			Eigen::MatrixXd PhaseCoord(PhaseDim, 1);
			PhaseCoord << x, p;
			return Predictors[ElementIndex]->predict_elements(PhaseCoord).value();
		}
		else
		{
			return 0.0;
		}
	};
	QuantumMatrix<std::complex<double>> result = QuantumMatrix<std::complex<double>>::Zero();
	for (int iElement = 0; iElement < NumElements; iElement++)
	{
		const int iPES = iElement / NumPES, jPES = iElement % NumPES;
		const double ElementValue = predict(iElement);
		if (iPES <= jPES)
		{
			result(jPES, iPES) += ElementValue;
		}
		else
		{
			result(iPES, jPES) += 1.0i * ElementValue;
		}
	}
	return result.selfadjointView<Eigen::Lower>();
}

double calculate_population(const Predictions& Predictors)
{
	double result = 0.0;
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		const int ElementIndex = iPES * (NumPES + 1);
		if (Predictors[ElementIndex].has_value())
		{
			result += Predictors[ElementIndex]->calculate_population();
		}
	}
	return result;
}

ClassicalPhaseVector calculate_1st_order_average(const Predictions& Predictors)
{
	ClassicalPhaseVector result = ClassicalPhaseVector::Zero();
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		const int ElementIndex = iPES * (NumPES + 1);
		if (Predictors[ElementIndex].has_value())
		{
			result += Predictors[ElementIndex]->calculate_1st_order_average();
		}
	}
	return result;
}

double calculate_total_energy_average(
	const Predictions& Predictors,
	const EigenVector<PhaseSpacePoint>& density,
	const ClassicalVector<double>& mass)
{
	double result = 0.0;
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		const int ElementIndex = iPES * (NumPES + 1);
		if (Predictors[ElementIndex].has_value())
		{
			result += Predictors[ElementIndex]->calculate_population() * calculate_total_energy_average(density, iPES, mass);
		}
	}
	return result;
}

/// @details The global purity is the weighted sum of purity of all elements, with weight 1 for diagonal and 2 for off-diagonal
double calculate_purity(const Predictions& Predictors)
{
	double result = 0.0;
	for (int iElement = 0; iElement < NumElements; iElement++)
	{
		if (Predictors[iElement].has_value())
		{
			if (iElement / NumPES == iElement % NumPES)
			{
				result += Predictors[iElement]->calculate_purity();
			}
			else
			{
				result += 2.0 * Predictors[iElement]->calculate_purity();
			}
		}
	}
	return result;
}
