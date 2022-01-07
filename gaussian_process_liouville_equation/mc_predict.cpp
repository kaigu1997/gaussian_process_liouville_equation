/// @file mc_predict.cpp
/// @brief Implementation of mc_predict.h

#include "stdafx.h"

#include "mc_predict.h"

#include "input.h"
#include "mc.h"
#include "pes.h"
#include "predict.h"

/// @details The result will be @"symmetric@".
AllPoints mc_points_generator(const AllPoints& density, const int NumPoints)
{
	using namespace std::literals::complex_literals;
	AllPoints result;
	static std::mt19937 engine(std::chrono::system_clock::now().time_since_epoch().count()); // Random number generator, using merseen twister with seed from time
	std::array<std::normal_distribution<double>, PhaseDim> dist;
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		for (int jPES = 0; jPES < NumPES; jPES++)
		{
			if (iPES <= jPES)
			{
				// for row-major, visit upper triangular elements earlier
				// so only deal with upper-triangular
				const int ElementIndex = iPES * NumPES + jPES;
				if (!density[ElementIndex].empty())
				{
					// distributions
					const ClassicalPhaseVector ave = [&pts = density[ElementIndex], iPES, jPES](void) -> ClassicalPhaseVector
					{
						ClassicalPhaseVector result;
						[[maybe_unused]] const auto& [x_max, p_max, rho_max] = *std::max_element(
							std::execution::par_unseq,
							pts.cbegin(),
							pts.cend(),
							[iPES, jPES](const PhaseSpacePoint& psp1, const PhaseSpacePoint& psp2) -> bool
							{
								return weight_function(std::get<2>(psp1)(iPES, jPES)) < weight_function(std::get<2>(psp2)(iPES, jPES));
							});
						result << x_max, p_max;
						return result;
					}();
					const ClassicalPhaseVector stddev = calculate_standard_deviation(density[ElementIndex]);
					spdlog::info("Sampling for ({},{}) element, center at {} with width {}.", iPES, jPES, ave.format(VectorFormatter), stddev.format(VectorFormatter));
					const double var_prod = stddev.prod();
					for (int iDim = 0; iDim < PhaseDim; iDim++)
					{
						dist[iDim] = std::normal_distribution<double>(ave[iDim], stddev[iDim]);
					}
					// generate
					result[ElementIndex] = EigenVector<PhaseSpacePoint>(NumPoints, std::make_tuple(ClassicalVector<double>::Zero(), ClassicalVector<double>::Zero(), QuantumMatrix<std::complex<double>>::Zero()));
					std::for_each(
						std::execution::par_unseq,
						result[ElementIndex].begin(),
						result[ElementIndex].end(),
						[&dist, &ave, &stddev, var_prod](PhaseSpacePoint& psp) -> void
						{
							auto& [x, p, rho] = psp;
							for (int iDim = 0; iDim < Dim; iDim++)
							{
								x[iDim] = dist[iDim](engine);
								p[iDim] = dist[iDim + Dim](engine);
							}
							ClassicalPhaseVector r;
							r << x, p;
							// normalized weight
							rho(0, 0) = std::exp(-((r - ave).array() / stddev.array()).square().sum() / 2.0) / (std::pow(2.0 * M_PI, Dim) * var_prod);
						});
				}
			}
			else
			{
				// for strict lower triangular elements, copy the upper part
				result[iPES * NumPES + jPES] = result[jPES * NumPES + iPES];
			}
		}
	}
	return result;
}

/// @details The monte carlo integral of observable O of element @f$ \rho^{\alpha\alpha}_W @f$ on this element is
///
/// @f[
/// \langle O\rangle=\frac{1}{N}\sum_{i=0}^{N-1}\frac{1}{w(x_i)}\rho^{\alpha\alpha}(x_i)O^{\alpha\alpha}(x_i)
/// @f]
///
/// The reduction is not combined in return because of STUPID clang-format.
double calculate_population_one_surface(const Kernel& kernel, const AllPoints& mc_points)
{
	assert(kernel.is_same_feature());
	double result = 0.0;
	int n_surface = 0;
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		const int ElementIndex = iPES * NumPES + iPES;
		if (!mc_points[ElementIndex].empty())
		{
			const double reduction =
				std::transform_reduce(
					std::execution::par_unseq,
					mc_points[ElementIndex].cbegin(),
					mc_points[ElementIndex].cend(),
					0.0,
					std::plus<double>(),
					[&kernel](const PhaseSpacePoint& psp) -> double
					{
						const auto& [x, p, rho] = psp;
						return predict_elements(kernel, construct_training_feature({x}, {p})).value() / rho(0, 0).real();
					});
			result += reduction / mc_points[ElementIndex].size();
			n_surface++;
		}
	}
	return result / n_surface;
}

ClassicalPhaseVector calculate_1st_order_average_one_surface(const Kernel& kernel, const AllPoints& mc_points)
{
	assert(kernel.is_same_feature());
	ClassicalPhaseVector result = ClassicalPhaseVector::Zero();
	int n_surface = 0;
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		const int ElementIndex = iPES * NumPES + iPES;
		if (!mc_points[ElementIndex].empty())
		{
			const ClassicalPhaseVector reduction =
				std::transform_reduce(
					std::execution::par_unseq,
					mc_points[ElementIndex].cbegin(),
					mc_points[ElementIndex].cend(),
					ClassicalPhaseVector(ClassicalPhaseVector::Zero()),
					std::plus<ClassicalPhaseVector>(),
					[&kernel](const PhaseSpacePoint& psp) -> ClassicalPhaseVector
					{
						const auto& [x, p, rho] = psp;
						ClassicalPhaseVector r;
						r << x, p;
						return predict_elements(kernel, r).value() / rho(0, 0).real() * r;
					});
			result += reduction / mc_points[ElementIndex].size();
			n_surface++;
		}
	}
	return result / n_surface / calculate_population_one_surface(kernel, mc_points);
}

double calculate_total_energy_average_one_surface(
	const Kernel& kernel,
	const AllPoints& mc_points,
	const ClassicalVector<double>& mass,
	const int PESIndex)
{
	assert(kernel.is_same_feature());
	double result = 0.0;
	int n_surface = 0;
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		const int ElementIndex = iPES * NumPES + iPES;
		if (!mc_points[ElementIndex].empty())
		{
			const double reduction =
				std::transform_reduce(
					std::execution::par_unseq,
					mc_points[ElementIndex].cbegin(),
					mc_points[ElementIndex].cend(),
					0.0,
					std::plus<double>(),
					[&kernel, &mass, PESIndex](const PhaseSpacePoint& psp) -> double
					{
						const auto& [x, p, rho] = psp;
						const double V = adiabatic_potential(x)[PESIndex];
						const double T = (p.array().abs2() / mass.array()).sum() / 2.0;
						return (T + V) * predict_elements(kernel, construct_training_feature({x}, {p})).value() / rho(0, 0).real();
					});
			result += reduction / mc_points[ElementIndex].size();
			n_surface++;
		}
	}
	return result / n_surface / calculate_population_one_surface(kernel, mc_points);
}

/// @details The monte carlo integral of observable O of one element is
///
/// @f[
/// \langle O\rangle=\frac{1}{N}\sum_{i=0}^{N-1}\frac{1}{w(x_i)}\sum_{\alpha}\rho^{\alpha\alpha}(x_i)O^{\alpha\alpha}(x_i)
/// @f]
///
/// The difference is that now it sums over all @f$ \alpha @f$, and the result is the average of @<O@> on all surfaces
double calculate_population(const OptionalKernels& Kernels, const AllPoints& mc_points)
{
	double result = 0.0;
	int n_surface = 0;
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		const int ElementIndex = iPES * NumPES + iPES;
		if (!mc_points[ElementIndex].empty())
		{
			const double reduction =
				std::transform_reduce(
					std::execution::par_unseq,
					mc_points[ElementIndex].cbegin(),
					mc_points[ElementIndex].cend(),
					0.0,
					std::plus<double>(),
					[&Kernels](const PhaseSpacePoint& psp) -> double
					{
						const auto& [x, p, rho] = psp;
						return predict_matrix(Kernels, x, p).trace().real() / rho(0, 0).real();
					});
			result += reduction / mc_points[ElementIndex].size();
			n_surface++;
		}
	}
	return result / n_surface;
}

ClassicalPhaseVector calculate_1st_order_average(const OptionalKernels& Kernels, const AllPoints& mc_points)
{
	ClassicalPhaseVector result = ClassicalPhaseVector::Zero();
	int n_surface = 0;
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		const int ElementIndex = iPES * NumPES + iPES;
		if (!mc_points[ElementIndex].empty())
		{
			const ClassicalPhaseVector reduction =
				std::transform_reduce(
					std::execution::par_unseq,
					mc_points[ElementIndex].cbegin(),
					mc_points[ElementIndex].cend(),
					ClassicalPhaseVector(ClassicalPhaseVector::Zero()),
					std::plus<ClassicalPhaseVector>(),
					[&Kernels](const PhaseSpacePoint& psp) -> ClassicalPhaseVector
					{
						const auto& [x, p, rho] = psp;
						ClassicalPhaseVector r = ClassicalPhaseVector::Zero();
						r << x, p;
						return predict_matrix(Kernels, x, p).trace().real() / rho(0, 0).real() * r;
					});
			result += reduction / mc_points[ElementIndex].size();
			n_surface++;
		}
	}
	return result / n_surface;
}

double calculate_total_energy_average(
	const OptionalKernels& Kernels,
	const AllPoints& mc_points,
	const ClassicalVector<double>& mass)
{
	double result = 0.0;
	int n_surface = 0;
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		const int ElementIndex = iPES * NumPES + iPES;
		if (!mc_points[ElementIndex].empty())
		{
			const double reduction =
				std::transform_reduce(
					std::execution::par_unseq,
					mc_points[ElementIndex].cbegin(),
					mc_points[ElementIndex].cend(),
					0.0,
					std::plus<double>(),
					[&Kernels, &mass](const PhaseSpacePoint& psp) -> double
					{
						const auto& [x, p, rho] = psp;
						const auto ppls = predict_matrix(Kernels, x, p).diagonal().real();
						const double V = (adiabatic_potential(x).array() * ppls.array()).sum();
						const double T = ppls.sum() * (p.array().abs2() / mass.array()).sum() / 2.0;
						return (T + V) / rho(0, 0).real();
					});
			result += reduction / mc_points[ElementIndex].size();
			n_surface++;
		}
	}
	return result / n_surface;
}

/// @details The integral is
///
/// @f[
/// S=\frac{(2\pi\hbar)^{\frac{D}{2}}}{N}\sum_{i=0}^{N-1}\frac{1}{w(x_i)}
/// \left(\sum_{\alpha=0}^{M-1}[\rho^{\alpha\alpha}(x_i)]^2+2\sum_{\alpha\neq\beta}[\rho^{\alpha\beta}(x_i)]^2\right)
/// @f]
double calculate_purity(const OptionalKernels& Kernels, const AllPoints& mc_points)
{
	double result = 0.0;
	int n_surface = 0;
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		const int ElementIndex = iPES * NumPES + iPES;
		if (!mc_points[ElementIndex].empty())
		{
			const double reduction =
				std::transform_reduce(
					std::execution::par_unseq,
					mc_points[ElementIndex].cbegin(),
					mc_points[ElementIndex].cend(),
					0.0,
					std::plus<double>(),
					[&Kernels](const PhaseSpacePoint& psp) -> double
					{
						const auto& [x, p, rho] = psp;
						return predict_matrix(Kernels, x, p).squaredNorm() / rho(0, 0).real();
					});
			result += reduction / mc_points[ElementIndex].size();
			n_surface++;
		}
	}
	return PurityFactor * result / n_surface;
}

/// @details The derivatives of observables O over parameters are
///
/// @f[
/// \frac{\partial\langle O\rangle}{\partial\theta^{\gamma\gamma}}
/// =\frac{1}{N}\sum_{i=0}^{N-1}\frac{1}{w(x_i)}\frac{\partial\rho^{\gamma\gamma}(x_i)}{\partial\theta^{\gamma\gamma}}O^{\gamma\gamma}(x_i)
/// @f]
///
/// And this function will calculate the derivative of population over parameters from all diagonal elements.
ParameterVector population_derivative(const OptionalKernels& Kernels, const AllPoints& mc_points)
{
	Eigen::VectorXd result = Eigen::VectorXd::Zero(NumPES * Kernel::NumTotalParameters);
	int n_surface = 0;
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		const int PointElementIndex = iPES * NumPES + iPES;
		if (!mc_points[PointElementIndex].empty())
		{
			const Eigen::VectorXd reduction =
				std::transform_reduce(
					std::execution::par_unseq,
					mc_points[PointElementIndex].cbegin(),
					mc_points[PointElementIndex].cend(),
					Eigen::VectorXd(Eigen::VectorXd::Zero(NumPES * Kernel::NumTotalParameters)),
					std::plus<Eigen::VectorXd>(),
					[&Kernels](const PhaseSpacePoint& psp) -> Eigen::VectorXd
					{
						const auto& [x, p, rho] = psp;
						Eigen::VectorXd result = Eigen::VectorXd::Zero(NumPES * Kernel::NumTotalParameters);
						for (int iPES = 0; iPES < NumPES; iPES++)
						{
							const int KernelElementIndex = iPES * NumPES + iPES;
							if (Kernels[KernelElementIndex].has_value())
							{
								result.block(iPES * Kernel::NumTotalParameters, 0, Kernel::NumTotalParameters, 1) = Eigen::VectorXd::Map(prediction_derivative(Kernels[KernelElementIndex].value(), x, p).data(), Kernel::NumTotalParameters);
							}
						}
						return result / rho(0, 0).real();
					});
			result += reduction / mc_points[PointElementIndex].size();
			n_surface++;
		}
	}
	result /= n_surface;
	return ParameterVector(result.data(), result.data() + NumPES * Kernel::NumTotalParameters);
}

ParameterVector total_energy_derivative(
	const OptionalKernels& Kernels,
	const AllPoints& mc_points,
	const ClassicalVector<double>& mass)
{
	Eigen::VectorXd result = Eigen::VectorXd::Zero(NumPES * Kernel::NumTotalParameters);
	int n_surface = 0;
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		const int PointElementIndex = iPES * NumPES + iPES;
		if (!mc_points[PointElementIndex].empty())
		{
			const Eigen::VectorXd reduction =
				std::transform_reduce(
					std::execution::par_unseq,
					mc_points[PointElementIndex].cbegin(),
					mc_points[PointElementIndex].cend(),
					Eigen::VectorXd(Eigen::VectorXd::Zero(NumPES * Kernel::NumTotalParameters)),
					std::plus<Eigen::VectorXd>(),
					[&Kernels, &mass](const PhaseSpacePoint& psp) -> Eigen::VectorXd
					{
						const auto& [x, p, rho] = psp;
						const QuantumVector<double> V = adiabatic_potential(x);
						const double T = (p.array().square() / mass.array()).sum() / 2.0;
						Eigen::VectorXd result = Eigen::VectorXd::Zero(NumPES * Kernel::NumTotalParameters);
						for (int iPES = 0; iPES < NumPES; iPES++)
						{
							const int KernelElementIndex = iPES * NumPES + iPES;
							if (Kernels[KernelElementIndex].has_value())
							{
								result.block(iPES * Kernel::NumTotalParameters, 0, Kernel::NumTotalParameters, 1) = Eigen::VectorXd::Map(prediction_derivative(Kernels[KernelElementIndex].value(), x, p).data(), Kernel::NumTotalParameters) * (T + V[iPES]);
							}
						}
						return result / rho(0, 0).real();
					});
			result += reduction / mc_points[PointElementIndex].size();
			n_surface++;
		}
	}
	result /= n_surface;
	return ParameterVector(result.data(), result.data() + NumPES * Kernel::NumTotalParameters);
}

/**
 * @details The derivatives of purity over parameters are
 *
 * @f{eqnarray}{
 * \frac{\partial S}{\partial\theta^{\gamma\gamma}}&=&\frac{2(2\pi\hbar)^{\frac{D}{2}}}{N}\sum_{i=0}^{N-1}
 * \frac{\rho^{\gamma\gamma}}{w(x_i)}\frac{\partial\rho^{\gamma\gamma}(x_i)}{\partial\theta^{\gamma\gamma}} \\
 * \frac{\partial S}{\partial\theta^{\gamma\delta}}&=&\frac{4(2\pi\hbar)^{\frac{D}{2}}}{N}\sum_{i=0}^{N-1}
 * \frac{\rho^{\gamma\delta}}{w(x_i)}\frac{\partial\rho^{\gamma\delta}(x_i)}{\partial\theta^{\gamma\delta}}
 * @f}
 * 
 * The difference is the factor 2 vs 4 for diagonal vs off-diagonal elements.
 * 
 * This function will calculate the derivative over parameters from all elements.
 */
ParameterVector purity_derivative(const OptionalKernels& Kernels, const AllPoints& mc_points)
{
	Eigen::VectorXd result = Eigen::VectorXd::Zero(NumElements * Kernel::NumTotalParameters);
	int n_element = 0;
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		for (int jPES = 0; jPES < NumPES; jPES++)
		{
			const int PointElementIndex = iPES * NumPES + iPES;
			if (!mc_points[PointElementIndex].empty())
			{
				const Eigen::VectorXd reduction =
					std::transform_reduce(
						std::execution::par_unseq,
						mc_points[PointElementIndex].cbegin(),
						mc_points[PointElementIndex].cend(),
						Eigen::VectorXd(Eigen::VectorXd::Zero(NumElements * Kernel::NumTotalParameters)),
						std::plus<Eigen::VectorXd>(),
						[&Kernels](const PhaseSpacePoint& psp) -> Eigen::VectorXd
						{
							const auto& [x, p, rho] = psp;
							Eigen::VectorXd result = Eigen::VectorXd::Zero(NumElements * Kernel::NumTotalParameters);
							for (int iiPES = 0; iiPES < NumPES; iiPES++)
							{
								for (int jjPES = 0; jjPES < NumPES; jjPES++)
								{
									const int KernelElementIndex = iiPES * NumPES + jjPES;
									if (Kernels[KernelElementIndex].has_value())
									{
										const int factor = (iiPES == jjPES ? 2 : 4);
										result.block(KernelElementIndex * Kernel::NumTotalParameters, 0, Kernel::NumTotalParameters, 1) = factor * predict_elements(Kernels[KernelElementIndex].value(), construct_training_feature({x}, {p})).value()
											* Eigen::VectorXd::Map(prediction_derivative(Kernels[KernelElementIndex].value(), x, p).data(), Kernel::NumTotalParameters);
									}
								}
							}
							return result / rho(0, 0).real();
						});
				result += reduction / mc_points[PointElementIndex].size();
				n_element++;
			}
		}
	}
	result *= PurityFactor / n_element;
	return ParameterVector(result.data(), result.data() + NumElements * Kernel::NumTotalParameters);
}
