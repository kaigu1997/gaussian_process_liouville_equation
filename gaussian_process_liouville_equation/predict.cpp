/// @file predict.cpp
/// @brief Implementation of predict.h

#include "stdafx.h"

#include "predict.h"

#include "complex_kernel.h"
#include "kernel.h"
#include "pes.h"

/// @details The reduction is not combined in return because of STUPID clang-format.
ClassicalPhaseVector calculate_1st_order_average_one_surface(const ElementPoints& density)
{
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
		result[iPES] = calculate_total_energy_average_one_surface(density[iPES * NumPES + iPES], mass, iPES);
	}
	return result;
}

QuantumMatrix<std::complex<double>> predict_matrix(const OptionalKernels& Kernels, const ClassicalPhaseVector& r)
{
	QuantumMatrix<std::complex<double>> result = QuantumMatrix<std::complex<double>>::Zero();
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES <= iPES; jPES++)
		{
			const std::size_t ElementIndex = iPES * NumPES + jPES;
			if (iPES == jPES)
			{
				// diagonal elements, convert to Kernel
				const Kernel& TrainingKernel = dynamic_cast<const Kernel&>(*Kernels[ElementIndex].get());
				result(iPES, jPES) = Kernel(r, TrainingKernel, false).get_prediction().value();
			}
			else
			{
				// off-diagonal elements, convert to ComplexKernel
				const ComplexKernel& TrainingKernel = dynamic_cast<const ComplexKernel&>(*Kernels[ElementIndex].get());
				result(iPES, jPES) = ComplexKernel(r, TrainingKernel, false).get_prediction().value();
			}
		}
	}
	return result.selfadjointView<Eigen::Lower>();
}

QuantumMatrix<std::complex<double>> predict_matrix_with_variance_comparison(const OptionalKernels& Kernels, const ClassicalPhaseVector& r)
{
	QuantumMatrix<std::complex<double>> result = QuantumMatrix<std::complex<double>>::Zero();
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES <= iPES; jPES++)
		{
			const std::size_t ElementIndex = iPES * NumPES + jPES;
			if (iPES == jPES)
			{
				// diagonal elements, convert to Kernel
				const Kernel& TrainingKernel = dynamic_cast<const Kernel&>(*Kernels[ElementIndex].get());
				result(iPES, jPES) = Kernel(r, TrainingKernel, false).get_prediction_compared_with_variance().value();
			}
			else
			{
				// off-diagonal elements, convert to ComplexKernel
				const ComplexKernel& TrainingKernel = dynamic_cast<const ComplexKernel&>(*Kernels[ElementIndex].get());
				result(iPES, jPES) = ComplexKernel(r, TrainingKernel, false).get_prediction_compared_with_variance().value();
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
		result += dynamic_cast<const Kernel&>(*Kernels[iPES * NumPES + iPES].get()).get_population();
	}
	return result;
}

ClassicalPhaseVector calculate_1st_order_average(const OptionalKernels& Kernels)
{
	ClassicalPhaseVector result = ClassicalPhaseVector::Zero();
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		result += dynamic_cast<const Kernel&>(*Kernels[iPES * NumPES + iPES].get()).get_1st_order_average();
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
		result += dynamic_cast<const Kernel&>(*Kernels[iPES * NumPES + iPES].get()).get_population() * Energies[iPES];
	}
	return result;
}

/// @details The global purity is the weighted sum of purity of all elements, with weight 1 for diagonal and 2 for off-diagonal
double calculate_purity(const OptionalKernels& Kernels)
{
	double result = 0.0;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES <= iPES; jPES++)
		{
			const std::size_t ElementIndex = iPES * NumPES + jPES;
			if (iPES == jPES)
			{
				result += dynamic_cast<const Kernel&>(*Kernels[ElementIndex].get()).get_purity();
			}
			else
			{
				result += 2.0 * dynamic_cast<const ComplexKernel&>(*Kernels[ElementIndex].get()).get_purity();
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
		const Kernel::ParameterArray<double>& grad = dynamic_cast<const Kernel&>(*Kernels[ElementIndex].get()).get_population_derivative();
		result.insert(result.cend(), grad.cbegin(), grad.cend());
	}
	return result;
}

ParameterVector total_energy_derivative(const OptionalKernels& Kernels, const QuantumVector<double>& Energies)
{
	ParameterVector result;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		const std::size_t ElementIndex = iPES * NumPES + iPES;
		Kernel::ParameterArray<double> grad = dynamic_cast<const Kernel&>(*Kernels[ElementIndex].get()).get_population_derivative();
		for (double& d : grad)
		{
			d *= Energies[iPES];
		}
		result.insert(result.cend(), grad.cbegin(), grad.cend());
	}
	return result;
}

ParameterVector purity_derivative(const OptionalKernels& Kernels)
{
	ParameterVector result;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES <= iPES; jPES++)
		{
			const std::size_t ElementIndex = iPES * NumPES + jPES;
			if (iPES == jPES)
			{
				const Kernel::ParameterArray<double>& grad = dynamic_cast<const Kernel&>(*Kernels[ElementIndex].get()).get_purity_derivative();
				result.insert(result.cend(), grad.cbegin(), grad.cend());
			}
			else
			{
				ComplexKernel::ParameterArray<double> grad = dynamic_cast<const ComplexKernel&>(*Kernels[ElementIndex].get()).get_purity_derivative();
				for (double& d : grad)
				{
					d *= 2;
				}
				result.insert(result.cend(), grad.cbegin(), grad.cend());
			}
		}
	}
	return result;
}

AllPoints generate_extra_points(const AllPoints& density, const std::size_t NumPoints, const DistributionFunction& distribution)
{
	using namespace std::literals::complex_literals;
	AllPoints result;
	static std::mt19937 engine(std::chrono::system_clock::now().time_since_epoch().count()); // Random number generator, using merseen twister with seed from time
	std::array<std::normal_distribution<double>, PhaseDim> normdists;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES < NumPES; jPES++)
		{
			if (iPES <= jPES)
			{
				// for row-major, visit upper triangular elements earlier
				// so only deal with upper-triangular
				const std::size_t ElementIndex = iPES * NumPES + jPES;
				const ClassicalPhaseVector stddev = calculate_standard_deviation_one_surface(density[ElementIndex]);
				for (std::size_t iDim = 0; iDim < PhaseDim; iDim++)
				{
					normdists[iDim] = std::normal_distribution<double>(0.0, stddev[iDim]);
				}
				// generation
				result[ElementIndex] = ElementPoints(NumPoints, std::make_tuple(ClassicalPhaseVector::Zero(), QuantumMatrix<std::complex<double>>::Zero()));
				const auto indices = xt::arange(result[ElementIndex].size());
				std::for_each(
					std::execution::par_unseq,
					indices.cbegin(),
					indices.cend(),
					[&density = density[ElementIndex], &result = result[ElementIndex], &normdists, &distribution](std::size_t iPoint) -> void
					{
						auto& [r, rho] = result[iPoint];
						r = std::get<0>(density[iPoint % density.size()]);
						for (std::size_t iDim = 0; iDim < PhaseDim; iDim++)
						{
							// a deviation from the center
							r[iDim] += normdists[iDim](engine);
						}
						// current distribution
						rho = distribution(r);
					});
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
