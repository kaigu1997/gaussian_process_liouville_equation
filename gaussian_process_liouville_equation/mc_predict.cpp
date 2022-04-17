/// @file mc_predict.cpp
/// @brief Implementation of mc_predict.h

#include "stdafx.h"

#include "mc_predict.h"

#include "input.h"
#include "mc.h"
#include "pes.h"
#include "predict.h"

/// @details The result will be @"symmetric@".
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
				if (!density[ElementIndex].empty())
				{
					// distributions
					const ClassicalPhaseVector ave = std::get<0>(*std::max_element(
						std::execution::par_unseq,
						density[ElementIndex].cbegin(),
						density[ElementIndex].cend(),
						[iPES, jPES](const PhaseSpacePoint& psp1, const PhaseSpacePoint& psp2) -> bool
						{
							return std::abs(std::get<1>(psp1)(iPES, jPES)) < std::abs(std::get<1>(psp2)(iPES, jPES));
						})); // ave for new generation, max of the current
					const ClassicalPhaseVector stddev = 1.1 * calculate_standard_deviation_one_surface(density[ElementIndex]);
					spdlog::info("Sampling for ({},{}) element, center at {} with width {}.", iPES, jPES, ave.format(VectorFormatter), stddev.format(VectorFormatter));
					for (std::size_t iDim = 0; iDim < PhaseDim; iDim++)
					{
						normdists[iDim] = std::normal_distribution<double>(ave[iDim], stddev[iDim]);
					}
					// generation
					result[ElementIndex] = ElementPoints(NumPoints, std::make_tuple(ClassicalPhaseVector::Zero(), QuantumMatrix<std::complex<double>>::Zero()));
					std::for_each(
						std::execution::par_unseq,
						result[ElementIndex].begin(),
						result[ElementIndex].end(),
						[&normdists, &distribution](PhaseSpacePoint& psp) -> void
						{
							auto& [r, rho] = psp;
							for (std::size_t iDim = 0; iDim < PhaseDim; iDim++)
							{
								r[iDim] = normdists[iDim](engine);
							}
							// current distribution
							rho = distribution(r);
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

/// @details First, copy from density to reach the needed number of points. @n
/// Then, do the Metropolis MCMC to reach the balance. @n
/// Finally, divide over the normalization factor, the integral of @f$ \rho^2 @f$.
AllPoints generate_monte_carlo_points(
	const MCParameters& MCParams,
	const OptionalKernels& Kernels,
	const AllPoints& density,
	const std::size_t NumPoints,
	const DistributionFunction& distribution)
{
	AllPoints result;
	// first, using density to fill result
	// and also calculate the normalization factor
	QuantumMatrix<double> RhoSquareIntegral = QuantumMatrix<double>::Zero();
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES < NumPES; jPES++)
		{
			if (iPES <= jPES)
			{
				// for row-major, visit upper triangular elements earlier
				// so only deal with upper-triangular
				const std::size_t ElementIndex = iPES * NumPES + jPES, SymElementIndex = jPES * NumPES + iPES;
				if (!density[ElementIndex].empty())
				{
					// construct the density with same r but different rho (by distribution)
					ElementPoints modified_density = density[ElementIndex];
					std::for_each(
						std::execution::par_unseq,
						modified_density.begin(),
						modified_density.end(),
						[&distribution](PhaseSpacePoint& psp) -> void
						{
							auto& [r, rho] = psp;
							rho = distribution(r);
						});
					const std::size_t DensitySize = density[ElementIndex].size();
					while (NumPoints - result[ElementIndex].size() > DensitySize)
					{
						result[ElementIndex].insert(
							result[ElementIndex].cend(),
							modified_density.cbegin(),
							modified_density.cend());
					}
					if (result[ElementIndex].size() < NumPoints)
					{
						result[ElementIndex].insert(
							result[ElementIndex].cend(),
							modified_density.cbegin(),
							modified_density.cbegin() + NumPoints - result[ElementIndex].size());
					}
					if (Kernels[ElementIndex].has_value())
					{
						RhoSquareIntegral(iPES, jPES) += Kernels[ElementIndex]->get_purity();
					}
					if (iPES != jPES && Kernels[SymElementIndex].has_value())
					{
						RhoSquareIntegral(iPES, jPES) += Kernels[SymElementIndex]->get_purity();
					}
				}
			}
			else
			{
				// for strict lower triangular elements, copy the upper part
				result[iPES * NumPES + jPES] = result[jPES * NumPES + iPES];
			}
		}
	}
	RhoSquareIntegral = (RhoSquareIntegral / PurityFactor).selfadjointView<Eigen::Upper>();
	// second, random walk
	monte_carlo_selection(MCParams, distribution, result);
	// last, divide over the normalization factor
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES < NumPES; jPES++)
		{
			if (iPES <= jPES)
			{
				// for row-major, visit upper triangular elements earlier
				// so only deal with upper-triangular
				const std::size_t ElementIndex = iPES * NumPES + jPES;
				if (!density[ElementIndex].empty())
				{
					std::for_each(
						std::execution::par_unseq,
						result[ElementIndex].begin(),
						result[ElementIndex].end(),
						[&RhoSquareIntegral](PhaseSpacePoint& psp) -> void
						{
							[[maybe_unused]] auto& [r, rho] = psp;
							rho.array() /= RhoSquareIntegral.array().cast<std::complex<double>>();
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

/// @brief To calculate population of one surface by monte carlo integral using one element
/// @param[in] kernel The kernel cooresponding to the surface for prediction
/// @param[in] Points The selected points for calculating mc integration
/// @param[in] iPES The row index of the element in density matrix ( @p Points corresponding to)
/// @param[in] jPES The column index of the element in density matrix ( @p Points corresponding to)
/// @return The population of one surface using one element
/// @details The monte carlo integral of observable O of element @f$ \rho^{\alpha\alpha}_W @f$ on this element is @n
/// @f[
/// \langle O\rangle=\frac{1}{N}\sum_{i=0}^{N-1}\frac{1}{w(x_i)}\rho^{\alpha\alpha}(x_i)O^{\alpha\alpha}(x_i)
/// @f] @n
/// The reduction is not combined in return because of STUPID clang-format.
static double calculate_population_one_surface_one_element(
	const Kernel& kernel,
	const ElementPoints& Points,
	const std::size_t iPES,
	const std::size_t jPES)
{
	assert(kernel.is_same_feature() && !Points.empty());
	const double result =
		std::transform_reduce(
			std::execution::par_unseq,
			Points.cbegin(),
			Points.cend(),
			0.0,
			std::plus<double>(),
			[&kernel, iPES, jPES](const PhaseSpacePoint& psp) -> double
			{
				const auto& [r, rho] = psp;
				return predict_elements(kernel, r).value() / std::abs(rho(iPES, jPES));
			});
	return result / Points.size();
}

/// @brief To calculate the @<x@> and @<p@> on one surface by monte carlo integral using one element
/// @param[in] kernel The kernel cooresponding to the surface for prediction
/// @param[in] Points The selected points for calculating mc integration
/// @param[in] iPES The row index of the element in density matrix ( @p Points corresponding to)
/// @param[in] jPES The column index of the element in density matrix ( @p Points corresponding to)
/// @return The @<x@> and @<p@> of one surface using one element
static ClassicalPhaseVector calculate_1st_order_average_one_surface_one_element(
	const Kernel& kernel,
	const ElementPoints& Points,
	const std::size_t iPES,
	const std::size_t jPES)
{
	assert(kernel.is_same_feature() && !Points.empty());
	const ClassicalPhaseVector result =
		std::transform_reduce(
			std::execution::par_unseq,
			Points.cbegin(),
			Points.cend(),
			ClassicalPhaseVector(ClassicalPhaseVector::Zero()),
			std::plus<ClassicalPhaseVector>(),
			[&kernel, iPES, jPES](const PhaseSpacePoint& psp) -> ClassicalPhaseVector
			{
				const auto& [r, rho] = psp;
				return predict_elements(kernel, r).value() / std::abs(rho(iPES, jPES)) * r;
			});
	return result / Points.size();
}

/// @brief To calculate the energy on one surface by monte carlo integration using one element
/// @param[in] kernel The kernel for prediction
/// @param[in] Points The selected points for calculating mc integration
/// @param[in] mass Mass of classical degree of freedom
/// @param[in] PESIndex The row and column index of the surface in density matrix ( @p kernel corresponding to)
/// @param[in] iPES The row index of the element in density matrix ( @p Points corresponding to)
/// @param[in] jPES The column index of the element in density matrix ( @p Points corresponding to)
/// @return The energy of one surface using one element
static double calculate_total_energy_average_one_surface_one_element(
	const Kernel& kernel,
	const ElementPoints& Points,
	const ClassicalVector<double>& mass,
	const std::size_t PESIndex,
	const std::size_t iPES,
	const std::size_t jPES)
{
	assert(kernel.is_same_feature() && !Points.empty());
	const double result =
		std::transform_reduce(
			std::execution::par_unseq,
			Points.cbegin(),
			Points.cend(),
			0.0,
			std::plus<double>(),
			[&kernel, &mass, PESIndex, iPES, jPES](const PhaseSpacePoint& psp) -> double
			{
				const auto& [r, rho] = psp;
				const auto [x, p] = split_phase_coordinate(r);
				const double V = adiabatic_potential(x)[PESIndex];
				const double T = (p.array().square() / mass.array()).sum() / 2.0;
				return (T + V) * predict_elements(kernel, r).value() / std::abs(rho(iPES, jPES));
			});
	return result / Points.size();
}

/// @brief To calculate the norm of one element by monte carlo integration using one element (the two elements need not be the same)
/// @param[in] Kernels The kernel for prediction
/// @param[in] Points The selected points for calculating mc integration
/// @param[in] iKernelPES The row index of the integrated element in density matrix ( @p Kernels corresponding to)
/// @param[in] jKernelPES The column index of the integrated element in density matrix ( @p Kernels corresponding to)
/// @param[in] iPointPES The row index of the element doing integral in density matrix ( @p Points corresponding to)
/// @param[in] jPointPES The column index of the element doing integral in density matrix ( @p Points corresponding to)
/// @return The norm of one element using one element
static double calculate_rho_square_one_element_one_element(
	const OptionalKernels& Kernels,
	const ElementPoints& Points,
	const std::size_t iKernelPES,
	const std::size_t jKernelPES,
	const std::size_t iPointPES,
	const std::size_t jPointPES)
{
	assert(!Points.empty());
	// count for both real and imaginary parts
	double result = 0.0;
	if (Kernels[iKernelPES * NumPES + jKernelPES].has_value())
	{
		result +=
			std::transform_reduce(
				std::execution::par_unseq,
				Points.cbegin(),
				Points.cend(),
				0.0,
				std::plus<double>(),
				[&kernel = Kernels[iKernelPES * NumPES + jKernelPES].value(), iPointPES, jPointPES](const PhaseSpacePoint& psp) -> double
				{
					const auto& [r, rho] = psp;
					return predict_elements(kernel, r).squaredNorm() / std::abs(rho(iPointPES, jPointPES));
				});
	}
	if (iKernelPES != jKernelPES && Kernels[jKernelPES * NumPES + iKernelPES].has_value())
	{
		result +=
			std::transform_reduce(
				std::execution::par_unseq,
				Points.cbegin(),
				Points.cend(),
				0.0,
				std::plus<double>(),
				[&kernel = Kernels[jKernelPES * NumPES + iKernelPES].value(), iPointPES, jPointPES](const PhaseSpacePoint& psp) -> double
				{
					const auto& [r, rho] = psp;
					return predict_elements(kernel, r).squaredNorm() / std::abs(rho(iPointPES, jPointPES));
				});
	}
	return result / Points.size();
}

double calculate_population_one_surface(const Kernel& kernel, const AllPoints& mc_points)
{
	assert(kernel.is_same_feature());
	double result = 0.0;
	std::size_t n_surface = 0;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		const std::size_t ElementIndex = iPES * NumPES + iPES;
		if (!mc_points[ElementIndex].empty())
		{
			result += calculate_population_one_surface_one_element(kernel, mc_points[ElementIndex], iPES, iPES);
			n_surface++;
		}
	}
	return result / n_surface;
}

ClassicalPhaseVector calculate_1st_order_average_one_surface(const Kernel& kernel, const AllPoints& mc_points)
{
	assert(kernel.is_same_feature());
	ClassicalPhaseVector result = ClassicalPhaseVector::Zero();
	std::size_t n_surface = 0;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		const std::size_t ElementIndex = iPES * NumPES + iPES;
		if (!mc_points[ElementIndex].empty())
		{
			result += calculate_1st_order_average_one_surface_one_element(kernel, mc_points[ElementIndex], iPES, iPES)
				/ calculate_population_one_surface_one_element(kernel, mc_points[ElementIndex], iPES, iPES);
			n_surface++;
		}
	}
	return result / n_surface;
}

double calculate_total_energy_average_one_surface(
	const Kernel& kernel,
	const AllPoints& mc_points,
	const ClassicalVector<double>& mass,
	const std::size_t PESIndex)
{
	assert(kernel.is_same_feature());
	double result = 0.0;
	std::size_t n_surface = 0;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		const std::size_t ElementIndex = iPES * NumPES + iPES;
		if (!mc_points[ElementIndex].empty())
		{
			result += calculate_total_energy_average_one_surface_one_element(kernel, mc_points[ElementIndex], mass, PESIndex, iPES, iPES)
				/ calculate_population_one_surface_one_element(kernel, mc_points[ElementIndex], iPES, iPES);
			n_surface++;
		}
	}
	return result / n_surface;
}

double calculate_purity_one_element(
	const OptionalKernels& Kernels,
	const AllPoints& mc_points,
	const std::size_t iPES,
	const std::size_t jPES)
{
	double result = 0.0;
	std::size_t n_element = 0;
	for (std::size_t iPointPES = 0; iPointPES < NumPES; iPointPES++)
	{
		for (std::size_t jPointPES = 0; jPointPES <= iPointPES; jPointPES++)
		{
			// only deal with lower-triangular
			// for off-diagonal elements, they are used in integral, as they are paired
			const std::size_t ElementIndex = iPointPES * NumPES + jPointPES;
			const std::size_t factor = iPointPES == jPointPES ? 1 : 2;
			if (!mc_points[ElementIndex].empty())
			{
				result += factor * calculate_rho_square_one_element_one_element(Kernels, mc_points[ElementIndex], iPES, jPES, iPointPES, jPointPES);
				n_element += factor;
			}
		}
	}
	return PurityFactor * result / n_element;
}

/// @brief To calculate population by monte carlo integral using one element
/// @param[in] Kernels An array of kernels for prediction, whose size is NumElements
/// @param[in] Points The selected points for calculating mc integration
/// @param[in] iPES The row index of the element in density matrix ( @p Points corresponding to)
/// @param[in] jPES The column index of the element in density matrix ( @p Points corresponding to)
/// @return The population using one element
/// @details The monte carlo integral of observable O of one element is @n
/// @f[
/// \langle O\rangle=\frac{1}{N}\sum_{i=0}^{N-1}\frac{1}{w(x_i)}\sum_{\alpha}\rho^{\alpha\alpha}(x_i)O^{\alpha\alpha}(x_i)
/// @f] @n
/// The difference is that now it sums over all @f$ \alpha @f$, and the result is the average of @<O@> on all surfaces
static double calculate_population_one_element(
	const OptionalKernels& Kernels,
	const ElementPoints& Points,
	const std::size_t iPES,
	const std::size_t jPES)
{
	assert(!Points.empty());
	double result = 0.0;
	for (std::size_t iKernelPES = 0; iKernelPES < NumPES; iKernelPES++)
	{
		const std::size_t ElementIndex = iKernelPES * NumPES + iKernelPES;
		if (Kernels[ElementIndex].has_value())
		{
			result += calculate_population_one_surface_one_element(Kernels[ElementIndex].value(), Points, iPES, jPES);
		}
	}
	return result;
}

/// @brief To calculate the @<x@> and @<p@> by monte carlo integral using one element
/// @param[in] Kernels An array of kernels for prediction, whose size is NumElements
/// @param[in] Points The selected points for calculating mc integration
/// @param[in] iPES The row index of the element in density matrix ( @p Points corresponding to)
/// @param[in] jPES The column index of the element in density matrix ( @p Points corresponding to)
/// @return The @<x@> and @<p@> using one element
static ClassicalPhaseVector calculate_1st_order_average_one_element(
	const OptionalKernels& Kernels,
	const ElementPoints& Points,
	const std::size_t iPES,
	const std::size_t jPES)
{
	assert(!Points.empty());
	ClassicalPhaseVector result = ClassicalPhaseVector::Zero();
	for (std::size_t iKernelPES = 0; iKernelPES < NumPES; iKernelPES++)
	{
		const std::size_t ElementIndex = iKernelPES * NumPES + iKernelPES;
		if (Kernels[ElementIndex].has_value())
		{
			result += calculate_1st_order_average_one_surface_one_element(Kernels[ElementIndex].value(), Points, iPES, jPES);
		}
	}
	return result;
}

/// @brief To calculate the total energy by monte carlo integral using one element
/// @param[in] Kernels An array of kernels for prediction, whose size is NumElements
/// @param[in] Points The selected points for calculating mc integration
/// @param[in] mass Mass of classical degree of freedom
/// @param[in] iPES The row index of the element in density matrix ( @p Points corresponding to)
/// @param[in] jPES The column index of the element in density matrix ( @p Points corresponding to)
/// @return The total energy using one element
static double calculate_total_energy_average_one_element(
	const OptionalKernels& Kernels,
	const ElementPoints& Points,
	const ClassicalVector<double>& mass,
	const std::size_t iPES,
	const std::size_t jPES)
{
	assert(!Points.empty());
	double result = 0.0;
	for (std::size_t iKernelPES = 0; iKernelPES < NumPES; iKernelPES++)
	{
		const std::size_t ElementIndex = iKernelPES * NumPES + iKernelPES;
		if (Kernels[ElementIndex].has_value())
		{
			result += calculate_total_energy_average_one_surface_one_element(Kernels[ElementIndex].value(), Points, mass, iKernelPES, iPES, jPES);
		}
	}
	return result;
}

/// @brief To calculate the trace of square of density matrix by monte carlo integral using one element
/// @param[in] Kernels An array of kernels for prediction, whose size is NumElements
/// @param[in] Points The selected points for calculating mc integration
/// @param[in] iPES The row index of the element in density matrix ( @p Points corresponding to)
/// @param[in] jPES The column index of the element in density matrix ( @p Points corresponding to)
/// @return The overall purity using one element
/// @details The integral is @n
/// @f[
/// S=\sum_{i=0}^{N-1}\frac{1}{w(x_i)}
/// \left(\sum_{\alpha=0}^{M-1}[\rho^{\alpha\alpha}(x_i)]^2+2\sum_{\alpha\neq\beta}[\rho^{\alpha\beta}(x_i)]^2\right)
/// @f]
static double calculate_rho_square_one_element(
	const OptionalKernels& Kernels,
	const ElementPoints& Points,
	const std::size_t iPES,
	const std::size_t jPES)
{
	assert(!Points.empty());
	double result = 0.0;
	for (std::size_t iKernelPES = 0; iKernelPES < NumPES; iKernelPES++)
	{
		for (std::size_t jKernelPES = 0; jKernelPES < NumPES; jKernelPES++)
		{
			const std::size_t factor = iKernelPES == jKernelPES ? 1 : 2;
			result += factor * calculate_rho_square_one_element_one_element(Kernels, Points, iKernelPES, jKernelPES, iPES, jPES);
		}
	}
	return result;
}

double calculate_population(const OptionalKernels& Kernels, const AllPoints& mc_points)
{
	double result = 0.0;
	std::size_t n_surface = 0;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		const std::size_t ElementIndex = iPES * NumPES + iPES;
		if (!mc_points[ElementIndex].empty())
		{
			result += calculate_population_one_element(Kernels, mc_points[ElementIndex], iPES, iPES);
			n_surface++;
		}
	}
	return result / n_surface;
}

ClassicalPhaseVector calculate_1st_order_average(const OptionalKernels& Kernels, const AllPoints& mc_points)
{
	ClassicalPhaseVector result = ClassicalPhaseVector::Zero();
	std::size_t n_surface = 0;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		const std::size_t ElementIndex = iPES * NumPES + iPES;
		if (!mc_points[ElementIndex].empty())
		{
			result += calculate_1st_order_average_one_element(Kernels, mc_points[ElementIndex], iPES, iPES);
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
	std::size_t n_surface = 0;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		const std::size_t ElementIndex = iPES * NumPES + iPES;
		if (!mc_points[ElementIndex].empty())
		{
			result += calculate_total_energy_average_one_element(Kernels, mc_points[ElementIndex], mass, iPES, iPES);
			n_surface++;
		}
	}
	return result / n_surface;
}

double calculate_purity(const OptionalKernels& Kernels, const AllPoints& mc_points)
{
	double result = 0.0;
	std::size_t n_elements = 0;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES <= iPES; jPES++)
		{
			// only deal with lower-triangular
			// for off-diagonal elements, they are used in integral twice, as they are paired
			const std::size_t ElementIndex = iPES * NumPES + jPES;
			const std::size_t factor = iPES == jPES ? 1 : 2;
			if (!mc_points[ElementIndex].empty())
			{
				result += factor * calculate_rho_square_one_element(Kernels, mc_points[ElementIndex], iPES, jPES);
				n_elements += factor;
			}
		}
	}
	return PurityFactor * result / n_elements;
}

/// @brief To calculate the derivative of monte carlo integrated population of one surface by one element over parameters
/// @param[in] kernel The kernel for prediction
/// @param[in] Points The selected points for calculating mc integration
/// @param[in] iPES The row index of the element in density matrix ( @p Points corresponding to)
/// @param[in] jPES The column index of the element in density matrix ( @p Points corresponding to)
/// @return The derivative of population of one surface by one element over parameters
/// @details The derivatives of observables O over parameters are @n
/// @f[
/// \frac{\partial\langle O\rangle}{\partial\theta^{\gamma\gamma}}
/// =\frac{1}{N}\sum_{i=0}^{N-1}\frac{1}{w(x_i)}\frac{\partial\rho^{\gamma\gamma}(x_i)}{\partial\theta^{\gamma\gamma}}O^{\gamma\gamma}(x_i)
/// @f] @n
/// And this function will calculate the derivative of population over parameters from all diagonal elements.
static ElementParameter population_derivative_one_surface_one_element(
	const Kernel& kernel,
	const ElementPoints& Points,
	const std::size_t iPES,
	const std::size_t jPES)
{
	assert(kernel.is_same_feature() && !Points.empty());
	const ElementParameter result =
		std::transform_reduce(
			std::execution::par_unseq,
			Points.cbegin(),
			Points.cend(),
			ElementParameter(ElementParameter::Zero()),
			std::plus<ElementParameter>(),
			[&kernel, iPES, jPES](const PhaseSpacePoint& psp) -> ElementParameter
			{
				const auto& [r, rho] = psp;
				return ElementParameter::Map(prediction_derivative(kernel, r).data()) / std::abs(rho(iPES, jPES));
			});
	return result / Points.size();
}

/// @brief To calculate the derivative of monte carlo integrated energy of one surface by one element over parameters
/// @param[in] kernel The kernel for prediction
/// @param[in] Points The selected points for calculating mc integration
/// @param[in] mass Mass of classical degree of freedom
/// @param[in] PESIndex The row and column index of the surface in density matrix ( @p kernel corresponding to)
/// @param[in] iPES The row index of the element in density matrix ( @p Points corresponding to)
/// @param[in] jPES The column index of the element in density matrix ( @p Points corresponding to)
/// @return The derivative of energy of one surface by one element over parameters
static ElementParameter total_energy_derivative_one_surface_one_element(
	const Kernel& kernel,
	const ElementPoints& Points,
	const ClassicalVector<double>& mass,
	const std::size_t PESIndex,
	const std::size_t iPES,
	const std::size_t jPES)
{
	assert(kernel.is_same_feature() && !Points.empty());
	const ElementParameter result =
		std::transform_reduce(
			std::execution::par_unseq,
			Points.cbegin(),
			Points.cend(),
			ElementParameter(ElementParameter::Zero()),
			std::plus<ElementParameter>(),
			[&kernel, &mass, PESIndex, iPES, jPES](const PhaseSpacePoint& psp) -> ElementParameter
			{
				const auto& [r, rho] = psp;
				const auto [x, p] = split_phase_coordinate(r);
				const double V = adiabatic_potential(x)[PESIndex];
				const double T = (p.array().square() / mass.array()).sum() / 2.0;
				return ElementParameter::Map(prediction_derivative(kernel, r).data()) * (T + V) / std::abs(rho(iPES, jPES));
			});
	return result / Points.size();
}

/// @brief To calculate the derivative of monte carlo integrated @f$ \rho_{\alpha\beta}^2 @f$ by one element over parameters
/// @param[in] kernel The kernel for prediction
/// @param[in] Points The selected points for calculating mc integration
/// @param[in] iPES The row index of the element in density matrix ( @p Points corresponding to)
/// @param[in] jPES The column index of the element in density matrix ( @p Points corresponding to)
/// @return The derivative @f$ \rho_{\alpha\beta}^2 @f$ by one element over parameters
/// @details The derivatives of the square of one element of density matrix over parameters are @n
/// @f[
/// \frac{\partial \rho_{\alpha\beta}^2}{\partial \theta^{alpha\beta}}
/// =\frac{2}{N}\sum_{i=1}^N{\frac{\rho_{\alpha\beta}(x_i)}{w(x_i)}\frac{\partial \rho_{\alpha\beta}(x_i)}{\partial \theta_{\alpha\beta}}}
/// @f] @n
/// The factor between this derivative and the derivative of purity is @f$ c(2\pi\hbar)^D @f$,
/// where @f$ D @f$ is the dimension of configuration space (half the dimension of phase space),
/// and @f$ c @f$ is 2 for diagonal elements, 4 for off-diagonal elements.
static ElementParameter rho_square_derivative_one_element_one_element(
	const Kernel& kernel,
	const ElementPoints& Points,
	const std::size_t iPES,
	const std::size_t jPES)
{
	assert(kernel.is_same_feature() && !Points.empty());
	const ElementParameter result =
		std::transform_reduce(
			std::execution::par_unseq,
			Points.cbegin(),
			Points.cend(),
			ElementParameter(ElementParameter::Zero()),
			std::plus<ElementParameter>(),
			[&kernel, iPES, jPES](const PhaseSpacePoint& psp) -> ElementParameter
			{
				const auto& [r, rho] = psp;
				return predict_elements(kernel, r).value() / std::abs(rho(iPES, jPES)) * ElementParameter::Map(prediction_derivative(kernel, r).data());
			});
	return 2.0 * result / Points.size();
}

ParameterVector population_derivative(const OptionalKernels& Kernels, const AllPoints& mc_points)
{
	Eigen::Matrix<double, NumPES * Kernel::NumTotalParameters, 1> result = Eigen::Matrix<double, NumPES * Kernel::NumTotalParameters, 1>::Zero();
	std::size_t n_surface = 0;
	for (std::size_t iPointPES = 0; iPointPES < NumPES; iPointPES++)
	{
		const std::size_t PointElementIndex = iPointPES * NumPES + iPointPES;
		if (!mc_points[PointElementIndex].empty())
		{
			for (std::size_t iKernelPES = 0; iKernelPES < NumPES; iKernelPES++)
			{
				const std::size_t KernelElementIndex = iKernelPES * NumPES + iKernelPES;
				if (Kernels[KernelElementIndex].has_value())
				{
					const ElementParameter PopulationDerivative =
						population_derivative_one_surface_one_element(
							Kernels[KernelElementIndex].value(),
							mc_points[PointElementIndex],
							iPointPES,
							iPointPES);
					result.block<Kernel::NumTotalParameters, 1>(iKernelPES * Kernel::NumTotalParameters, 0) += PopulationDerivative;
				}
			}
			n_surface++;
		}
	}
	result /= n_surface;
	return ParameterVector(result.data(), result.data() + NumPES * Kernel::NumTotalParameters);
}

/// @details As average energy is calculated as @f$ \langle E\rangle=I_E/I_1 @f$,
/// the derivative would therefore be @n
/// @f[
/// \frac{\partial\langle E\rangle}{\partial\theta}=\frac{1}{I_1^2}
/// \left(\frac{\partial I_E}{\partial\theta}I_1-I_E\frac{\partial I_1}{\partial\theta}\right)
/// @f] @n
ParameterVector total_energy_derivative(
	const OptionalKernels& Kernels,
	const AllPoints& mc_points,
	const ClassicalVector<double>& mass)
{
	Eigen::Matrix<double, NumPES * Kernel::NumTotalParameters, 1> result = Eigen::Matrix<double, NumPES * Kernel::NumTotalParameters, 1>::Zero();
	std::size_t n_surface = 0;
	for (std::size_t iPointPES = 0; iPointPES < NumPES; iPointPES++)
	{
		const std::size_t PointElementIndex = iPointPES * NumPES + iPointPES;
		if (!mc_points[PointElementIndex].empty())
		{
			const double NormalizationFactor =
				calculate_population_one_element(
					Kernels,
					mc_points[PointElementIndex],
					iPointPES,
					iPointPES);
			const double TotalEnergy =
				calculate_total_energy_average_one_element(
					Kernels,
					mc_points[PointElementIndex],
					mass,
					iPointPES,
					iPointPES);
			for (std::size_t iKernelPES = 0; iKernelPES < NumPES; iKernelPES++)
			{
				const std::size_t KernelElementIndex = iKernelPES * NumPES + iKernelPES;
				if (Kernels[KernelElementIndex].has_value())
				{
					const ElementParameter TotalEnergyDerivative =
						total_energy_derivative_one_surface_one_element(
							Kernels[KernelElementIndex].value(),
							mc_points[PointElementIndex],
							mass,
							iKernelPES,
							iPointPES,
							iPointPES);
					const ElementParameter PopulationDerivative =
						population_derivative_one_surface_one_element(
							Kernels[KernelElementIndex].value(),
							mc_points[PointElementIndex],
							iPointPES,
							iPointPES);
					result.block<Kernel::NumTotalParameters, 1>(iKernelPES * Kernel::NumTotalParameters, 0) +=
						(TotalEnergyDerivative * NormalizationFactor - TotalEnergy * PopulationDerivative) / power<2>(NormalizationFactor);
				}
			}
			n_surface++;
		}
	}
	result /= n_surface;
	return ParameterVector(result.data(), result.data() + NumPES * Kernel::NumTotalParameters);
}

ParameterVector purity_derivative(const OptionalKernels& Kernels, const AllPoints& mc_points)
{
	Eigen::Matrix<double, NumElements * Kernel::NumTotalParameters, 1> result = Eigen::Matrix<double, NumElements * Kernel::NumTotalParameters, 1>::Zero();
	std::size_t n_element = 0;
	for (std::size_t iPointPES = 0; iPointPES < NumPES; iPointPES++)
	{
		for (std::size_t jPointPES = 0; jPointPES <= iPointPES; jPointPES++)
		{
			// only deal with lower-triangular
			// for off-diagonal elements, they are used in integral twice, as they are paired
			const std::size_t PointElementIndex = iPointPES * NumPES + jPointPES;
			const std::size_t PointFactor = iPointPES == jPointPES ? 1 : 2;
			if (!mc_points[PointElementIndex].empty())
			{
				for (std::size_t iKernelPES = 0; iKernelPES < NumPES; iKernelPES++)
				{
					for (std::size_t jKernelPES = 0; jKernelPES < NumPES; jKernelPES++)
					{
						const std::size_t KernelElementIndex = iKernelPES * NumPES + jKernelPES;
						if (Kernels[KernelElementIndex].has_value())
						{
							const ElementParameter RhoSquareDerivative =
								rho_square_derivative_one_element_one_element(
									Kernels[KernelElementIndex].value(),
									mc_points[PointElementIndex],
									iPointPES,
									jPointPES);
							// for off-diagonal elements, they contributes to purity twice
							const double ElementFactor = iKernelPES == jKernelPES ? 1 : 2;
							result.block<Kernel::NumTotalParameters, 1>(KernelElementIndex * Kernel::NumTotalParameters, 0) +=
								PointFactor * ElementFactor * RhoSquareDerivative;
						}
					}
				}
				n_element += PointFactor;
			}
		}
	}
	result *= PurityFactor / n_element;
	return ParameterVector(result.data(), result.data() + NumElements * Kernel::NumTotalParameters);
}
