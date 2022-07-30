/// @file output.cpp
/// @brief Implementation of output.h

#include "stdafx.h"

#include "output.h"

#include "complex_kernel.h"
#include "input.h"
#include "kernel.h"
#include "mc.h"
#include "opt.h"
#include "predict.h"

/// @brief To output std vector
/// @tparam T The data type in the vector
/// @param[inout] os The output stream
/// @param[in] vec The vector
/// @return @p os after output
template <typename T>
inline std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec)
{
	if (vec.cbegin() != vec.cend())
	{
		std::copy(vec.cbegin(), vec.cend() - 1, std::ostream_iterator<T>(os, " "));
		os << *(vec.cend() - 1);
	}
	return os;
}

/// @details For each surfaces and for total, output
/// its @<x@> and @<p@>, population and energy calculated by
/// analytical integral, direct averaging and monte carlo
/// integral. For those that is nonexistent (population by
/// direct averaging and energy by analytical integral), output NAN. @n
/// Then, output total purity by analytical integral and monte carlo integral. @n
/// For the nomenclature, prm = parameter (analytical integral),
/// ave = direct averaing, and mci = monte carlo integral.
void output_average(
	std::ostream& os,
	const OptionalKernels& Kernels,
	const AllPoints& density,
	const ClassicalVector<double>& mass)
{
	// average: time, population, x, p, V, T, E of each PES
	ClassicalPhaseVector r_ave_all = ClassicalPhaseVector::Zero();
	double e_ave_all = 0.0;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		const std::size_t ElementIndex = iPES * NumPES + iPES;
		const ClassicalPhaseVector r_prm = calculate_1st_order_average_one_surface(Kernels[ElementIndex]);
		const ClassicalPhaseVector r_ave = calculate_1st_order_average_one_surface(density[ElementIndex]);
		const double ppl_prm = calculate_population_one_surface(Kernels[ElementIndex]);
		const double e_ave = calculate_total_energy_average_one_surface(density[ElementIndex], mass, iPES);
		// output for analytcal integral by parameters; energy is NAN
		os << ' ' << r_prm.format(VectorFormatter) << ' ' << ppl_prm;
		// output for direct averaging; population is NAN
		os << ' ' << r_ave.format(VectorFormatter) << ' ' << e_ave;
		r_ave_all += ppl_prm * r_ave;
		e_ave_all += ppl_prm * e_ave;
	}
	const ClassicalPhaseVector r_prm_all = calculate_1st_order_average(Kernels);
	// r_ave_all is already calculated
	const double ppl_prm_all = calculate_population(Kernels);
	// e_ave_all is already calculated
	// output for analytcal integral by parameters; energy is NAN
	os << ' ' << (r_prm_all / ppl_prm_all).format(VectorFormatter) << ' ' << ppl_prm_all;
	// output for direct averaging; population is NAN
	os << ' ' << (r_ave_all / ppl_prm_all).format(VectorFormatter) << ' ' << e_ave_all / ppl_prm_all;
	// output purity
	QuantumMatrix<double> prt_prm = QuantumMatrix<double>::Zero();
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES <= iPES; jPES++)
		{
			const std::size_t ElementIndex = iPES * NumPES + jPES;
			if (iPES == jPES)
			{
				prt_prm(iPES, jPES) = dynamic_cast<const Kernel&>(*Kernels[ElementIndex].get()).get_purity();
			}
			else
			{
				prt_prm(iPES, jPES) = dynamic_cast<const ComplexKernel&>(*Kernels[ElementIndex].get()).get_purity();
			}
		}
	}
	prt_prm = prt_prm.selfadjointView<Eigen::Lower>();
	// output elementwise
	os << ' ' << prt_prm.format(VectorFormatter);
	// output sum
	const double prt_prm_all = calculate_purity(Kernels);
	os << ' ' << prt_prm_all;
	// finish
	os << std::endl;
}

void output_param(std::ostream& os, const Optimization& Optimizer)
{
	const QuantumArray<ParameterVector> lb = Optimizer.get_lower_bounds(), param = Optimizer.get_parameters(), ub = Optimizer.get_upper_bounds();
	for (std::size_t iElement = 0; iElement < NumElements; iElement++)
	{
		os << lb[iElement] << '\n';
		os << param[iElement] << '\n';
		os << ub[iElement] << '\n';
	}
	os << '\n';
}

void output_point(std::ostream& os, const AllPoints& Points)
{
	const std::size_t NumPoints = Points[0].size();
	const auto indices = xt::arange(NumPoints);
	const QuantumMatrix<bool> IsSmall = is_very_small(Points);
	QuantumArray<PhasePoints> point_coordinates;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES < NumPES; jPES++)
		{
			const std::size_t ElementIndex = iPES * NumPES + jPES;
			if (iPES <= jPES)
			{
				point_coordinates[ElementIndex] = PhasePoints::Zero(PhasePoints::RowsAtCompileTime, NumPoints);
				// for row-major, visit upper triangular elements earlier
				// so selection for the upper triangular elements
				const ElementPoints& ElementPoints = Points[ElementIndex];
				if (!IsSmall(iPES, jPES))
				{
					std::for_each(
						std::execution::par_unseq,
						indices.cbegin(),
						indices.cend(),
						[&ElementPoints, &element_point_coordinates = point_coordinates[ElementIndex]](const std::size_t iPoint) -> void
						{
							[[maybe_unused]] const auto& [r, rho] = ElementPoints[iPoint];
							element_point_coordinates.col(iPoint) = r;
						});
				}
				// else 0, as already set
			}
			else
			{
				// for strict lower triangular elements, copy the upper part
				point_coordinates[ElementIndex] = point_coordinates[jPES * NumPES + iPES];
			}
			os << point_coordinates[ElementIndex].format(MatrixFormatter) << '\n';
		}
	}
	os << '\n';
}

void output_phase(std::ostream& phase, std::ostream& variance, const OptionalKernels& Kernels, const PhasePoints& PhaseGrids)
{
	QuantumArray<std::optional<ComplexKernel>> offdiag_predictor;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES < NumPES; jPES++)
		{
			const std::size_t ElementIndex = iPES * NumPES + jPES;
			if (iPES < jPES)
			{
				const std::size_t SymmetricElementIndex = jPES * NumPES + iPES;
				offdiag_predictor[SymmetricElementIndex].emplace(PhaseGrids, dynamic_cast<const ComplexKernel&>(*Kernels[SymmetricElementIndex].get()), false);
				phase << offdiag_predictor[SymmetricElementIndex]->get_prediction().real().format(VectorFormatter);
				variance << offdiag_predictor[SymmetricElementIndex]->get_variance().format(VectorFormatter);
			}
			else if (iPES > jPES)
			{
				phase << offdiag_predictor[ElementIndex]->get_prediction().imag().format(VectorFormatter);
				variance << offdiag_predictor[ElementIndex]->get_variance().format(VectorFormatter);
			}
			else // if (iPES == jPES)
			{
				const Kernel k(PhaseGrids, dynamic_cast<const Kernel&>(*Kernels[ElementIndex].get()), false);
				phase << k.get_prediction().format(VectorFormatter);
				variance << k.get_variance().format(VectorFormatter);
			}
			phase << '\n';
			variance << '\n';
		}
	}
	phase << '\n';
	variance << '\n';
}

void output_autocor(
	std::ostream& os,
	MCParameters& MCParams,
	const DistributionFunction& dist,
	const AllPoints& density)
{
	const AutoCorrelations step_autocor = autocorrelation_optimize_steps(MCParams, dist, density);
	const AutoCorrelations displ_autocor = autocorrelation_optimize_displacement(MCParams, dist, density);
	for (std::size_t iElement = 0; iElement < NumElements; iElement++)
	{
		os << step_autocor[iElement].format(VectorFormatter) << ' ' << NAN << ' ' << displ_autocor[iElement].format(VectorFormatter) << '\n';
	}
	os << '\n';
}

void output_logging(
	std::ostream& os,
	const double time,
	const Optimization::Result& OptResult,
	const MCParameters& MCParams,
	const std::chrono::duration<double>& CPUTime)
{
	const std::time_t CurrentTime = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
	const auto& [error, Steps] = OptResult;
	os << time << ' ' << error << ' ' << MCParams.get_num_MC_steps() << ' ' << MCParams.get_max_displacement();
	os << ' ' << Steps << ' ' << CPUTime.count() << ' ' << std::put_time(std::localtime(&CurrentTime), "%F %T %Z") << std::endl;
}
