/// @file output.cpp
/// @brief Implementation of output.h

#include "stdafx.h"

#include "output.h"

#include "complex_kernel.h"
#include "evolve.h"
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
	const Kernels& AllKernels,
	const AllPoints& density,
	const ClassicalVector<double>& mass
)
{
	// average: time, population, x, p, V, T, E of each PES
	ClassicalPhaseVector r_ave_all = ClassicalPhaseVector::Zero();
	double e_ave_all = 0.0;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		const ClassicalPhaseVector r_prm = calculate_1st_order_average_one_surface(AllKernels(iPES));
		const ClassicalPhaseVector r_ave = calculate_1st_order_average_one_surface(density(iPES));
		const double ppl_prm = calculate_population_one_surface(AllKernels(iPES));
		const double e_ave = calculate_total_energy_average_one_surface(density(iPES), mass, iPES);
		// output for analytcal integral by parameters; energy is NAN
		os << ' ' << r_prm.format(VectorFormatter) << ' ' << ppl_prm;
		// output for direct averaging; population is NAN
		os << ' ' << r_ave.format(VectorFormatter) << ' ' << e_ave;
		r_ave_all += ppl_prm * r_ave;
		e_ave_all += ppl_prm * e_ave;
	}
	const ClassicalPhaseVector r_prm_all = AllKernels.calculate_1st_order_average();
	// r_ave_all is already calculated
	const double ppl_prm_all = AllKernels.calculate_population();
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
			if (iPES == jPES)
			{
				prt_prm(iPES, jPES) = AllKernels(iPES).get_purity();
			}
			else
			{
				prt_prm(iPES, jPES) = AllKernels(iPES, jPES).get_purity();
			}
		}
	}
	prt_prm = prt_prm.selfadjointView<Eigen::Lower>();
	// output elementwise
	os << ' ' << prt_prm.format(VectorFormatter);
	// output sum
	const double prt_prm_all = AllKernels.calculate_purity();
	os << ' ' << prt_prm_all;
	// finish
	os << std::endl;
}

void output_param(std::ostream& os, const Optimization& Optimizer)
{
	const QuantumStorage<ParameterVector> lb = Optimizer.get_lower_bounds(), param = Optimizer.get_parameters(), ub = Optimizer.get_upper_bounds();
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES <= iPES; jPES++)
		{
			os << lb(iPES, jPES) << '\n';
			os << param(iPES, jPES) << '\n';
			os << ub(iPES, jPES) << '\n';
		}
	}
	os << '\n';
}

void output_point(std::ostream& coord, std::ostream& value, const AllPoints& density, const AllPoints& extra_points)
{
	const std::size_t NumPoints = density(0).size(), NumExtraPoints = extra_points(0).size(), NumTotalPoints = NumPoints + NumExtraPoints;
	const auto indices = xt::arange(NumPoints), extra_indices = xt::arange(NumExtraPoints);
	const QuantumMatrix<bool> IsSmall = is_very_small(density);
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES <= iPES; jPES++)
		{
			PhasePoints point_coordinates = PhasePoints::Zero(PhasePoints::RowsAtCompileTime, NumTotalPoints);
			Eigen::VectorXcd point_weight = Eigen::VectorXcd::Zero(NumTotalPoints);
			if (!IsSmall(iPES, jPES))
			{
				std::for_each(
					std::execution::par_unseq,
					indices.cbegin(),
					indices.cend(),
					[&element_density = density(iPES, jPES), &point_coordinates, &point_weight](const std::size_t iPoint) -> void
					{
						point_coordinates.col(iPoint) = element_density[iPoint].get<0>();
						point_weight(iPoint) = element_density[iPoint].get_exact_element();
					}
				);
				std::for_each(
					std::execution::par_unseq,
					extra_indices.cbegin(),
					extra_indices.cend(),
					[&element_density = extra_points(iPES, jPES), NumPoints, &point_coordinates, &point_weight](const std::size_t iPoint) -> void
					{
						point_coordinates.col(NumPoints + iPoint) = element_density[iPoint].get<0>();
						point_weight(NumPoints + iPoint) = element_density[iPoint].get_exact_element();
					}
				);
			}
			// else 0, as already set
			// then output
			coord << point_coordinates.format(MatrixFormatter) << '\n';
			value << point_weight.real().format(VectorFormatter) << '\n';
			value << point_weight.imag().format(VectorFormatter) << '\n';
		}
	}
	coord << '\n';
	value << '\n';
}

void output_phase(
	std::ostream& phase,
	std::ostream& phasefactor,
	std::ostream& variance,
	const ClassicalVector<double>& mass,
	const double dt,
	const std::size_t NumTicks,
	const Kernels& AllKernels,
	const PhasePoints& PhaseGrids
)
{
	const std::size_t NumPoints = PhaseGrids.cols();
	const auto indices = xt::arange(NumPoints);
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES <= iPES; jPES++)
		{
			if (iPES == jPES)
			{
				const Kernel k(PhaseGrids, AllKernels(iPES), false);
				phase << k.get_prediction_compared_with_variance().format(VectorFormatter) << '\n';
				phase << Eigen::VectorXd::Zero(NumPoints).format(VectorFormatter) << '\n';
				phasefactor << Eigen::VectorXd::Zero(NumPoints).format(VectorFormatter) << '\n';
				variance << k.get_variance().format(VectorFormatter) << '\n';
			}
			else
			{
				const ComplexKernel ck(PhaseGrids, AllKernels(iPES, jPES), false);
				const Eigen::VectorXcd& pred = ck.get_prediction_compared_with_variance();
				Eigen::VectorXd phase_factor = Eigen::VectorXd::Zero(NumPoints);
				std::for_each(
					std::execution::par_unseq,
					indices.cbegin(),
					indices.cend(),
					[&pred, &phase_factor, &mass, dt, NumTicks, &PhaseGrids, iPES, jPES](std::size_t iPoint) -> void
					{
						if (pred[iPoint] != 0.0)
						{
							phase_factor[iPoint] = get_phase_factor(PhaseGrids.col(iPoint), mass, dt, NumTicks, iPES, jPES);
						}
					}
				);
				phase << pred.real().format(VectorFormatter) << '\n';
				phase << pred.imag().format(VectorFormatter) << '\n';
				phasefactor << phase_factor.format(VectorFormatter) << '\n';
				variance << ck.get_variance().format(VectorFormatter) << '\n';
			}
		}
	}
	phase << '\n';
	variance << '\n';
}

void output_autocor(
	std::ostream& os,
	MCParameters& MCParams,
	const DistributionFunction& distribution,
	const AllPoints& density
)
{
	const AutoCorrelations step_autocor = autocorrelation_optimize_steps(MCParams, distribution, density);
	const AutoCorrelations displ_autocor = autocorrelation_optimize_displacement(MCParams, distribution, density);
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES <= iPES; jPES++)
		{
			os << step_autocor(iPES, jPES).format(VectorFormatter) << ' ' << NAN << ' ' << displ_autocor(iPES, jPES).format(VectorFormatter) << '\n';
		}
	}
	os << '\n';
}

void output_logging(
	std::ostream& os,
	const double time,
	const Optimization::Result& OptResult,
	const MCParameters& MCParams,
	const std::chrono::duration<double>& CPUTime
)
{
	const std::time_t CurrentTime = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
	const auto& [error, Steps] = OptResult;
	os << time << ' ' << error << ' ' << MCParams.get_num_MC_steps() << ' ' << MCParams.get_max_displacement();
	os << ' ' << Steps << ' ' << CPUTime.count() << ' ' << std::put_time(std::localtime(&CurrentTime), "%F %T %Z") << std::endl;
}
