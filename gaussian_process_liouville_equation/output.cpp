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
#include "storage.h"

/// @brief To output std vector
/// @tparam T The data type in the vector
/// @param[inout] os The output stream
/// @param[in] vec The vector
/// @return @p os after output
template <typename T>
static inline std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec)
{
	if (vec.cbegin() != vec.cend())
	{
		std::copy(vec.cbegin(), vec.cend() - 1, std::ostream_iterator<T>(os, " "));
		os << *(vec.cend() - 1);
	}
	return os;
}

/// @details For each surfaces and for total, output its
/// population, @<x@> and @<p@>, and energy calculated by
/// direct averaging and monte carlo integral. For those that
/// is nonexistent (analytical integrated energy), output NAN. @n
/// Then, output elementwise purity and total purity
/// by analytical integral and monte carlo integral. @n
/// For the nomenclature, prm = parameter (analytical
/// integral), and mci = monte carlo integral.
void output_average(
	std::ostream& os,
	const TrainingKernels& AllKernels,
	const AllPoints& density,
	const ClassicalVector<double>& mass,
	const double PurityFactor
)
{
	// average: population, x, p, E of each PES
	// output elementwise
	const QuantumVector<double> ppl_mci_each = calculate_population_each_surface(density);
	const QuantumVector<double> e_mci_each = calculate_total_energy_average_each_surface(density, mass);
	for (const std::size_t iPES : std::ranges::iota_view{0ul, NumPES})
	{
		// output for analytcal integral by parameters; energy is NAN
		if (AllKernels(iPES).has_value())
		{
			os << ' ' << calculate_population_one_surface(AllKernels(iPES).value());
			os << ' ' << calculate_1st_order_average_one_surface(AllKernels(iPES).value()).format(VectorFormatter);
		}
		else
		{
			// output for analytcal integral by parameters; energy is NAN
			os << ' ' << 0.0;
			os << ' ' << (ClassicalPhaseVector::Zero() * NAN).format(VectorFormatter);
		}
		os << ' ' << NAN;
		// output for monte carlo integral
		os << ' ' << ppl_mci_each[iPES];
		if (!density(iPES).empty())
		{
			os << ' ' << calculate_1st_order_average_one_surface(density(iPES)).format(VectorFormatter);
		}
		else
		{
			os << ' ' << (ClassicalPhaseVector::Zero() * NAN).format(VectorFormatter);
		}
		os << ' ' << e_mci_each[iPES];
	}
	// output sum
	// output for analytcal integral by parameters
	const double ppl_prm_all = AllKernels.calculate_population();
	os << ' ' << ppl_prm_all;
	os << ' ' << (AllKernels.calculate_1st_order_average() / ppl_prm_all).format(VectorFormatter);
	os << ' ' << AllKernels.calculate_total_energy_average(e_mci_each) / ppl_prm_all;
	// output for monte carlo integral
	const double ppl_mci_all = ppl_mci_each.sum();
	os << ' ' << ppl_mci_all;
	os << ' ' << (calculate_1st_order_average_all_surface(density) / ppl_mci_all).format(VectorFormatter);
	os << ' ' << calculate_total_energy_average_all_surface(density, mass) / ppl_mci_all;

	// output purity
	// output for analytical integral
	QuantumMatrix<double> prt_prm = QuantumMatrix<double>::Zero();
	for (const std::size_t iPES : std::ranges::iota_view{0ul, NumPES})
	{
		for (const std::size_t jPES : std::ranges::iota_view{0ul, iPES + 1})
		{
			if (iPES == jPES)
			{
				prt_prm(iPES, jPES) = AllKernels(iPES).has_value() ? AllKernels(iPES)->get_purity() : 0.0;
			}
			else
			{
				prt_prm(iPES, jPES) = AllKernels(iPES, jPES).has_value() ? AllKernels(iPES, jPES)->get_purity() : 0.0;
			}
		}
	}
	prt_prm = prt_prm.selfadjointView<Eigen::Lower>();
	os << ' ' << prt_prm.format(VectorFormatter);
	os << ' ' << AllKernels.calculate_purity();
	// output for monte carlo integral
	const QuantumMatrix<double> prt_mci = calculate_purity_each_element(density) * PurityFactor;
	os << ' ' << prt_mci.format(VectorFormatter);
	os << ' ' << prt_mci.sum();
	// finish
	os << std::endl;
}

void output_param(std::ostream& os, const Optimization& Optimizer)
{
	const QuantumStorage<ParameterVector> lb = Optimizer.get_lower_bounds(), param = Optimizer.get_parameters(), ub = Optimizer.get_upper_bounds();
	for (const std::size_t iPES : std::ranges::iota_view{0ul, NumPES})
	{
		for (const std::size_t jPES : std::ranges::iota_view{0ul, iPES + 1})
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
	for (const std::size_t iPES : std::ranges::iota_view{0ul, NumPES})
	{
		for (const std::size_t jPES : std::ranges::iota_view{0ul, iPES + 1})
		{
			PhasePoints point_coordinates = PhasePoints::Zero(PhasePoints::RowsAtCompileTime, NumTotalPoints);
			Eigen::VectorXcd point_weight = Eigen::VectorXcd::Zero(NumTotalPoints);
			if (!density(iPES, jPES).empty())
			{
				std::for_each(
					std::execution::par_unseq,
					indices.cbegin(),
					indices.cend(),
					[&element_density = density(iPES, jPES), &point_coordinates, &point_weight](const std::size_t iPoint) -> void
					{
						const auto& [r, rho] = element_density[iPoint];
						point_coordinates.col(iPoint) = r;
						point_weight(iPoint) = rho;
					}
				);
				std::for_each(
					std::execution::par_unseq,
					extra_indices.cbegin(),
					extra_indices.cend(),
					[&element_density = extra_points(iPES, jPES), NumPoints, &point_coordinates, &point_weight](const std::size_t iPoint) -> void
					{
						const auto& [r, rho] = element_density[iPoint];
						point_coordinates.col(NumPoints + iPoint) = r;
						point_weight(NumPoints + iPoint) = rho;
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
	std::ostream& variance,
	const TrainingKernels& AllKernels,
	const PhasePoints& PhaseGrids
)
{
	const std::size_t NumPoints = PhaseGrids.cols();
	auto output_zero = [&phase, &variance, NumPoints]() -> void
	{
		phase << Eigen::VectorXd::Zero(NumPoints).format(VectorFormatter) << '\n';
		phase << Eigen::VectorXd::Zero(NumPoints).format(VectorFormatter) << '\n';
		variance << Eigen::VectorXd::Zero(NumPoints).format(VectorFormatter) << '\n';
	};
	const auto indices = xt::arange(NumPoints);
	for (const std::size_t iPES : std::ranges::iota_view{0ul, NumPES})
	{
		for (const std::size_t jPES : std::ranges::iota_view{0ul, iPES + 1})
		{
			if (iPES == jPES)
			{
				if (AllKernels(iPES).has_value())
				{
					const PredictiveKernel k(PhaseGrids, AllKernels(iPES).value(), false);
					phase << k.get_cutoff_prediction().format(VectorFormatter) << '\n';
					phase << Eigen::VectorXd::Zero(NumPoints).format(VectorFormatter) << '\n';
					variance << k.get_variance().format(VectorFormatter) << '\n';
				}
				else
				{
					output_zero();
				}
			}
			else
			{
				if (AllKernels(iPES, jPES).has_value())
				{
					const PredictiveComplexKernel ck(PhaseGrids, AllKernels(iPES, jPES).value(), false);
					const Eigen::VectorXcd& pred = ck.get_cutoff_prediction();
					phase << pred.real().format(VectorFormatter) << '\n';
					phase << pred.imag().format(VectorFormatter) << '\n';
					variance << ck.get_variance().format(VectorFormatter) << '\n';
				}
				else
				{
					output_zero();
				}
			}
		}
	}
	phase << '\n';
	variance << '\n';
}

void output_logging(
	std::ostream& os,
	const double time,
	const Optimization::Result& OptResult,
	const QuantumStorage<MCParameters>& MCParams,
	const std::chrono::duration<double>& CPUTime,
	const TrainingKernels& AllKernels
)
{
	const auto& [error, Steps, OptType] = OptResult;
	// output current time
	os << time;
	// output the time consumption
	os << ' ' << CPUTime.count();
	// output the steps and max displacement for Metropolis MCMC
	for (const std::size_t iPES : std::ranges::iota_view{0ul, NumPES})
	{
		for (const std::size_t jPES : std::ranges::iota_view{0ul, iPES + 1})
		{
			os << ' ' << MCParams(iPES, jPES).get_num_MC_steps();
		}
	}
	for (const std::size_t iPES : std::ranges::iota_view{0ul, NumPES})
	{
		for (const std::size_t jPES : std::ranges::iota_view{0ul, iPES + 1})
		{
			os << ' ' << MCParams(iPES, jPES).get_max_displacement();
		}
	}
	// output the rescale factor for each kernel
	for (const std::size_t iPES : std::ranges::iota_view{0ul, NumPES})
	{
		for (const std::size_t jPES : std::ranges::iota_view{0ul, iPES + 1})
		{
			if (iPES == jPES)
			{
				if (AllKernels(iPES).has_value())
				{
					os << ' ' << AllKernels(iPES)->get_rescale_factor();
				}
				else
				{
					os << ' ' << NAN;
				}
			}
			else
			{
				if (AllKernels(iPES, jPES).has_value())
				{
					os << ' ' << AllKernels(iPES, jPES)->get_rescale_factor();
				}
				else
				{
					os << ' ' << NAN;
				}
			}
		}
	}
	// output the error from optimization
	os << ' ' << error;
	// output the number of steps for optimization
	os << ' ' << Steps;
	// output the kind of optimization
	os << ' ' << OptType;
	// output current system time
	const std::time_t CurrentTime = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
	os << ' ' << std::put_time(std::localtime(&CurrentTime), "%F %T %Z") << std::endl;
}
