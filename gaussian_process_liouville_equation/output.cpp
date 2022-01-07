/// @file output.cpp
/// @brief Implementation of output.h

#include "stdafx.h"

#include "output.h"

#include "input.h"
#include "kernel.h"
#include "mc.h"
#include "mc_predict.h"
#include "opt.h"
#include "predict.h"

/// @details For each surfaces and for total, output 
/// its <x> and <p>, population and energy calculated by
/// analytical integral, direct averaging and monte carlo
/// integral. For those that is nonexistent (population by
/// direct averaging and energy by analytical integral), output NAN.
///
/// Then, output total purity by analytical integral and monte carlo integral.
///
/// For the nomenclature, prm = parameter (analytical integral),
/// ave = direct averaing, and mci = monte carlo integral.
void output_average(
	std::ostream& os,
	const OptionalKernels& Kernels,
	const AllPoints& density,
	const AllPoints& mc_points,
	const ClassicalVector<double>& mass)
{
	// average: time, population, x, p, V, T, E of each PES
	double ppl_prm_all = 0.;
	ClassicalPhaseVector r_prm_all = ClassicalPhaseVector::Zero();
	ClassicalPhaseVector r_ave_all = ClassicalPhaseVector::Zero();
	double e_ave_all = 0.0;
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		const int ElementIndex = iPES * (NumPES + 1);
		assert(!density[ElementIndex].empty() == Kernels[ElementIndex].has_value() && density[ElementIndex].empty() == mc_points[ElementIndex].empty());
		if (!density[ElementIndex].empty())
		{
			const double ppl_prm = Kernels[ElementIndex]->get_population();
			const ClassicalPhaseVector r_prm = Kernels[ElementIndex]->get_1st_order_average();
			const ClassicalPhaseVector r_ave = calculate_1st_order_average(density[ElementIndex]);
			const double e_ave = calculate_total_energy_average(density[ElementIndex], iPES, mass);
			// output for analytcal integral by parameters; energy is NAN
			os << ' ' << r_prm.format(VectorFormatter) << ' ' << ppl_prm << ' ' << NAN;
			// output for direct averaging; population is NAN
			os << ' ' << r_ave.format(VectorFormatter) << ' ' << NAN << ' ' << e_ave;
			// output for monte carlo integral
			os << ' ' << calculate_1st_order_average_one_surface(Kernels[ElementIndex].value(), mc_points).format(VectorFormatter);
			os << ' ' << calculate_population_one_surface(Kernels[ElementIndex].value(), mc_points);
			os << ' ' << calculate_total_energy_average_one_surface(Kernels[ElementIndex].value(), mc_points, mass, iPES);
			ppl_prm_all += ppl_prm;
			r_prm_all += ppl_prm * r_prm;
			r_ave_all += ppl_prm * r_ave;
			e_ave_all += ppl_prm * e_ave;
		}
		else
		{
			// output for analytcal integral by parameters; energy is NAN
			os << ' ' << ClassicalPhaseVector::Zero().format(VectorFormatter) << ' ' << 0.0 << ' ' << NAN;
			// output for direct averaging; population is NAN
			os << ' ' << ClassicalPhaseVector::Zero().format(VectorFormatter) << ' ' << NAN << ' ' << 0.0;
			// output for monte carlo integral
			os << ' ' << ClassicalPhaseVector::Zero().format(VectorFormatter) << ' ' << 0.0 << ' ' << 0.0;
		}
	}
	const double ppl_mci_all = calculate_population(Kernels, mc_points);
	// output for analytcal integral by parameters; energy is NAN
	os << ' ' << (r_prm_all / ppl_prm_all).format(VectorFormatter) << ' ' << ppl_prm_all << ' ' << NAN;
	// output for direct averaging; population is NAN
	os << ' ' << (r_ave_all / ppl_prm_all).format(VectorFormatter) << ' ' << NAN << ' ' << e_ave_all / ppl_prm_all;
	// output for monte carlo integral
	os << ' ' << (calculate_1st_order_average(Kernels, mc_points) / ppl_mci_all).format(VectorFormatter);
	os << ' ' << ppl_mci_all;
	os << ' ' << (calculate_total_energy_average(Kernels, mc_points, mass) / ppl_mci_all);
	// output purity
	os << ' ' << calculate_purity(Kernels) << ' ' << NAN << ' ' << calculate_purity(Kernels, mc_points) << std::endl;
}

void output_param(std::ostream& os, const QuantumArray<ParameterVector>& Params)
{
	for (int iElement = 0; iElement < NumElements; iElement++)
	{
		for (double d : Params[iElement])
		{
			os << ' ' << d;
		}
		os << '\n';
	}
	os << '\n';
}

void output_point(std::ostream& os, const AllPoints& Points)
{
	const int NumPoints = get_num_points(Points);
	const std::vector<int> indices = get_indices(NumPoints);
	Eigen::MatrixXd result(PhaseDim * NumElements, NumPoints);
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		for (int jPES = 0; jPES < NumPES; jPES++)
		{
			if (iPES <= jPES)
			{
				// for row-major, visit upper triangular elements earlier
				// so selection for the upper triangular elements
				const int ElementIndex = iPES * NumPES + jPES;
				const EigenVector<PhaseSpacePoint>& ElementPoints = Points[ElementIndex];
				const int StartRow = ElementIndex * PhaseDim;
				if (!ElementPoints.empty())
				{
					std::for_each(
						std::execution::par_unseq,
						indices.cbegin(),
						indices.cend(),
						[&ElementPoints, &result, StartRow](int iPoint) -> void
						{
							[[maybe_unused]] const auto& [x, p, rho] = ElementPoints[iPoint];
							result.block<PhaseDim, 1>(StartRow, iPoint) << x, p;
						});
				}
				else
				{
					result.block(StartRow, 0, PhaseDim, NumPoints) = Eigen::MatrixXd::Zero(PhaseDim, NumPoints);
				}
			}
			else
			{
				// for strict lower triangular elements, copy the upper part
				result.block((iPES * NumPES + jPES) * NumPES, 0, PhaseDim, NumPoints) = result.block((jPES * NumPES + iPES) * NumPES, 0, PhaseDim, NumPoints);
			}
		}
	}
	os << result.format(MatrixFormatter) << "\n\n";
}

void output_phase(std::ostream& phase, std::ostream& variance, const OptionalKernels& Kernels, const Eigen::MatrixXd& PhaseGrids)
{
	for (int iElement = 0; iElement < NumElements; iElement++)
	{
		if (Kernels[iElement].has_value())
		{
			phase << predict_elements(Kernels[iElement].value(), PhaseGrids).format(VectorFormatter);
			variance << predict_variances(Kernels[iElement].value(), PhaseGrids).format(VectorFormatter);
		}
		else
		{
			phase << Eigen::VectorXd::Zero(PhaseGrids.cols()).format(VectorFormatter);
			variance << Eigen::VectorXd::Zero(PhaseGrids.cols()).format(VectorFormatter);
		}
		phase << '\n';
		variance << '\n';
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
	for (int iElement = 0; iElement < NumElements; iElement++)
	{
		os << step_autocor[iElement].format(VectorFormatter) << displ_autocor[iElement].format(VectorFormatter) << '\n';
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
	for (int step : Steps)
	{
		os << ' ' << step;
	}
	os << ' ' << CPUTime.count() << ' ' << std::put_time(std::localtime(&CurrentTime), "%F %T %Z") << std::endl;
}