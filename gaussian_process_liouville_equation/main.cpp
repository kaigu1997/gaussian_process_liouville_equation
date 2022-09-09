/// @file main.cpp
/// @brief The main driver

#include "stdafx.h"

#include "evolve.h"
#include "input.h"
#include "kernel.h"
#include "mc.h"
#include "opt.h"
#include "output.h"
#include "pes.h"
#include "predict.h"

/// @brief The main function
/// @return 0, except for cases of exceptions
int main(void)
{
	// initialization: save some input parameters, using initial distribution to select points, and initial output
	std::chrono::high_resolution_clock::time_point begin = std::chrono::high_resolution_clock::now(), end = begin;
	spdlog::set_pattern("[%=4Y-%=2m-%=2d %=8T.%=9F %=6z][Thread %t][Process %P][Logger %n][%=8l] %v");
	spdlog::set_default_logger(spdlog::stderr_color_mt("console"));
	const InitialParameters InitParams = read_input("input");				  // get mass, r0, sigma_r0, dt, output time
	const ClassicalPhaseVector& r0 = InitParams.get_r0();					  // save initial phase space point
	const ClassicalPhaseVector& SigmaR0 = InitParams.get_sigma_r0();		  // save initial standard deviation
	const PhasePoints& PhaseGrids = InitParams.get_phase_space_points();	  // save output grids in phase space
	const ClassicalVector<double>& mass = InitParams.get_mass();			  // save mass directly
	const std::size_t TotalTicks = InitParams.calc_total_ticks();			  // calculate total time in unit of dt
	const std::size_t OutputFreq = InitParams.get_output_freq();			  // save output frequency
	const std::size_t ReoptFreq = InitParams.get_reoptimize_freq();			  // save parameter re-optimization frequency
	const double dt = InitParams.get_dt();									  // save dt directly
	const std::size_t NumPoints = InitParams.get_number_of_selected_points(); // save the number of points for evolving/optimizing
	const std::size_t NumExtraPoints = NumPoints * 5;						  // The number of points extra selected for density fitting
	// get initial distribution
	MCParameters MCParams(SigmaR0.minCoeff());
	static constexpr std::array<double, NumPES> InitialPopulation = {1.0, 0.0};
	static constexpr std::array<double, NumPES> InitialPhaseFactor = {0.0, 0.0};
	const DistributionFunction initdist = [&r0, &SigmaR0](const ClassicalPhaseVector& r, const std::size_t RowIndex, const std::size_t ColIndex) -> std::complex<double>
	{
		return initial_distribution(r0, SigmaR0, r, RowIndex, ColIndex, InitialPopulation, InitialPhaseFactor);
	};
	AllPoints density; // generated from MC from initial distribution
	{
		for (std::size_t iPES = 0; iPES < NumPES; iPES++)
		{
			for (std::size_t jPES = 0; jPES <= iPES; jPES++)
			{
				if (InitialPopulation[iPES] > 0.0 && InitialPopulation[jPES] > 0.0)
				{
					density(iPES, jPES).insert(density(iPES, jPES).cend(), NumPoints, PhaseSpacePoint(r0, initdist(r0, iPES, jPES), 0));
				}
			}
		}
		monte_carlo_selection(MCParams, mass, dt, 0, initdist, density);
	}
	QuantumMatrix<bool> IsSmall = is_very_small(density);	 // whether each element of density matrix is close to 0 everywhere or not
	QuantumMatrix<bool> IsSmallOld = IsSmall;				 // whether each element is small at last moment; initially, not used
	QuantumMatrix<bool> IsNew = QuantumMatrix<bool>::Zero(); // whether some element used to be small but is not small now; initially, not used
	bool IsCouple = false;									 // whether the dynamics is adiabatic or not; initially, not used
	bool IsOptimized = true;								 // whether there has been optimization in this dt or not
	// calculate initial energy and purity
	const double TotalEnergy = [&density, &mass](void) -> double
	{
		const QuantumVector<double> Population = QuantumVector<double>::Map(InitialPopulation.data()).array().square();
		return Population.dot(calculate_total_energy_average_each_surface(density, mass)) / Population.sum();
	}();
	const double Purity = 1.0;
	spdlog::info("{}Initial Energy = {}; Initial Purity = {}", indents<0>::apply(), TotalEnergy, Purity);
	// sample extra points for density fitting
	AllPoints extra_points = generate_extra_points(density, NumExtraPoints, mass, dt, 0, initdist);
	// generate initial parameters and predictors
	Optimization optimizer(InitParams, TotalEnergy, Purity);
	spdlog::info("{}Optimization initially...", indents<0>::apply());
	Optimization::Result opt_result = optimizer.optimize(density, extra_points);
	std::unique_ptr<Kernels> all_kernels = std::make_unique<Kernels>(optimizer.get_parameters(), density);
	const DistributionFunction& predict_distribution = [&all_kernels](const ClassicalPhaseVector& r, const std::size_t RowIndex, const std::size_t ColIndex) -> std::complex<double>
	{
		assert(all_kernels);
		if (RowIndex == ColIndex)
		{
			return Kernel(r, (*all_kernels)(RowIndex), false).get_prediction_compared_with_variance().value();
		}
		else
		{
			return ComplexKernel(r, (*all_kernels)(RowIndex, ColIndex), false).get_prediction_compared_with_variance().value();
		}
	}; // predict current step distribution, used for resampling of density
	// output: average, parameters, selected points, gridded phase space
	std::ofstream average("ave.txt");
	std::ofstream param("param.txt", std::ios_base::binary);
	std::ofstream point("coord.txt", std::ios_base::binary);
	std::ofstream value("value.txt", std::ios_base::binary);
	std::ofstream phase("phase.txt", std::ios_base::binary);
	std::ofstream phasefactor("argument.txt", std::ios_base::binary);
	std::ofstream variance("var.txt", std::ios_base::binary);
	std::ofstream autocor("autocor.txt", std::ios_base::binary);
	std::ofstream logging("run.log");
	auto output = [&](const std::size_t NumTicks) -> void
	{
		end = std::chrono::high_resolution_clock::now();
		spdlog::info("{}Start output...", indents<0>::apply());
		const double time = NumTicks * dt;
		output_average(average, *all_kernels, density, mass);
		output_param(param, optimizer);
		output_point(point, value, density, extra_points);
		output_phase(phase, phasefactor, variance, mass, dt, NumTicks, *all_kernels, PhaseGrids);
		output_autocor(autocor, MCParams, predict_distribution, density);
		output_logging(logging, time, opt_result, MCParams, end - begin);
		spdlog::info(
			"{}T = {:>6.2f}, error = {:>+22.15e}, norm = {:>+22.15e}, energy = {:>+22.15e}, purity = {:>+22.15e}",
			indents<1>::apply(),
			time,
			std::get<0>(opt_result),
			all_kernels->calculate_population(),
			all_kernels->calculate_total_energy_average(calculate_total_energy_average_each_surface(density, mass)),
			all_kernels->calculate_purity()
		);
		spdlog::info("{}Finish output", indents<0>::apply());
		begin = end;
	};
	output(0);

	// evolution
	for (std::size_t iTick = 1; iTick <= TotalTicks; iTick++)
	{
		// evolve
		// both density and extra points needs evolution, as they are both used for prediction
		evolve(density, mass, dt, iTick, predict_distribution);
		evolve(extra_points, mass, dt, iTick, predict_distribution);
		IsNew = IsSmallOld - IsSmall;
		IsCouple = is_coupling(density, mass);
		IsOptimized = false;
		// reselect; nothing new, nothing happens
		if (IsNew.any())
		{
			new_element_point_selection(density, extra_points, IsNew, mass, dt, iTick, predict_distribution);
			spdlog::info("Optimization at T = {} because of element change", iTick * dt);
			opt_result = optimizer.optimize(density, extra_points);
			all_kernels = std::make_unique<Kernels>(optimizer.get_parameters(), density);
			extra_points = generate_extra_points(density, NumExtraPoints, mass, dt, iTick, predict_distribution);
			IsOptimized = true;
		}
		// judge if it is time to re-optimize, or there are new elements having population
		if (iTick % ReoptFreq == 0)
		{
			// model training if non-adiabatic
			if (IsCouple && !IsOptimized)
			{
				spdlog::info("Optimization at T = {} as the re-optimization required by non-adiabaticity", iTick * dt);
				opt_result = optimizer.optimize(density, extra_points);
				all_kernels = std::make_unique<Kernels>(optimizer.get_parameters(), density);
				extra_points = generate_extra_points(density, NumExtraPoints, mass, dt, iTick, predict_distribution);
				IsOptimized = true;
			}
			if (iTick % OutputFreq == 0)
			{
				if (!IsOptimized)
				{
					spdlog::info("Optimization at T = {} for output", iTick * dt);
					// not coupled, optimize for output
					opt_result = optimizer.optimize(density, extra_points);
					all_kernels = std::make_unique<Kernels>(optimizer.get_parameters(), density);
					extra_points = generate_extra_points(density, NumExtraPoints, mass, dt, iTick, predict_distribution);
					IsOptimized = true;
				}
				// output time
				output(iTick);
				// judge whether to finish or not: any of the dimension go over -x0
				const auto x_tot = calculate_1st_order_average_one_surface(density(0)).block<Dim, 1>(0, 0);
				const auto x0 = r0.block<Dim, 1>(0, 0);
				if ((x_tot.array() > -x0.array()).any())
				{
					break;
				}
			}
		}
		// update the training set for prediction
		if (!IsOptimized)
		{
			all_kernels = std::make_unique<Kernels>(optimizer.get_parameters(), density);
		}
		IsSmallOld = IsSmall;
	}

	// finalization
	average.close();
	param.close();
	point.close();
	value.close();
	phase.close();
	phasefactor.close();
	variance.close();
	autocor.close();
	logging.close();
}
