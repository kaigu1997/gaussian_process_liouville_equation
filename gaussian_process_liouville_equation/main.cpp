/// @file main.cpp
/// @brief The main driver

#include "stdafx.h"

#include "evolve.h"
#include "input.h"
#include "kernel.h"
#include "mc.h"
#include "mc_predict.h"
#include "opt.h"
#include "output.h"
#include "pes.h"
#include "predict.h"

int main()
{
	// initialization: save some input parameters, using initial distribution to select points, and initial output
	std::chrono::high_resolution_clock::time_point begin = std::chrono::high_resolution_clock::now(), end = begin;
	spdlog::set_pattern("[%=4Y-%=2m-%=2d %=8T.%=9F %=6z][Thread %t][Process %P][Logger %n][%=8l] %v");
	spdlog::set_default_logger(spdlog::stderr_color_mt("console"));
	const InitialParameters InitParams("input");							  // get mass, r0, sigma_r0, dt, output time
	const ClassicalPhaseVector& r0 = InitParams.get_r0();					  // save initial phase space point
	const ClassicalPhaseVector& SigmaR0 = InitParams.get_sigma_r0();		  // save initial standard deviation
	const Eigen::MatrixXd& PhaseGrids = InitParams.get_phase_space_points();  // save output grids in phase space
	const ClassicalVector<double>& mass = InitParams.get_mass();			  // save mass directly
	const std::size_t TotalTicks = InitParams.calc_total_ticks();			  // calculate total time in unit of dt
	const std::size_t OutputFreq = InitParams.get_output_freq();			  // save output frequency
	const std::size_t ReoptFreq = InitParams.get_reoptimize_freq();			  // save parameter re-optimization frequency
	const double dt = InitParams.get_dt();									  // save dt directly
	const std::size_t NumPoints = InitParams.get_number_of_selected_points(); // save the number of points for evolving/optimizing
	const std::size_t NumMCPoints = NumPoints * 5;							  // The number of points for monte carlo integral
	// get initial distribution
	MCParameters MCParams(InitParams);
	static constexpr std::array<double, NumPES> InitialPopulation = {0.5, 0.5};
	const DistributionFunction initdist = std::bind(initial_distribution, r0, SigmaR0, InitialPopulation, std::placeholders::_1);
	auto generate_initial_points = [&MCParams, &r0, &initdist](std::size_t NumPoints) -> AllPoints
	{
		AllPoints result;
		for (std::size_t iPES = 0; iPES < NumPES; iPES++)
		{
			if (InitialPopulation[iPES] > 0.0)
			{
				result[iPES * NumPES + iPES] = EigenVector<PhaseSpacePoint>(NumPoints, std::make_tuple(r0, initdist(r0)));
			}
		}
		monte_carlo_selection(MCParams, initdist, result); // generated from MC from initial distribution
		return result;
	};
	AllPoints density = generate_initial_points(NumPoints);;
	QuantumMatrix<bool> IsSmall = is_very_small(density);	 // whether each element of density matrix is close to 0 everywhere or not
	QuantumMatrix<bool> IsSmallOld = IsSmall;				 // whether each element is small at last moment; initially, not used
	QuantumMatrix<bool> IsNew = QuantumMatrix<bool>::Zero(); // whether some element used to be small but is not small now; initially, not used
	bool IsCouple = false;									 // whether the dynamics is adiabatic or not; initially, not used
	// calculate initial energy and purity
	const double TotalEnergy = [&density, &mass](void) -> double
	{
		const double SumWeight = std::reduce(InitialPopulation.begin(), InitialPopulation.end());
		double result = 0.0;
		for (std::size_t iPES = 0; iPES < NumPES; iPES++)
		{
			result += InitialPopulation[iPES] * calculate_total_energy_average_one_surface(density[iPES * NumPES + iPES], mass, iPES);
		}
		return result / SumWeight;
	}();
	const double Purity = QuantumVector<double>::Map(InitialPopulation.data()).squaredNorm() / std::pow(std::reduce(InitialPopulation.cbegin(), InitialPopulation.cend()), 2);
	spdlog::info("Initial Energy = {:>+13.6e}; Initial Purity = {:>+13.6e}", TotalEnergy, Purity);
	// sample extra points for density fitting
	AllPoints extra_points = generate_initial_points(NumPoints);
	// generate initial parameters and predictors
	Optimization optimizer(InitParams, TotalEnergy, Purity);
	Optimization::Result opt_result = optimizer.optimize(density, extra_points);
	OptionalKernels kernels;
	construct_predictors(kernels, optimizer.get_parameters(), density, extra_points);
	const DistributionFunction& predict_distribution = std::bind(
		predict_matrix,
		std::cref(kernels),
		std::placeholders::_1); // predict current step distribution
	const DistributionFunction& backward_evolve_predict_distribution = std::bind(
		non_adiabatic_evolve_predict,
		std::placeholders::_1,
		mass,
		dt,
		std::cref(kernels));
	// sample points for mc integral
	AllPoints mc_points = generate_initial_points(NumMCPoints);
	// output: average, parameters, selected points, gridded phase space
	std::ofstream average("ave.txt");
	std::ofstream param("param.txt", std::ios_base::binary);
	std::ofstream densel("prm_point.txt", std::ios_base::binary); // densel = density selected; prm = parameter (optimization)
	std::ofstream extra("xtr_point.txt", std::ios_base::binary);  // xtr = extra
	std::ofstream mcint("mci_point.txt", std::ios_base::binary);  // mcint = monte carlo integral
	std::ofstream phase("phase.txt", std::ios_base::binary);
	std::ofstream variance("var.txt", std::ios_base::binary);
	std::ofstream autocor("autocor.txt", std::ios_base::binary);
	std::ofstream logging("run.log");
	auto output = [&](const double time) -> void
	{
		end = std::chrono::high_resolution_clock::now();
		average << time;
		output_average(average, kernels, density, mc_points, mass);
		output_param(param, optimizer);
		output_point(densel, density);
		output_point(extra, extra_points);
		output_point(mcint, mc_points);
		output_phase(phase, variance, kernels, PhaseGrids);
		output_autocor(autocor, MCParams, predict_distribution, density);
		output_logging(logging, time, opt_result, MCParams, end - begin);
		spdlog::info(
			"T = {:>4.0f}, error = {:>+13.6e}, norm = {:>+13.6e}, energy = {:>+13.6e}, purity = {:>+13.6e}",
			time,
			std::get<0>(opt_result),
			calculate_population(kernels, extra_points),
			calculate_total_energy_average(kernels, extra_points, mass),
			calculate_purity(kernels, extra_points));
		begin = end;
	};
	output(0.0);

	// evolution
	for (std::size_t iTick = 1; iTick <= TotalTicks; iTick++)
	{
		// evolve
		evolve(density, mass, dt, kernels);
		evolve(extra_points, mass, dt, kernels);
		evolve(mc_points, mass, dt, kernels);
		IsSmall = is_very_small(density);
		IsNew = IsSmallOld.array() > IsSmall.array();
		IsCouple = is_coupling(density, mass);
		// reselect; nothing new, nothing happens
		if (IsNew.any())
		{
			new_element_point_selection(density, IsNew);
			extra_points = extra_points_generator(density, NumPoints, backward_evolve_predict_distribution);
			opt_result = optimizer.optimize(density, extra_points);
			construct_predictors(kernels, optimizer.get_parameters(), density, extra_points);
		}
		// judge if it is time to re-optimize, or there are new elements having population
		if (iTick % ReoptFreq == 0)
		{
			// model training if non-adiabatic
			if (IsCouple)
			{
				extra_points = extra_points_generator(density, NumPoints, backward_evolve_predict_distribution);
				opt_result = optimizer.optimize(density, extra_points);
				construct_predictors(kernels, optimizer.get_parameters(), density, extra_points);
			}
			if (iTick % OutputFreq == 0)
			{
				if (IsCouple)
				{
					// if in the coupling region, reselect points
					monte_carlo_selection(MCParams, predict_distribution, density);
				}
				else
				{
					// not coupled, optimize for output
					extra_points = extra_points_generator(density, NumPoints, backward_evolve_predict_distribution);
					opt_result = optimizer.optimize(density, extra_points);
					construct_predictors(kernels, optimizer.get_parameters(), density, extra_points);
				}
				monte_carlo_selection(MCParams, predict_distribution, mc_points);
				// output time
				output(iTick * dt);
				// judge whether to finish or not: any of the dimension go over -x0
				const auto x_tot = calculate_1st_order_average_one_surface(density[0]).block<Dim, 1>(0, 0);
				const auto x0 = r0.block<Dim, 1>(0, 0);
				if ((x_tot.array() > -x0.array()).any())
				{
					break;
				}
			}
		}
		else
		{
			// otherwise, update the training set for prediction
			construct_predictors(kernels, optimizer.get_parameters(), density, extra_points);
		}
		IsSmallOld = IsSmall;
	}

	// finalization
	average.close();
	param.close();
	densel.close();
	extra.close();
	mcint.close();
	phase.close();
	variance.close();
	autocor.close();
	logging.close();
}
