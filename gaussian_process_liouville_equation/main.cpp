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
	const Parameters Params("input");									 // get mass, x0, p0, sigma_x0, sigma_p0, dt, output time
	const ClassicalVector<double>& x0 = Params.get_x0();				 // save initial positions
	const ClassicalVector<double>& p0 = Params.get_p0();				 // save initial momentum
	const Eigen::MatrixXd& PhaseGrids = Params.get_phase_space_points(); // save output grids in phase space
	const ClassicalVector<double>& mass = Params.get_mass();			 // save mass directly
	const int TotalTicks = Params.calc_total_ticks();					 // calculate total time in unit of dt
	const int OutputFreq = Params.get_output_freq();					 // save output frequency
	const int ReoptFreq = Params.get_reoptimize_freq();					 // save parameter re-optimization frequency
	const double dt = Params.get_dt();									 // save dt directly
	const int NumPoints = Params.get_number_of_selected_points();		 // save the number of points for evolving/optimizing
	const int NumMCPoints = NumPoints * 5;								 // The number of points for mc integral
	// get initial distribution
	const std::array<double, NumPES> InitialPopulation = {0.5, 0.5};
	const DistributionFunction initdist = std::bind(initial_distribution, Params, InitialPopulation, std::placeholders::_1, std::placeholders::_2);
	AllPoints density;
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		if (InitialPopulation[iPES] > 0.0)
		{
			density[iPES * NumPES + iPES] = EigenVector<PhaseSpacePoint>(NumPoints, std::make_tuple(x0, p0, initdist(x0, p0)));
		}
	}
	QuantumMatrix<bool> IsSmall = is_very_small(density);	 // whether each element of density matrix is close to 0 everywhere or not
	QuantumMatrix<bool> IsSmallOld = IsSmall;				 // whether each element is small at last moment; initially, not used
	QuantumMatrix<bool> IsNew = QuantumMatrix<bool>::Zero(); // whether some element used to be small but is not small now; initially, not used
	bool IsCouple = false;									 // whether the dynamics is adiabatic or not; initially, not used
	MCParameters MCParams(Params);
	monte_carlo_selection(Params, MCParams, initdist, density); // generated from MC from initial distribution
	// calculate initial energy and purity
	const double TotalEnergy = [&InitialPopulation, &density, &mass](void) -> double
	{
		const double SumWeight = std::reduce(InitialPopulation.begin(), InitialPopulation.end());
		double result = 0.0;
		for (int iPES = 0; iPES < NumPES; iPES++)
		{
			result += InitialPopulation[iPES] * calculate_total_energy_average_one_surface(density[iPES * NumPES + iPES], mass, iPES);
		}
		return result / SumWeight;
	}();
	const double Purity = QuantumVector<double>(InitialPopulation.data()).squaredNorm() / std::pow(std::reduce(InitialPopulation.cbegin(), InitialPopulation.cend()), 2);
	spdlog::info("Initial Energy = {:>+13.6e}; Initial Purity = {:>+13.6e}", TotalEnergy, Purity);
	// sample points for calculate MC average
	AllPoints mc_points = mc_points_generator(density, NumMCPoints);
	// generate initial parameters and predictors
	Optimization optimizer(Params, TotalEnergy, Purity);
	Optimization::Result opt_result = optimizer.optimize(density, mc_points);
	OptionalKernels kernels;
	construct_predictors(kernels, optimizer.get_parameters(), density);
	const DistributionFunction& evolve_predict_distribution = std::bind(
		non_adiabatic_evolve_predict,
		std::placeholders::_1,
		std::placeholders::_2,
		mass,
		dt,
		std::cref(kernels)); // predict next step distribution
	const DistributionFunction& predict_distribution = std::bind(
		predict_matrix,
		std::cref(kernels),
		std::placeholders::_1,
		std::placeholders::_2); // predict current step distribution
	// output: average, parameters, selected points, gridded phase space
	std::ofstream average("ave.txt");
	std::ofstream param("param.txt", std::ios_base::binary);
	std::ofstream densel("prm_point.txt", std::ios_base::binary); // densel = density selected; prm = parameter (optimization)
	std::ofstream mcint("mci_point.txt", std::ios_base::binary); // mcint = monte carlo integral
	std::ofstream phase("phase.txt", std::ios_base::binary);
	std::ofstream variance("var.txt", std::ios_base::binary);
	std::ofstream autocor("autocor.txt", std::ios_base::binary);
	std::ofstream logging("run.log");
	auto output = [&](const double time, const DistributionFunction& dist) -> void
	{
		end = std::chrono::high_resolution_clock::now();
		average << time;
		output_average(average, kernels, density, mc_points, mass);
		output_param(param, optimizer);
		output_point(densel, density);
		output_point(mcint, mc_points);
		output_phase(phase, variance, kernels, PhaseGrids);
		output_autocor(autocor, MCParams, dist, density);
		output_logging(logging, time, opt_result, MCParams, end - begin);
		spdlog::info(
			"T = {:>4.0f}, error = {:>+13.6e}, norm = {:>+13.6e}, energy = {:>+13.6e}, purity = {:>+13.6e}",
			time,
			std::get<0>(opt_result),
			calculate_population(kernels, mc_points),
			calculate_total_energy_average(kernels, mc_points, mass),
			calculate_purity(kernels, mc_points));
		begin = end;
	};
	output(0.0, initdist);

	// evolution
	for (int iTick = 1; iTick <= TotalTicks; iTick++)
	{
		// evolve
		evolve(density, mass, dt, kernels);
		IsSmall = is_very_small(density);
		IsNew = IsSmallOld.array() > IsSmall.array();
		IsCouple = is_coupling(density, mass);
		// reselect; nothing new, nothing happens
		if (IsNew.any())
		{
			new_element_point_selection(density, IsNew);
			opt_result = optimizer.optimize(density, mc_points);
			construct_predictors(kernels, optimizer.get_parameters(), density);
		}
		// judge if it is time to re-optimize, or there are new elements having population
		if (iTick % ReoptFreq == 0)
		{
			// model training if non-adiabatic
			if (IsCouple)
			{
				mc_points = mc_points_generator(density, NumMCPoints);
				opt_result = optimizer.optimize(density, mc_points);
				construct_predictors(kernels, optimizer.get_parameters(), density);
			}
			if (iTick % OutputFreq == 0)
			{
				if (IsCouple)
				{
					// if in the coupling region, reselect points
					monte_carlo_selection(Params, MCParams, predict_distribution, density, true);
				}
				else
				{
					// not coupled, optimize for output
					mc_points = mc_points_generator(density, NumMCPoints);
					opt_result = optimizer.optimize(density, mc_points);
					construct_predictors(kernels, optimizer.get_parameters(), density);
				}
				// output time
				output(iTick * dt, predict_distribution);
				// judge whether to finish or not: any of the dimension go over -x0
				const ClassicalVector<double> x_tot = calculate_1st_order_average(kernels, mc_points).block<Dim, 1>(0, 0);
				if ((x_tot.array() > -x0.array()).any())
				{
					break;
				}
			}
		}
		else
		{
			// otherwise, update the training set for prediction
			construct_predictors(kernels, optimizer.get_parameters(), density);
		}
		IsSmallOld = IsSmall;
	}

	// finalization
	average.close();
	param.close();
	densel.close();
	mcint.close();
	phase.close();
	variance.close();
	autocor.close();
	logging.close();
}
