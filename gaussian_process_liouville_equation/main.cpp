/// @file main.cpp
/// @brief The main driver

#include "stdafx.h"

#include "evolve.h"
#include "input.h"
#include "kernel.h"
#include "mc.h"
#include "mc_ave.h"
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
	// get initial distribution and energy
	const std::array<double, NumPES> InitialPopulation = {0.5, 0.5};
	EigenVector<PhaseSpacePoint> density(
		Params.get_number_of_selected_points() * NumPoints,
		std::make_tuple(x0, p0, initial_distribution(Params, InitialPopulation, x0, p0))); // initial density
	QuantumMatrix<bool> IsSmall = is_very_small(density);								   // whether each element of density matrix is close to 0 everywhere or not
	QuantumMatrix<bool> IsSmallOld = IsSmall;											   // whether each element is small at last moment
	QuantumMatrix<bool> IsNew = IsSmallOld.array() > IsSmall.array();					   // whether some element used to be small but is not small now
	bool IsCouple = false;																   // whether the dynamics is adiabatic or not
	MCParameters MCParams(Params);
	const DistributionFunction initdist = std::bind(initial_distribution, Params, InitialPopulation, std::placeholders::_1, std::placeholders::_2);
	monte_carlo_selection(Params, MCParams, initdist, IsSmall, density); // generated from MC from initial distribution
	const double TotalEnergy = [&](void) -> double
	{
		const double SumWeight = std::accumulate(InitialPopulation.begin(), InitialPopulation.end(), 0.);
		double result = 0.;
		for (int iPES = 0; iPES < NumPES; iPES++)
		{
			result += InitialPopulation[iPES] * calculate_total_energy_average(density, iPES, mass);
		}
		return result / SumWeight;
	}(); // Total energy, calculated from MC average
	const double Purity =
		std::accumulate(
			InitialPopulation.cbegin(),
			InitialPopulation.cend(),
			0.,
			[](const double sum, const double elem) -> double
			{
				return sum + elem * elem;
			})
		/ std::pow(std::accumulate(InitialPopulation.cbegin(), InitialPopulation.cend(), 0.), 2); // Purity, sum of squared weight over square of sum weight
	spdlog::info("Initial Energy = {:>+13.6e}; Initial Purity = {:>+13.6e}", TotalEnergy, Purity);
	// generate initial parameters and predictors
	Optimization optimizer(Params);
	Predictions predictors;
	Optimization::Result opt_result = optimizer.optimize(density, mass, TotalEnergy, Purity, IsSmall);
	construct_predictors(predictors, optimizer, IsSmall, density);
	const DistributionFunction& evolve_predict_distribution = std::bind(
		non_adiabatic_evolve_predict,
		std::placeholders::_1,
		std::placeholders::_2,
		mass,
		dt,
		std::cref(predictors)); // predict next step distribution
	const DistributionFunction& predict_distribution = std::bind(
		predict_matrix,
		std::cref(predictors),
		std::placeholders::_1,
		std::placeholders::_2); // predict current step distribution
	// output: average, parameters, selected points, gridded phase space
	std::ofstream average("ave.txt");
	std::ofstream param("param.txt", std::ios_base::binary);
	std::ofstream point("point.txt", std::ios_base::binary);
	std::ofstream phase("phase.txt", std::ios_base::binary);
	std::ofstream autocor("autocor.txt", std::ios_base::binary);
	std::ofstream logging("run.log");
	auto output = [&](const double time, const DistributionFunction& dist) -> void
	{
		end = std::chrono::high_resolution_clock::now();
		output_average(average, time, predictors, density, mass);
		output_param(param, optimizer);
		output_point(point, density);
		output_phase(phase, predictors, PhaseGrids);
		output_autocor(autocor, MCParams, dist, IsSmall, density);
		output_logging(logging, time, opt_result, MCParams, end - begin);
		spdlog::info(
			"T = {:>4.0f}, error = {:>+13.6e}, norm = {:>+13.6e}, energy = {:>+13.6e}, purity = {:>+13.6e}",
			time,
			std::get<0>(opt_result),
			calculate_population(predictors),
			calculate_total_energy_average(predictors, density, mass),
			calculate_purity(predictors));
		begin = end;
	};
	output(0.0, initdist);

	// evolution
	for (int iTick = 1; iTick <= TotalTicks; iTick++)
	{
		// evolve
		evolve(density, MCParams.get_num_points(), mass, dt, predictors);
		IsSmall = is_very_small(density);
		IsNew = IsSmallOld.array() > IsSmall.array();
		IsCouple = std::any_of(
			std::execution::par_unseq,
			density.cbegin(),
			density.cend(),
			[&mass](const PhaseSpacePoint& psp) -> bool
			{
				const auto& [x, p, rho] = psp;
				return is_coupling(x, p, mass).any();
			});
		// judge if it is time to re-optimize, or there are new elements having population
		if (iTick % ReoptFreq == 0 || IsNew.any())
		{
			// reselect; nothing new, nothing happens
			new_element_point_selection(density, IsNew, NumPoints);
			// model training if non-adiabatic
			if (IsCouple)
			{
				opt_result = optimizer.optimize(density, mass, TotalEnergy, Purity, IsSmall);
				construct_predictors(predictors, optimizer, IsSmall, density);
			}
			if (iTick % OutputFreq == 0)
			{
				if (IsCouple)
				{
					// if in the coupling region, reselect points
					monte_carlo_selection(Params, MCParams, predict_distribution, IsSmall, density);
				}
				else
				{
					// not coupled, optimize for output
					opt_result = optimizer.optimize(density, mass, TotalEnergy, Purity, IsSmall);
					construct_predictors(predictors, optimizer, IsSmall, density);
				}
				// output time
				output(iTick * dt, predict_distribution);
				// judge whether to finish or not: any of the dimension go over -x0
				const ClassicalVector<double> x_tot = calculate_1st_order_average(predictors).block<Dim, 1>(0, 0);
				if ((x_tot.array() > -x0.array()).any())
				{
					break;
				}
			}
		}
		else
		{
			// otherwise, update the training set for prediction
			construct_predictors(predictors, optimizer, IsSmall, density);
		}
		IsSmallOld = IsSmall;
	}

	// finalization
	average.close();
	param.close();
	point.close();
	phase.close();
	autocor.close();
	logging.close();
	return 0;
}
