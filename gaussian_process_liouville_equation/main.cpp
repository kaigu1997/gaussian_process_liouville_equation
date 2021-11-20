/// @file main.cpp
/// @brief the main driver

#include "stdafx.h"

#include "evolve.h"
#include "io.h"
#include "mc.h"
#include "opt.h"
#include "pes.h"

int main()
{
	// initialization: save some input parameters, using initial distribution to select points, and initial output
	std::chrono::high_resolution_clock::time_point begin = std::chrono::high_resolution_clock::now(), end = begin;
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
	Optimization optimizer(Params);
	const double TotalEnergy = [&](void) -> double
	{
		const double SumWeight = std::accumulate(InitialPopulation.begin(), InitialPopulation.end(), 0.);
		double result = 0.;
		for (int iPES = 0; iPES < NumPES; iPES++)
		{
			result += InitialPopulation[iPES] * (calculate_kinetic_average(density, iPES, mass) + calculate_potential_average(density, iPES));
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
	std::clog << "Initial Energy = " << TotalEnergy << "\nInitial Purity = " << Purity << std::endl;
	// generate initial parameters
	const DistributionFunction& evolve_predict_distribution = std::bind(
		non_adiabatic_evolve_predict,
		std::placeholders::_1,
		std::placeholders::_2,
		mass,
		dt,
		std::cref(IsSmall),
		std::cref(optimizer)); // predict next step distribution
	const DistributionFunction& predict_distribution = std::bind(
		&Optimization::predict_matrix,
		&optimizer,
		std::cref(IsSmall),
		std::placeholders::_1,
		std::placeholders::_2); // predict current step distribution
	auto [err, steps] = optimizer.optimize(density, mass, TotalEnergy, Purity, IsSmall);
	double ppl = optimizer.normalize(density, IsSmall);
	optimizer.update_training_set(density, IsSmall);
	// output: average, parameters, selected points, gridded phase space
	std::ofstream average("ave.txt");
	std::ofstream param("param.txt", std::ios_base::binary);
	std::ofstream point("point.txt", std::ios_base::binary);
	std::ofstream phase("phase.txt", std::ios_base::binary);
	std::ofstream autocor("autocor.txt", std::ios_base::binary);
	std::ofstream log("run.log");
	auto output = [&](const double time, const double error, const std::vector<int>& Steps) -> void
	{
		static const Eigen::IOFormat VectorFormatter(Eigen::StreamPrecision, Eigen::DontAlignCols, " ", " ", "", "", " ", "");
		static const Eigen::IOFormat MatrixFormatter(Eigen::StreamPrecision, Eigen::DontAlignCols, " ", "\n", " ", "", "", "");
		// average: time, population, x, p, V, T, E of each PES
		double ppl_all = 0.;
		ClassicalPhaseVector r_prm_all = ClassicalPhaseVector::Zero();
		ClassicalPhaseVector r_mc_all = ClassicalPhaseVector::Zero();
		double T_all = 0.;
		double V_all = 0.;
		average << time;
		for (int iPES = 0; iPES < NumPES; iPES++)
		{
			const double ppl_prm = optimizer.calculate_population(IsSmall, iPES); // prm = parameter
			const ClassicalPhaseVector r_prm = optimizer.calculate_1st_order_average(IsSmall, iPES);
			const ClassicalPhaseVector r_mc = calculate_1st_order_average(density, iPES);
			const double T_mc = calculate_kinetic_average(density, iPES, mass), V_mc = calculate_potential_average(density, iPES);
			average << ' ' << ppl_prm << r_prm.format(VectorFormatter) << r_mc.format(VectorFormatter) << ' ' << T_mc << ' ' << V_mc << ' ' << V_mc + T_mc;
			ppl_all += ppl_prm;
			r_prm_all += ppl_prm * r_prm;
			r_mc_all += ppl_prm * r_mc;
			T_all += ppl_prm * T_mc;
			V_all += ppl_prm * V_mc;
		}
		average << ' ' << ppl_all << r_prm_all.format(VectorFormatter) << r_mc_all.format(VectorFormatter) << ' ' << T_all << ' ' << V_all << ' ' << V_all + T_all;
		average << ' ' << optimizer.calculate_purity(IsSmall) << std::endl;
		for (int iElement = 0; iElement < NumElements; iElement++)
		{
			for (double d : optimizer.get_parameter(iElement))
			{
				param << ' ' << d;
			}
			param << '\n';
		}
		param << '\n';
		point << print_point(density, NumPoints).format(MatrixFormatter) << "\n\n";
		for (int iElement = 0; iElement < NumElements; iElement++)
		{
			phase << optimizer.print_element(IsSmall, PhaseGrids, iElement).format(VectorFormatter) << '\n';
		}
		phase << '\n';
		// output autocorrelation
		AutoCorrelations step_autocor = autocorrelation_optimize_steps(MCParams, initdist, IsSmall, density);
		AutoCorrelations displ_autocor = autocorrelation_optimize_displacement(MCParams, initdist, IsSmall, density);
		for (int iElement = 0; iElement < NumElements; iElement++)
		{
			autocor << step_autocor[iElement].format(VectorFormatter) << displ_autocor[iElement].format(VectorFormatter) << '\n';
		}
		autocor << '\n';
		end = std::chrono::high_resolution_clock::now();
		log << time << ' ' << error << ' ' << ppl << ' ' << MCParams.get_num_MC_steps() << ' ' << MCParams.get_max_dispalcement();
		for (auto step : Steps)
		{
			log << ' ' << step;
		}
		log << ' ' << std::chrono::duration<double>(end - begin).count() << ' ' << print_time << std::endl;
		begin = end;
	};
	output(0.0, err, steps);

	// evolution
	for (int iTick = 1; iTick <= TotalTicks; iTick++)
	{
		// evolve
		evolve(density, MCParams.get_num_points(), mass, dt, IsSmall, optimizer);
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
		if (iTick % ReoptFreq == 0 || IsNew.any() == true)
		{
			// reselect; nothing new, nothing happens
			new_element_point_selection(density, IsNew, NumPoints);
			// model training if non-adiabatic
			if (IsCouple == true)
			{
				std::tie(err, steps) = optimizer.optimize(density, mass, TotalEnergy, Purity, IsSmall);
				ppl = optimizer.normalize(density, IsSmall);
				optimizer.update_training_set(density, IsSmall);
			}
			if (iTick % OutputFreq == 0)
			{
				if (IsCouple == true)
				{
					// if in the coupling region, reselect points
					monte_carlo_selection(Params, MCParams, predict_distribution, IsSmall, density);
				}
				else
				{
					// not coupled, optimize for output
					std::tie(err, steps) = optimizer.optimize(density, mass, TotalEnergy, Purity, IsSmall);
					ppl = optimizer.normalize(density, IsSmall);
					optimizer.update_training_set(density, IsSmall);
				}
				// output time
				output(iTick * dt, err, steps);
				// judge whether to finish or not: any of the dimension go over -x0
				ClassicalVector<double> x_tot = ClassicalVector<double>::Zero();
				for (int iPES = 0; iPES < NumPES; iPES++)
				{
					x_tot += calculate_1st_order_average(density, iPES).block<Dim, 1>(0, 0) * optimizer.calculate_population(IsSmall, iPES);
				}
				if ((x_tot.array() > -x0.array()).any() == true)
				{
					break;
				}
			}
		}
		else
		{
			// otherwise, update the training set for prediction
			optimizer.update_training_set(density, IsSmall);
		}
		IsSmallOld = IsSmall;
	}

	// finalization
	average.close();
	param.close();
	point.close();
	phase.close();
	autocor.close();
	return 0;
}
