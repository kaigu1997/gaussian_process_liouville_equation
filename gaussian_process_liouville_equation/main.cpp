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
	const Parameters Params("input");									 // get mass, x0, p0, sigma_x0, sigma_p0, dt, output time
	const ClassicalDoubleVector& x0 = Params.get_x0();					 // save initial positions
	const ClassicalDoubleVector& p0 = Params.get_p0();					 // save initial momentum
	const Eigen::MatrixXd& PhaseGrids = Params.get_phase_space_points(); // save output grids in phase space
	const ClassicalDoubleVector& mass = Params.get_mass();				 // save mass directly
	const int TotalTicks = Params.calc_total_ticks();					 // calculate total time in unit of dt
	const int OutputFreq = Params.get_output_freq();					 // save output frequency
	const int ReoptFreq = Params.get_reoptimize_freq();					 // save hyperparameter re-optimization frequency
	const double dt = Params.get_dt();									 // save dt directly
	const int NumPoints = Params.get_number_of_selected_points();		 // save the number of points for evolving/optimizing
	// get initial distribution
	EvolvingDensity density; // initial density
	for (int i = 0; i < Params.get_number_of_selected_points(); i++)
	{
		density.push_back(std::make_tuple(x0, p0, initial_distribution(Params, x0, p0))); // put the initial point into density
	}
	QuantumBoolMatrix IsSmall = QuantumBoolMatrix::Ones();			// whether each element of density matrix is close to 0 everywhere or not
	QuantumBoolMatrix IsSmallOld = IsSmall;							// whether each element is small at last moment
	QuantumBoolMatrix IsNew = IsSmallOld.array() > IsSmall.array(); // whether some element used to be small but is not small now
	IsSmall(0, 0) = false;
	MCParameters MCParams(Params);
	const DistributionFunction initdist = std::bind(initial_distribution, Params, std::placeholders::_1, std::placeholders::_2);
	monte_carlo_selection(Params, MCParams, initdist, IsSmall, density, false); // generated from MC from initial distribution
	// generate initial hyperparameters
	Optimization optimizer(Params);
	const DistributionFunction& evolve_predict_distribution = std::bind(
		evolve_predict,
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
	double err = optimizer.optimize(density, IsSmall);
	double ppl = optimizer.normalize(density, IsSmall);
	optimizer.update_training_set(density, IsSmall);
	// initial output: average, hyperparameters, selected points, gridded phase space
	const Eigen::IOFormat VectorFormatter(Eigen::StreamPrecision, Eigen::DontAlignCols, " ", " ", "", "", " ", "");
	const Eigen::IOFormat MatrixFormatter(Eigen::StreamPrecision, Eigen::DontAlignCols, " ", "\n", " ", "", "", "");
	std::ofstream average("ave.txt");
	std::ofstream hyperparam("param.txt", std::ios_base::binary);
	std::ofstream point("point.txt", std::ios_base::binary);
	std::ofstream phase("phase.txt", std::ios_base::binary);
	std::ofstream autocor("autocor.txt", std::ios_base::binary);
	// average: time, population, x, p, V, T, E of each PES
	average << 0.;
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		const auto& [ppl, x, p, V, T] = optimizer.calculate_average(mass, IsSmall, iPES);
		average << ' ' << ppl << x.format(VectorFormatter) << p.format(VectorFormatter) << ' ' << V << ' ' << T << ' ' << V + T;
	}
	average << std::endl;
	for (int iElement = 0; iElement < NumElements; iElement++)
	{
		for (double d : optimizer.get_hyperparameter(iElement))
		{
			hyperparam << ' ' << d;
		}
		hyperparam << '\n';
	}
	hyperparam << '\n';
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
	std::cout << 0 << ' ' << err << ' ' << ppl << ' ' << MCParams.get_num_MC_steps() << ' ' << MCParams.get_max_dispalcement() << ' ' << print_time << std::endl;

	// evolution
	for (int iTick = 1; iTick <= TotalTicks; iTick++)
	{
		// evolve; using Trotter expansion, combined with MC selection
		monte_carlo_selection(Params, MCParams, evolve_predict_distribution, IsSmall, density);
		IsSmall = is_very_small(density);
		IsNew = IsSmallOld.array() > IsSmall.array();
		// judge if it is time to re-optimize, or there are new elements having population
		if (iTick % ReoptFreq == 0 || IsNew.any() == true)
		{
			// reselect; nothing new, nothing happens
			new_element_point_selection(density, IsNew, NumPoints);
			// model training
			err = optimizer.optimize(density, IsSmall);
			ppl = optimizer.normalize(density, IsSmall);
			optimizer.update_training_set(density, IsSmall);
			if (iTick % OutputFreq == 0)
			{
				// output time
				// output average
				ClassicalDoubleVector x_tot = ClassicalDoubleVector::Zero();
				average << iTick * dt;
				for (int iPES = 0; iPES < NumPES; iPES++)
				{
					const auto& [ppl, x, p, V, T] = optimizer.calculate_average(mass, IsSmall, iPES);
					average << ' ' << ppl << x.format(VectorFormatter) << p.format(VectorFormatter) << ' ' << V << ' ' << T << ' ' << V + T;
					x_tot += x * ppl;
				}
				average << std::endl;
				// output hyperparameters
				for (int iElement = 0; iElement < NumElements; iElement++)
				{
					for (double d : optimizer.get_hyperparameter(iElement))
					{
						hyperparam << ' ' << d;
					}
					hyperparam << '\n';
				}
				hyperparam << '\n';
				// output point
				point << print_point(density, NumPoints).format(MatrixFormatter) << "\n\n";
				// output phase
				for (int iElement = 0; iElement < NumElements; iElement++)
				{
					phase << optimizer.print_element(IsSmall, PhaseGrids, iElement).format(VectorFormatter) << '\n';
				}
				phase << '\n';
				// output autocorrelation
				step_autocor = autocorrelation_optimize_steps(MCParams, initdist, IsSmall, density);
				displ_autocor = autocorrelation_optimize_displacement(MCParams, initdist, IsSmall, density);
				for (int iElement = 0; iElement < NumElements; iElement++)
				{
					autocor << step_autocor[iElement].format(VectorFormatter) << displ_autocor[iElement].format(VectorFormatter) << '\n';
				}
				autocor << '\n';
				std::cout << iTick * dt << ' ' << err << ' ' << ppl << ' ' << MCParams.get_num_MC_steps() << ' ' << MCParams.get_max_dispalcement() << ' ' << print_time << std::endl;
				// judge whether to finish or not: any of the dimension go over -x0
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
	hyperparam.close();
	point.close();
	phase.close();
	autocor.close();
	return 0;
}
