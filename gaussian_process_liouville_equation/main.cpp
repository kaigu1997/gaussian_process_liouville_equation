/// @file main.cpp
/// @brief the main driver

#include "stdafx.h"

#include "evolve.h"
#include "io.h"
#include "mc.h"
#include "pes.h"

int main()
{
	// initialization: save some input parameters, using initial distribution to select points, and initial output
	const Parameters params("input");									 // get mass, x0, p0, sigma_x0, sigma_p0, dt, output time
	const ClassicalDoubleVector& x0 = params.get_x0();					 // save initial positions
	const ClassicalDoubleVector& p0 = params.get_p0();					 // save initial momentum
	const Eigen::MatrixXd& PhaseGrids = params.get_phase_space_points(); // save output grids in phase space
	const ClassicalDoubleVector& mass = params.get_mass();				 // save mass directly
	const int TotalTicks = params.calc_total_ticks();					 // calculate total time in unit of dt
	const int OutputFreq = params.get_output_freq();					 // save output frequency
	const int ReoptFreq = params.get_reoptimize_freq();					 // save hyperparameter re-optimization frequency
	const double dt = params.get_dt();									 // save dt directly
	// get initial distribution
	EvolvingDensity density; // initial density
	for (int i = 0; i < params.get_number_of_selected_points(); i++)
	{
		density.push_back(std::make_tuple(x0, p0, initial_distribution(params, x0, p0))); // put the initial point into density
	}
	QuantumBoolMatrix IsSmall = QuantumBoolMatrix::Ones(); // matrix that saves whether each element of density matrix is close to 0 everywhere or not
	IsSmall(0, 0) = false;
	monte_carlo_selection(params, std::bind(initial_distribution, std::cref(params), std::placeholders::_1, std::placeholders::_2), IsSmall, density); // generated from MC from initial distribution
	// generate initial hyperparameters
	Optimization optimizer(params);
	double marg_ll = optimizer.optimize(density, IsSmall);
	const DistributionFunction& predict_distribution = std::bind(evolve_predict, std::placeholders::_1, std::placeholders::_2, mass, dt, std::cref(IsSmall), std::cref(optimizer));
	// initial output: average, hyperparameters, selected points, gridded phase space
	std::clog << 0 << ' ' << marg_ll << ' ' << print_time << std::endl;
	const Eigen::IOFormat VectorFormatter(Eigen::StreamPrecision, Eigen::DontAlignCols, " ", " ", "", "", " ", "");
	const Eigen::IOFormat MatrixFormatter(Eigen::StreamPrecision, Eigen::DontAlignCols, " ", "\n", " ", "", "", "");
	std::ofstream average("ave.txt");
	std::ofstream hyperparam("param.txt", std::ios_base::binary);
	std::ofstream point("point.txt", std::ios_base::binary);
	std::ofstream phase("phase.txt", std::ios_base::binary);
	// average: time, population, x, p, V, T of each PES
	average << 0 << ' ' << 1.0 << x0.format(VectorFormatter) << p0.format(VectorFormatter) << ' ' << adiabatic_potential(x0)[0] << ' ' << (p0.array().abs2() / mass.array()).sum() / 2.0;
	for (int iPES = 1; iPES < NumPES; iPES++)
	{
		average << ' ' << 0.0 << ClassicalDoubleVector::Zero().format(VectorFormatter) << ClassicalDoubleVector::Zero().format(VectorFormatter) << ' ' << 0.0 << ' ' << 0.0;
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
	point << print_point(density).format(MatrixFormatter) << "\n\n";
	for (int iPoint = 0; iPoint < PhaseGrids.cols(); iPoint++)
	{
		phase << ' ' << initial_distribution(params, PhaseGrids.block<Dim, 1>(0, iPoint), PhaseGrids.block<Dim, 1>(Dim, iPoint))(0, 0).real();
	}
	phase << '\n';
	for (int iElement = 1; iElement < NumElements; iElement++)
	{
		for (int iPoint = 0; iPoint < PhaseGrids.cols(); iPoint++)
		{
			phase << ' ' << 0;
		}
		phase << '\n';
	}
	phase << '\n';

	// evolution
	for (int iTick = 1; iTick <= TotalTicks; iTick++)
	{
		// evolve; using Trotter expansion
		// TODO: 5 steps splitting vs 7 steps
		IsSmall = is_very_small(density);
		monte_carlo_selection(params, predict_distribution, IsSmall, density);
		if (iTick % ReoptFreq == 0)
		{
			// reselect, model training
			marg_ll = optimizer.optimize(density, IsSmall);
			if (iTick % OutputFreq == 0)
			{
				// output time
				std::clog << iTick * dt << ' ' << marg_ll << ' ' << print_time << std::endl;
				// output average
				ClassicalDoubleVector x_tot = ClassicalDoubleVector::Zero();
				double ppl_tot = 0.0;
				average << iTick * dt;
				for (int iPES = 0; iPES < NumPES; iPES++)
				{
					const auto& [ppl, x, p, V, T] = optimizer.calculate_average(mass, IsSmall, iPES);
					average << ' ' << ppl << x.format(VectorFormatter) << p.format(VectorFormatter) << ' ' << V << ' ' << T;
					x_tot += x * ppl;
					ppl_tot += ppl;
				}
				average << std::endl;
				x_tot /= ppl_tot;
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
				point << print_point(density).format(MatrixFormatter) << "\n\n";
				// output phase
				for (int iElement = 0; iElement < NumElements; iElement++)
				{
					phase << optimizer.print_element(IsSmall, PhaseGrids, iElement).format(VectorFormatter) << '\n';
				}
				phase << '\n';
				// judge whether to finish or not: any of the dimension go over -x0
				if ((x_tot.array() > -x0.array()).any() == true)
				{
					break;
				}
			}
		}
	}

	// finalization
	average.close();
	hyperparam.close();
	point.close();
	phase.close();
	return 0;
}
