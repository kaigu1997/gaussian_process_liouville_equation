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
	const int ReselectFreq = params.get_reselect_freq();				 // save point re-selection frequency
	const double dt = params.get_dt();									 // save dt directly
	const double dt_over_2 = dt / 2.0;
	const double dt_over_4 = dt / 4.0;
	// get initial distribution
	EvolvingDensity density; // initial density
	density.push_back(std::make_tuple(x0, p0, initial_distribution(params, x0, p0))); // put the initial point into density
	QuantumBoolMatrix IsSmall = QuantumBoolMatrix::Ones();							  // matrix that saves whether each element of density matrix is close to 0 everywhere or not
	IsSmall(0, 0) = false;
	monte_carlo_selection(params, std::bind(initial_distribution, std::cref(params), std::placeholders::_1, std::placeholders::_2), IsSmall, density); // generated from MC from initial distribution
	// generate initial hyperparameters
	Optimization optimizer(params);
	optimizer.optimize(density, IsSmall);
	const DistributionFunction& predict_distribution = std::bind(&Optimization::predict, &optimizer, IsSmall, std::placeholders::_1, std::placeholders::_2);
	// initial output: average, hyperparameters, selected points, gridded phase space
	std::clog << 0 << ' ' << print_time << std::endl;
	const Eigen::IOFormat VectorFormatter(Eigen::StreamPrecision, Eigen::DontAlignCols, " ", " ", "", "", " ", "");
	const Eigen::IOFormat MatrixFormatter(Eigen::StreamPrecision, Eigen::DontAlignCols, " ", "\n", " ", "", "", "");
	std::ofstream average("ave.txt");
	std::ofstream hyperparam("param.txt", std::ios_base::binary);
	std::ofstream point("point.txt", std::ios_base::binary);
	std::ofstream phase("phase.txt", std::ios_base::binary);
	average << 0 << ' ' << 1.0 << x0.format(VectorFormatter) << p0.format(VectorFormatter) << ' ' << adiabatic_potential(x0)[0] << ' ' << (p0.array().abs2() / mass.array()).sum() / 2.0;
	for (int iPES = 1; iPES < NumPES; iPES++)
	{
		average << ' ' << 0.0 << ClassicalDoubleVector::Zero().format(VectorFormatter) << ClassicalDoubleVector::Zero().format(VectorFormatter) << ' ' << 0.0 << ' ' << 0.0;
	}
	average << '\n';
	for (int iElement = 0; iElement < NumElement; iElement++)
	{
		for (double d : optimizer.get_hyperparameter(iElement))
		{
			hyperparam << ' ' << d;
		}
		hyperparam << '\n';
	}
	hyperparam << '\n';
	point << print_point(density).format(MatrixFormatter) << '\n' << '\n';
	for (int iPoint = 0; iPoint < PhaseGrids.cols(); iPoint++)
	{
		phase << ' ' << initial_distribution(params, PhaseGrids.block<Dim, 1>(0, iPoint), PhaseGrids.block<Dim, 1>(Dim, iPoint))(0, 0).real();
	}
	phase << '\n';
	for (int iElement = 1; iElement < NumElement; iElement++)
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
		quantum_liouville(density, mass, dt_over_4);
		classical_position_liouville(density, mass, dt_over_2);
		quantum_liouville(density, mass, dt_over_4);
		classical_momentum_liouville(density, optimizer, dt);
		quantum_liouville(density, mass, dt_over_4);
		classical_position_liouville(density, mass, dt_over_2);
		quantum_liouville(density, mass, dt_over_4);
		if (iTick % ReselectFreq == 0)
		{
			// reselect, model training
			IsSmall = is_very_small(density);
			std::cout << IsSmall.format(VectorFormatter) << std::endl;
			optimizer.optimize(density, IsSmall);
			std::cout << 2 << std::endl;
			monte_carlo_selection(params, predict_distribution, IsSmall, density);
			std::cout << 3 << std::endl;
			if (iTick % OutputFreq == 0)
			{
				// output time
				std::clog << iTick * dt << ' ' << print_time << std::endl;
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
				average << '\n';
				x_tot /= ppl_tot;
				// output hyperparameters
				for (int iElement = 0; iElement < NumElement; iElement++)
				{
					for (double d : optimizer.get_hyperparameter(iElement))
					{
						hyperparam << ' ' << d;
					}
					hyperparam << '\n';
				}
				hyperparam << '\n';
				// output point
				point << print_point(density).format(MatrixFormatter) << '\n'
					  << '\n';
				// output phase
				for (int iElement = 0; iElement < NumElement; iElement++)
				{
					phase << optimizer.print_element(PhaseGrids, iElement).format(VectorFormatter) << '\n';
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
