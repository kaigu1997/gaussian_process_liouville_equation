﻿/// @file main.cpp
/// @brief Test of GPR to rebuild the phase space distribution
///
/// This file is the combination of all functions (including the main function)
/// to rebuild the phase space distribution.

#include "gpr.h"
#include "io.h"

/// @brief The main driver
int main()
{
	std::clog << std::setprecision(std::numeric_limits<double>::digits10 + 1);
	std::clog.sync_with_stdio(true);

	// read input
	Eigen::VectorXd x = read_coord("x.txt");
	const int nx = x.size();
	Eigen::VectorXd p = read_coord("p.txt");
	const int np = p.size();
	Eigen::VectorXd t = read_coord("t.txt");
	const int nt = t.size();
	const double dx = (x[nx - 1] - x[0]) / nx, dp = (p[np - 1] - p[0]) / np;
	// set the io streams
	std::ifstream phase("phase.txt");	// phase space distribution for input
	std::ofstream sim("sim.txt");		// simulated phase space distribution
	std::ofstream choose("choose.txt"); // chosen point for simulation
	std::ofstream log("log.txt");		// time, MSE and -log(Marginal likelihood)

	// constant initialization
	// the gradient and non-gradient algorithm to use
	const nlopt::algorithm NonGradient = nlopt::algorithm::LN_NELDERMEAD;
	const nlopt::algorithm Gradient = nlopt::algorithm::LD_TNEWTON_PRECOND_RESTART;
	// the kernel to use
	const KernelTypeList TypesOfKernels = { shogun::EKernelType::K_DIAG, shogun::EKernelType::K_GAUSSIANARD };
	// initial phase space distribution, for some calculations: energy, initial hyperparameters and their bounds
	SuperMatrix Density, PhaseFitting;
	for (int i = 0; i < NumPES; i++)
	{
		for (int j = 0; j < NumPES; j++)
		{
			Density[i][j].resize(nx, np);
			PhaseFitting[i][j].resize(nx, np);
		}
	}
	read_density(phase, Density);
	// initial energy
	const double InitialEnergy = calculate_potential_energy_from_grid(Density, x, p).trace() + calculate_kinetic_energy_from_grid(Density, x, p).trace();
	// the bounds and initial values for hyperparameters, which is fixed
	const std::vector<ParameterVector>& Initials = set_initial_value(TypesOfKernels, Density, x, p);
	const ParameterVector& LowerBound = Initials[0];
	const ParameterVector& UpperBound = Initials[1];
	const ParameterVector& InitialHyperparameters = Initials[2];
	ParameterVector OldHyperparameters = InitialHyperparameters;

	for (int ti = 0; ti < nt; ti++)
	{
		std::clog << "T = " << t[ti] << std::endl;
		// read the PWTDM, rho
		// as phase is used for the initial value, skip its first reading
		if (ti != 0)
		{
			read_density(phase, Density);
		}

		// check if some elements are 0 everywhere
		const QuantumMatrixBool& IsSmall = is_very_small_everywhere(Density);
		// generate training set
		const FullTrainingSet& Training = generate_training_set(Density, IsSmall, x, p);
		const SuperMatrix& TrainingFeatures = Training.first;
		VectorMatrix TrainingLabels = Training.second;

		// optimize hyperparameters. Derivative-free algorithm first, then derivative method; subset first, then full set
		const ParameterVector& NonGradientOptimized
			= optimize(
				TrainingFeatures,
				TrainingLabels,
				TypesOfKernels,
				IsSmall,
				NonGradient,
				LowerBound,
				UpperBound,
				OldHyperparameters);
		const ParameterVector& FinalHyperparameters
			= optimize(
				TrainingFeatures,
				TrainingLabels,
				TypesOfKernels,
				IsSmall,
				Gradient,
				LowerBound,
				UpperBound,
				NonGradientOptimized);
		std::copy(FinalHyperparameters.cbegin(), FinalHyperparameters.cend() - 1, OldHyperparameters.begin());

		// fit on the grids, and calculate corresponding MSE, population and energy
		predict_phase(
			PhaseFitting,
			TrainingFeatures,
			TrainingLabels,
			TypesOfKernels,
			IsSmall,
			x,
			p,
			FinalHyperparameters);
		const QuantumMatrixDouble& MSEWithoutConstraints = mean_squared_error(Density, PhaseFitting);
		const QuantumVectorDouble& PopulationWithoutConstraintsFromGrid = calculate_population_from_grid(PhaseFitting, dx, dp);
		const QuantumVectorDouble& PopulationWithoutConstraintsFromParam = calculate_population_from_gpr(TrainingFeatures, TrainingLabels, TypesOfKernels, IsSmall, FinalHyperparameters);
		const QuantumVectorDouble& PotentialWithoutConstraintsFromGrid = calculate_potential_energy_from_grid(PhaseFitting, x, p);
		const QuantumVectorDouble& PotentialWithoutConstraintsFromParam = calculate_potential_energy_from_gpr(TrainingFeatures, TrainingLabels, TypesOfKernels, IsSmall, FinalHyperparameters);
		const QuantumVectorDouble& KineticsWithoutConstraintsFromGrid = calculate_kinetic_energy_from_grid(PhaseFitting, x, p);
		const QuantumVectorDouble& KineticsWithoutConstraintsFromParam = calculate_kinetic_energy_from_gpr(TrainingFeatures, TrainingLabels, TypesOfKernels, IsSmall, FinalHyperparameters);

		// make the system obey the constraints (modify on density and labels), and calculate corresponding MSE, population and energy
		obey_conservation(
			PhaseFitting,
			TrainingFeatures,
			TrainingLabels,
			TypesOfKernels,
			IsSmall,
			InitialEnergy,
			x,
			p,
			FinalHyperparameters);
		const QuantumMatrixDouble& MSE = mean_squared_error(Density, PhaseFitting);
		const QuantumVectorDouble& PopulationFromGrid = calculate_population_from_grid(PhaseFitting, dx, dp);
		const QuantumVectorDouble& PopulationFromParam = calculate_population_from_gpr(TrainingFeatures, TrainingLabels, TypesOfKernels, IsSmall, FinalHyperparameters);
		const QuantumVectorDouble& PotentialFromGrid = calculate_potential_energy_from_grid(PhaseFitting, x, p);
		const QuantumVectorDouble& PotentialFromParam = calculate_potential_energy_from_gpr(TrainingFeatures, TrainingLabels, TypesOfKernels, IsSmall, FinalHyperparameters);
		const QuantumVectorDouble& KineticsFromGrid = calculate_kinetic_energy_from_grid(PhaseFitting, x, p);
		const QuantumVectorDouble& KineticsFromParam = calculate_kinetic_energy_from_gpr(TrainingFeatures, TrainingLabels, TypesOfKernels, IsSmall, FinalHyperparameters);

		// output: selected points, phase space distribution, MSE, hyperparameters, -log(likelihood), ...
		print_point(choose, TrainingFeatures);
		log << t[ti];
		// output -ln(marginal likelihood)
		const int NoParam = FinalHyperparameters.size() - 1;
		log << ' ' << FinalHyperparameters[NoParam];
		// output hyperparameters
		for (int i = 0; i < NoParam; i++)
		{
			log << ' ' << FinalHyperparameters[i];
		}
		// output MSEs with fitted phase space distribution
		for (int iPES = 0; iPES < NumPES; iPES++)
		{
			for (int jPES = 0; jPES < NumPES; jPES++)
			{
				for (int iGrid = 0; iGrid < nx; iGrid++)
				{
					for (int jGrid = 0; jGrid < np; jGrid++)
					{
						sim << ' ' << PhaseFitting[iPES][jPES](iGrid, jGrid);
					}
				}
				sim << '\n';
				log << ' ' << MSEWithoutConstraints(iPES, jPES) << ' ' << MSE(iPES, jPES);
			}
		}
		sim << '\n';
		// output population and energy
		const QuantumVectorDouble& ExactPopulation = calculate_population_from_grid(Density, dx, dp);
		const QuantumVectorDouble& ExactPotential = calculate_potential_energy_from_grid(Density, x, p);
		const QuantumVectorDouble& ExactKinetics = calculate_kinetic_energy_from_grid(Density, x, p);
		for (int i = 0; i < NumPES; i++)
		{
			// population->potential->kinetics
			log << ' ' << ExactPopulation[i]
				<< ' ' << PopulationWithoutConstraintsFromGrid[i] << ' ' << PopulationWithoutConstraintsFromParam[i]
				<< ' ' << PopulationFromGrid[i] << ' ' << PopulationFromParam[i]
				<< ' ' << ExactPotential[i]
				<< ' ' << PotentialWithoutConstraintsFromGrid[i] << ' ' << PotentialWithoutConstraintsFromParam[i]
				<< ' ' << PotentialFromGrid[i] << ' ' << PotentialFromParam[i]
				<< ' ' << ExactKinetics[i]
				<< ' ' << KineticsWithoutConstraintsFromGrid[i] << ' ' << KineticsWithoutConstraintsFromParam[i]
				<< ' ' << KineticsFromGrid[i] << ' ' << KineticsFromParam[i];
		}
		log << std::endl;
	}
	phase.close();
	sim.close();
	log.close();
	return 0;
}
