/// @file main.cpp
/// @brief Test of GPR to rebuild the phase space distribution
///
/// This file is the combination of all functions (including the main function)
/// to rebuild the phase space distribution.

#include "gpr.h"
#include "io.h"

/// @brief The main driver
int main(const int argc, const char* argv[])
{
	if (argc < 2)
	{
		std::cerr << "Arguments not enough!\n";
		return 1;
	}
	std::clog << std::setprecision(std::numeric_limits<double>::digits10 + 1);
	std::clog.sync_with_stdio(true);
	// read input
	Eigen::VectorXd x = read_coord("x.txt");
	const int nx = x.size();
	Eigen::VectorXd p = read_coord("p.txt");
	const int np = p.size();
	const int t = std::stoi(argv[1]);

	std::ifstream phase(std::string(argv[1]) + ".txt");	// phase space distribution for input
	std::ofstream sim(std::string(argv[1]) + "sim.txt");		// simulated phase space distribution
	std::ofstream choose(std::string(argv[1]) + "choose.txt"); // chosen point for simulation
	std::ofstream log(std::string(argv[1]) + "log.txt");		// time, MSE and -log(Marginal likelihood)
	// then read the rho
	double tmp;
	SuperMatrix Density, PhaseFitting;
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		for (int jPES = 0; jPES < NumPES; jPES++)
		{
			Density[iPES][jPES].resize(nx, np);
			PhaseFitting[iPES][jPES].resize(nx, np);
			for (int iGrid = 0; iGrid < nx; iGrid++)
			{
				for (int jGrid = 0; jGrid < np; jGrid++)
				{
					phase >> Density[iPES][jPES](iGrid, jGrid);
				}
			}
		}
	}
	// the gradient and non-gradient algorithm to use
	const nlopt::algorithm NonGradient = nlopt::algorithm::LN_NELDERMEAD;
	const nlopt::algorithm Gradient = nlopt::algorithm::LD_TNEWTON_PRECOND_RESTART;
	// the kernel to use
	const KernelTypeList TypesOfKernels = { shogun::EKernelType::K_DIAG, shogun::EKernelType::K_GAUSSIANARD };
	// the bounds and initial values for hyperparameters, which is fixed
	const std::vector<ParameterVector>& Initials = set_initial_value(TypesOfKernels, Density, x, p);
	const ParameterVector& LowerBound = Initials[0];
	const ParameterVector& UpperBound = Initials[1];
	const ParameterVector& InitialHyperparameters = Initials[2];

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
			InitialHyperparameters);
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
	predict_phase(
			PhaseFitting,
			TrainingFeatures,
			TrainingLabels,
			TypesOfKernels,
			IsSmall,
			x,
			p,
			FinalHyperparameters);
	const QuantumMatrixDouble& MSE = mean_squared_error(Density, PhaseFitting);
	log << t;
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
			log << ' ' << MSE(iPES, jPES);
		}
	}
	sim << '\n';
	log << ' ' << FinalHyperparameters[FinalHyperparameters.size() - 1] << std::endl;

	print_point(choose, TrainingFeatures);
	choose << '\n';

	phase.close();
	sim.close();
	log.close();
	return 0;
}
