/// @file main.cpp
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
	shogun::init_shogun_with_defaults();
	// read input
	Eigen::VectorXd x = read_coord("x.txt");
	const int nx = x.size();
	Eigen::VectorXd p = read_coord("p.txt");
	const int np = p.size();
	Eigen::VectorXd t = read_coord("t.txt");
	const int nt = t.size();

	std::ifstream phase("phase.txt");	// phase space distribution for input
	std::ofstream sim("sim.txt");		// simulated phase space distribution
	std::ofstream choose("choose.txt"); // chosen point for simulation
	std::ofstream log("log.txt");		// time, MSE and -log(Marginal likelihood)
	for (int ti = 0; ti < nt; ti++)
	{
		std::clog << t[ti] << std::endl;
		// read the PWTDM, rho
		const SuperMatrix& Density = read_density(phase, nx, np);
		// fit
		const FittingResult& result = fit(Density, x, p);
		// output the fitting
		const SuperMatrix& PhaseFitting = std::get<0>(result);
		const QuantumMatrixD& MSE = mean_squared_error(Density, PhaseFitting);
		log << t[ti];
		for (int iPES = 0; iPES < NumPES; iPES++)
		{
			for (int jPES = 0; jPES < NumPES; jPES++)
			{
				for (int iGrid = 0; iGrid < nx; iGrid++)
				{
					for (int jGrid = 0; jGrid < np; jGrid++)
					{
						sim << ' ' << PhaseFitting(iGrid, jGrid)(iPES, jPES);
					}
				}
				sim << '\n';
				log << ' ' << MSE(iPES, jPES);
			}
		}
		sim << '\n';
		log << ' ' << std::get<2>(result) << std::endl;

		print_point(choose, std::get<1>(result), x, p);
		choose << '\n';
	}
	phase.close();
	sim.close();
	log.close();
	shogun::exit_shogun();
	return 0;
}
