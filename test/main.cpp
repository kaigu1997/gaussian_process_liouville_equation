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
	std::cout << std::setprecision(std::numeric_limits<double>::digits10 + 1);
	std::cout.sync_with_stdio(true);
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
		// then read the rho00 and rho11
		double tmp;
		Eigen::MatrixXd rho0(nx, np), rho1(nx, np), rho_re(nx, np), rho_im(nx, np);
		// rho00
		for (int i = 0; i < nx; i++)
		{
			for (int j = 0; j < np; j++)
			{
				phase >> rho0(i, j) >> tmp;
			}
		}
		// rho01 and rho10
		for (int i = 0; i < nx; i++)
		{
			for (int j = 0; j < np; j++)
			{
				phase >> rho_re(i, j) >> rho_im(i, j);
			}
		}
		for (int i = 0; i < nx; i++)
		{
			for (int j = 0; j < np; j++)
			{
				phase >> tmp;
				rho_re(i, j) = (rho_re(i, j) + tmp) / 2.0;
				phase >> tmp;
				rho_im(i, j) = (rho_im(i, j) - tmp) / 2.0;
			}
		}
		// rho11
		for (int i = 0; i < nx; i++)
		{
			for (int j = 0; j < np; j++)
			{
				phase >> rho1(i, j) >> tmp;
			}
		}
		log << t(ti);

		std::clog << "\nT = " << t(ti) << "\nrho[0][0]:\n";
		const FittingResult& rho0_result = fit(rho0, x, p);
		const Eigen::MatrixXd& rho0_fit = std::get<0>(rho0_result);
		for (int i = 0; i < nx; i++)
		{
			sim << ' ' << rho0_fit.row(i);
		}
		sim << '\n';
		print_point(choose, std::get<1>(rho0_result), x, p);
		log << ' ' << (rho0 - rho0_fit).array().square().sum() << ' ' << std::get<2>(rho0_result);

		std::clog << "\nRe(rho[0][1]):\n";
		const FittingResult& rho_re_result = fit(rho_re, x, p);
		const Eigen::MatrixXd& rho_re_fit = std::get<0>(rho_re_result);
		for (int i = 0; i < nx; i++)
		{
			sim << ' ' << rho_re_fit.row(i);
		}
		sim << '\n';
		print_point(choose, std::get<1>(rho_re_result), x, p);
		log << ' ' << (rho_re - rho_re_fit).array().square().sum() << ' ' << std::get<2>(rho_re_result);

		std::clog << "\nIm(rho[0][1]):\n";
		const FittingResult& rho_im_result = fit(rho_im, x, p);
		const Eigen::MatrixXd& rho_im_fit = std::get<0>(rho_im_result);
		for (int i = 0; i < nx; i++)
		{
			sim << ' ' << rho_im_fit.row(i);
		}
		sim << '\n';
		print_point(choose, std::get<1>(rho_im_result), x, p);
		log << ' ' << (rho_im - rho_im_fit).array().square().sum() << ' ' << std::get<2>(rho_im_result);

		std::clog << "\nrho[1][1]:\n";
		const FittingResult& rho1_result = fit(rho1, x, p);
		const Eigen::MatrixXd& rho1_fit = std::get<0>(rho1_result);
		for (int i = 0; i < nx; i++)
		{
			sim << ' ' << rho1_fit.row(i);
		}
		sim << '\n';
		print_point(choose, std::get<1>(rho1_result), x, p);
		log << ' ' << (rho1 - rho1_fit).array().square().sum() << ' ' << std::get<2>(rho1_result);

		log << std::endl;
		sim << '\n';
		choose << '\n';
	}
	phase.close();
	sim.close();
	log.close();
	shogun::exit_shogun();
	return 0;
}
