/// @file main.cpp
/// @brief test of GPR to rebuild the phase space distribution
///
/// This file is the combination of all functions (including the main function)
/// to rebuild the phase space distribution.

#include "gpr.h"
#include "io.h"

int main()
{
	cout << setprecision(numeric_limits<double>::digits10 + 1);
	cout.sync_with_stdio(true);
	// read input
	VectorXd x = read_coord("x.txt");
	const int nx = x.size();
	VectorXd p = read_coord("p.txt");
	const int np = p.size();
	VectorXd t = read_coord("t.txt");
	const int nt = t.size();

	ifstream phase("phase.txt");   // phase space distribution for input
	ofstream sim("sim.txt");	   // simulated phase space distribution
	ofstream choose("choose.txt"); // chosen point for simulation
	ofstream log("log.txt");	   // time, MSE and -log(Marginal likelihood)
	for (int ti = 0; ti < nt; ti++)
	{
		// then read the rho00 and rho11
		double tmp;
		MatrixXd rho0(nx, np), rho1(nx, np), rho_re(nx, np), rho_im(nx, np);
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
		cout << "\nT = " << t(ti) << "\nrho[0][0]:\n";
		fit(rho0, x, p, sim, choose, log);
		cout << "\nRe(rho[0][1]):\n";
		fit(rho_re, x, p, sim, choose, log);
		cout << "\nIm(rho[0][1]):\n";
		fit(rho_im, x, p, sim, choose, log);
		cout << "\nrho[1][1]:\n";
		fit(rho1, x, p, sim, choose, log);
		log << endl;
		sim << '\n';
		choose << '\n';
	}
	phase.close();
	sim.close();
	log.close();
	return 0;
}
