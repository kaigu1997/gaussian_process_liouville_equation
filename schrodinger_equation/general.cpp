/// @file general.cpp
/// @brief Implementation of general.h

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <ctime>
#include <iostream>
#include <memory>
#include <mkl.h>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include "general.h"
#include "pes.h"

#include <mkl.h>
#ifndef EIGEN_USE_MKL_ALL
#define EIGEN_USE_MKL_ALL
#endif // ! EIGEN_USE_MKL_ALL
#include <Eigen/Eigen>

using namespace std;
using namespace Eigen;

// utility functions

/// Do the cutoff, e.g. 0.2493 -> 0.125, 1.5364 -> 1
/// 
/// Transform to the nearest 2 power, i.e. 2^(-2), 2^0, etc
double cutoff(const double val)
{
	return exp2(static_cast<int>(floor(log2(val))));
}

int pow_minus_one(const int n)
{
	return n % 2 == 0 ? 1 : -1;
}



// I/O functions

/// The file is one line declarator and one line the value,
/// so using a buffer string to read the declarator
/// and the rest of the second line.
double read_double(istream& is)
{
	static string buffer;
	static double temp;
	getline(is, buffer);
	is >> temp;
	getline(is, buffer);
	return temp;
}

ostream& show_time(ostream& os)
{
	auto time = chrono::system_clock::to_time_t(chrono::system_clock::now());
	os << ctime(&time);
	return os;
}


// evolution related function

/// psi(x)=exp(-(x-x0)^2/4/sigma_x^2+i*p0*x/hbar)/[(2*pi)^(1/4)*sqrt(sigma_x)]
///
/// as the wavefunction is on grid, the normalization factor could be different
///
/// this function is also only called once, for initialization of the adiabatic wavefunction
VectorXcd wavefunction_initialization
(
	const VectorXd& XCoordinate,
	const double x0,
	const double p0,
	const double SigmaX
)
{
	const int NGrids = XCoordinate.size();
	const double dx = (XCoordinate[NGrids - 1] - XCoordinate[0]) / (NGrids - 1);
	// the wavefunction
	VectorXcd Psi(NGrids * NumPES);
	// for higher PES, the initial wavefunction is zero
	memset(Psi.data() + NGrids, 0, NGrids * (NumPES - 1) * sizeof(Complex));
	// for ground state, it is a gaussian. psi(x)=A*exp(-(x-x0)^2/4sigmax^2+ip0x/hbar)
#pragma omp parallel for default(none) shared(Psi, XCoordinate) schedule(static, 1)
	for (int i = 0; i < NGrids; i++)
	{
		const double& x = XCoordinate[i];
		Psi[i] = exp(-pow((x - x0) / 2 / SigmaX, 2) + p0 * x / hbar * 1.0i) / sqrt(sqrt(2.0 * pi) * SigmaX);
	}
	// normalization
	const double NormFactor = sqrt(Psi.squaredNorm() * dx);
#pragma omp parallel for default(none) shared(Psi, XCoordinate) schedule(static, 1)
	for (int i = 0; i < NGrids; i++)
	{
		Psi[i] /= NormFactor;
	}
	return Psi;
}

/// Construct the diabatic Hamiltonian consisting of 3 parts:
///
/// 1. diabatic potential (the subsystem/electron Hamiltonian), which is real-symmetric
///
/// 2. bath/nucleus kinetic energy by infinite order finite difference, which is also real-symmetric
///
/// 3. absorbed boundary condition when it is used, which is pure complex on diagonal elements
///
/// This function is called only once for the initialization of Hamiltonian
MatrixXcd Hamiltonian_construction
(
	const VectorXd& XCoordinate,
	const double dx,
	const double mass,
	const double xmin,
	const double xmax,
	const double AbsorbingRegionLength
)
{
	const int NGrids = XCoordinate.size();
	MatrixXcd Hamiltonian(NGrids * NumPES, NGrids * NumPES);
	memset(Hamiltonian.data(), 0, Hamiltonian.size() * sizeof(Complex));
	// 1. V_{mm'}(R_n), n=n'
#pragma omp parallel for default(none) shared(Hamiltonian, XCoordinate) schedule(static, 1)
	for (int n = 0; n < NGrids; n++)
	{
		const PESMatrix& Vn = diabatic_potential(XCoordinate[n]);
		for (int m = 0; m < NumPES; m++)
		{
			for (int mm = 0; mm < NumPES; mm++)
			{
				Hamiltonian(m * NGrids + n, mm * NGrids + n) += Vn(m, mm);
			}
		}
	}
	// 2. d2/dx2 (over all pes) and absorbing (if have)
	switch (TestBoundaryCondition)
	{
	case Absorbing:
#pragma omp parallel for default(none) shared(Hamiltonian, XCoordinate) schedule(static, 1)
		for (int n = 0; n < NGrids; n++)
		{
			const double&& An = absorbing_potential(mass, xmin, xmax, AbsorbingRegionLength, XCoordinate[n]);
			for (int m = 0; m < NumPES; m++)
			{
				Hamiltonian(m * NGrids + n, m * NGrids + n) -= 1.0i * An;
			}
		}
	case Reflective:
#pragma omp parallel for default(none) shared(Hamiltonian) schedule(static, 1)
		for (int m = 0; m < NumPES; m++)
		{
			for (int n = 0; n < NGrids; n++)
			{
				for (int nn = 0; nn < NGrids; nn++)
				{
					if (nn == n)
					{
						// T_{ii}=\frac{\pi^2\hbar^2}{6m\Delta x^2}
						Hamiltonian(m * NGrids + n, m * NGrids + nn) += pow(pi * hbar / dx, 2) / 6.0 / mass;
					}
					else
					{
						// T_{ij}=\frac{(-1)^{i-j}\hbar^2}{(i-j)^2m\Delta x^2}
						Hamiltonian(m * NGrids + n, m * NGrids + nn) += pow_minus_one(nn - n) * pow(hbar / dx / (nn - n), 2) / mass;
					}
				}
			}
		}
		break;
	case Periodic:
		const double L = XCoordinate[NGrids - 1] - XCoordinate[0];
#pragma omp parallel for default(none) shared(Hamiltonian) schedule(static, 1)
		for (int m = 0; m < NumPES; m++)
		{
			for (int n = 0; n < NGrids; n++)
			{
				for (int nn = 0; nn < NGrids; nn++)
				{
					if (nn == n)
					{
						// T_{ii}=\frac{\pi^2\hbar^2((2N+1)^2-1)}{6mL^2}
						Hamiltonian(m * NGrids + n, m * NGrids + nn) += pow(pi * hbar / L, 2) / 6.0 / mass * (NGrids * NGrids - 1);
					}
					else
					{
						// T_{ij}=\frac{(-1)^{i-j}\pi^2\hbar^2\cos(\frac{(i-j)\pi}{2N+1})}{mL^2\sin[2](\frac{(i-j)\pi}{2N+1})}
						const double diff = (n - nn) * pi / NGrids;
						Hamiltonian(m * NGrids + n, m * NGrids + nn) += pow_minus_one(nn - n) * cos(diff) * pow(pi * hbar / L / sin(diff), 2) / mass;
					}
				}
			}
		}
		break;
	}
	return Hamiltonian;
}

/// Constructor. It allocates memory, and if there is no ABC, diagonalize Hamiltonian
Evolution::Evolution
(
	const MatrixXcd& DiaH,
	const VectorXcd& Psi0,
	const double TimeStep
):
	dim(DiaH.rows()),
	Hamiltonian(DiaH),
	Intermediate1(dim),
	Intermediate2(dim),
	PsiAtTimeT(dim),
	solver(DiaH),
	EigVec(TestBoundaryCondition != Absorbing? solver.eigenvectors() : MatrixXcd(0,0)),
	EigVal(TestBoundaryCondition != Absorbing ? solver.eigenvalues() : VectorXd(0)),
	dt(TimeStep),
	RK4kBeta(Complex(dt) / RK4Parameter),
	RK4PsiAlpha(Complex(dt / 6.0) * RK4Parameter)
{
	if (TestBoundaryCondition != Absorbing)
	{
		Intermediate1 = EigVec.adjoint() * Psi0;
	}
}

Evolution::~Evolution(void)
{
}

/// If w/ ABC, using RK4 to evolve.
/// k1=f(y,t), k2=f(y+dt/2*k1, t+dt/2), k3=f(y+dt/2*k2,t+dt/2),
/// k4=f(y+dt*k3, t+dt), y(t+dt)=y(t)+dt/6*(k1+2*k2+2*k3+k4).
/// Here f(y,t)=hat(H)/i/hbar*psi
///
/// If w/o ABC, the initial diagonalized wavefunction have been
/// calculated. psi(t)_dia=exp(-iHt/hbar)*psi(0)_dia
/// =C*exp(-iHd*t/hbar)*C^T*psi(0)_dia=C*exp(-iHd*t/hbar)*psi(0)_diag
void Evolution::evolve(VectorXcd& Psi, const double Time)
{
	if (TestBoundaryCondition != Absorbing)
	{
		// no ABC, using the diagonalized wavefunction

		// calculate psi(t)_diag = exp(-iH(diag)*t/hbar) * psi(0)_diag, each diagonal element be the eigenvalue
		// here, psi(0)_diag is Intermediate1, initialized in constructor; psi(t)_diag is Intermediate2 
		Intermediate2 = ((-EigVal * Time / hbar * 1.0i).array().exp() * Intermediate1.array()).matrix();
		// calculate DiabaticPsi=C1*psi_t_diag
		PsiAtTimeT = EigVec * Intermediate2;
	}
	else
	{
		// with ABC, using RK4

		// copy the wavefunction at time t-dt, leave the input as constant
		PsiAtTimeT = Psi;
		// k1=f(y,t), k2=f(y+dt/2*k1, t+dt/2), k3=f(y+dt/2*k2,t+dt/2)
		// k4=f(y+dt*k3, t+dt), y(t+dt)=y(t)+dt/6*(k1+2*k2+2*k3+k4)
		// here f(y,t)=hat(H)/i/hbar*psi
		// Intermediate1 is y+dt/n*ki, Intermediate2 is ki=f(Intermediate1)=H*I1/ihbar
		// then I2 is swapped with I1, so I1=psi/ihbar+dt/ni/ihbar*I1 by calling clabs_zaxpby
		memset(Intermediate1.data(), 0, dim * sizeof(Complex));
		for (int i = 0; i < 4; i++)
		{
			// k(i)=(psi(t)+dt/ni*k(i-1))/i/hbar
			Intermediate2 = Hamiltonian * (Psi + dt / RK4Parameter[i] * Intermediate1) / 1.0i / hbar;
			// change the k' and k
			swap(Intermediate1, Intermediate2);
			// add to the wavefunction: y(t+dt)=y(t)+dt/6*ni*ki
			Psi += dt / 6.0 * RK4Parameter[i] * Intermediate1;
		}
	}
	// after evolution, copy into the Psi in parameter list
	Psi = PsiAtTimeT;
}


/// output each element's modular square in a line, separated with space, w/ \n @ eol
void output_grided_population(ostream& os, const VectorXcd Psi)
{
	for (int i = 0; i < Psi.size(); i++)
	{
		os << ' ' << (Psi[i] * conj(Psi[i])).real();
	}
	os << endl;
}

/// the eigenvalues of potential matrix
using PESVector = Matrix<double, NumPES, 1>;

/// @brief calculate the adiabatic energy on each position grid
/// @param XCoordinate the coordinate of each grid on x direction, i.e., x_i
/// @return a container, each element is a VectorNd, N is NumPES, the vector is the adiabatic potential
static vector<PESVector> adiabatic_energy(const VectorXd& XCoordinate)
{
	// the number of grids
	const int NGrids = XCoordinate.size();
	// the container
	vector<PESVector> result(NGrids);
	for (int i = 0; i < NGrids; i++)
	{
		// calculate the diagonalized eigen(adiabatic) potential energy
		SelfAdjointEigenSolver<PESMatrix> solver(diabatic_potential(XCoordinate[i]));
		result[i] = solver.eigenvalues();
	}
	return result;
}

/// Output one element of ComplexMatrix in a line, change column first, then row
///
/// That is to say,
///
/// (line 1) rho00(x0,p0) rho00(x0,p1) rho00(x0,p2) ... rho00(x0,pn) ... rho00(xn,pn)
///
/// (line 2) rho01(x0,p0) rho01(x0,p1) rho01(x0,p2) ... rho01(x0,pn) ... rho01(xn,pn)
///
/// and so on, and a blank line at the end.
///
/// The region of momentum is the Fourier transformation of position, and moved to centered at initial momentum.
///
/// In other words, p in p0+pi*hbar/dx*[-1,1], dp=2*pi*hbar/(xmax-xmin)
tuple<double, double, double> output_phase_space_distribution
(
	ostream& os,
	const VectorXd& XCoordinate,
	const VectorXd& PCoordinate,
	const double mass,
	const double p0,
	const VectorXcd& Psi
)
{
	// NGrids is the number of grids in wavefunction
	const int NGrids = XCoordinate.size();
	const double dx = (XCoordinate[NGrids - 1] - XCoordinate[0]) / (NGrids - 1);
	const double dp = (PCoordinate[NGrids - 1] - PCoordinate[0]) / (NGrids - 1);
	// solve the adiabatic energies on each grids
	static const auto AdiabaticPotential = adiabatic_energy(XCoordinate);
	// average energy, position, momentum
	double E = 0, x = 0, p = 0;
	// PhaseSpaceDistribution saves phase space distribution of one element of PWTDM
	static MatrixXcd PhaseSpaceDistribution(NGrids, NGrids);
	// Wigner Transformation: P(x,p)=int{dy*exp(2ipy/hbar)<x-y|rho|x+y>}/(pi*hbar)
	// the interval of p is p0+pi*hbar/dx*[-1,1), dp=2*pi*hbar/(xmax-xmin)
	// loop over pes first
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		for (int jPES = 0; jPES < NumPES; jPES++)
		{
#pragma omp parallel for default(none) shared(Psi, PCoordinate, iPES, jPES, PhaseSpaceDistribution) schedule(static, 1)
			// loop over x
			for (int xi = 0; xi < NGrids; xi++)
			{
				// loop over p
				for (int pj = 0; pj < NGrids; pj++)
				{
					const double p = PCoordinate[pj];
					PhaseSpaceDistribution(xi, pj) = 0;
					// do the numerical integral
					switch (TestBoundaryCondition)
					{
					case Absorbing:
					case Reflective:
						// 0 <= (x+y, x-y) < NGrids 
						for (int yk = max(-xi, xi + 1 - NGrids); yk <= min(xi, NGrids - 1 - xi); yk++)
						{
							const double y = yk * dx;
							PhaseSpaceDistribution(xi, pj) += exp(2.0 * p * y / hbar * 1.0i) * Psi[xi - yk + iPES * NGrids] * conj(Psi[xi + yk + jPES * NGrids]);
						}
						break;
					case Periodic:
						// when y=L/2, psi(x+y)=psi(x-y), gives 2 wavepackets in phase spaces, so do not integrate near that
						for (int yk = -NGrids / 3; yk <= NGrids / 3; yk++)
						{
							const double y = yk * dx;
							PhaseSpaceDistribution(xi, pj) += exp(2.0 * p * y / hbar * 1.0i) * Psi[(xi - yk + NGrids) % NGrids + iPES * NGrids] * conj(Psi[(xi + yk + NGrids) % NGrids + jPES * NGrids]);
						}
						break;
					}
				}
			}
			PhaseSpaceDistribution *= dx / (pi * hbar);
			// output
			for (int i = 0; i < NGrids; i++)
			{
				for (int j = 0; j < NGrids; j++)
				{
					os << ' ' << PhaseSpaceDistribution(i, j).real() << ' ' << PhaseSpaceDistribution(i, j).imag();
				}
			}
			os << '\n';
			// calculate average on diagonal elements of PWTDM
			if (iPES == jPES)
			{
#pragma omp parallel for default(none) shared(iPES, XCoordinate, PCoordinate, AdiabaticPotential, PhaseSpaceDistribution) reduction(+:E,x,p) schedule(static, 1)
				for (int i = 0; i < NGrids; i++)
				{
					// grid on the same row have same position, and potential energy
					x += PhaseSpaceDistribution.row(i).sum().real() * XCoordinate[i];
					E += PhaseSpaceDistribution.row(i).sum().real() * AdiabaticPotential[i][iPES];
					// grid on the same column have same momentum, and kinetic energy
					p += PhaseSpaceDistribution.col(i).sum().real() * PCoordinate[i];
					E += PhaseSpaceDistribution.col(i).sum().real() * pow(PCoordinate[i], 2) / 2.0 / mass;
				}
			}
		}
	}
	os << endl;
	return make_tuple(E * dx * dp, x * dx * dp, p * dx * dp);
}

/// @brief construct the first order derivative derived from finite difference
/// @param NGrids the number of grids
/// @param dx the grid spacing
/// @return the 1st order derivative matrix
static MatrixXd derivative(const int NGrids, const double dx)
{
	MatrixXd result(NumPES * NGrids, NumPES * NGrids);
	memset(result.data(), 0, result.size() * sizeof(double));
#pragma omp parallel for default(none) shared(result) schedule(static, 1)
	for (int i = 0; i < NumPES; i++)
	{
		for (int j = 0; j < NGrids; j++)
		{
			for (int k = 0; k < NGrids; k++)
			{
				if (j != k)
				{
					result(j + i * NGrids, k + i * NGrids) = pow_minus_one(j - k) / dx / (j - k);
				}
			}
		}
	}
	return result;
}

/// return <E>, <x>, then <p>
///
/// <x> = sum_i{x*|psi(x)|^2}
///
/// <E,p> = <psi|A|psi>
tuple<double, double, double> calculate_average
(
	const VectorXd& XCoordinate,
	const double dx,
	const double mass,
	const VectorXcd& Psi
)
{
	// NGrids is the number of grids in wavefunction
	const int NGrids = XCoordinate.size();
	// the number of elements in psi
	const int dim = Psi.size();
	// construct the p and H matrix
	static const MatrixXcd P = -1.0i * hbar * derivative(NGrids, dx);
	static const MatrixXcd H = Hamiltonian_construction
	(
		XCoordinate,
		dx,
		mass
	);
	// first, <E>
	const double E = Psi.dot(H * Psi).real() * dx;
	// next, <x>
	double x = 0.0;
	for (int i = 0; i < NumPES; i++)
	{
		for (int j = 0; j < NGrids; j++)
		{
			x += XCoordinate[j] * (Psi[i * NGrids + j] * conj(Psi[i * NGrids + j])).real();
		}
	}
	x *= dx;
	// finally, <p>, same as <E>
	const double p = Psi.dot(P * Psi).real() * dx;
	return make_tuple(E, x, p);
}

void calculate_population
(
	const int NGrids,
	const double dx,
	const VectorXcd& Psi,
	double* const Population
)
{
	Complex InnerProduct;
	// calculate the inner product of each PES
	// again, calling cblas_zdotc_sub rather than cblas_znrm2 due to higher accuracy
	for (int i = 0; i < NumPES; i++)
	{
		Population[i] = Psi.segment(i * NGrids, NGrids).squaredNorm() * dx;
	}
}
