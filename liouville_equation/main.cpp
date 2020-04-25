/// @file main.cpp
/// @brief The main driver
/// 
/// The purpose of this program is to give
/// an exact solution of quantum mechanic problem
/// using Mixed Quantum-Classical Liouville Equation
/// (MQCLE) by Discrete Variable Representation (DVR).
/// It requires C++17 or newer C++ standards when compiling
/// and needs connection to Intel(R) Math Kernel Library (MKL).
/// Error code criteria: 1XX for matrix, 
/// 2XX for pes, 3XX for general, and 4XX for main.

#include <chrono>
#include <ctime>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mkl.h>
#include <numeric>
#include <tuple>
#include "general.h"
#include "pes.h"
#include "matrix.h"
using namespace std;


/// The main driver
int main(void)
{
    // initialize: read input and calculate cooresponding constants
    // including the number of grids, their coordinates, etc
    cout.sync_with_stdio(false);
    clog.sync_with_stdio(false);
    cerr.sync_with_stdio(false);
    // in: the input file
    ifstream in("input");
    in.sync_with_stdio(false);
    // read mass: the mass of the bath
    const double mass = read_double(in);
    // read initial wavepacket info
    // the center/width of the wavepacket
    // calculate SigmaX by SigmaP using minimum uncertainty rule
    const double x0 = read_double(in);
    const double p0 = read_double(in);
    const double SigmaP = read_double(in);
    const double SigmaX = hbar / 2.0 / SigmaP;
    // 99.7% initial momentum in this region: p0+-3SigmaP
    // calculate the region of momentum by p0 and SigmaP
    const double p0min = p0 - 3.0 * SigmaP;
    const double p0max = p0 + 3.0 * SigmaP;
    clog << "The particle weighes " << mass << " a.u.,\n"
        << "starting from " << x0 << " with initial momentum " << p0 << ".\n"
        << "Initial width of x and p are " << SigmaX << " and " << SigmaP << ", respectively." << endl;
    
    // read interaction region
    const double xmin = read_double(in);
    const double xmax = read_double(in);
    const double TotalPositionLength = xmax - xmin;
    // read grid spacing, should be "~ 4 to 5 grids per de Broglie wavelength"
    // and then do the cut off, e.g. 0.2493 -> 0.2, 1.5364 -> 1
    // and the number of grids are thus determined
    const double dx = cutoff(min(read_double(in), PlanckH / p0max / 2.0));
    //const double dx = read_double(in);
    // NGrids: number of grids in [xmin, xmax], also in [pmin, pmax]
    const int NGrids = static_cast<int>(TotalPositionLength / dx) + 1;
    // momentum region is determined by fourier transformation:
    // p in p0+pi*hbar/dx/2*[-1,1), dp=pi*hbar/(xmax-xmin)
    const double pmin = p0 - pi * hbar / dx / 2.0;
    const double pmax = p0 + pi * hbar / dx / 2.0;
    const double TotalMomentumLength = pmax - pmin;
    const double dp = TotalMomentumLength / static_cast<double>(NGrids - 1);
    // NoPSGrids: Number of Phase Space Grids
    const int NoPSGrids = NGrids * NGrids;
    // dim: total number of elements (dimension) in L/rho
    const int dim = NoPSGrids * NoMatrixElement;

    // Position/Momentum contains each grid coordinate, one in a line
    ofstream Position("x.txt"), Momentum("p.txt");
    Position.sync_with_stdio(false);
    Momentum.sync_with_stdio(false);
    // the coordinates of the grids, i.e. value of xi/pj
    double* GridPosition = new double[NGrids];
    double* GridMomentum = new double[NGrids];
    // calculate the grid coordinates, and print them
    for (int i = 0; i < NGrids; i++)
    {
        GridPosition[i] = (xmin * (NGrids - 1 - i) + xmax * i) / (NGrids - 1);
        Position << GridPosition[i] << '\n';
        GridMomentum[i] = (pmin * (NGrids - 1 - i) + pmax * i) / (NGrids - 1);
        Momentum << GridMomentum[i] << '\n';
    }
    clog << "dx = " << dx << ", dp = " << dp << ", and there are overall " << NGrids << " grids\n"
        << "in [" << xmin << ", " << xmax <<"] for x and [" << pmin << ", " << pmax << "] for p.\n";
    Position.close();
    Momentum.close();

    // save the V, F and D matrices
    const RealMatrix* const* const Potential = calculate_potential_on_grids(NGrids, GridPosition);
    const RealMatrix* const* const Force = calculate_force_on_grids(NGrids, GridPosition);
    const RealMatrix* const* const Coupling = calculate_coupling_on_grids(NGrids, GridPosition);

    // total time is based on the length/speed*coe
    // being a rough approx, corrected during running
    const double TotalTime = TotalPositionLength / (p0 / mass) * 2.0;
    // read evolving time and output time, in unit of a.u.
    const double OutputTime = read_double(in);
    // criteria of dt: dt*dE<=hbar/2, dE=sigmap*sqrt(p^2-sigmap^2/16)/m=sigmap*p/m
    const double dt = cutoff(min(read_double(in), hbar / 500.0 / (SigmaP * p0 / mass)));
    // finish reading
    in.close();
    // calculate corresponding dt of the above (how many dt they have)
    const int TotalStep = static_cast<int>(TotalTime / dt);
    const int OutputStep = static_cast<int>(OutputTime / dt);
    clog << "dt = " << dt << ", and there are overall " << TotalStep << " time steps." << endl;

    // Steps contains when is each step, also one in a line
    ofstream Steps("t.txt");
    Steps.sync_with_stdio(false);
    // Output gives the partial wigner-transformed density matrix
    // In Phase space, each line is the PS-distribution at a moment:
    // rho[0][0](x0,p0,t0), rho[0][0](x0,p1,t0), ... rho[0][0](x1,p0,t0), ...
    // (continue) rho[0][1](x0,p0,t0), ... rho[1][0](x0,p0,t0), ... rho[n][n](xN,pN,t0)
    // (new line) rho[0][0](x0,p0,t1), ...
    ofstream Output("phase.txt");
    Output.sync_with_stdio(false);
    // log contains average <E>, <x>, and <p> with corresponding time
    ofstream Log("averages.txt");
    Log.sync_with_stdio(false);

    // population on each PES, and the population on each PES at last output moment
    double Population[NumPES] = {1.0};
    // memory allocation: density matrix
    ComplexMatrixMatrix rho(NGrids, NumPES);
    // construct the initial adiabatic PWTDM: gaussian on the ground state PES
    // rho[0][0](x,p,0)=exp(-(x-x0)^2/2sigma_x-(p-p0)^2/2sigma_p)/(pi*hbar)
    // initially in the adiabatic basis
    density_matrix_initialization
    (
        NGrids,
        GridPosition,
        GridMomentum,
        dx,
        dp,
        x0,
        p0,
        SigmaX,
        SigmaP,
        rho
    );
    // the evolving representation
    const Representation EvolveBasis = Diabatic;
    // after initialization, calculate the averages and populations, and output
    // using structured binding in C++17
    Steps << 0 << endl;
    Output << rho << endl;
    auto [LastE, LastX, LastP] = calculate_average
    (
        rho,
        NGrids,
        Potential,
        GridPosition,
        GridMomentum,
        mass,
        dx,
        dp,
        Adiabatic
    );
    Log << 0 << ' ' << LastE << ' ' << LastX << ' ' << LastP;
    calculate_population
    (
        NGrids,
        dx,
        dp,
        rho,
        Population
    );
    for (int i = 0; i < NumPES; i++)
    {
        Log << ' ' << Population[i];
    }
    Log << endl;
    // transform to diabatic basis for evolution
    basis_transform[Adiabatic][EvolveBasis](rho, NGrids, GridPosition);
    clog << "Finish diagonalization and memory allocation.\n" << show_time << endl;

    // evolve: Trotter expansion
    // rho(t+dt)=exp(-iLQdt/2)exp(-iLRdt/2)exp(-iLPdt)exp(-iLRdt/2)exp(-iLQdt/2)rho(t)
    // -iLQrho=-i/hbar[H-ihbarP/M*D,rho], -iLRrho=-P/M*drho/dR, -iLPrho=-(F*drho/dP+drho/dP*F)/2
    // derivatives are calculated by infinite order finite difference
    for (int iStep = 1; iStep <= TotalStep; iStep++)
    {
        // evolve
        // 1. Quantum Liouville, -iLQ*rho=-i/hbar[V-i*hbar*P/m*D, rho]
        // for diabatic basis, D=0, so simply trans to adia basis
        // exp(-iLQt)rho_dia=exp(-iVd t/hbar)*rho_adia*exp(iVd t/hbar), t=dt/2
        quantum_liouville_propagation
        (
            rho,
            NGrids,
            Potential,
            Coupling,
            GridPosition,
            GridMomentum,
            mass,
            dt / 2.0,
            EvolveBasis
        );
        // 2. Classical position Liouville, -iLRrho=-P/M*drho/dR
        // so exp(-iLRt)=exp(-P/M*d/dR*t), t=dt/2
        classical_position_liouville_propagator
        (
            rho,
            NGrids,
            GridMomentum,
            mass,
            TotalPositionLength,
            dx,
            dt / 2.0
        );
        // 3. Classical Momentum Liouville, under force basis,
        // -iLQrho=-(Fd*drho/dP+drho/dP*Fd)/2
        // so exp(-iLQt)=exp(-(Fdaa+Fdbb)/2*d/dP*t)
        // transform the density matrix to force basis
        classical_momentum_liouville_propagator
        (
            rho,
            NGrids,
            Force,
            GridPosition,
            TotalMomentumLength,
            dp,
            dt,
            EvolveBasis
        );
        // 4. Classical position Liouville again
        classical_position_liouville_propagator
        (
            rho,
            NGrids,
            GridMomentum,
            mass,
            TotalPositionLength,
            dx,
            dt / 2.0
        );
        // 5. Quantum Liouville again
        quantum_liouville_propagation
        (
            rho,
            NGrids,
            Potential,
            Coupling,
            GridPosition,
            GridMomentum,
            mass,
            dt / 2.0,
            EvolveBasis
        );

        // for the output case
        if (iStep % OutputStep == 0)
        {
            const double Time = iStep * dt;
            basis_transform[EvolveBasis][Adiabatic](rho, NGrids, GridPosition);
            Steps << Time << endl;
            // output the whole density matrix
            Output << rho << endl;

            // calculate <E>, <x>, <p> ...
            // using structured binding in C++17
            const auto& [E_bar, x_bar, p_bar] = calculate_average
            (
                rho,
                NGrids,
                Potential,
                GridPosition,
                GridMomentum,
                mass,
                dx,
                dp,
                Adiabatic
            );
            // ... then output
            Log << Time << ' ' << E_bar << ' ' << x_bar << ' ' << p_bar;
            calculate_population
            (
                NGrids,
                dx,
                dp,
                rho,
                Population
            );
            for (int i = 0; i < NumPES; i++)
            {
                Log << ' ' << Population[i];
            }
            Log << endl;
            // compare with last time and see the difference
            // once it passes the boundary / going across the symmetric interaction region, think the evolution is over
            if (x_bar > 0 && ((x_bar - LastX) * p0 < 0 || x_bar > -x0))
            {
                // the wavepacket changes its direction, stop evolving
                break;
            }
            // set the last values
            LastE = E_bar;
            LastX = x_bar;
            LastP = p_bar;
            // transform to diabatic basis again for next revolution
            basis_transform[Adiabatic][EvolveBasis](rho, NGrids, GridPosition);
        }
    }
    // after evolution, print time and frees the resources
    clog << "Finish evolution.\n" << show_time << endl;
    Steps.close();
    Output.close();
    Log.close();

    // print the final info
    if (TestModel == DAC)
    {
        cout << log(p0 * p0 / 2.0 / mass);
    }
    else
    {
        cout << p0;
    }
    calculate_population(NGrids, dx, dp, rho, Population);
    for (int i = 0; i < NumPES; i++)
    {
        cout << ' ' << Population[i];
    }
    cout << '\n';

    // end. free the memory.    
    delete[] GridPosition;
    delete[] GridMomentum;
	return 0;
}
