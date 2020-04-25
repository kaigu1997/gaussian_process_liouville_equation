/// @file main.cpp
/// @brief the main driver
///
/// The purpose of this program is to give
/// an exact solution of quantum mechanic problem
/// using Discrete Variable Representation (DVR)
/// in [1]J. Chem. Phys., 1992, 96(3): 1982-1991,
/// with Absorbing Boundary Condition in
/// [2]J. Chem. Phys., 2002, 117(21): 9552-9559
/// and [3]J. Chem. Phys., 2004, 120(5): 2247-2254.
/// This program could be used to solve
/// exact solution under diabatic basis ONLY.
/// It requires C++17 or newer C++ standards when compiling
/// and needs connection to Intel(R) Math Kernel Library
/// (MKL) by whatever methods: icpc/msvc/gcc -I.
/// Error code criteria: 1XX for matrix, 
/// 2XX for general, 3XX for pes, and 4XX for main.

#include <chrono>
#include <ctime>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mkl.h>
#include <numeric>
#include <utility>
#include "general.h"
#include "pes.h"
#include "matrix.h"
using namespace std;

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
    clog << "The particle weighes " << mass << " a.u.,\n"
        << "starting from " << x0 << " with initial momentum " << p0 << ".\n"
        << "Initial width of x and p are " << SigmaX << " and " << SigmaP << ", respectively.\n";
    // 99.7% initial momentum in this region: p0+-3SigmaP
    // calculate the region of momentum by p0 and SigmaP
    const double p0min = p0 - 3.0 * SigmaP;
    const double p0max = p0 + 3.0 * SigmaP;

    // read interaction region
    const double xmin = read_double(in);
    const double xmax = read_double(in);
    // read grid spacing, should be "~ 4 to 5 grids per de Broglie wavelength"
    // and then do the cut off, e.g. 0.2493 -> 0.2, 1.5364 -> 1
    // and the number of grids are thus determined
    const double dx = cutoff(min(read_double(in), PlanckH / p0max / 5.0));
    // grids in [xmin, xmax]
    const int InteractingGrid = static_cast<int>((xmax - xmin) / dx) + 1;

    // read whether have absorb potential or not
    const bool Absorbed = read_absorb(in);
    // absorbing region length from [2] and [3], determined by p0
    const double AbsorbingRegionLength = [&]
    {
        if (Absorbed == true)
        {
            return PlanckH / p0min;
        }
        else
        {
            return 0.0;
        }
    }();
    // total length is the interaction region + absorbing region
    const double TotalLength = xmax - xmin + 2.0 * AbsorbingRegionLength;
    // grids in [xmin-arl, xmin) or (xmax, xmax+arl]. no absorb -> 0 grids
    const int AbsorbingGrid = static_cast<int>(AbsorbingRegionLength / dx);
    // NGrids: number of grids in [xmin-arl, xmax+arl]
    const int NGrids = InteractingGrid + 2 * AbsorbingGrid;
    // dim: total number of elements (dimension) in Psi/H
    const int dim = NGrids * NumPES;

    // the coordinates of the grids, i.e. value of xi
    double* GridCoordinate = new double[NGrids];
    // XGrids contains each grid coordinate, one in a line
    ofstream XGrids("x.txt");
    XGrids.sync_with_stdio(false);
    // PGrids contains the calculated p values in the phase space distribution
    ofstream PGrids("p.txt");
    PGrids.sync_with_stdio(false);
    // pmin and pmax are the minimum/maximum calculate p value in the phase space distribution
    // they are centered at p0, dp=pi*hbar/L, Lp=pi*hbar/dx
    // in main function, they are just to output to the file, and are not passed to any functions
    const double pmin = p0 - pi * hbar / dx / 2.0;
    const double pmax = p0 + pi * hbar / dx / 2.0;
    // calculate the grid coordinates, and print them
    for (int i = 0; i < NGrids; i++)
    {
        GridCoordinate[i] = xmin + dx * (i - AbsorbingGrid);
        XGrids << GridCoordinate[i] << '\n';
        PGrids << ((NGrids - 1 - i) * pmin + i * pmax) / (NGrids - 1) << '\n';
    }
    XGrids.close();
    PGrids.close();
    clog << "dx = " << dx << ", and there is overall " << NGrids << " grids from "
        << GridCoordinate[0] << " to " << GridCoordinate[NGrids - 1] << ".\n";

    // read evolving time and output time, in unit of a.u.
    // total time is estimated as a uniform linear motion
    const double TotalTime = (xmax - xmin) / (p0 / mass) * 2.0;
    const double OutputTime = read_double(in);
    // read dt. criteria is from J. Comput. Phys., 1983, 52(1): 35-53.
    const double dt = [&]
    {
        if (Absorbed == false)
        {
            return OutputTime;
        }
        else
        {
            return cutoff(min(read_double(in), hbar / 500.0 / (SigmaP * p0 / mass)));
        }
    }();
    // finish reading
    in.close();
    // calculate corresponding dt of the above (how many dt they have)
    const int TotalStep = static_cast<int>(TotalTime / dt);
    const int OutputStep = static_cast<int>(OutputTime / dt);
    clog << "dt = " << dt << ", and there is overall " << TotalStep << " time steps." << endl;

    // allocate for the adiabatic/diabatic representation wavefunction
    Complex* AdiabaticPsi = new Complex[dim];
    Complex* DiabaticPsi = new Complex[dim];
    // construct the initial adiabatic wavepacket: gaussian on the ground state PES
    // psi(x)=exp(-((x-x0)/2sigma_x)^2+i*p0*x/hbar)/sqrt(sqrt(2*pi)*sigma_x)
    wavefunction_initialization
    (
        NGrids,
        GridCoordinate,
        dx,
        x0,
        p0,
        SigmaX,
        AdiabaticPsi
    );
    // TransformationMatrix makes dia to adia
    const ComplexMatrix TransformationMatrix = diabatic_to_adiabatic(NGrids, GridCoordinate);
    // and transform to diabatic representation C^T*psi(dia)*C=psi(adia), so psi(dia)=C*psi(adia)
    cblas_zgemv
    (
        CblasRowMajor,
        CblasNoTrans,
        dim,
        dim,
        &Alpha,
        TransformationMatrix.data(),
        dim,
        AdiabaticPsi,
        1,
        &Beta,
        DiabaticPsi,
        1
    );
    // the population on each PES
    double Population[NumPES] = { 1.0 };
    // the population on each PES at last output moment
    double OldPopulation[NumPES] = { 0 };
    // construct the Hamiltonian and then the evolution class.
    // Diabatic Hamiltonian used for propagator: dc/dt=-iHc/hbar => c(t)=e^(-iHt)c(0)
    // c is the coefficient, c[m*NGrids+n] is the nth grid on the mth surface
    Evolution EvolveObject = Evolution
    (
        Absorbed,
        Hamiltonian_construction
        (
            NGrids,
            GridCoordinate,
            dx,
            mass,
            Absorbed,
            xmin,
            xmax,
            AbsorbingRegionLength
        ),
        DiabaticPsi,
        dt
    );
    auto [LastE, LastX, LastP] = make_tuple(p0 * p0 / 2.0 / mass, x0, p0);

    // psi: the grided wavefunction^2 (population on each grid)
    // In psi, each line is the wavefunction at a moment. In each line,
    // the order is t, psi(t)[0].real, psi(t)[0].imag, psi(t)[1].real, ...
    ofstream PsiOutput("psi.txt");
    PsiOutput.sync_with_stdio(false);
    // phase: the grided phase space distribution
    // In Phase space, each line is the PS-distribution at a moment:
    // rho00(x0,p0), rho00(x0,p1), ... rho00(x0,pn), ... rho00(xn,pn) (end line)
    // rho01(x0,p0), ... rho01(xn, pn) (end line)
    // ... rho0n(xn, pn) (end line)
    // ... rhonn(xn, pn) (end line)
    // (blank line)
    // (next moment grided phase space distribution)
    ofstream PhaseOutput("phase.txt");
    PhaseOutput.sync_with_stdio(false);
    // log contains average <E>, <x>, and <p> with corresponding time
    ofstream Log("averages.txt");
    Log.sync_with_stdio(false);
    // Steps contains when is each step, also one in a line
    ofstream Steps("t.txt");
    Steps.sync_with_stdio(false);
    clog << "Finish initialization. Begin evolving." << endl << show_time << endl;


    // evolution
    for (int iStep = 0; iStep <= TotalStep; iStep++)
    {
        // output the time
        const double Time = iStep * dt;

        // check if output is needed
        if (iStep % OutputStep == 0)
        {
            Steps << Time << endl;
            // calculate adiabatic wavefunction
            cblas_zgemv
            (
                CblasRowMajor,
                CblasConjTrans,
                dim,
                dim,
                &Alpha,
                TransformationMatrix.data(),
                dim,
                DiabaticPsi,
                1,
                &Beta,
                AdiabaticPsi,
                1
            );
            // print population on each grid
            output_grided_population(PsiOutput, NGrids, AdiabaticPsi);
            // print phase space distribution
            output_phase_space_distribution
            (
                PhaseOutput,
                NGrids,
                GridCoordinate,
                dx,
                p0,
                AdiabaticPsi
            );
            // calcuate <E>, <x> and <p>
            // using structured binding in C++17
            auto [AverageE, AverageX, AverageP] = calculate_average
            (
                NGrids,
                GridCoordinate,
                dx,
                mass,
                AdiabaticPsi
            );
            // ... then output                
            Log << Time << ' ' << AverageE << ' ' << AverageX << ' ' << AverageP;
            // calculate population on each PES
            calculate_population(NGrids, dx, AdiabaticPsi, Population);
            // ... then output
            for (int i = 0; i < NumPES; i++)
            {
                Log << ' ' << Population[i];
            }
            Log << endl;

            // compare with the last moment to see if stopping evolving
            if (AverageX > 0.0)
            {
                // after passing the center, do the judgement
                if (AverageX > -x0)
                {
                    cerr << "GET OUT OF INTERACTING REGION, STOP EVOLVING AT " << Time << endl;
                    break;
                }
                if ((AverageX - LastX) * p0 < 0)
                {
                    cerr << "DIRECTION REVERSED DUE TO REFLECTION / PBC, STOP EVOLVING AT " << Time << endl;
                    break;
                }
                if (Absorbed == true && accumulate(Population, Population + NumPES, 0.0) < PplLim)
                {
                    cerr << "ALMOST ALL POPULATION HAVE BEEN ABSORBED, STOP EVOLVING AT " << Time << endl;
                    break;
                }
                int NotChangedPES = 0;
                for (int i = 0; i < NumPES; i++)
                {
                    if (abs(Population[i] - OldPopulation[i]) < ChangeLim)
                    {
                        NotChangedPES++;
                    }
                }
                if (NotChangedPES == NumPES)
                {
                    cerr << "POPULATION ON EACH PES IS STABLE. STOP EVOLVING AT " << Time << endl;
                    break;
                }
            }

            // if not stop, copy the datas
            LastE = AverageE;
            LastX = AverageX;
            LastP = AverageP;
            memcpy(OldPopulation, Population, NumPES * sizeof(double));
        }

        // evolve to next moment
        EvolveObject.evolve(DiabaticPsi, Time + dt);
    }


    // after evolution, close the files
    Steps.close();
    Log.close();
    PhaseOutput.close();
    PsiOutput.close();
    // and output
    clog << "Finish evolution." << endl << show_time << endl << endl;
    if (TestModel == DAC)
    {
        cout << log(p0 * p0 / 2.0 / mass);
    }
    else
    {
        cout << p0;
    }
    calculate_population(NGrids, dx, AdiabaticPsi, Population);
    for (int i = 0; i < NumPES; i++)
    {
        cout << ' ' << Population[i];
    }
    cout << '\n';
    // free the memory
    delete[] AdiabaticPsi;
    delete[] DiabaticPsi;
    delete[] GridCoordinate;
	return 0;
}
