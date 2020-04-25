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
#include "general.h"
#include "matrix.h"
#include "pes.h"
using namespace std;

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

/// Judge by the string in the input.
///
/// If the string is "on", use ABC, return true.
///
/// If the string is "off", not use ABC, return false.
///
/// Otherwise, the input is invalid, kills the program
bool read_absorb(istream& is)
{
    string buffer, WhetherAbsorb;
    bool Absorbed;
    getline(is, buffer);
    is >> WhetherAbsorb;
    getline(is, buffer);
    if (strcmp(WhetherAbsorb.c_str(), "on") == 0)
    {
        Absorbed = true;
    }
    else if (strcmp(WhetherAbsorb.c_str(), "off") == 0)
    {
        Absorbed = false;
    }
    else
    {
        cerr << "UNKNOWN CASE OF ABSORBING POTENTIAL" << endl;
        exit(200);
    }
    return Absorbed;
}

ostream& show_time(ostream& os)
{
    auto time = chrono::system_clock::to_time_t(chrono::system_clock::now());
    os << ctime(&time);
    return os;
}


// evolution related function
/// Construct the diabatic Hamiltonian consisting of 3 parts:
///
/// 1. diabatic potential (the subsystem/electron Hamiltonian), which is real-symmetric
///
/// 2. bath/nucleus kinetic energy by infinite order finite difference, which is also real-symmetric
///
/// 3. absorbed boundary condition when it is used, which is pure complex on diagonal elements
///
/// This function is called only once for the initialization of Hamiltonian
ComplexMatrix Hamiltonian_construction
(
    const int NGrids,
    const double* const GridCoordinate,
    const double dx,
    const double mass,
    const bool absorbed,
    const double xmin,
    const double xmax,
    const double AbsorbingRegionLength
)
{
    ComplexMatrix Hamiltonian(NGrids * NumPES);
    // 1. V_{mm'}(R_n), n=n'
    for (int n = 0; n < NGrids; n++)
    {
        const RealMatrix& Vn = diabatic_potential(GridCoordinate[n]);
        for (int m = 0; m < NumPES; m++)
        {
            for (int mm = 0; mm < NumPES; mm++)
            {
                Hamiltonian[m * NGrids + n][mm * NGrids + n] += Vn[m][mm];
            }
        }
    }
    // 2. d2/dx2 (over all pes)
    for (int m = 0; m < NumPES; m++)
    {
        for (int n = 0; n < NGrids; n++)
        {
            for (int nn = 0; nn < NGrids; nn++)
            {
                if (nn == n)
                {
                    Hamiltonian[m * NGrids + n][m * NGrids + nn] += pow(pi * hbar / dx, 2) / 6.0 / mass;
                }
                else
                {
                    Hamiltonian[m * NGrids + n][m * NGrids + nn] += pow_minus_one(nn - n) * pow(hbar / dx / (nn - n), 2) / mass;
                }
            }
        }
    }
    // 3. absorbing potential
    if (absorbed == true)
    {
        for (int n = 0; n < NGrids; n++)
        {
            const double&& An = absorbing_potential(mass, xmin, xmax, AbsorbingRegionLength, GridCoordinate[n]);
            for (int m = 0; m < NumPES; m++)
            {
                Hamiltonian[m * NGrids + n][m * NGrids + n] -= 1.0i * An;
            }
        }
    }
    return Hamiltonian;
}

/// psi(x)=exp(-(x-x0)^2/4/sigma_x^2+i*p0*x/hbar)/[(2*pi)^(1/4)*sqrt(sigma_x)]
///
/// as the wavefunction is on grid, the normalization factor could be different
///
/// this function is also only called once, for initialization of the adiabatic wavefunction
void wavefunction_initialization
(
    const int NGrids,
    const double* const GridCoordinate,
    const double dx,
    const double x0,
    const double p0,
    const double SigmaX,
    Complex* const Psi
)
{
    // for higher PES, the initial wavefunction is zero
    memset(Psi + NGrids, 0, NGrids * (NumPES - 1) * sizeof(Complex));
    // for ground state, it is a gaussian. psi(x)=A*exp(-(x-x0)^2/4sigmax^2+ip0x/hbar)
    for (int i = 0; i < NGrids; i++)
    {
        const double& x = GridCoordinate[i];
        Psi[i] = exp(-pow((x - x0) / 2 / SigmaX, 2) + p0 * x / hbar * 1.0i) / sqrt(sqrt(2.0 * pi) * SigmaX);
    }
    // normalization
    Complex PsiSquare;
    // calling cblas_zdotc_sub rather than cblas_znrm2 due to higher accuracy
    cblas_zdotc_sub
    (
        NGrids,
        Psi,
        1,
        Psi,
        1,
        &PsiSquare
    );
    const double NormFactor = sqrt(PsiSquare.real() * dx);
    for (int i = 0; i < NGrids; i++)
    {
        Psi[i] /= NormFactor;
    }
}

/// Constructor. It allocates memory, and if there is no ABC, diagonalize Hamiltonian
Evolution::Evolution
(
    const bool IsAbsorb,
    const ComplexMatrix& DiaH,
    const Complex* const Psi0,
    const double TimeStep
):
    Absorbed(IsAbsorb),
    dim(DiaH.length_of_matrix()),
    Hamiltonian(DiaH),
    Intermediate1(new Complex[dim]),
    Intermediate2(new Complex[dim]),
    PsiAtTimeT(new Complex[dim]),
    EigVec(Hamiltonian),
    EigVal(new double[dim]),
    dt(TimeStep),
    RK4kBeta(Complex(dt) / RK4Parameter),
    RK4PsiAlpha(Complex(dt / 6.0) * RK4Parameter)
{
    if (Absorbed == false)
    {
        // diagonalization
        const int status = LAPACKE_zheev
        (
            LAPACK_ROW_MAJOR,
            'V',
            'U',
            dim,
            reinterpret_cast<MKL_Complex16*>(EigVec.data()),
            dim,
            EigVal
        );
        if (status > 0)
        {
            cerr << "FAILING DIAGONALIZE HAMILTONIAN WITHOUT ABSORBING POTENTIAL" << endl;
            exit(201);
        }

        // then calculate the initial wavefunction under the energy eigenbasis
        cblas_zgemv
        (
            CblasRowMajor,
            CblasConjTrans,
            dim,
            dim,
            &Alpha,
            EigVec.data(),
            dim,
            Psi0,
            1,
            &Beta,
            Intermediate1,
            1
        );
    }
}

Evolution::~Evolution(void)
{
    delete[] EigVal;
    delete[] PsiAtTimeT;
    delete[] Intermediate2;
    delete[] Intermediate1;
}

/// If w/ ABC, using RK4 to evolve.
/// k1=f(y,t), k2=f(y+dt/2*k1, t+dt/2), k3=f(y+dt/2*k2,t+dt/2),
/// k4=f(y+dt*k3, t+dt), y(t+dt)=y(t)+dt/6*(k1+2*k2+2*k3+k4).
/// Here f(y,t)=hat(H)/i/hbar*psi
///
/// If w/o ABC, the initial diagonalized wavefunction have been
/// calculated. psi(t)_dia=exp(-iHt/hbar)*psi(0)_dia
/// =C*exp(-iHd*t/hbar)*C^T*psi(0)_dia=C*exp(-iHd*t/hbar)*psi(0)_diag
void Evolution::evolve(Complex* Psi, const double Time)
{
    if (Absorbed == false)
    {
        // no ABC, using the diagonalized wavefunction

        // calculate psi(t)_diag = exp(-iH(diag)*t/hbar) * psi(0)_diag, each diagonal element be the eigenvalue
        // here, psi(0)_diag is Intermediate1, initialized in constructor; psi(t)_diag is Intermediate2 
#pragma omp parallel for default(none) shared(Intermediate1, Intermediate2, EigVal) schedule(static)
        for (int i = 0; i < dim; i++)
        {
            Intermediate2[i] = exp(-EigVal[i] * Time / hbar * 1.0i) * Intermediate1[i];
        }
        // calculate DiabaticPsi=C1*psi_t_diag
        cblas_zgemv
        (
            CblasRowMajor,
            CblasNoTrans,
            dim,
            dim,
            &Alpha,
            EigVec.data(),
            dim,
            Intermediate2,
            1,
            &Beta,
            PsiAtTimeT,
            1
        );
    }
    else
    {
        // with ABC, using RK4

        // copy the wavefunction at time t-dt, leave the input as constant
        memcpy(PsiAtTimeT, Psi, dim * sizeof(Complex));
        // k1=f(y,t), k2=f(y+dt/2*k1, t+dt/2), k3=f(y+dt/2*k2,t+dt/2)
        // k4=f(y+dt*k3, t+dt), y(t+dt)=y(t)+dt/6*(k1+2*k2+2*k3+k4)
        // here f(y,t)=hat(H)/i/hbar*psi
        // Intermediate1 is y+dt/n*ki, Intermediate2 is ki=f(Intermediate1)=H*I1/ihbar
        // then I2 is swapped with I1, so I1=psi/ihbar+dt/ni/ihbar*I1 by calling clabs_zaxpby
        memset(Intermediate1, 0, dim * sizeof(Complex));
        for (int i = 0; i < 4; i++)
        {
            // k(i)=(psi(t)+dt/ni*k(i-1))/i/hbar
            cblas_zaxpby
            (
                dim,
                &Alpha,
                Psi,
                1,
                &RK4kBeta[i],
                Intermediate1,
                1
            );
            // k'(i)=H*k(i)
            cblas_zgemv
            (
                CblasRowMajor,
                CblasNoTrans,
                dim,
                dim,
                &RK4kAlpha,
                Hamiltonian.data(),
                dim,
                Intermediate1,
                1,
                &Beta,
                Intermediate2,
                1
            );
            // change the k' and k
            swap(Intermediate1, Intermediate2);
            // add to the wavefunction: y(t+dt)=y(t)+dt/6*ni*ki
            cblas_zaxpy
            (
                dim,
                &RK4PsiAlpha[i],
                Intermediate1,
                1,
                PsiAtTimeT,
                1
            );
        }
    }
    // after evolution, copy into the Psi in parameter list
    memcpy(Psi, PsiAtTimeT, dim * sizeof(Complex));
}


/// output each element's modular square in a line, separated with space, w/ \n @ eol
void output_grided_population(ostream& os, const int NGrids, const Complex* const Psi)
{
    for (int i = 0; i < NGrids * NumPES; i++)
    {
        os << ' ' << (Psi[i] * conj(Psi[i])).real();
    }
    os << endl;
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
void output_phase_space_distribution
(
    ostream& os,
    const int NGrids,
    const double* const GridCoordinate,
    const double dx,
    const double p0,
    const Complex* const psi
)
{
    const double pmin = p0 - pi * hbar / dx / 2.0;
    const double pmax = p0 + pi * hbar / dx / 2.0;
    // Wigner Transformation: P(x,p)=int{dy*exp(2ipy/hbar)<x-y|rho|x+y>}/(pi*hbar)
    // the interval of p is p0+pi*hbar/dx*[-1,1), dp=2*pi*hbar/(xmax-xmin)
    // loop over pes first
    for (int iPES = 0; iPES < NumPES; iPES++)
    {
        for (int jPES = 0; jPES < NumPES; jPES++)
        {
            // loop over x
            for (int xi = 0; xi < NGrids; xi++)
            {
                // loop over p
                for (int pj = 0; pj < NGrids; pj++)
                {
                    const double p = ((NGrids - 1 - pj) * pmin + pj * pmax) / (NGrids - 1);
                    // do the numerical integral
                    Complex integral = 0.0;
                    // 0 <= (x+y, x-y) < NGrids 
                    for (int yk = max(-xi, xi + 1 - NGrids); yk <= min(xi, NGrids - 1 - xi); yk++)
                    {
                        const double y = yk * dx;
                        integral += exp(2.0 * p * y / hbar * 1.0i) * psi[xi - yk + iPES * NGrids] * conj(psi[xi + yk + jPES * NGrids]);
                    }
                    integral *= dx / pi / hbar;
                    os << ' ' << integral.real() << ' ' << integral.imag();
                }
            }
            os << '\n';
        }
    }
    os << endl;
}

void calculate_population
(
    const int NGrids,
    const double dx,
    const Complex* const Psi,
    double* const Population
)
{
    Complex InnerProduct;
    // calculate the inner product of each PES
    // again, calling cblas_zdotc_sub rather than cblas_znrm2 due to higher accuracy
    for (int i = 0; i < NumPES; i++)
    {
        cblas_zdotc_sub
        (
            NGrids,
            Psi + i * NGrids,
            1,
            Psi + i * NGrids,
            1,
            &InnerProduct
        );
        Population[i] = InnerProduct.real() * dx;
    }
}

/// @brief construct the first order derivative derived from finite difference
/// @param NGrids the number of grids
/// @param dx the grid spacing
/// @return the 1st order derivative matrix
static RealMatrix derivative(const int NGrids, const double dx)
{
    RealMatrix result(NumPES * NGrids);
    for (int i = 0; i < NumPES; i++)
    {
        for (int j = 0; j < NGrids; j++)
        {
            for (int k = 0; k < NGrids; k++)
            {
                if (j != k)
                {
                    result[j + i * NGrids][k + i * NGrids] = pow_minus_one(j - k) / dx / (j - k);
                }
            }
        }
    }
    return result;
}

/// return <E>, <x>, then <p>
///
/// <x> = dx*sum_i{x*|psi(x)|^2}
///
/// <E,p> = dx * <psi|A|psi>
tuple<double, double, double> calculate_average
(
    const int NGrids,
    const double* const GridCoordinate,
    const double dx,
    const double mass,
    const Complex* const Psi
)
{
    // the number of elements in psi
    const int dim = NumPES * NGrids;
    // construct the p and H matrix
    static const ComplexMatrix P = -1.0i * hbar * ComplexMatrix(derivative(NGrids, dx));
    static const ComplexMatrix H = Hamiltonian_construction
    (
        NGrids,
        GridCoordinate,
        dx,
        mass
    );
    static Complex* MatMul = new Complex[dim];
    Complex result;
    double x = 0.0, p = 0.0, E = 0.0;
    // first, <E>
    cblas_zhemv
    (
        CblasRowMajor,
        CblasUpper,
        dim,
        &Alpha,
        H.data(),
        dim,
        Psi,
        1,
        &Beta,
        MatMul,
        1
    );
    cblas_zdotc_sub
    (
        dim,
        Psi,
        1,
        MatMul,
        1,
        &result
    );
    E = result.real() * dx;
    // next, <x>
    for (int i = 0; i < NumPES; i++)
    {
        for (int j = 0; j < NGrids; j++)
        {
            x += GridCoordinate[j] * (Psi[i * NGrids + j] * conj(Psi[i * NGrids + j])).real();
        }
    }
    x *= dx;
    // finally, <p>, same as <E>
    cblas_zhemv
    (
        CblasRowMajor,
        CblasUpper,
        dim,
        &Alpha,
        P.data(),
        dim,
        Psi,
        1,
        &Beta,
        MatMul,
        1
    );
    cblas_zdotc_sub
    (
        dim,
        Psi,
        1,
        MatMul,
        1,
        &result
    );
    p = result.real() * dx;
    return make_tuple(E, x, p);
}
