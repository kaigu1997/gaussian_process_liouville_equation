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
#include <numeric>
#include <omp.h>
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

/// Print current time in RFC 2822 format
ostream& show_time(ostream& os)
{
    auto time = chrono::system_clock::to_time_t(chrono::system_clock::now());
    os << ctime(&time);
    return os;
}


// evolution related function

/// Initialize the PWTDM (Partial Wigner-Transformed Density Matrix) with a gaussian wavepacket
/// 
/// i.e. the ground state is exp(-(x-x0)^2/2sigma_x^2-(p-p0)^2/2sigma_p^2)/(2*pi*sigma_x*sigma_p)
/// and all the other PES and off-diagonal elements are 0, and then normalize it
void density_matrix_initialization
(
    const int NGrids,
    const double* const GridPosition,
    const double* const GridMomentum,
    const double dx,
    const double dp,
    const double x0,
    const double p0,
    const double SigmaX,
    const double SigmaP,
    ComplexMatrixMatrix& rho_adia
)
{
    // NormFactor is for normalization
    double NormFactor = 0;
    for (int i = 0; i < NGrids; i++)
    {
        const double& xi = GridPosition[i];
        for (int j = 0; j < NGrids; j++)
        {
            const double& pj = GridMomentum[j];
            // for non-[0][0] elements, the initial density matrix is zero
            rho_adia[i][j] *= 0.0;
            // for ground state, it is a gaussian. rho[0][0](x,p,0)=exp(-(x-x0)^2/2sigma_x^2-(p-p0)^2/2sigma_p^2)/(2*pi*sigma_x*sigma_p)
            rho_adia[i][j][0][0] = exp(-(pow((xi - x0) / SigmaX, 2) + pow((pj - p0) / SigmaP, 2)) / 2.0) / (2.0 * pi * SigmaX * SigmaP);
            NormFactor += rho_adia[i][j][0][0].real();
        }
    }
    NormFactor *= dx * dp;
    // normalization, because of numerical error
    for (int i = 0; i < NGrids; i++)
    {
        for (int j = 0; j < NGrids; j++)
        {
            rho_adia[i][j][0][0] /= NormFactor;
        }
    }
}

void calculate_population
(
    const int NGrids,
    const double dx,
    const double dp,
    const ComplexMatrixMatrix& rho_adia,
    double* const Population
)
{
    // calculate the inner product of each PES
    for (int a = 0; a < NumPES; a++)
    {
        Population[a] = 0;
        for (int i = 0; i < NGrids; i++)
        {
            for (int j = 0; j < NGrids; j++)
            {
                Population[a] += rho_adia[i][j][a][a].real();
            }
        }
        Population[a] *= dx * dp;
    }
}

tuple<double, double, double> calculate_average
(
    const ComplexMatrixMatrix& rho,
    const int NGrids,
    const RealMatrix* const* const Potential,
    const double* const GridPosition,
    const double* const GridMomentum,
    const double mass,
    const double dx,
    const double dp,
    const Representation BasisOfRho
)
{
    double energy = 0, x = 0, p = 0;
    // calculate: <B>=sum_{i,j}(B(xi,pj)*P(xi,pj))*dx*dp
    for (int i = 0; i < NGrids; i++)
    {
        const double& xi = GridPosition[i];
        for (int j = 0; j < NGrids; j++)
        {
            const double& pj = GridMomentum[j];
            // loop over diagonal elements only
            for (int a = 0; a < NumPES; a++)
            {
                const double& ppl = rho[i][j][a][a].real();
                energy += ppl * (Potential[BasisOfRho][i][a][a] + pj * pj / 2.0 / mass);
                x += ppl * xi;
                p += ppl * pj;
            }
        }
    }
    return make_tuple(energy * dx * dp, x * dx * dp, p * dx * dp);
}

/// Evolve the quantum liouville: exp(-iLQt)rho
///
/// -iLQrho=-i/hbar[V-i*hbar*P/Mass*D,rho]
///
/// Input t should be dt/2
void quantum_liouville_propagation
(
    ComplexMatrixMatrix& rho,
    const int NGrids,
    const RealMatrix* const* const Potential,
    const RealMatrix* const* const Coupling,
    const double* const GridPosition,
    const double* const GridMomentum,
    const double mass,
    const double dt,
    const Representation BasisOfRho
)
{
    if (BasisOfRho == Diabatic)
    {
        // for diabatic basis, D=0, simple transform to adiabatic representation
        // exp(-iLQt)rho_dia=to_dia(exp(-iVd t/hbar)*rho_adia*exp(iVd t/hbar))
        basis_transform[Diabatic][Adiabatic](rho, NGrids, GridPosition);
#pragma omp parallel for default(none) shared(rho) schedule(static)
        for (int i = 0; i < NGrids; i++)
        {
            const RealMatrix& H = Potential[Adiabatic][i];
            for (int j = 0; j < NGrids; j++)
            {
                for (int a = 0; a < NumPES; a++)
                {
                    const double& Ea = H[a][a];
                    // no operation needed for diagonal elements
                    for (int b = a + 1; b < NumPES; b++)
                    {
                        const double& Eb = H[b][b];
                        rho[i][j][a][b] *= exp((Eb - Ea) * dt / hbar * 1.0i);
                        rho[i][j][b][a] *= exp((Ea - Eb) * dt / hbar * 1.0i);
                    }
                }
            }
        }
        basis_transform[Adiabatic][Diabatic](rho, NGrids, GridPosition);
    }
    else
    {
        // for other basis, no tricks to play
        // exp(-iLQt)rho=exp(-iH't/hbar)*rho*exp(iH't/hbar)
        // =C*exp(-iH'd t/hbar)*C^T*rho*C*exp(iH'd t/hbar)*C^T
#pragma omp parallel for default(none) shared(rho, cerr) schedule(static)
        for (int i = 0; i < NGrids; i++)
        {
            const double& x = GridPosition[i];
            const ComplexMatrix& H = Potential[BasisOfRho][i];
            const ComplexMatrix& D = Coupling[BasisOfRho][i];
            for (int j = 0; j < NGrids; j++)
            {
                const double& p = GridMomentum[j];
                ComplexMatrix EigVec = H - Complex(0.0, hbar * p / mass) * D;
                double EigVal[NumPES];
                if (LAPACKE_zheev(LAPACK_ROW_MAJOR, 'V', 'U', NumPES, reinterpret_cast<MKL_Complex16*>(EigVec.data()), NumPES, EigVal) != 0)
                {
                    cerr << "UNABLE TO DIAGONALIZE THE QUANTUM LIOUVILLE AT X=" << x << " AND P=" << p << endl;
                    exit(300);
                }
                Complex EigDenEig[NumPES * NumPES];
                // first, C^T*rho*C
                cblas_zhemm(CblasRowMajor, CblasLeft, CblasUpper, NumPES, NumPES, &Alpha, rho[i][j].data(), NumPES, EigVec.data(), NumPES, &Beta, EigDenEig, NumPES);
                cblas_zgemm(CblasRowMajor, CblasConjTrans, CblasNoTrans, NumPES, NumPES, NumPES, &Alpha, EigVec.data(), NumPES, EigDenEig, NumPES, &Beta, rho[i][j].data(), NumPES);
                // make it hermitian
                rho[i][j].hermitize();
                // second, exp(-iH'd t/hbar)*C^T*rho*C*exp(iH'd t/hbar)
                for (int a = 0; a < NumPES; a++)
                {
                    // no operation needed for diagonal elements
                    for (int b = a + 1; b < NumPES; b++)
                    {
                        rho[i][j][a][b] *= exp((EigVal[b] - EigVal[a]) * dt / hbar * 1.0i);
                        rho[i][j][b][a] *= exp((EigVal[a] - EigVal[b]) * dt / hbar * 1.0i);
                    }
                }
                // make itself hermitian for next basis transformation
                rho[i][j].hermitize();
                // last, C*exp(-iH'd t/hbar)*C^T*rho*C*exp(iH'd t/hbar)*C^T
                cblas_zhemm(CblasRowMajor, CblasRight, CblasUpper, NumPES, NumPES, &Alpha, rho[i][j].data(), NumPES, EigVec.data(), NumPES, &Beta, EigDenEig, NumPES);
                cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasConjTrans, NumPES, NumPES, NumPES, &Alpha, EigDenEig, NumPES, EigVec.data(), NumPES, &Beta, rho[i][j].data(), NumPES);
                // make it hermitian again
                rho[i][j].hermitize();
            }
        }
    }
}

/// evolve the classical position liouville: exp(-iLRt)rho
///
/// -iLRrho=-P/M*drho/dR, and the derivative is done via FFT in MKL
///
/// rho(k,t)=exp(-P/M*2*k*pi*i/L*t)rho(k,0), k=-N/2 to N/2, using Periodic Boundary
///
/// input t should be dt/2 as well
void classical_position_liouville_propagator
(
    ComplexMatrixMatrix& rho,
    const int NGrids,
    const double* const GridMomentum,
    const double mass,
    const double TotalPositionLength,
    const double dx,
    const double dt
)
{
    // set the FFT handle
    DFTI_DESCRIPTOR_HANDLE PositionFFT = nullptr;
    // initialize
    MKL_LONG status = DftiCreateDescriptor(&PositionFFT, DFTI_DOUBLE, DFTI_COMPLEX, 1, NGrids);
    if (status != 0 && DftiErrorClass(status, DFTI_NO_ERROR) != 0)
    {
        cerr << "UNABLE TO INITIALIZE FFT OF CLASSICAL POSITION LIOUVILLE BECAUSE "
            << DftiErrorMessage(status) << endl;
        exit(301);
    }
    // set parameters: backward scale, to make backward the inverse of forward
    status = DftiSetValue(PositionFFT, DFTI_BACKWARD_SCALE, 1.0 / NGrids);
    if (status != 0 && DftiErrorClass(status, DFTI_NO_ERROR) == 0)
    {
        cerr << "UNABLE TO SET PARAMETER OF FFT OF CLASSICAL POSITION LIOUVILLE BECAUSE "
            << DftiErrorMessage(status) << endl;
        exit(301);
    }
    // set parameters: replacement, to transform to a new space
    status = DftiSetValue(PositionFFT, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    if (status != 0 && DftiErrorClass(status, DFTI_NO_ERROR) == 0)
    {
        cerr << "UNABLE TO SET PARAMETER OF FFT OF CLASSICAL POSITION LIOUVILLE BECAUSE "
            << DftiErrorMessage(status) << endl;
        exit(301);
    }
    // after setting parameters, commit the descriptor
    status = DftiCommitDescriptor(PositionFFT);
    if (status != 0 && DftiErrorClass(status, DFTI_NO_ERROR) == 0)
    {
        cerr << "UNABLE TO COMMIT FFT OF CLASSICAL POSITION LIOUVILLE BECAUSE "
            << DftiErrorMessage(status) << endl;
        exit(301);
    }

#pragma omp parallel for default(none) shared(PositionFFT, rho, cerr) private(status) schedule(static)
    for (int j = 0; j < NGrids; j++)
    {
        // the evolving matrix is the same for a and b, so construct and diagonalize here
        const double& p = GridMomentum[j];
        for (int a = 0; a < NumPES; a++)
        {
            for (int b = 0; b < NumPES; b++)
            {
                // construct the vector: rho_W^{ab}(.,P_j)
                Complex* DifferentPosition = new Complex[NGrids];
                Complex* TransformedPosition = new Complex[NGrids];
                // assign the value
                for (int i = 0; i < NGrids; i++)
                {
                    DifferentPosition[i] = rho[i][j][a][b];
                }
                // evolve
                // calculate the forward FFT
                status = DftiComputeForward(PositionFFT, DifferentPosition, TransformedPosition);
                if (status != 0 && DftiErrorClass(status, DFTI_NO_ERROR) == 0)
                {
                    cerr << "UNABLE TO CALCULATE FORWARD FFT OF CLASSICAL POSITION LIOUVILLE BECAUSE "
                        << DftiErrorMessage(status) << endl;
                    exit(301);
                }
                // multiply with exp(-P/M*2*k*pi*i/L * dt), k=-N/2 to N/2
                for (int k = 0; k < NGrids / 2; k++)
                {
                    TransformedPosition[k] *= exp(-p / mass * 2 * k * pi * 1.0i / TotalPositionLength * dt);
                }
                for (int k = NGrids / 2; k < NGrids; k++)
                {
                    TransformedPosition[k] *= exp(-p / mass * 2 * (k - NGrids) * pi * 1.0i / TotalPositionLength * dt);
                }
                // after exp, doing backward FFT
                status = DftiComputeBackward(PositionFFT, TransformedPosition, DifferentPosition);
                if (status != 0 && DftiErrorClass(status, DFTI_NO_ERROR) == 0)
                {
                    cerr << "UNABLE TO CALCULATE BACKWARD FFT OF CLASSICAL POSITION LIOUVILLE BECAUSE "
                        << DftiErrorMessage(status) << endl;
                    exit(301);
                }
                // assign value back
                for (int i = 0; i < NGrids; i++)
                {
                    rho[i][j][a][b] = DifferentPosition[i];
                }
                // free the memory
                delete[] DifferentPosition;
                delete[] TransformedPosition;
            }
        }
        // to make density matrix on each grid hermitian
        for (int i = 0; i < NGrids; i++)
        {
            rho[i][j].hermitize();
        }
    }

    // after FFT, release the descriptor
    status = DftiFreeDescriptor(&PositionFFT);
    if (status != 0 && DftiErrorClass(status, DFTI_NO_ERROR) == 0)
    {
        cerr << "UNABLE TO FREE FFT OF CLASSICAL POSITION LIOUVILLE BECAUSE "
            << DftiErrorMessage(status) << endl;
        exit(301);
    }
}

/// evolve the classical momentum liouville: exp(-iLPt)rho
///
/// -iLPrho=-1/2(F*drho/dP+drho/dP*F)=-(Faa+Fbb)/2*drho/dP
/// if under force basis.
///
/// input t should be dt
void classical_momentum_liouville_propagator
(
    ComplexMatrixMatrix& rho,
    const int NGrids,
    const RealMatrix* const* const Force,
    const double* const GridPosition,
    const double TotalMomentumLength,
    const double dp,
    const double dt,
    const Representation BasisOfRho
)
{
    // transform to force basis first
    basis_transform[BasisOfRho][ForceBasis](rho, NGrids, GridPosition);

    // set the FFT handle
    DFTI_DESCRIPTOR_HANDLE MomentumFFT = nullptr;
    // initialize
    MKL_LONG status = DftiCreateDescriptor(&MomentumFFT, DFTI_DOUBLE, DFTI_COMPLEX, 1, NGrids);
    if (status != 0 && DftiErrorClass(status, DFTI_NO_ERROR) != 0)
    {
        cerr << "UNABLE TO INITIALIZE FFT OF CLASSICAL POSITION LIOUVILLE BECAUSE "
            << DftiErrorMessage(status) << endl;
        exit(301);
    }
    // set parameters: backward scale, to make backward the inverse of forward
    status = DftiSetValue(MomentumFFT, DFTI_BACKWARD_SCALE, 1.0 / NGrids);
    if (status != 0 && DftiErrorClass(status, DFTI_NO_ERROR) == 0)
    {
        cerr << "UNABLE TO SET PARAMETER OF FFT OF CLASSICAL POSITION LIOUVILLE BECAUSE "
            << DftiErrorMessage(status) << endl;
        exit(301);
    }
    // set parameters: replacement, to transform to a new space
    status = DftiSetValue(MomentumFFT, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    if (status != 0 && DftiErrorClass(status, DFTI_NO_ERROR) == 0)
    {
        cerr << "UNABLE TO SET PARAMETER OF FFT OF CLASSICAL POSITION LIOUVILLE BECAUSE "
            << DftiErrorMessage(status) << endl;
        exit(301);
    }
    // after setting parameters, commit the descriptor
    status = DftiCommitDescriptor(MomentumFFT);
    if (status != 0 && DftiErrorClass(status, DFTI_NO_ERROR) == 0)
    {
        cerr << "UNABLE TO COMMIT FFT OF CLASSICAL POSITION LIOUVILLE BECAUSE "
            << DftiErrorMessage(status) << endl;
        exit(301);
    }

#pragma omp parallel for default(none) shared(MomentumFFT, rho, cerr) private(status) schedule(static)
    for (int i = 0; i < NGrids; i++)
    {
        // the eigen forces
        const RealMatrix& F = Force[ForceBasis][i];
        for (int a = 0; a < NumPES; a++)
        {
            const double& Fa = F[a][a];
            for (int b = 0; b < NumPES; b++)
            {
                const double& Fb = F[b][b];
                // construct the vector: rho_W^{ab}(R_i,.)
                Complex* DifferentMomentum = new Complex[NGrids];
                Complex* TransformedMomentum = new Complex[NGrids];
                // assign the value
                for (int j = 0; j < NGrids; j++)
                {
                    DifferentMomentum[j] = rho[i][j][a][b];
                }
                // evolve
                // calculate the forward FFT
                status = DftiComputeForward(MomentumFFT, DifferentMomentum, TransformedMomentum);
                if (status != 0 && DftiErrorClass(status, DFTI_NO_ERROR) == 0)
                {
                    cerr << "UNABLE TO CALCULATE FORWARD FFT OF CLASSICAL POSITION LIOUVILLE BECAUSE "
                        << DftiErrorMessage(status) << endl;
                    exit(301);
                }
                // multiply with exp(-(Faa+Fbb)/2*2*k*pi*i/L*dt)=exp(-(Fa+Fb)*k*pi*i/L*dt), k=-N/2 to N/2
                for (int k = 0; k < NGrids / 2; k++)
                {
                    TransformedMomentum[k] *= exp(-(Fa + Fb) * k * pi * 1.0i / TotalMomentumLength * dt);
                }
                for (int k = NGrids / 2; k < NGrids; k++)
                {
                    TransformedMomentum[k] *= exp(-(Fa + Fb) * (k - NGrids) * pi * 1.0i / TotalMomentumLength * dt);
                }
                // after exp, doing backward FFT
                status = DftiComputeBackward(MomentumFFT, TransformedMomentum, DifferentMomentum);
                if (status != 0 && DftiErrorClass(status, DFTI_NO_ERROR) == 0)
                {
                    cerr << "UNABLE TO CALCULATE BACKWARD FFT OF CLASSICAL POSITION LIOUVILLE BECAUSE "
                        << DftiErrorMessage(status) << endl;
                    exit(301);
                }
                // assign value back
                for (int j = 0; j < NGrids; j++)
                {
                    rho[i][j][a][b] = DifferentMomentum[j];
                }
                // free the memory
                delete[] DifferentMomentum;
                delete[] TransformedMomentum;
            }
        }
        for (int j = 0; j < NGrids; j++)
        {
            rho[i][j].hermitize();
        }
    }

    // after FFT, release the descriptor
    status = DftiFreeDescriptor(&MomentumFFT);
    if (status != 0 && DftiErrorClass(status, DFTI_NO_ERROR) == 0)
    {
        cerr << "UNABLE TO FREE FFT OF CLASSICAL POSITION LIOUVILLE BECAUSE "
            << DftiErrorMessage(status) << endl;
        exit(301);
    }

    // finally transform back to diabatic basis
    basis_transform[ForceBasis][BasisOfRho](rho, NGrids, GridPosition);
}
