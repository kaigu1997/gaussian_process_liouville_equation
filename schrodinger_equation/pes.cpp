// implementation of pes.h:
// diabatic PES and absorbing potential

#include <cmath>
#include "general.h"
#include "matrix.h"
#include "pes.h"



// Diabatic reprersentation
// parameters of Tully's 1st model, Simple Avoided Crossing (SAC)
static const double SAC_A = 0.01; ///< A in SAC model
static const double SAC_B = 1.6; ///< B in SAC model
static const double SAC_C = 0.005; ///< C in SAC model
static const double SAC_D = 1.0; ///< D in SAC model
// parameters of Tully's 2nd model, Dual Avoided Crossing (DAC)
static const double DAC_A = 0.10; ///< A in DAC model
static const double DAC_B = 0.28; ///< B in DAC model
static const double DAC_C = 0.015; ///< C in DAC model
static const double DAC_D = 0.06; ///< D in DAC model
static const double DAC_E = 0.05; ///< E in DAC model
// parameters of Tully's 3rd model, Extended Coupling with Reflection (ECR)
static const double ECR_A = 6e-4; ///< A in ECR model
static const double ECR_B = 0.10; ///< B in ECR model
static const double ECR_C = 0.90; ///< C in ECR model

/// Subsystem diabatic Hamiltonian, being the potential of the bath
RealMatrix diabatic_potential(const double x)
{
    RealMatrix Potential(NumPES);
    switch (TestModel)
    {
    case SAC: // Tully's 1st model
        Potential[0][1] = Potential[1][0] = SAC_C * exp(-SAC_D * x * x);
        Potential[0][0] = sgn(x) * SAC_A * (1.0 - exp(-sgn(x) * SAC_B * x));
        Potential[1][1] = -Potential[0][0];
        break;
    case DAC: // Tully's 2nd model
        Potential[0][1] = Potential[1][0] = DAC_C * exp(-DAC_D * x * x);
        Potential[1][1] = DAC_E - DAC_A * exp(-DAC_B * x * x);
        break;
    case ECR: // Tully's 3rd model
        Potential[0][0] = ECR_A;
        Potential[1][1] = -ECR_A;
        Potential[0][1] = Potential[1][0] = ECR_B * (1 - sgn(x) * (exp(-sgn(x) * ECR_C * x) - 1));
        break;
    }
    return Potential;
}

static const double c = sqrt(2) * comp_ellint_1(1.0 / sqrt(2)); ///< the constant used in absorbing potential

/// In the interacting region [xmin, xmax], E is zero 
///
/// Otherwise, E=hbar^2/2m*(2*pi/arl)^2*y(x), where
/// x=2*d*kmin*(r-r1)=c(r-r1)/arl, and
/// d=c/4pi, so arl=2pi/kmin=h/pmin.
///
/// Here, the r is the parameter x in the function call
///
/// y(x)=sqrt(pow(cn(x/sqrt(2),1/sqrt(2)),-4)-1)
/// ~4/(c-x)^2+4/(c+x)^2-8/c^2 from
/// J. Chem. Phys., 2004, 120(5): 2247-2254,
///
/// c=sqrt(2)*K(1/sqrt(2))
double absorbing_potential
(
    const double mass,
    const double xmin,
    const double xmax,
    const double AbsorbingRegionLength,
    const double x
)
{
    // in the interacting region, no AP
    if (x > xmin && x < xmax)
    {
        return 0;
    }
    // otherwise, return E(x)_ii
    const double xx = c * (x < xmin ? x - xmin : x - xmax) / AbsorbingRegionLength;
    return pow(PlanckH / AbsorbingRegionLength, 2) * 2.0 / mass
        * (1.0 / pow(c - xx, 2) + 1.0 / pow(c + xx, 2) - 2.0 / pow(c, 2));
}

/// i.e. C^T*psi(dia)=psi(adia), which diagonalize PES only (instead of diagonal H)
ComplexMatrix diabatic_to_adiabatic(const int NGrids, const double* const GridCoordinate)
{
    ComplexMatrix TransformationMatrix(NGrids * NumPES);
    // EigVal stores the eigenvalues
    double EigVal[NumPES];
    
    for (int i = 0; i < NGrids; i++)
    {
        // EigVec stores the V before diagonalization and 
        // transformation matrix after diagonalization
        RealMatrix EigVec = diabatic_potential(GridCoordinate[i]);
        // diagonal each grid
        if (LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', NumPES, EigVec.data(), NumPES, EigVal) > 0)
        {
            cerr << "FAILING DIAGONALIZE DIABATIC POTENTIAL AT " << GridCoordinate[i] << endl;
            exit(300);
        }
        // copy the transformation info to the whole matrix
        for (int j = 0; j < NumPES; j++)
        {
            for (int k = 0; k < NumPES; k++)
            {
                TransformationMatrix[j * NGrids + i][k * NGrids + i] = EigVec[j][k];
            }
        }
    }
    return TransformationMatrix;
}
