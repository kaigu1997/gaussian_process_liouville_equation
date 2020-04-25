/// @file pes.cpp
/// @brief Implementation of pes.h: diabatic PES and absorbing potential

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <mkl.h>
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
/// @param x the bath DOF position
/// @return the potential matrix, which is real symmetric
/// @see diabatic_force(), diabatic_hesse(), diabatic_coupling(),
/// @see potential, adiabatic_potential(), force_basis_potential()
static RealMatrix diabatic_potential(const double x)
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

/// Diabatic force, the analytical derivative (F=-dH/dR=-dV/dR)
/// @param x the bath DOF position
/// @return the force matrix, which is real symmetric
/// @see diabatic_potential(), diabatic_hesse(), diabatic_coupling(),
/// @see force, adiabatic_force(), force_basis_force()
static RealMatrix diabatic_force(const double x)
{
    RealMatrix Force(NumPES);
    switch (TestModel)
    {
    case SAC: // Tully's 1st model
        Force[0][1] = Force[1][0] = 2.0 * SAC_C * SAC_D * x * exp(-SAC_D * x * x);
        Force[0][0] = -SAC_A * SAC_B * exp(-sgn(x) * SAC_B * x);
        Force[1][1] = -Force[0][0];
        break;
    case DAC: // Tully's 2nd model
        Force[0][1] = Force[1][0] = 2 * DAC_C * DAC_D * x * exp(-DAC_D * x * x);
        Force[1][1] = -2 * DAC_A * DAC_B * x * exp(-DAC_B * x * x);
        break;
    case ECR: // Tully's 3rd model
        Force[0][1] = Force[1][0] = -ECR_B * ECR_C * exp(-sgn(x) * ECR_C * x);
        break;
    }
    return Force;
}

/// Diabatic hessian matrix, the second derivative over position (Hesse=d2H/dR2=d2V/dR2)
/// @param x the bath DOF position
/// @return the hessian matrix, which is real symmetric
/// @see diabatic_potential(), diabatic_force(), diabatic_coupling(), force_basis_hesse()
static RealMatrix diabatic_hesse(const double x)
{
    RealMatrix Hesse(NumPES);
    switch (TestModel)
    {
    case SAC: // Tully's 1st model
        Hesse[0][1] = Hesse[1][0] = 2 * SAC_C * SAC_D * (2 * SAC_D * x * x - 1) * exp(-SAC_D * x * x);
        Hesse[0][0] = -sgn(x) * SAC_A * SAC_B * SAC_B * exp(-sgn(x) * SAC_B * x);
        Hesse[1][1] = -Hesse[0][0];
        break;
    case DAC: // Tully's 2nd model
        Hesse[0][1] = Hesse[1][0] = 2 * DAC_C * DAC_D * (2 * DAC_D * x * x - 1) * exp(-DAC_D * x * x);
        Hesse[1][1] = -2 * DAC_A * DAC_B * (2 * DAC_B * x * x - 1) * exp(-DAC_B * x * x);
        break;
    case ECR: // Tully's 3rd model
        Hesse[0][1] = Hesse[1][0] = -sgn(x) * ECR_B * ECR_C * ECR_C * exp(-sgn(x) * ECR_C * x);
        break;
    }
    return Hesse;
}

/// Non-Adiabatic Coupling (NAC) matrix under diabatic basis, which is a zero-matrix
/// @param x the bath DOF position, which is useless here
/// @return a zero-matrix
/// @see diabatic_potential(), diabatic_force(), diabatic_hesse(),
/// @see coupling, adiabatic_coupling(), force_basis_coupling()
static RealMatrix diabatic_coupling(const double x)
{
    return RealMatrix(NumPES);
}


// Adiabatic representation

/// Transformation matrix from diabatic matrix to adiabatic one
///
/// i.e. C^T*M(dia)*C=M(adia), which diagonalize PES only.
///
/// This is the transformation matrix at a certain place.
///
/// @param x the bath DOF position
/// @return the transformation matrix at this position, the C matrix, which is real orthogonal
/// @see diabatic_to_force_basis_at_one_point(),
/// @see basis_transform, diabatic_to_adiabatic(), adiabatic_to_diabatic(),
/// @see adiabatic_potential(), adiabatic_force(), adiabatic_coupling()
static RealMatrix diabatic_to_adiabatic_at_one_point(const double x)
{
    RealMatrix EigVec = diabatic_potential(x);
    double EigVal[NumPES];
    if (LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', NumPES, EigVec.data(), NumPES, EigVal) != 0)
    {
        cerr << "UNABLE TO CALCULATE ADIABATIC REPRESENTATION AT " << x << endl;
        exit(200);
    }
    return EigVec;
}

/// The eigenvalues of the diabatic potential
///
/// i.e., the diagonalized potential matrix
///
/// @param x the bath DOF position
/// @return the adiabatic potential matrix at this position, which is real diagonal
/// @see diabatic_to_adiabatic_at_one_point(),
/// @see potential, diabatic_potential(), force_basis_potential(),
/// @see adiabatic_force(), adiabatic_coupling()
static RealMatrix adiabatic_potential(const double x)
{
    RealMatrix EigVec = diabatic_potential(x);
    double EigVal[NumPES];
    if (LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'N', 'U', NumPES, EigVec.data(), NumPES, EigVal) != 0)
    {
        cerr << "UNABLE TO CALCULATE ADIABATIC HAMILTONIAN AT " << x << endl;
        exit(201);
    }
    RealMatrix result(NumPES);
    for (int i = 0; i < NumPES; i++)
    {
        result[i][i] = EigVal[i];
    }
    return result;    
}

/// The adiabatic force calculated from diabatic force
///
/// i.e., F(adia)=C^T*F(dia)^C, C the transformation matrix
///
/// @param x the bath DOF position
/// @return the adiabatic force matrix at this position, which is real symmetric
/// @see diabatic_to_adiabatic_at_one_point(),
/// @see force, diabatic_force(), force_basis_force(),
/// @see adiabatic_potential(), adiabatic_coupling()
static RealMatrix adiabatic_force(const double x)
{
    const RealMatrix TransformMatrix = diabatic_to_adiabatic_at_one_point(x);
    double FoEig[NumPES * NumPES];
    RealMatrix result(NumPES);
    cblas_dsymm(CblasRowMajor, CblasLeft, CblasUpper, NumPES, NumPES, 1.0, diabatic_force(x).data(), NumPES, TransformMatrix.data(), NumPES, 0.0, FoEig, NumPES);
    cblas_dgemm(CblasRowMajor, CblasConjTrans, CblasNoTrans, NumPES, NumPES, NumPES, 1.0, TransformMatrix.data(), NumPES, FoEig, NumPES, 0.0, result.data(), NumPES);
    result.symmetrize();
    return result;
}

/// The Non-Adiabatic Coupling (NAC) matrix under adiabatic representation
///
/// i.e. the D matrix, whose element is dij=Fij/(Ei-Ej), F the force, E the potential
///
/// @param x the bath DOF position
/// @return the adiabatic NAC matrix at this position, which is real anti-symmetric
/// @see coupling, diabatic_coupling(), force_basis_coupling(),
/// @see adiabatic_potential(), adiabatic_force()
static RealMatrix adiabatic_coupling(const double x)
{
    RealMatrix result(NumPES);
    const RealMatrix Force = adiabatic_force(x), Energy = adiabatic_potential(x);
    for (int i = 0; i < NumPES; i++)
    {
        for (int j = 0; j < NumPES; j++)
        {
            if (j < i)
            {
                result[i][j] = -result[j][i];
            }
            else if (j == i)
            {
                result[i][j] = 0;
            }
            else
            {
                result[i][j] = Force[i][j] / (Energy[i][i] - Energy[j][j]);
            }
            
        }
    }
    return result;
}


// Force basis

/// Transformation matrix from diabatic matrix to force basis one
///
/// i.e. C^T*M(dia)*C=M(force), which diagonalize the force matrix only.
///
/// This is the transformation matrix at a certain place.
///
/// @param x the bath DOF position
/// @return the transformation matrix at this position, the C matrix, which is real orthogonal
/// @see diabatic_to_adiabatic_at_one_point(),
/// @see basis_transform, diabatic_to_force_basis(), force_basis_to_diabatic(),
/// @see force_basis_potential(), force_basis_force(), force_basis_hesse(), force_basis_coupling()
static RealMatrix diabatic_to_force_basis_at_one_point(const double x)
{
    RealMatrix EigVec = diabatic_force(x);
    double EigVal[NumPES];
    if (LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', NumPES, EigVec.data(), NumPES, EigVal) != 0)
    {
        cerr << "UNABLE TO CALCULATE ADIABATIC REPRESENTATION AT " << x << endl;
        exit(202);
    }
    return EigVec;
}

/// The force basis potential calculated from diabatic force
///
/// i.e., V(force)=C^T*V(dia)^C, C the transformation matrix
///
/// @param x the bath DOF position
/// @return the force basis potential matrix at this position, which is real symmetric
/// @see diabatic_to_force_basis_at_one_point(),
/// @see potential, diabatic_potential(), adiabatic_potential(),
/// @see force_basis_force(), force_basis_hesse(), force_basis_coupling()
static RealMatrix force_basis_potential(const double x)
{
    const RealMatrix TransformMatrix = diabatic_to_force_basis_at_one_point(x);
    double VEig[NumPES * NumPES];
    RealMatrix result(NumPES);
    cblas_dsymm(CblasRowMajor, CblasLeft, CblasUpper, NumPES, NumPES, 1.0, diabatic_potential(x).data(), NumPES, TransformMatrix.data(), NumPES, 0.0, VEig, NumPES);
    cblas_dgemm(CblasRowMajor, CblasConjTrans, CblasNoTrans, NumPES, NumPES, NumPES, 1.0, TransformMatrix.data(), NumPES, VEig, NumPES, 0.0, result.data(), NumPES);
    result.symmetrize();
    return result;
}

/// The eigenvalues of the diabatic force
///
/// i.e., the diagonalized force matrix
///
/// @param x the bath DOF position
/// @return the force basis force matrix at this position, which is real diagonal
/// @see diabatic_to_force_basis_at_one_point(),
/// @see force, diabatic_force(), adiabatic_force(),
/// @see force_basis_potential(), force_basis_hesse(), force_basis_coupling()
static RealMatrix force_basis_force(const double x)
{
    RealMatrix EigVec = diabatic_force(x);
    double EigVal[NumPES];
    if (LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'N', 'U', NumPES, EigVec.data(), NumPES, EigVal) != 0)
    {
        cerr << "UNABLE TO CALCULATE ADIABATIC REPRESENTATION AT " << x << endl;
        exit(203);
    }
    RealMatrix result(NumPES);
    for (int i = 0; i < NumPES; i++)
    {
        result[i][i] = EigVal[i];
    }
    return result;
}

/// The force basis hesse calculated from diabatic hesse
///
/// i.e., He(force)=C^T*He(dia)^C, C the transformation matrix
///
/// @param x the bath DOF position
/// @return the force basis hessian matrix at this position, which is real symmetric
/// @see diabatic_to_force_basis_at_one_point(), diabatic_hesse(),
/// @see force_basis_potential(), force_basis_force(), force_basis_coupling()
static RealMatrix force_basis_hesse(const double x)
{
    const RealMatrix TransformMatrix = diabatic_to_force_basis_at_one_point(x);
    double HeEig[NumPES * NumPES];
    RealMatrix result(NumPES);
    cblas_dsymm(CblasRowMajor, CblasLeft, CblasUpper, NumPES, NumPES, 1.0, diabatic_hesse(x).data(), NumPES, TransformMatrix.data(), NumPES, 0.0, HeEig, NumPES);
    cblas_dgemm(CblasRowMajor, CblasConjTrans, CblasNoTrans, NumPES, NumPES, NumPES, 1.0, TransformMatrix.data(), NumPES, HeEig, NumPES, 0.0, result.data(), NumPES);
    result.symmetrize();
    return result;
}

/// The Non-Adiabatic Coupling (NAC) matrix under force basis representation
///
/// i.e. the D matrix, whose element is dij=He[i][j]/(Fi-Fj), He the hesse, F the force
///
/// @param x the bath DOF position
/// @return the force basis NAC matrix at this position, which is real anti-symmetric
/// @see coupling, diabatic_coupling(), adiabatic_coupling(),
/// @see force_basis_force(), force_basis_hesse()
static RealMatrix force_basis_coupling(const double x)
{
    RealMatrix result(NumPES);
    const RealMatrix Hesse = force_basis_hesse(x), Force = force_basis_force(x);
    for (int i = 0; i < NumPES; i++)
    {
        for (int j = 0; j < NumPES; j++)
        {
            if (j < i)
            {
                result[i][j] = -result[j][i];
            }
            else if (j == i)
            {
                result[i][j] = 0;
            }
            else
            {
                result[i][j] = Hesse[i][j] / (Force[i][i] - Force[j][j]);
            }
            
        }
    }
    return result;
}


// transform the whole matrix

/// Transform from one basis to the same basis, so doing nothing
/// @param rho the grided density matrix
/// @param NGrids the number of grids on one dimention (overall NGrids^2 sub-DM in rho)
/// @param GridPosition the array saved the position of each grid, i.e., x_i
/// @see basis_transform,
/// @see diabatic_to_adiabatic(), diabatic_to_force_basis(),
/// @see adiabatic_to_diabatic(), adiabatic_to_force_basis(),
/// @see force_basis_to_diabatic(), force_basis_to_adiabatic()
static void do_nothing(ComplexMatrixMatrix& rho, const int NGrids, const double* const GridPosition)
{
}

/// Transform from diabatic basis to adiabatic basis
///
/// The transformation matrices are saved for less diagonalization
///
/// @param rho the grided diabatic density matrix
/// @param NGrids the number of grids on one dimention (overall NGrids^2 sub-DM in rho)
/// @param GridPosition the array saved the position of each grid, i.e., x_i
/// @see basis_transform, do_nothing(), adiabatic_to_diabatic(),
/// @see diabatic_to_force_basis(), force_basis_to_adiabatic()
static void diabatic_to_adiabatic(ComplexMatrixMatrix& rho, const int NGrids, const double* const GridPosition)
{
    static bool FirstRun = true;
    static ComplexMatrix* TransformMatrices = nullptr;
    static int LastNGrids = NGrids;
    static const double* LastGridPosition = GridPosition;
    static allocator<ComplexMatrix> MatrixAllocator;
    if (FirstRun == true)
    {
        // for the first run, do diagonalization
        TransformMatrices = MatrixAllocator.allocate(NGrids);
#pragma omp parallel for default(none) shared(TransformMatrices) schedule(static)
        for (int i = 0; i < NGrids; i++)
        {
            const ComplexMatrix TransformMatrix = diabatic_to_adiabatic_at_one_point(GridPosition[i]);
            uninitialized_copy(&TransformMatrix, &TransformMatrix + 1, TransformMatrices + i);
        }
        FirstRun = false;
    }
    else if(NGrids != LastNGrids || LastGridPosition != GridPosition)
    {
        // changed case, re-diagonalization
        destroy(TransformMatrices, TransformMatrices + LastNGrids);
        if (NGrids != LastNGrids)
        {
            MatrixAllocator.deallocate(TransformMatrices, LastNGrids);
            TransformMatrices = MatrixAllocator.allocate(NGrids);
        }
#pragma omp parallel for default(none) shared(TransformMatrices) schedule(static)
        for (int i = 0; i < NGrids; i++)
        {
            const ComplexMatrix TransformMatrix = diabatic_to_adiabatic_at_one_point(GridPosition[i]);
            uninitialized_copy(&TransformMatrix, &TransformMatrix + 1, TransformMatrices + i);
        }
        LastNGrids = NGrids;
        LastGridPosition = GridPosition;
    }
    // rho(adia)=C^T*rho(dia)*C
    Complex MatTrans[NumPES * NumPES];
#pragma omp parallel for default(none) shared(TransformMatrices, rho) private(MatTrans) schedule(static)
    for (int i = 0; i < NGrids; i++)
    {
        const ComplexMatrix& TransformationMatrix = TransformMatrices[i];
        for (int j = 0; j < NGrids; j++)
        {
            cblas_zhemm(CblasRowMajor, CblasLeft, CblasUpper, NumPES, NumPES, &Alpha, rho[i][j].data(), NumPES, TransformationMatrix.data(), NumPES, &Beta, MatTrans, NumPES);
            cblas_zgemm(CblasRowMajor, CblasConjTrans, CblasNoTrans, NumPES, NumPES, NumPES, &Alpha, TransformationMatrix.data(), NumPES, MatTrans, NumPES, &Beta, rho[i][j].data(), NumPES);
            rho[i][j].hermitize();
        }
    }
}

/// Transform from diabatic basis to force basis
///
/// The transformation matrices are saved for less diagonalization
///
/// @param rho the diabatic grided density matrix
/// @param NGrids the number of grids on one dimention (overall NGrids^2 sub-DM in rho)
/// @param GridPosition the array saved the position of each grid, i.e., x_i
/// @see basis_transform, do_nothing(), force_basis_to_diabatic(),
/// @see diabatic_to_adiabatic(), adiabatic_to_force_basis()
static void diabatic_to_force_basis(ComplexMatrixMatrix& rho, const int NGrids, const double* const GridPosition)
{
    static bool FirstRun = true;
    static ComplexMatrix* TransformMatrices = nullptr;
    static int LastNGrids = NGrids;
    static const double* LastGridPosition = GridPosition;
    static allocator<ComplexMatrix> MatrixAllocator;
    if (FirstRun == true)
    {
        // for the first run, do diagonalization
        TransformMatrices = MatrixAllocator.allocate(NGrids);
#pragma omp parallel for default(none) shared(TransformMatrices) schedule(static)
        for (int i = 0; i < NGrids; i++)
        {
            const ComplexMatrix TransformMatrix = diabatic_to_force_basis_at_one_point(GridPosition[i]);
            uninitialized_copy(&TransformMatrix, &TransformMatrix + 1, TransformMatrices + i);
        }
        FirstRun = false;
    }
    else if(NGrids != LastNGrids || LastGridPosition != GridPosition)
    {
        // changed case, re-diagonalization
        destroy(TransformMatrices, TransformMatrices + LastNGrids);
        if (NGrids != LastNGrids)
        {
            MatrixAllocator.deallocate(TransformMatrices, LastNGrids);
            TransformMatrices = MatrixAllocator.allocate(NGrids);
        }
#pragma omp parallel for default(none) shared(TransformMatrices) schedule(static)
        for (int i = 0; i < NGrids; i++)
        {
            const ComplexMatrix TransformMatrix = diabatic_to_force_basis_at_one_point(GridPosition[i]);
            uninitialized_copy(&TransformMatrix, &TransformMatrix + 1, TransformMatrices + i);
        }
        LastNGrids = NGrids;
        LastGridPosition = GridPosition;
    }
    // rho(force)=C^T*rho(dia)*C
    Complex MatTrans[NumPES * NumPES];
#pragma omp parallel for default(none) shared(TransformMatrices, rho) private(MatTrans) schedule(static)
    for (int i = 0; i < NGrids; i++)
    {
        const ComplexMatrix& TransformationMatrix = TransformMatrices[i];
        for (int j = 0; j < NGrids; j++)
        {
            cblas_zhemm(CblasRowMajor, CblasLeft, CblasUpper, NumPES, NumPES, &Alpha, rho[i][j].data(), NumPES, TransformationMatrix.data(), NumPES, &Beta, MatTrans, NumPES);
            cblas_zgemm(CblasRowMajor, CblasConjTrans, CblasNoTrans, NumPES, NumPES, NumPES, &Alpha, TransformationMatrix.data(), NumPES, MatTrans, NumPES, &Beta, rho[i][j].data(), NumPES);
            rho[i][j].hermitize();
        }
    }
}

/// Transform from adiabatic basis to diabatic basis
///
/// The transformation matrices are saved for less diagonalization
///
/// @param rho the adiabatic grided density matrix
/// @param NGrids the number of grids on one dimention (overall NGrids^2 sub-DM in rho)
/// @param GridPosition the array saved the position of each grid, i.e., x_i
/// @see basis_transform, do_nothing(), diabatic_to_adiabatic(),
/// @see adiabatic_to_force_basis(), force_basis_to_diabatic()
static void adiabatic_to_diabatic(ComplexMatrixMatrix& rho, const int NGrids, const double* const GridPosition)
{
    static bool FirstRun = true;
    static ComplexMatrix* TransformMatrices = nullptr;
    static int LastNGrids = NGrids;
    static const double* LastGridPosition = GridPosition;
    static allocator<ComplexMatrix> MatrixAllocator;
    if (FirstRun == true)
    {
        // for the first run, do diagonalization
        TransformMatrices = MatrixAllocator.allocate(NGrids);
#pragma omp parallel for default(none) shared(TransformMatrices) schedule(static)
        for (int i = 0; i < NGrids; i++)
        {
            const ComplexMatrix TransformMatrix = diabatic_to_adiabatic_at_one_point(GridPosition[i]);
            uninitialized_copy(&TransformMatrix, &TransformMatrix + 1, TransformMatrices + i);
        }
        FirstRun = false;
    }
    else if(NGrids != LastNGrids || LastGridPosition != GridPosition)
    {
        // changed case, re-diagonalization
        destroy(TransformMatrices, TransformMatrices + LastNGrids);
        if (NGrids != LastNGrids)
        {
            MatrixAllocator.deallocate(TransformMatrices, LastNGrids);
            TransformMatrices = MatrixAllocator.allocate(NGrids);
        }
#pragma omp parallel for default(none) shared(TransformMatrices) schedule(static)
        for (int i = 0; i < NGrids; i++)
        {
            const ComplexMatrix TransformMatrix = diabatic_to_adiabatic_at_one_point(GridPosition[i]);
            uninitialized_copy(&TransformMatrix, &TransformMatrix + 1, TransformMatrices + i);
        }
        LastNGrids = NGrids;
        LastGridPosition = GridPosition;
    }
    // rho(dia)=C*rho(adia)*C^T
    Complex TransMat[NumPES * NumPES];
#pragma omp parallel for default(none) shared(TransformMatrices, rho) private(TransMat) schedule(static)
    for (int i = 0; i < NGrids; i++)
    {
        const ComplexMatrix& TransformationMatrix = TransformMatrices[i];
        for (int j = 0; j < NGrids; j++)
        {
            cblas_zhemm(CblasRowMajor, CblasRight, CblasUpper, NumPES, NumPES, &Alpha, rho[i][j].data(), NumPES, TransformationMatrix.data(), NumPES, &Beta, TransMat, NumPES);
            cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasConjTrans, NumPES, NumPES, NumPES, &Alpha, TransMat, NumPES, TransformationMatrix.data(), NumPES, &Beta, rho[i][j].data(), NumPES);
            rho[i][j].hermitize();
        }
    }
}

/// Transform from adiabatic basis to force basis
///
/// which is done by calling existing functions
///
/// @param rho the adiabatic grided density matrix
/// @param NGrids the number of grids on one dimention (overall NGrids^2 sub-DM in rho)
/// @param GridPosition the array saved the position of each grid, i.e., x_i
/// @see basis_transform, do_nothing(), force_basis_to_adiabatic(),
/// @see adiabatic_to_diabatic(), diabatic_to_force_basis()
static void adiabatic_to_force_basis(ComplexMatrixMatrix& rho, const int NGrids, const double* const GridPosition)
{
    adiabatic_to_diabatic(rho, NGrids, GridPosition);
    diabatic_to_force_basis(rho, NGrids, GridPosition);
}

/// Transform from force basis to diabatic basis
///
/// The transformation matrices are saved for less diagonalization
///
/// @param rho the force basis grided density matrix
/// @param NGrids the number of grids on one dimention (overall NGrids^2 sub-DM in rho)
/// @param GridPosition the array saved the position of each grid, i.e., x_i
/// @see basis_transform, do_nothing(), diabatic_to_force_basis(),
/// @see force_basis_to_adiabatic(), adiabatic_to_diabatic()
static void force_basis_to_diabatic(ComplexMatrixMatrix& rho, const int NGrids, const double* const GridPosition)
{
    static bool FirstRun = true;
    static ComplexMatrix* TransformMatrices = nullptr;
    static int LastNGrids = NGrids;
    static const double* LastGridPosition = GridPosition;
    static allocator<ComplexMatrix> MatrixAllocator;
    if (FirstRun == true)
    {
        // for the first run, do diagonalization
        TransformMatrices = MatrixAllocator.allocate(NGrids);
#pragma omp parallel for default(none) shared(TransformMatrices) schedule(static)
        for (int i = 0; i < NGrids; i++)
        {
            const ComplexMatrix TransformMatrix = diabatic_to_force_basis_at_one_point(GridPosition[i]);
            uninitialized_copy(&TransformMatrix, &TransformMatrix + 1, TransformMatrices + i);
        }
        FirstRun = false;
    }
    else if(NGrids != LastNGrids || LastGridPosition != GridPosition)
    {
        // changed case, re-diagonalization
        destroy(TransformMatrices, TransformMatrices + LastNGrids);
        if (NGrids != LastNGrids)
        {
            MatrixAllocator.deallocate(TransformMatrices, LastNGrids);
            TransformMatrices = MatrixAllocator.allocate(NGrids);
        }
#pragma omp parallel for default(none) shared(TransformMatrices) schedule(static)
        for (int i = 0; i < NGrids; i++)
        {
            const ComplexMatrix TransformMatrix = diabatic_to_force_basis_at_one_point(GridPosition[i]);
            uninitialized_copy(&TransformMatrix, &TransformMatrix + 1, TransformMatrices + i);
        }
        LastNGrids = NGrids;
        LastGridPosition = GridPosition;
    }
    // rho(dia)=C*rho(force)*C^T
    Complex TransMat[NumPES * NumPES];
#pragma omp parallel for default(none) shared(TransformMatrices, rho) private(TransMat) schedule(static)
    for (int i = 0; i < NGrids; i++)
    {
        const ComplexMatrix& TransformationMatrix = TransformMatrices[i];
        for (int j = 0; j < NGrids; j++)
        {
            cblas_zhemm(CblasRowMajor, CblasRight, CblasUpper, NumPES, NumPES, &Alpha, rho[i][j].data(), NumPES, TransformationMatrix.data(), NumPES, &Beta, TransMat, NumPES);
            cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasConjTrans, NumPES, NumPES, NumPES, &Alpha, TransMat, NumPES, TransformationMatrix.data(), NumPES, &Beta, rho[i][j].data(), NumPES);
            rho[i][j].hermitize();
        }
    }
}

/// Transform from force basis to adiabatic basis
///
/// which is done by calling existing functions
///
/// @param rho the force basis grided density matrix
/// @param NGrids the number of grids on one dimention (overall NGrids^2 sub-DM in rho)
/// @param GridPosition the array saved the position of each grid, i.e., x_i
/// @see basis_transform, do_nothing(), adiabatic_to_force_basis(),
/// @see force_basis_to_diabatic(), diabatic_to_adiabatic()
static void force_basis_to_adiabatic(ComplexMatrixMatrix& rho, const int NGrids, const double* const GridPosition)
{
    force_basis_to_diabatic(rho, NGrids, GridPosition);
    diabatic_to_adiabatic(rho, NGrids, GridPosition);
}


// save in array
extern const function<RealMatrix(const double)> potential[NoBasis] = { diabatic_potential, adiabatic_potential, force_basis_potential }; ///< function object: potential (V of environment, H of subsystem); saved in array
extern const function<RealMatrix(const double)> force[NoBasis] = { diabatic_force, adiabatic_force, force_basis_force }; ///< function object: force (F=-dV/dR); saved in array
extern const function<RealMatrix(const double)> coupling[NoBasis] = { diabatic_coupling, adiabatic_coupling, force_basis_coupling }; ///< function object: non-adiabatic coupling (dij=<i|d/dR|j>); saved in array

RealMatrix** calculate_potential_on_grids(const int NGrids, const double* const GridPosition)
{
    allocator<RealMatrix> MatrixAllocator;
    RealMatrix** result = new RealMatrix * [NoBasis];
    for (int i = 0; i < NoBasis; i++)
    {
        result[i] = MatrixAllocator.allocate(NGrids);
#pragma omp parallel for default(none) shared(result, i) schedule(static)
        for (int j = 0; j < NGrids; j++)
        {
            const double& x = GridPosition[j];
            const RealMatrix H = potential[i](x);
            uninitialized_copy(&H, &H + 1, result[i] + j);
        }
    }
    return result;
}

RealMatrix** calculate_force_on_grids(const int NGrids, const double* const GridPosition)
{
    allocator<RealMatrix> MatrixAllocator;
    RealMatrix** result = new RealMatrix * [NoBasis];
    for (int i = 0; i < NoBasis; i++)
    {
        result[i] = MatrixAllocator.allocate(NGrids);
#pragma omp parallel for default(none) shared(result, i) schedule(static)
        for (int j = 0; j < NGrids; j++)
        {
            const double& x = GridPosition[j];
            const RealMatrix F = force[i](x);
            uninitialized_copy(&F, &F + 1, result[i] + j);
        }
    }
    return result;
}

RealMatrix** calculate_coupling_on_grids(const int NGrids, const double* const GridPosition)
{
    allocator<RealMatrix> MatrixAllocator;
    RealMatrix** result = new RealMatrix * [NoBasis];
    for (int i = 0; i < NoBasis; i++)
    {
        result[i] = MatrixAllocator.allocate(NGrids);
#pragma omp parallel for default(none) shared(result, i) schedule(static)
        for (int j = 0; j < NGrids; j++)
        {
            const double& x = GridPosition[j];
            const RealMatrix D = coupling[i](x);
            uninitialized_copy(&D, &D + 1, result[i] + j);
        }
    }
    return result;
}

extern const function<void(ComplexMatrixMatrix&, int, const double* const)> basis_transform[NoBasis][NoBasis] = 
{
    { do_nothing, diabatic_to_adiabatic, diabatic_to_force_basis },
    { adiabatic_to_diabatic, do_nothing, adiabatic_to_force_basis },
    { force_basis_to_diabatic, force_basis_to_adiabatic, do_nothing }
}; ///< function object: basis transformation matrices; saved in 2d array
