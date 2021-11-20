/// @file pes.cpp
/// @brief Implementation of pes.h: diabatic PES and absorbing potential

#include "stdafx.h"

#include "pes.h"

/// Sign function: return -1 for negative, 1 for positive, and 0 for 0
/// @param val A value of any type that have '<' and '>' and could construct 0.0
/// @return The sign of the value
template <typename valtype>
static inline int sgn(const valtype& val)
{
	return (val > valtype(0.0)) - (val < valtype(0.0));
}


// Diabatic reprersentation
// parameters of Tully's 1st model, Simple Avoided Crossing (SAC)
static const double SAC_A = 0.01;  ///< A in SAC model
static const double SAC_B = 1.6;   ///< B in SAC model
static const double SAC_C = 0.005; ///< C in SAC model
static const double SAC_D = 1.0;   ///< D in SAC model
// parameters of Tully's 2nd model, Dual Avoided Crossing (DAC)
static const double DAC_A = 0.10;  ///< A in DAC model
static const double DAC_B = 0.28;  ///< B in DAC model
static const double DAC_C = 0.015; ///< C in DAC model
static const double DAC_D = 0.06;  ///< D in DAC model
static const double DAC_E = 0.05;  ///< E in DAC model
// parameters of Tully's 3rd model, Extended Coupling with Reflection (ECR)
static const double ECR_A = 6e-4; ///< A in ECR model
static const double ECR_B = 0.10; ///< B in ECR model
static const double ECR_C = 0.90; ///< C in ECR model

/// @brief Subsystem diabatic Hamiltonian, being the potential of the bath
/// @param x Position of classical degree of freedom
/// @return The potential matrix, which is real symmetric
/// @see diabatic_force(), diabatic_hessian(), diabatic_coupling(),
/// @see potential, adiabatic_potential(), force_basis_potential()
static QuantumMatrix<double> diabatic_potential(const ClassicalVector<double>& x)
{
	QuantumMatrix<double> Potential;
	switch (TestModel)
	{
	case SAC: // Tully's 1st model
		Potential(0, 0) = sgn(x[0]) * SAC_A * (1.0 - std::exp(-sgn(x[0]) * SAC_B * x[0]));
		Potential(1, 1) = -Potential(0, 0);
		Potential(0, 1) = Potential(1, 0) = SAC_C * std::exp(-SAC_D * x[0] * x[0]);
		break;
	case DAC: // Tully's 2nd model
		Potential(0, 0) = 0;
		Potential(1, 1) = DAC_E - DAC_A * std::exp(-DAC_B * x[0] * x[0]);
		Potential(0, 1) = Potential(1, 0) = DAC_C * std::exp(-DAC_D * x[0] * x[0]);
		break;
	case ECR: // Tully's 3rd model
		Potential(0, 0) = ECR_A;
		Potential(1, 1) = -ECR_A;
		Potential(0, 1) = Potential(1, 0) = ECR_B * (1 - sgn(x[0]) * (std::exp(-sgn(x[0]) * ECR_C * x[0]) - 1));
		break;
	}
	return Potential;
}

/// @brief Diabatic force, the analytical derivative (F=-dH/dR=-dV/dR)
/// @param x Position of classical degree of freedom
/// @return The force tensor, which is real symmetric for each element
/// @see diabatic_potential(), diabatic_hessian(), diabatic_coupling(),
/// @see force, adiabatic_force(), force_basis_force()
static Tensor3d diabatic_force(const ClassicalVector<double>& x)
{
	Tensor3d Force;
	switch (TestModel)
	{
	case SAC: // Tully's 1st model
		Force[0](0, 0) = -SAC_A * SAC_B * std::exp(-sgn(x[0]) * SAC_B * x[0]);
		Force[0](1, 1) = -Force[0](0, 0);
		Force[0](0, 1) = Force[0](1, 0) = 2.0 * SAC_C * SAC_D * x[0] * std::exp(-SAC_D * x[0] * x[0]);
		break;
	case DAC: // Tully's 2nd model
		Force[0](0, 0) = 0;
		Force[0](1, 1) = -2 * DAC_A * DAC_B * x[0] * std::exp(-DAC_B * x[0] * x[0]);
		Force[0](0, 1) = Force[0](1, 0) = 2 * DAC_C * DAC_D * x[0] * std::exp(-DAC_D * x[0] * x[0]);
		break;
	case ECR: // Tully's 3rd model
		Force[0](0, 0) = Force[0](1, 1) = 0;
		Force[0](0, 1) = Force[0](1, 0) = -ECR_B * ECR_C * std::exp(-sgn(x[0]) * ECR_C * x[0]);
		break;
	}
	return Force;
}


// Adiabatic representation
/// @brief Transformation matrix from diabatic representation to adiabatic one
/// @param x Position of classical degree of freedom
/// @return The transformation matrix at this position, the C matrix, which is real orthogonal
/// @see basis_transform, diabatic_to_force_basis_matrix(),
/// @see adiabatic_potential(), adiabatic_force(), adiabatic_coupling()
/// @details The return Matrix C following C^T*M(dia)*C=M(adia),
/// which diagonalizes the subsystem Hamiltonian (PES) only
/// and is the transformation matrix at a certain position \vec{x}.
static QuantumMatrix<double> diabatic_to_adiabatic_matrix(const ClassicalVector<double>& x)
{
	const Eigen::SelfAdjointEigenSolver<QuantumMatrix<double>> solver(diabatic_potential(x));
	return solver.eigenvectors();
}

/// @see diabatic_to_adiabatic_matrix, diabatic_potential()
/// @details Calculate from the diabatic potential by calculating the eigenvalues as the diagonal elements
QuantumVector<double> adiabatic_potential(const ClassicalVector<double>& x)
{
	const Eigen::SelfAdjointEigenSolver<QuantumMatrix<double>> solver(diabatic_potential(x));
	return solver.eigenvalues();
}

/// @see diabatic_to_adiabatic_matrix(), diabatic_force()
/// @details Calculate from diabatic force. On each direction, F(adia)=C^T*F(dia)^C, C the transformation matrix
Tensor3d adiabatic_force(const ClassicalVector<double>& x)
{
	Tensor3d AdiaForce;
	const QuantumMatrix<double>& TransformMatrix = diabatic_to_adiabatic_matrix(x);
	const Tensor3d& DiaForce = diabatic_force(x);
	for (int i = 0; i < Dim; i++)
	{
		AdiaForce[i] = (TransformMatrix.adjoint() * DiaForce[i] * TransformMatrix).selfadjointView<Eigen::Lower>();
	}
	return AdiaForce;
}

/// @details Calculate the D matrix, whose element is \f$ d^{\alpha}_{ij}=F^{\alpha}_{ij}/(E_i-E_j) \f$
/// where \f$ F^{\alpha} \f$ is the force on the alpha classical dimension, and E the potential
Tensor3d adiabatic_coupling(const ClassicalVector<double>& x)
{
	Tensor3d NAC;
	const Eigen::VectorXd& E = adiabatic_potential(x);
	const Tensor3d& F = adiabatic_force(x);
	for (int i = 0; i < Dim; i++)
	{
		NAC[i] = QuantumMatrix<double>::Zero();
		for (int j = 1; j < NumPES; j++)
		{
			for (int k = 0; k < j; k++)
			{
				NAC[i](j, k) = F[i](j, k) / (E[j] - E[k]);
			}
		}
		NAC[i] = NAC[i].triangularView<Eigen::StrictlyLower>();
	}
	return NAC;
}

ClassicalVector<double> tensor_slice(const Tensor3d& tensor, const int row, const int col)
{
	ClassicalVector<double> result;
	for (int iDim = 0; iDim < Dim; iDim++)
	{
		result[iDim] = tensor[iDim](row, col);
	}
	return result;
}

// off-diagonal force basis
/// @brief Transformation matrix from diabatic representation to force basis
/// @param x Position of classical degree of freedom
/// @paran idx The index of the classical dimension where the force matrix is diagonal
/// @return The transformation matrix at this position, the C matrix, which is real orthogonal
/// @see basis_transform, diabatic_to_adiabatic_matrix(),
/// @see force_basis_potential(), force_basis_force(), force_basis_hessian()
/// @details The return Matrix C following C^T*M(adia)*C=M(force), which diagonalizes force on the first direction
/// and is the transformation matrix at a certain position \vec{x}.
static QuantumMatrix<double> adiabatic_to_force_basis_matrix(const ClassicalVector<double>& x, const int idx)
{
	const Eigen::SelfAdjointEigenSolver<QuantumMatrix<double>> solver(adiabatic_force(x)[idx]);
	return solver.eigenvectors();
}

/// @see diabatic_force(), adiabatic_force()
///
/// Calculate by diagonalize the first element of the diabatic force.
/// On other directions, F(force)=C^T*F(dia)*C, C thetransformation matrix
QuantumVector<double> force_basis_force(const ClassicalVector<double>& x, const int idx)
{
	const Eigen::SelfAdjointEigenSolver<QuantumMatrix<double>> solver(adiabatic_force(x)[idx]);
	return solver.eigenvalues();
}


// basis transformation
/// @see diabatic_to_adiabatic_matrix(), adiabatic_to_force_basis_matrix()
/// @details The indices (to and from) are in the range [0, Dim]. [0, Dim) indicates
/// it is one of the force basis, and idx == Dim indicates adiabatic basis.
QuantumMatrix<std::complex<double>> basis_transform(
	const QuantumMatrix<std::complex<double>>& rho,
	const ClassicalVector<double>& x,
	const int idx_from,
	const int idx_to)
{
	QuantumMatrix<std::complex<double>> result = rho;
	if (idx_from != idx_to)
	{
		// if the same, then in and out are the same matrix, no transformation needed
		if (idx_from != Representation::Adiabatic)
		{
			// from one of the force basises to adiabatic basis. rho(adia)=C*rho(force)*C^T
			const QuantumMatrix<std::complex<double>> TransformMatrix = adiabatic_to_force_basis_matrix(x, idx_from).cast<std::complex<double>>();
			result = TransformMatrix * result.selfadjointView<Eigen::Lower>() * TransformMatrix.adjoint();
		}
		if (idx_to != Representation::Adiabatic)
		{
			// from adiabatic basis to one of the force basises. rho(force)=C^T*rho(adia)*C
			const QuantumMatrix<std::complex<double>> TransformMatrix = adiabatic_to_force_basis_matrix(x, idx_to).cast<std::complex<double>>();
			result = TransformMatrix.adjoint() * result.selfadjointView<Eigen::Lower>() * TransformMatrix;
		}
	}
	make_self_adjoint(result);
	return result;
}
