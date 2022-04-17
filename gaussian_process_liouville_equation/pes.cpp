/// @file pes.cpp
/// @brief Implementation of pes.h: diabatic PES and absorbing potential

#include "stdafx.h"

#include "pes.h"


// Diabatic reprersentation
// parameters of Tully's 1st model, Simple Avoided Crossing (SAC)
static constexpr double SAC_A = 0.01;  ///< A in SAC model
static constexpr double SAC_B = 1.6;   ///< B in SAC model
static constexpr double SAC_C = 0.005; ///< C in SAC model
static constexpr double SAC_D = 1.0;   ///< D in SAC model
// parameters of Tully's 2nd model, Dual Avoided Crossing (DAC)
static constexpr double DAC_A = 0.10;  ///< A in DAC model
static constexpr double DAC_B = 0.28;  ///< B in DAC model
static constexpr double DAC_C = 0.015; ///< C in DAC model
static constexpr double DAC_D = 0.06;  ///< D in DAC model
static constexpr double DAC_E = 0.05;  ///< E in DAC model
// parameters of Tully's 3rd model, Extended Coupling with Reflection (ECR)
static constexpr double ECR_A = 6e-4; ///< A in ECR model
static constexpr double ECR_B = 0.10; ///< B in ECR model
static constexpr double ECR_C = 0.90; ///< C in ECR model

/// @brief Subsystem diabatic Hamiltonian, being the potential of the bath
/// @param[in] x Position of classical degree of freedom
/// @return The potential matrix, which is real symmetric
/// @sa diabatic_force()
static QuantumMatrix<double> diabatic_potential(const ClassicalVector<double>& x)
{
	QuantumMatrix<double> Potential;
	switch (TestModel)
	{
	case SAC: // Tully's 1st model
		Potential(0, 0) = sgn(x[0]) * SAC_A * (1.0 - std::exp(-sgn(x[0]) * SAC_B * x[0]));
		Potential(1, 1) = -Potential(0, 0);
		Potential(0, 1) = Potential(1, 0) = SAC_C * std::exp(-SAC_D * power<2>(x[0]));
		break;
	case DAC: // Tully's 2nd model
		Potential(0, 0) = 0;
		Potential(1, 1) = DAC_E - DAC_A * std::exp(-DAC_B * power<2>(x[0]));
		Potential(0, 1) = Potential(1, 0) = DAC_C * std::exp(-DAC_D * power<2>(x[0]));
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
/// @param[in] x Position of classical degree of freedom
/// @return The force tensor, which is real symmetric for each element
/// @sa diabatic_potential()
static Tensor3d diabatic_force(const ClassicalVector<double>& x)
{
	Tensor3d Force;
	switch (TestModel)
	{
	case SAC: // Tully's 1st model
		Force[0](0, 0) = -SAC_A * SAC_B * std::exp(-sgn(x[0]) * SAC_B * x[0]);
		Force[0](1, 1) = -Force[0](0, 0);
		Force[0](0, 1) = Force[0](1, 0) = 2.0 * SAC_C * SAC_D * x[0] * std::exp(-SAC_D * power<2>(x[0]));
		break;
	case DAC: // Tully's 2nd model
		Force[0](0, 0) = 0;
		Force[0](1, 1) = -2 * DAC_A * DAC_B * x[0] * std::exp(-DAC_B * power<2>(x[0]));
		Force[0](0, 1) = Force[0](1, 0) = 2 * DAC_C * DAC_D * x[0] * std::exp(-DAC_D * power<2>(x[0]));
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
/// @param[in] x Position of classical degree of freedom
/// @return The transformation matrix at this position, the C matrix, which is real orthogonal
/// @sa adiabatic_to_force_basis_matrix(),
/// @sa adiabatic_potential(), adiabatic_force(), adiabatic_coupling()
/// @details The return matrix @f$ C @f$ following
/// @f$ C^{\mathsf{T}}M_{\mathrm{dia}}C=M_{\mathrm{adia}} @f$,
/// which diagonalizes the subsystem Hamiltonian (PES) only
/// and is the transformation matrix at a certain position x.
static QuantumMatrix<double> diabatic_to_adiabatic_matrix(const ClassicalVector<double>& x)
{
	const Eigen::SelfAdjointEigenSolver<QuantumMatrix<double>> solver(diabatic_potential(x));
	return solver.eigenvectors();
}

/// @sa diabatic_to_adiabatic_matrix(), diabatic_potential()
/// @details This function calculates from the diabatic potential by calculating the eigenvalues as the diagonal elements
QuantumVector<double> adiabatic_potential(const ClassicalVector<double>& x)
{
	const Eigen::SelfAdjointEigenSolver<QuantumMatrix<double>> solver(diabatic_potential(x));
	return solver.eigenvalues();
}

/// @sa diabatic_to_adiabatic_matrix(), diabatic_force(), adiabatic_potential()
/// @details This function calculates from diabatic force. On each direction,
/// @f$ F_{\mathrm{adia}}=C^{\mathsf{T}}F_{\mathrm{dia}}C @f$,
/// where C is the transformation matrix
Tensor3d adiabatic_force(const ClassicalVector<double>& x)
{
	Tensor3d AdiaForce;
	const QuantumMatrix<double>& TransformMatrix = diabatic_to_adiabatic_matrix(x);
	const Tensor3d& DiaForce = diabatic_force(x);
	for (std::size_t i = 0; i < Dim; i++)
	{
		AdiaForce[i] = (TransformMatrix.adjoint() * DiaForce[i] * TransformMatrix).selfadjointView<Eigen::Lower>();
	}
	return AdiaForce;
}

/// @sa adiabatic_potential(), adiabatic_coupling()
/// @details This function calculates the D matrix, whose element is @f$ d^{\alpha}_{ij}=F^{\alpha}_{ij}/(E_i-E_j) @f$
/// where @f$ F^{\alpha} @f$ is the force on the alpha classical dimension, and E the potential
Tensor3d adiabatic_coupling(const ClassicalVector<double>& x)
{
	Tensor3d NAC;
	const Eigen::VectorXd& E = adiabatic_potential(x);
	const Tensor3d& F = adiabatic_force(x);
	for (std::size_t i = 0; i < Dim; i++)
	{
		NAC[i] = QuantumMatrix<double>::Zero();
		for (std::size_t j = 1; j < NumPES; j++)
		{
			for (std::size_t k = 0; k < j; k++)
			{
				NAC[i](j, k) = F[i](j, k) / (E[j] - E[k]);
			}
		}
		NAC[i] = NAC[i].triangularView<Eigen::StrictlyLower>();
	}
	return NAC;
}

ClassicalVector<double> tensor_slice(const Tensor3d& tensor, const std::size_t row, const std::size_t col)
{
	ClassicalVector<double> result;
	for (std::size_t iDim = 0; iDim < Dim; iDim++)
	{
		result[iDim] = tensor[iDim](row, col);
	}
	return result;
}

// off-diagonal force basis
/// @brief Transformation matrix from diabatic representation to force basis
/// @param[in] x Position of classical degree of freedom
/// @param[in] idx The index of the classical dimension where the force matrix is diagonal
/// @return The transformation matrix at this position, the C matrix, which is real orthogonal
/// @sa basis_transform(), diabatic_to_adiabatic_matrix(), force_basis_force()
/// @details The return matrix @f$ C @f$ following
/// @f$ C^{\mathsf{T}}M_{\mathrm{adia}}C=M_{\mathrm{force }i} @f$,
/// which diagonalizes force on the idx-th direction,
/// and is the transformation matrix at a certain position x.
static QuantumMatrix<double> adiabatic_to_force_basis_matrix(const ClassicalVector<double>& x, const std::size_t idx)
{
	const Eigen::SelfAdjointEigenSolver<QuantumMatrix<double>> solver(adiabatic_force(x)[idx]);
	return solver.eigenvectors();
}

/// @sa adiabatic_force(), adiabatic_to_force_basis_matrix()
/// @details This function calculates the force of the given direction by
/// diagonalization of the corresponding (idx) direction of the adiabatic force.
QuantumVector<double> force_basis_force(const ClassicalVector<double>& x, const std::size_t idx)
{
	const Eigen::SelfAdjointEigenSolver<QuantumMatrix<double>> solver(adiabatic_force(x)[idx]);
	return solver.eigenvalues();
}


// basis transformation

/// @brief To make the complex matrix self-adjoint
/// @param[inout] mat The matrix
static inline void make_self_adjoint(QuantumMatrix<std::complex<double>>& mat)
{
	mat.diagonal() = mat.diagonal().real();
	mat = mat.selfadjointView<Eigen::Lower>();
}

/// @sa diabatic_to_adiabatic_matrix(), adiabatic_to_force_basis_matrix()
/// @details The indices (to and from) are in the range [0, Dim]. [0, Dim) indicates
/// it is one of the force basis, and idx == Dim indicates adiabatic basis.
QuantumMatrix<std::complex<double>> basis_transform(
	const QuantumMatrix<std::complex<double>>& rho,
	const ClassicalVector<double>& x,
	const std::size_t idx_from,
	const std::size_t idx_to)
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
