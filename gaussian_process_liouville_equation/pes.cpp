/// @file pes.cpp
/// @brief Implementation of pes.h: diabatic PES and absorbing potential

#include "stdafx.h"

#include "pes.h"


// Diabatic reprersentation
// parameters of Tully's 1st model, Simple Avoided Crossing (SAC)
/// @brief A in SAC model
static constexpr double SAC_A = 0.01;
/// @brief B in SAC model
static constexpr double SAC_B = 1.6;
/// @brief C in SAC model
static constexpr double SAC_C = 0.005;
/// @brief D in SAC model
static constexpr double SAC_D = 1.0;
// parameters of Tully's 2nd model, Dual Avoided Crossing (DAC)
/// @brief A in DAC model
static constexpr double DAC_A = 0.10;
/// @brief B in DAC model
static constexpr double DAC_B = 0.28;
/// @brief C in DAC model
static constexpr double DAC_C = 0.015;
/// @brief D in DAC model
static constexpr double DAC_D = 0.06;
/// @brief E in DAC model
static constexpr double DAC_E = 0.05;
// parameters of Tully's 3rd model, Extended Coupling with Reflection (ECR)
/// @brief A in ECR model
static constexpr double ECR_A = 6e-4;
/// @brief B in ECR model
static constexpr double ECR_B = 0.10;
/// @brief C in ECR model
static constexpr double ECR_C = 0.90;

/// @brief Subsystem diabatic Hamiltonian, being the potential of the bath
/// @param[in] x Position of classical degree of freedom
/// @return The potential matrix, which is real symmetric
/// @sa diabatic_force()
static QuantumMatrix<double> diabatic_potential(const ClassicalVector<double>& x)
{
	QuantumMatrix<double> Potential = QuantumMatrix<double>::Zero();
	switch (TestModel)
	{
	case SAC: // Tully's 1st model
		Potential(0, 0) = sgn(x[0]) * SAC_A * (1.0 - std::exp(-sgn(x[0]) * SAC_B * x[0]));
		Potential(1, 1) = -Potential(0, 0);
		Potential(0, 1) = Potential(1, 0) = SAC_C * std::exp(-SAC_D * power<2>(x[0]));
		break;
	case DAC: // Tully's 2nd model
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
	Tensor3d Force = xt::zeros<Tensor3d::value_type>(Tensor3d::shape_type{});
	switch (TestModel)
	{
	case SAC: // Tully's 1st model
		Force(0, 0, 0) = -SAC_A * SAC_B * std::exp(-sgn(x[0]) * SAC_B * x[0]);
		Force(0, 1, 1) = -Force(0, 0, 0);
		Force(0, 0, 1) = Force(0, 1, 0) = 2.0 * SAC_C * SAC_D * x[0] * std::exp(-SAC_D * power<2>(x[0]));
		break;
	case DAC: // Tully's 2nd model
		Force(0, 1, 1) = -2 * DAC_A * DAC_B * x[0] * std::exp(-DAC_B * power<2>(x[0]));
		Force(0, 0, 1) = Force(0, 1, 0) = 2 * DAC_C * DAC_D * x[0] * std::exp(-DAC_D * power<2>(x[0]));
		break;
	case ECR: // Tully's 3rd model
		Force(0, 0, 1) = Force(0, 1, 0) = -ECR_B * ECR_C * std::exp(-sgn(x[0]) * ECR_C * x[0]);
		break;
	}
	return Force;
}


// Adiabatic representation
/// @brief Transformation matrix from diabatic representation to adiabatic one
/// @param[in] x Position of classical degree of freedom
/// @return The transformation matrix at this position, the C matrix, which is real orthogonal
/// @sa adiabatic_potential(), adiabatic_force(), adiabatic_coupling()
/// @details The return matrix @f$ C @f$ following
/// @f$ C^{\mathsf{T}}M_{\mathrm{dia}}C=M_{\mathrm{adia}} @f$,
/// which diagonalizes the subsystem Hamiltonian (PES) only
/// and is the transformation matrix at a certain position x.
static QuantumMatrix<double> diabatic_to_adiabatic_matrix(const ClassicalVector<double>& x)
{
	if constexpr (NumPES == 1)
	{
		return QuantumMatrix<double>::Identity();
	}
	else if constexpr (NumPES == 2)
	{
		const QuantumMatrix<double> diaH = diabatic_potential(x);
		QuantumMatrix<double> result;
		result.row(0) << -1.0, 1.0;
		result.row(0) *= std::sqrt(std::norm(diaH(0, 0) - diaH(1, 1)) + 4.0 * std::norm(diaH(0, 1)));
		result.row(0).array() += diaH(0, 0) - diaH(1, 1);
		result.row(0) /= 2.0 * diaH(1, 0);
		result.row(1).setOnes();
		result.array().rowwise() /= result.array().colwise().norm();
		return result;
	}
	else
	{
		const Eigen::SelfAdjointEigenSolver<QuantumMatrix<double>> solver(diabatic_potential(x));
		return solver.eigenvectors();
	}
}

/// @sa diabatic_to_adiabatic_matrix(), diabatic_potential()
/// @details This function calculates from the diabatic potential by calculating the eigenvalues as the diagonal elements
QuantumVector<double> adiabatic_potential(const ClassicalVector<double>& x)
{
	if constexpr (NumPES == 1)
	{
		return diabatic_potential(x);
	}
	else if constexpr (NumPES == 2)
	{
		const QuantumMatrix<double> diaH = diabatic_potential(x);
		QuantumVector<double> result;
		result << -1.0, 1.0;
		result *= std::sqrt(std::norm(diaH(0, 0) - diaH(1, 1)) + std::norm(2.0 * diaH(1, 0)));
		result.array() += diaH(0, 0) + diaH(1, 1);
		result /= 2.0;
		return result;
	}
	else
	{
		const Eigen::SelfAdjointEigenSolver<QuantumMatrix<double>> solver(diabatic_potential(x));
		return solver.eigenvalues();
	}
}

/// @sa diabatic_to_adiabatic_matrix(), diabatic_force(), adiabatic_potential()
/// @details This function calculates from diabatic force. On each direction,
/// @f$ F_{\mathrm{adia}}=C^{\mathsf{T}}F_{\mathrm{dia}}C @f$,
/// where C is the transformation matrix
Tensor3d adiabatic_force(const ClassicalVector<double>& x)
{
	Tensor3d result = diabatic_force(x);
	const QuantumMatrix<double>& TransformMatrix = diabatic_to_adiabatic_matrix(x);
	for (const std::size_t iDim : std::ranges::iota_view{0ul, Dim})
	{
		// slice the force out using Eigen mapping
		// because both of them are row-major, no stride needed
		Eigen::Map<QuantumMatrix<double>> iDimAdiabaticForceMatrix(xt::view(result, iDim, xt::all(), xt::all()).data());
		// calculate its adiabatic result by matmul
		iDimAdiabaticForceMatrix = (TransformMatrix.adjoint() * iDimAdiabaticForceMatrix * TransformMatrix).selfadjointView<Eigen::Lower>();
	}
	return result;
}

/// @sa adiabatic_potential(), adiabatic_coupling()
/// @details This function calculates the D matrix, whose element is @f$ d^{\alpha}_{ij}=F^{\alpha}_{ij}/(E_i-E_j) @f$
/// where @f$ F^{\alpha} @f$ is the force on the alpha classical dimension, and E the potential
Tensor3d adiabatic_coupling(const ClassicalVector<double>& x)
{
	Tensor3d NAC = xt::zeros<Tensor3d::value_type>(Tensor3d::shape_type{});
	const Eigen::VectorXd& E = adiabatic_potential(x);
	const Tensor3d& F = adiabatic_force(x);
	for (const std::size_t i : std::ranges::iota_view{0ul, Dim})
	{
		for (const std::size_t j : std::ranges::iota_view{1ul, NumPES})
		{
			for (const std::size_t k : std::ranges::iota_view{0ul, j})
			{
				NAC(i, j, k) = F(i, j, k) / (E[j] - E[k]);
				NAC(i, k, j) = -NAC(i, j, k);
			}
		}
	}
	return NAC;
}
