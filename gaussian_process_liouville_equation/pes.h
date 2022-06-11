/// @file pes.h
/// @brief Declaration of Potential Energy Surfaces (PES) and their derivatives

#ifndef PES_H
#define PES_H

#include "stdafx.h"

/// @brief Sign function: return -1 for negative, 1 for positive, and 0 for 0
/// @param[in] val A value of any type that have '<' and '>' and could construct 0.0
/// @return The sign of the value
template <typename valtype>
inline constexpr int sgn(const valtype& val)
{
	return (val > valtype(0.0)) - (val < valtype(0.0));
}

/// Different basis
enum Representation
{
	// Before are Force basis, the -dH/dR operator is diagonal
	Adiabatic = Dim ///< Adiabatic basis, subsystem Hamiltonian is diagonal
};

/// Different models
enum Model
{
	SAC, ///< Simple Avoided Crossing, tully's first model
	DAC, ///< Dual Avoided Crossing, tully's second model
	ECR	 ///< Extended Coupling with Reflection, tully's third model
};

#ifndef TestModel
constexpr Model TestModel = DAC; ///< The model to use
#endif						 // !TestModel

/// @brief To calculate the diagonalized subsystem Hamiltonian matrix
/// @param[in] x Position of classical degree of freedom
/// @return The adiabatic potential matrix at this position, which is real diagonal
/// @sa adiabatic_force(), adiabatic_coupling()
QuantumVector<double> adiabatic_potential(const ClassicalVector<double>& x);

/// @brief To calculate the Non-Adiabatic Coupling (NAC) under the adiabatic basis
/// @param[in] x Position of classical degree of freedom
/// @return The NAC tensor at this position, whose each element is real-symmetric
/// @sa adiabatic_potential(), adiabatic_force()
Tensor3d adiabatic_coupling(const ClassicalVector<double>& x);

/// @brief To calculate the forces (-dH/dR) under the adiabatic basis
/// @param[in] x Position of classical degree of freedom
/// @return The force tensor of each direction
/// @sa adiabatic_potential(), adiabatic_coupling()
Tensor3d adiabatic_force(const ClassicalVector<double>& x);

/// @brief To transform density matrix from one basis to another
/// @param[in] rho The partial Wigner-transformed denstiy matrix, a self-adjoint complex matrix
/// @param[in] x Position of classical degree of freedom
/// @param[in] idx_from The index indicating the representation rho in
/// @param[in] idx_to The index indicating the representation the return value in
/// @return Another self-adjoint denstiy matrix, in the representation indicated by the parameter
QuantumMatrix<std::complex<double>> basis_transform(
	const QuantumMatrix<std::complex<double>>& rho,
	const ClassicalVector<double>& x,
	const std::size_t idx_from,
	const std::size_t idx_to);

#endif // !PES_H
