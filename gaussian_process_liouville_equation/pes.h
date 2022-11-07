/// @file pes.h
/// @brief Declaration of Potential Energy Surfaces (PES) and their derivatives

#ifndef PES_H
#define PES_H

#include "stdafx.h"

/// @brief Sign function: return -1 for negative, 1 for positive, and 0 for 0
/// @tparam T The type of the value
/// @param[in] val A value of any type that have '<' and '>' and could construct 0.0
/// @return The sign of the value
template <typename T>
inline constexpr int sgn(const T& val)
{
	return (val > static_cast<T>(0)) - (val < static_cast<T>(0));
}

/// @brief Different basis
enum Representation
{
	// Before are Force basis, the -dH/dR operator is diagonal
	/// @brief Adiabatic basis, subsystem Hamiltonian is diagonal
	Adiabatic = Dim
};

/// @brief Different models
enum Model
{
	/// @brief Simple Avoided Crossing, tully's first model
	SAC,
	/// @brief Dual Avoided Crossing, tully's second model
	DAC,
	/// @brief Extended Coupling with Reflection, tully's third model
	ECR
};

#ifndef TestModel
/// @brief The model to use
constexpr Model TestModel = DAC;
#endif // !TestModel

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

#endif // !PES_H
