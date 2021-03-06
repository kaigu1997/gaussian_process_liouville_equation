/// @file pes.h
/// @brief Declaration of Potential Energy Surfaces (PES) and their derivatives

#ifndef PES_H
#define PES_H

#include "stdafx.h"

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
const Model TestModel = DAC; ///< The model to use
#endif						 // !TestModel

/// @brief To calculate the diagonalized subsystem Hamiltonian matrix
/// @param[in] x Position of classical degree of freedom
/// @return The adiabatic potential matrix at this position, which is real diagonal
/// @see adiabatic_force(), adiabatic_coupling()
QuantumDoubleVector adiabatic_potential(const ClassicalDoubleVector& x);

/// @brief To calculate the Non-Adiabatic Coupling (NAC) under the adiabatic basis
/// @param[in] x Position of classical degree of freedom
/// @return The NAC tensor at this position, whose each element is real-symmetric
/// @see adiabatic_potential(), adiabatic_force()
Tensor3d adiabatic_coupling(const ClassicalDoubleVector& x);

/// @brief To calculate the forces (-dH/dR) under the adiabatic basis
/// @param[in] x Position of classical degree of freedom
/// @return The force tensor of each direction
/// @see adiabatic_potential(), adiabatic_coupling()
Tensor3d adiabatic_force(const ClassicalDoubleVector& x);

/// @brief To get a slice of the tensor
/// @param[in] tensor The tensor which the slice is from
/// @param[in] row The row index of the tensor
/// @param[in] col The column index of the tensor
/// @return A vector, that is tensor(0, row, col) to tensor(idx1_max, row, col)
ClassicalDoubleVector tensor_slice(const Tensor3d& tensor, const int row, const int col);

/// @brief To transform density matrix from one basis to another
/// @param[in] rho The PWTDM, a self-adjoint complex matrix
/// @param[in] x Position of classical degree of freedom
/// @param[in] idx_from The index indicating the representation rho in
/// @param[in] idx_to The index indicating the representation the return value in
/// @return Another self-adjoint denstiy matrix, in the representation indicated by the parameter
QuantumComplexMatrix basis_transform(
	const QuantumComplexMatrix& rho,
	const ClassicalDoubleVector& x,
	const int idx_from,
	const int idx_to);

/// @brief To make the complex matrix self-adjoint
/// @param[inout] mat The matrix
inline void make_self_adjoint(QuantumComplexMatrix& mat)
{
	mat.diagonal() = mat.diagonal().real();
	mat = mat.selfadjointView<Eigen::Lower>();
}

#endif // !PES_H
