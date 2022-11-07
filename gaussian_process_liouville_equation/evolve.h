/// @file evolve.h
/// @brief Interface of evolution functions: quantum/position/momentum Liouville superoperators.

#ifndef EVOLVE_H
#define EVOLVE_H

#include "stdafx.h"

#include "storage.h"

/// @brief To evolve all selected points forward
/// @param[inout] density The vector containing all known density matrix
/// @param[in] mass Mass of classical degree of freedom
/// @param[in] dt Time interval
/// @param[in] distribution Phase space distribution function
void evolve(
	AllPoints& density,
	const ClassicalVector<double>& mass,
	const double dt,
	const DistributionFunction& distribution
);

/// @brief To predict the density matrix of the given point from a new element of density matrix
/// @param[in] r Phase space coordinates of classical degree of freedom
/// @param[in] mass Mass of classical degree of freedom
/// @param[in] dt Time interval
/// @param[in] distribution Phase space distribution function
/// @param[in] RowIndex RowIndex Index of row of the element in density matrix. Must be a valid index (< @p NumPES )
/// @param[in] ColIndex RowIndex Index of row of the element in density matrix. Must be a valid index and no more than @p RowIndex
/// @return The density at the given point of the given new element in density matrix. @n
/// If the coupling is negligible at the given point, it gives 0 density.
std::complex<double> new_point_predict(
	const ClassicalPhaseVector& r,
	const ClassicalVector<double>& mass,
	const double dt,
	const DistributionFunction& distribution,
	const std::size_t RowIndex,
	const std::size_t ColIndex
);

/// @brief To check each element is small or not
/// @param[in] density The selected points in phase space for each element of density matrices
/// @param[in] mass Mass of classical degree of freedom
/// @param[in] dt Time interval
/// @param[in] distribution Phase space distribution function
/// @return Triangularly stored boolean type containing each element of density matrix is small or not.
QuantumStorage<bool> is_very_small(
	const AllPoints& density,
	const ClassicalVector<double>& mass,
	const double dt,
	const DistributionFunction& distribution
);

#endif // !EVOLVE_H
