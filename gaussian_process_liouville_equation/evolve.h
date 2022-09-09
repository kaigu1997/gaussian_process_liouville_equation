/// @file evolve.h
/// @brief Interface of evolution functions: quantum/position/momentum Liouville superoperators.

#ifndef EVOLVE_H
#define EVOLVE_H

#include "stdafx.h"

#include "predict.h"

/// @brief To judge if current point have large coupling in case of 2-level system
/// @param[in] density The vector containing all known density matrix
/// @param[in] mass Mass of classical degree of freedom
/// @return Whether have coupling or not
bool is_coupling(const AllPoints& density, const ClassicalVector<double>& mass);

/// @brief To evolve back to calculate the phase factor (initial phase factor is regarded as 0)
/// @param[in] r Current phase space corrdinates
/// @param[in] mass Mass of classical degree of freedom
/// @param[in] dt Time interval
/// @param[in] NumTicks Number of @p dt since T=0; i.e., now is @p NumTicks * @p dt
/// @param[in] RowIndex RowIndex Index of row of the element in density matrix. Must be a valid index (< @p NumPES )
/// @param[in] ColIndex RowIndex Index of row of the element in density matrix. Must be a valid index (< @p NumPES )
/// @return Current phase factor
double get_phase_factor(
	const ClassicalPhaseVector& r,
	const ClassicalVector<double>& mass,
	const double dt,
	const std::size_t NumTicks,
	const std::size_t RowIndex,
	const std::size_t ColIndex
);

/// @brief To evolve all selected points forward
/// @param[inout] density The vector containing all known density matrix
/// @param[in] mass Mass of classical degree of freedom
/// @param[in] dt Time interval
/// @param[in] NumTicks Number of @p dt since T=0; i.e., now is @p NumTicks * @p dt
/// @param[in] distribution Phase space distribution function
void evolve(
	AllPoints& density,
	const ClassicalVector<double>& mass,
	const double dt,
	const std::size_t NumTicks,
	const DistributionFunction& distribution
);

#endif // !EVOLVE_H
