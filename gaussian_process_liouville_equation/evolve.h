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

/// @brief To evolve all selected points forward
/// @param[inout] density The vector containing all known density matrix
/// @param[in] mass Mass of classical degree of freedom
/// @param[in] dt Time interval
/// @param[in] Kernels An array of predictors for prediction, whose size is NumElements
void evolve(
	AllPoints& density,
	const ClassicalVector<double>& mass,
	const double dt,
	const OptionalKernels& Kernels);

#endif // !EVOLVE_H
