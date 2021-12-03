/// @file evolve.h
/// @brief Interface of evolution functions: quantum/position/momentum Liouville superoperators.

#ifndef EVOLVE_H
#define EVOLVE_H

#include "stdafx.h"

#include "predict.h"

/// @brief To judge if current point have large coupling in case of 2-level system
/// @param[in] x Position of classical degree of freedom
/// @param[in] p Momentum of classical degree of freedom
/// @param[in] mass Mass of classical degree of freedom
/// @return Whether have coupling in any of the directions
ClassicalVector<bool> is_coupling(
	const ClassicalVector<double>& x,
	const ClassicalVector<double>& p,
	const ClassicalVector<double>& mass);

/// @brief To evolve all selected points forward
/// @param[inout] density The vector containing all known density matrix
/// @param[in] NumPoints The number of selected points for each of the density matrix element
/// @param[in] mass Mass of classical degree of freedom
/// @param[in] dt Time interval
/// @param[in] IsSmall The matrix that saves whether each element is small or not
/// @param[in] Predictors An array of predictors for prediction, whose size is NumElements
void evolve(
	EigenVector<PhaseSpacePoint>& density,
	const int NumPoints,
	const ClassicalVector<double>& mass,
	const double dt,
	const Predictions& Predictors);

/// @brief To predict the density matrix of the given point after evolving 1 time step
/// @param[in] x Position of classical degree of freedom
/// @param[in] p Momentum of classical degree of freedom
/// @param[in] mass Mass of classical degree of freedom
/// @param[in] dt Time interval
/// @param[in] IsSmall The matrix that saves whether each element is small or not
/// @param[in] Predictors An array of predictors for prediction, whose size is NumElements
/// @return The density matrix at the given point after evolving
QuantumMatrix<std::complex<double>> non_adiabatic_evolve_predict(
	const ClassicalVector<double>& x,
	const ClassicalVector<double>& p,
	const ClassicalVector<double>& mass,
	const double dt,
	const Predictions& Predictors);

#endif // !EVOLVE_H
