/// @file evolve.h
/// @brief Interface of evolution functions: quantum/position/momentum Liouville superoperators.

#ifndef EVOLVE_H
#define EVOLVE_H

#include "opt.h"

/// @brief To predict the density matrix of the given point after evolving 1 time step
/// @param[in] x Position of classical degree of freedom
/// @param[in] p Momentum of classical degree of freedom
/// @param[in] mass Mass of classical degree of freedom
/// @param[in] dt Time interval
/// @param[in] IsSmall The matrix that saves whether each element is small or not
/// @param[in] optimizer The predictor that predicts original density matrix elements
/// @return The density matrix at the given point after evolving
QuantumComplexMatrix evolve_predict(
	const ClassicalDoubleVector& x,
	const ClassicalDoubleVector& p,
	const ClassicalDoubleVector& mass,
	const double dt,
	const QuantumBoolMatrix IsSmall,
	const Optimization& optimizer);

#endif // !EVOLVE_H
