/// @file evolve.h
/// @brief Interface of evolution functions: quantum/position/momentum Liouville superoperators.

#ifndef EVOLVE_H
#define EVOLVE_H

#include "mc.h"

/// @brief Evolve the selected phase points and their PWTDMs with the quantum Liouville superoperator
/// @param[inout] density The selected phase points with their PWTDMs, each point will generate a new PWTDM
/// @param[in] mass Mass of classical degree of freedom
/// @param[in] dt The time-step this superoperator will evolve
void quantum_liouville(EvolvingDensity& density, const ClassicalDoubleVector& mass, const double dt);

/// @brief Evolve the selected phase points and their PWTDMs with the classical position Liouville superoperator
/// @param[inout] density The selected phase points with their PWTDMs, each point will generate a new PWTDM
/// @param[in] mass Mass of classical degree of freedom
/// @param[in] dt The time-step this superoperator will evolve
void classical_position_liouville(EvolvingDensity& density, const ClassicalDoubleVector& mass, const double dt);

/// @brief Evolve the selected phase points and their PWTDMs with the classical momentum Liouville superoperator
/// @param[inout] density The selected phase points with their PWTDMs, generally each point will generate more tha one new PWTDMs
/// @param[in] optimizer The optimizer containing the selected points, hyperparameters, and kernel matrices
/// @param[in] dt The time-step this superoperator will evolve
void classical_momentum_liouville(EvolvingDensity& density, const Optimization& optimizer, const double dt);

#endif // !EVOLVE_H
