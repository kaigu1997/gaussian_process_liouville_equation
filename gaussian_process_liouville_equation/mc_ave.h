/// @file mc_ave.h
/// @brief Interface to the monte carlo averages

#ifndef MC_AVE_H
#define MC_AVE_H

#include "stdafx.h"

/// @brief To calculate the average position and momentum of one diagonal element by monte carlo integration
/// @param[in] density The selected density matrices
/// @param[in] PESIndex The index of the potential energy surface, corresponding to (PESIndex, PESIndex) in density matrix
/// @return Average position and momentum
ClassicalPhaseVector calculate_1st_order_average(const EigenVector<PhaseSpacePoint>& density, const int PESIndex);

/// @brief To calculate the average kinetic energy of one diagonal element by monte carlo integration
/// @param[in] density The selected density matrices
/// @param[in] PESIndex The index of the potential energy surface, corresponding to (PESIndex, PESIndex) in density matrix
/// @param[in] mass Mass of classical degree of freedom
/// @return Average kinetic energy
double calculate_kinetic_energy_average(const EigenVector<PhaseSpacePoint>& density, const int PESIndex, const ClassicalVector<double>& mass);

/// @brief To calculate the average potential energy of one diagonal element by monte carlo integration
/// @param[in] density The selected density matrices
/// @param[in] PESIndex The index of the potential energy surface, corresponding to (PESIndex, PESIndex) in density matrix
/// @return Average potential energy
double calculate_potential_energy_average(const EigenVector<PhaseSpacePoint>& density, const int PESIndex);

/// @brief To calculate the total energy, simply a wrapper
/// @param[in] density The selected density matrices
/// @param[in] PESIndex The index of the potential energy surface, corresponding to (PESIndex, PESIndex) in density matrix
/// @param[in] mass Mass of classical degree of freedom
/// @return Averaged total energy
inline double calculate_total_energy_average(const EigenVector<PhaseSpacePoint>& density, const int PESIndex, const ClassicalVector<double>& mass)
{
	return calculate_kinetic_energy_average(density, PESIndex, mass) + calculate_potential_energy_average(density, PESIndex);
}

#endif
