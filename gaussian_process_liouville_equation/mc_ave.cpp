/// @file mc_ave.cpp
/// @brief Implementation of mc_ave.h

#include "stdafx.h"

#include "mc_ave.h"

#include "pes.h"

ClassicalPhaseVector calculate_1st_order_average(const EigenVector<PhaseSpacePoint>& density, const int PESIndex)
{
	const int NumPoints = density.size() / NumElements;
	const int BeginIndex = PESIndex * (NumPES + 1) * NumPoints;
	ClassicalPhaseVector r = ClassicalPhaseVector::Zero();
	for (int iPoint = BeginIndex; iPoint < BeginIndex + NumPoints; iPoint++)
	{
		const auto& [x, p, rho] = density[iPoint];
		for (int iDim = 0; iDim < Dim; iDim++)
		{
			r[iDim] += x[iDim];
			r[iDim + Dim] += p[iDim];
		}
	}
	return r / NumPoints;
}

double calculate_kinetic_energy_average(const EigenVector<PhaseSpacePoint>& density, const int PESIndex, const ClassicalVector<double>& mass)
{
	const int NumPoints = density.size() / NumElements;
	const int BeginIndex = PESIndex * (NumPES + 1) * NumPoints;
	double T = 0.0;
	for (int iPoint = BeginIndex; iPoint < BeginIndex + NumPoints; iPoint++)
	{
		const auto& [x, p, rho] = density[iPoint];
		T += p.dot((p.array() / mass.array()).matrix()) / 2.0;
	}
	return T / NumPoints;
}

double calculate_potential_energy_average(const EigenVector<PhaseSpacePoint>& density, const int PESIndex)
{
	const int NumPoints = density.size() / NumElements;
	const int BeginIndex = PESIndex * (NumPES + 1) * NumPoints;
	double V = 0.0;
	for (int iPoint = BeginIndex; iPoint < BeginIndex + NumPoints; iPoint++)
	{
		const auto& [x, p, rho] = density[iPoint];
		V += adiabatic_potential(x)[PESIndex];
	}
	return V / NumPoints;
}