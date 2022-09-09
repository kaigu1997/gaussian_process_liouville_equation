/// @file storage.cpp
/// @brief Implementation of storage.h

#include "stdafx.h"

#include "storage.h"

using namespace std::literals::complex_literals;

PhaseSpacePoint::PhaseSpacePoint(const ClassicalPhaseVector& R, std::complex<double> DenMatElm, double theta):
	r(R), rho(DenMatElm / std::exp(1.0i * theta)), adiabatic_theta(theta)
{
}

PhaseSpacePoint::PhaseSpacePoint(ClassicalPhaseVector&& R, std::complex<double> DenMatElm, double theta):
	r(std::move(R)), rho(DenMatElm / std::exp(1.0i * theta)), adiabatic_theta(theta)
{
}

std::complex<double> PhaseSpacePoint::get_exact_element(void) const
{
	return rho * std::exp(1.0i * adiabatic_theta);
}

void PhaseSpacePoint::set_density(std::complex<double> DenMatElm)
{
	rho = DenMatElm / std::exp(1.0i * adiabatic_theta);
}