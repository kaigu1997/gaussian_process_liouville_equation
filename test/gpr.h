/// @file gpr.h
/// @brief the header file containing gaussian process regression interfaces

#pragma once
#ifndef GPR_H
#define GPR_H

#include "stdafx.h"

/// @brief Calculate the difference between GPR and exact value
/// @param[in] data The exact value
/// @param[in] x The gridded position coordinates
/// @param[in] p The gridded momentum coordinates
/// @param[inout] sim The output stream for simulated phase space distribution
/// @param[inout] choose The output stream for the chosen point
/// @param[inout] log The output stream for log info (mse, -log(marg ll))
void fit(
	const MatrixXd& data,
	const VectorXd& x,
	const VectorXd& p,
	ostream& sim,
	ostream& choose,
	ostream& log);

#endif // !GPR_H
