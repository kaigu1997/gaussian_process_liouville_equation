/// @file gpr.h
/// @brief the header file containing gaussian process regression interfaces

#pragma once
#ifndef GPR_H
#define GPR_H

#include "stdafx.h"

/// calculate the difference between GPR and exact value
/// @param[in] data the exact value
/// @param[in] x the gridded position coordinates
/// @param[in] p the gridded momentum coordinates
/// @param[inout] sim the output stream for simulated phase space distribution
/// @param[inout] choose the output stream for the chosen point
/// @param[inout] log the output stream for log info (mse, -log(marg ll))
void fit
(
	const MatrixXd& data,
	const VectorXd& x,
	const VectorXd& p,
	ostream& sim,
	ostream& choose,
	ostream& log
);

#endif // !GPR_H

