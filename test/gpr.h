/// @file gpr.h
/// @brief The header file containing gaussian process regression interfaces

#pragma once
#ifndef GPR_H
#define GPR_H

#include "stdafx.h"

/// @brief Calculate the difference between GPR and exact value
/// @param[in] data The exact value
/// @param[in] x The gridded position coordinates
/// @param[in] p The gridded momentum coordinates
/// @return The fitting result tuple
FittingResult fit(const Eigen::MatrixXd& data, const Eigen::VectorXd& x, const Eigen::VectorXd& p);

#endif // !GPR_H
