/// @file gpr.h
/// @brief The header file containing gaussian process regression interfaces

#pragma once
#ifndef GPR_H
#define GPR_H

#include "stdafx.h"

/// @brief Calculate the mean squared error
/// @param[in] lhs The left-hand side PWTDM
/// @param[in] rhs The right-hand side PWTDM
/// @return The MSE between lhs and rhs on each element of PWTDM
QuantumMatrixD mean_squared_error(const SuperMatrix& lhs, const SuperMatrix& rhs);

/// @brief Calculate the difference between GPR and exact value
/// @param[in] data The gridded whole PWTDM
/// @param[in] x The gridded position coordinates
/// @param[in] p The gridded momentum coordinates
/// @return The fitting result tuple
FittingResult fit(const SuperMatrix& data, const Eigen::VectorXd& x, const Eigen::VectorXd& p);

#endif // !GPR_H
