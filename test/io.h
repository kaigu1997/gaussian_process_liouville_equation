/// @file io.h
/// @brief the header file containing interface of all io functions

#pragma once
#ifndef IO_H
#define IO_H

#include "stdafx.h"

/// @brief Read the coordinate (of x, p and t) from a file
/// @param[in] filename The name of the input file
/// @return The vector containing the data
VectorXd read_coord(const string& filename);

/// @brief Output the hyperparameters
/// @param[inout] out The output stream
/// @param[in] x The gsl vector contatining all the hyperparameters
/// @param[in] indent The level of indentation, or how many tabs to print before the line
void print_kernel(ostream& os, const gsl_vector* x, const int indent);

/// @brief Output the chosen point to an output stream
/// @param[inout] out The output stream
/// @param[in] point The chosen points
/// @param[in] x The gridded position coordinates
/// @param[in] p The gridded momentum coordinates
void print_point(
	ostream& out,
	const set<pair<int, int>>& point,
	const VectorXd& x,
	const VectorXd& p);

#endif // !IO_H
