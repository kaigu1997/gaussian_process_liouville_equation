/// @file io.h
/// @brief the header file containing interface of all io functions 

#pragma once
#ifndef IO_H
#define IO_H

#include "stdafx.h"

/// to read the coordinate (of x, p and t) from a file
/// @param[in] filename the name of the input file
/// @return the vector containing the data
VectorXd read_coord(const string& filename);

/// output the chosen point to an output stream
/// @param[in] point the chosen points
/// @param[in] x the gridded position coordinates
/// @param[in] p the gridded momentum coordinates
/// @param[inout] out the output stream
void output_point
(
	const set<pair<int, int>>& point,
	const VectorXd& x,
	const VectorXd& p,
	ostream& out
);

/// print kernel information. If a combined kernel, print its inner
/// @param[in] kernel_ptr a raw pointer to a CKernel object to be printed
/// @param[in] layer the layer of the kernel
void print_kernel(CKernel* kernel_ptr, int layer);

#endif // !IO_H
