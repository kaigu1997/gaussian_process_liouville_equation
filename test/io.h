/// @file io.h
/// @brief The header file containing interface of all io functions

#pragma once
#ifndef IO_H
#define IO_H

#include "stdafx.h"

/// @brief Give a string of n repeated tabs
/// @param[in] IndentLevel The number of tabs
/// @return The string with given number of tabs
std::string indent(const int IndentLevel);

/// @brief Read the coordinate (of x, p and t) from a file
/// @param[in] filename The name of the input file
/// @return The vector containing the data
Eigen::VectorXd read_coord(const std::string& filename);

/// @brief Read the whole PWTDM from a input stream to the supermatrix
/// @param[inout] in The input stream
/// @param[inout] ExactDistribution The whole PWTDM (of a certain moment)
void read_density(std::istream& in, SuperMatrix& ExactDistribution);

/// @brief Output the chosen point to an output stream
/// @param[inout] out The output stream
/// @param[in] TrainingFeatures The features of the training set
void print_point(std::ostream& out, const SuperMatrix& TrainingFeatures);

/// @brief Output the information (mainly hyperparameters) of the kernel
/// @param[inout] out The output stream
/// @param[in] TypesOfKernels The vector containing all the kernel type used in optimization
/// @param[in] Hyperparameters The hyperparameters of all kernels (magnitude and other)
/// @param[in] IndentLevel The number of tabs before print the kernel
void print_kernel(
	std::ostream& out,
	const KernelTypeList& TypesOfKernels,
	const std::vector<double>& Hyperparameters,
	const int IndentLevel);

#endif // !IO_H
