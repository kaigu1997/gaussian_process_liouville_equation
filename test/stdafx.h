/// @file stdafx.h
/// @brief The header file containing all headers and constants.
///
/// This file will be the precompiled header(PCH), named
/// after STanDard Application Framework eXtensions.

#pragma once
#ifndef STDAFX_H
#define STDAFX_H

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#ifndef EIGEN_USE_MKL_ALL
#define EIGEN_USE_MKL_ALL
#endif
#include <Eigen/Eigen>
#undef EIGEN_USE_MKL_ALL
#include <nlopt.hpp>
#pragma warning(push, 0)
#pragma warning(disable : 3346 654)
#include <shogun/base/init.h>
#include <shogun/base/some.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/ConstKernel.h>
#include <shogun/kernel/DiagKernel.h>
#include <shogun/kernel/GaussianARDKernel.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#pragma warning(pop)

const int NPoint = 200;		  ///< The size of training set
const int Dim = 1;			  ///< The dimension of the system, half the dimension of the phase space
const int PhaseDim = Dim * 2; ///< The dimension of the phase space, twice the dimension of the system
const int NumPES = 2;		  ///< The number of potential energy surfaces

/// Vector containing the type of all kernels
using KernelTypeList = std::vector<shogun::EKernelType>;
/// The set containing the index (instead of coordinate) of the selected points
using PointSet = std::set<std::pair<int, int>>;
/// The vector containing hyperparameters (or similarly: bounds, gradient, etc)
using ParameterVector = std::vector<double>;
/// The matrix used in quantum system (e.g. H, rho) with double values
using QuantumMatrixD = Eigen::Matrix<double, NumPES, NumPES>;
/// The matrix used in quantum system (e.g. H, rho) with complex values
using QuantumMatrixC = Eigen::Matrix<std::complex<double>, NumPES, NumPES>;
/// The matrix of matrix, used for PWTDM or Hamiltonian, etc
using SuperMatrix = Eigen::Matrix<QuantumMatrixD, Eigen::Dynamic, Eigen::Dynamic>;
/// The result of fitting: the fitted matrix, the selected points, and the -log(marginal likelihood)
using FittingResult = std::tuple<SuperMatrix, PointSet, double>;

#endif // !STDAFX_H
