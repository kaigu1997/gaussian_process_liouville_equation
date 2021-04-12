/// @file stdafx.h
/// @brief The header file containing all headers and constants.

#ifndef STDAFX_H
#define STDAFX_H

#ifndef EIGEN_USE_MKL_ALL
#define EIGEN_USE_MKL_ALL
#endif // !EIGEN_USE_MKL_ALL

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <execution>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <Eigen/Eigen>
#include <Eigen/StdVector>
#include <boost/numeric/odeint.hpp>
#include <nlopt.hpp>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/ConstKernel.h>
#include <shogun/kernel/DiagKernel.h>
#include <shogun/kernel/GaussianARDKernel.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>

#include <mkl.h>

// mathmatical/physical constants
const double hbar = 1.0;				  ///< Reduced Planck constant in a.u.
const double Pi = 3.14159265358979323846; ///< As its name

// DOF constants
const int NumPES = 2;					 ///< The number of quantum degree (potential energy surfaces)
const int Dim = 1;						 ///< The dimension of the system, half the dimension of the phase space
const int NumElements = NumPES * NumPES; ///< The number of elements of matrices in quantum degree (e.g. H, rho)
const int PhaseDim = Dim * 2;			 ///< The dimension of the phase space, twice the dimension of the system

// typedefs
/// Judge if all grids of certain density matrix element are very small or not
using QuantumBoolMatrix = Eigen::Matrix<bool, NumPES, NumPES>;
/// The vector used in quantum system (e.g. H.diag(), rho.diag()) with double values
using QuantumDoubleVector = Eigen::Matrix<double, NumPES, 1>;
/// The matrix used in quantum system (e.g. H, rho) with double values
using QuantumDoubleMatrix = Eigen::Matrix<double, NumPES, NumPES>;
/// The matrix used in quantum system (e.g. H, rho) with complex<double> values
using QuantumComplexMatrix = Eigen::Matrix<std::complex<double>, NumPES, NumPES>;
/// The vector used in classical degree with bool values
using ClassicalBoolVector = Eigen::Matrix<bool, Dim, 1>;
/// The vector used in classical degree (e.g. mass, x0, p0) with double values
using ClassicalDoubleVector = Eigen::Matrix<double, Dim, 1>;
/// 3d tensor type for Force and NAC, based on Matrix with extra dimenstion from classical degree
using Tensor3d = Eigen::Matrix<QuantumDoubleMatrix, Dim, 1>;
/// The vector containing branched classical position/momentum
using ClassicalVectors = std::vector<ClassicalDoubleVector, Eigen::aligned_allocator<ClassicalDoubleVector>>;
/// The type for phase space point, first being x, second being p, third being the partial wigner-transformed density matrix
using PhaseSpacePoint = std::tuple<ClassicalDoubleVector, ClassicalDoubleVector, QuantumComplexMatrix>;
/// All selected density matrices for evolution; eigen-related type need special allocator
using EvolvingDensity = std::vector<PhaseSpacePoint, Eigen::aligned_allocator<PhaseSpacePoint>>;
/// The function prototype to do MC selection
using DistributionFunction = std::function<QuantumComplexMatrix(const ClassicalDoubleVector&, const ClassicalDoubleVector&)>;

#endif // !STDAFX_H
