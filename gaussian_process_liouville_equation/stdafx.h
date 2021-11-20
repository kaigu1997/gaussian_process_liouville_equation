/// @file stdafx.h
/// @brief The header file containing all headers and constants. stdafx = STanDard Application Framework eXtensions

#ifndef STDAFX_H
#define STDAFX_H

#ifndef EIGEN_USE_MKL_ALL
#define EIGEN_USE_MKL_ALL
#endif // !EIGEN_USE_MKL_ALL

#include <algorithm>
#include <array>
#include <cassert>
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
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <Eigen/Eigen>
#include <Eigen/StdVector>
#include <nlopt.hpp>

#include <mkl.h>

// mathmatical/physical constants
const double hbar = 1.0; ///< Reduced Planck constant in a.u.

// DOF constants
const int NumPES = 2;									  ///< The number of quantum degree (potential energy surfaces)
const int NumElements = NumPES * NumPES;				  ///< The number of elements of matrices in quantum degree (e.g. H, rho)
const int NumDiagonalElements = NumPES;					  ///< The number of diagonal elements
const int NumOffDiagonalElements = NumPES * (NumPES - 1); ///< The number of off-diagonal elements; overall numpes^2 elements
const int Dim = 1;										  ///< The dimension of the system, half the dimension of the phase space
const int PhaseDim = Dim * 2;							  ///< The dimension of the phase space, twice the dimension of the system

// typedefs
/// Matrices used for quantum degree
template <typename T>
using QuantumMatrix = Eigen::Matrix<T, NumPES, NumPES>;
/// Vectors used for quantum degree
template <typename T>
using QuantumVector = Eigen::Matrix<T, NumPES, 1>;
/// Vectors used for classical degree
template <typename T>
using ClassicalVector = Eigen::Matrix<T, Dim, 1>;
/// std::vector with eigen allocator
template <typename T>
using EigenVector = std::vector<T, Eigen::aligned_allocator<T>>;
/// The vector to depict a phase space point (i.e. coordinate)
using ClassicalPhaseVector = Eigen::Matrix<double, 2 * Dim, 1>;
/// 3d tensor type for Force and NAC, based on Matrix with extra dimenstion from classical degree
using Tensor3d = ClassicalVector<QuantumMatrix<double>>;
/// The type for phase space point, first being x, second being p, third being the partial wigner-transformed density matrix
using PhaseSpacePoint = std::tuple<ClassicalVector<double>, ClassicalVector<double>, QuantumMatrix<std::complex<double>>>;
/// The function prototype to do MC selection
using DistributionFunction = std::function<QuantumMatrix<std::complex<double>>(const ClassicalVector<double>&, const ClassicalVector<double>&)>;

#endif // !STDAFX_H
