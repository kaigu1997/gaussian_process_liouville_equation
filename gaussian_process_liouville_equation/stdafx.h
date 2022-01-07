/// @file stdafx.h
/// @brief The header file containing all headers and constants. stdafx = STanDard Application Framework eXtensions

#ifndef STDAFX_H
#define STDAFX_H

#ifndef EIGEN_USE_MKL_ALL
#define EIGEN_USE_MKL_ALL
#endif // !EIGEN_USE_MKL_ALL

#ifndef SPDLOG_ACTIVE_LEVEL
#ifdef NDEBUG
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_OFF
#else
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE
#endif // !NDEBUG
#endif // !SPDLOG_ACTIVE_LEVEL

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
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
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <Eigen/Eigen>
#include <Eigen/StdVector>
#include <nlopt.hpp>
#include <spdlog/fmt/ostr.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <mkl.h>

// mathmatical/physical constants
const double hbar = 1.0; ///< Reduced Planck constant in a.u.

// DOF constants
const int NumPES = 2;					 ///< The number of quantum degree (potential energy surfaces)
const int NumElements = NumPES * NumPES; ///< The number of elements of matrices in quantum degree (e.g. H, rho)
const int Dim = 1;						 ///< The dimension of the system, half the dimension of the phase space
const int PhaseDim = Dim * 2;			 ///< The dimension of the phase space, twice the dimension of the system

// other constants
const double PurityFactor = std::pow(2.0 * M_PI * hbar, Dim); ///< The factor for purity. @f$ S=(2\pi\hbar)^D\int\mathrm{d}\Gamma\mathrm{Tr}\rho^2 @f$

// formatters
const Eigen::IOFormat VectorFormatter(Eigen::StreamPrecision, Eigen::DontAlignCols, " ", " ", "", "", "", "");	///< Formatter for output vector
const Eigen::IOFormat MatrixFormatter(Eigen::StreamPrecision, Eigen::DontAlignCols, " ", "\n", "", "", "", ""); ///< Formatter for output matrix

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
/// std::array of quantum degree
template <typename T>
using QuantumArray = std::array<T, NumElements>;
/// The vector to depict a phase space point (i.e. coordinate)
using ClassicalPhaseVector = Eigen::Matrix<double, 2 * Dim, 1>;
/// 3d tensor type for Force and NAC, based on Matrix with extra dimenstion from classical degree
using Tensor3d = ClassicalVector<QuantumMatrix<double>>;
/// The type for phase space point, first being x, second being p, third being the partial wigner-transformed density matrix
using PhaseSpacePoint = std::tuple<ClassicalVector<double>, ClassicalVector<double>, QuantumMatrix<std::complex<double>>>;
/// The function prototype to do MC selection
using DistributionFunction = std::function<QuantumMatrix<std::complex<double>>(const ClassicalVector<double>&, const ClassicalVector<double>&)>;
/// Sets of selected phase space points of all density matrix elements
using AllPoints = QuantumArray<EigenVector<PhaseSpacePoint>>;

// frequently used inline functions
/// @brief Similar to range(N) in python
/// @param[in] N The number of indices
/// @return A vector, whose element is 0 to N-1
inline std::vector<int> get_indices(const int N)
{
	std::vector<int> result(N);
	std::iota(result.begin(), result.end(), 0);
	return result;
}

/// @brief The weight function for sorting / MC selection
/// @param x The input variable
/// @return The weight of the input, which is the square of it
static inline double weight_function(const std::complex<double>& x)
{
	return std::norm(x);
}

#endif // !STDAFX_H
