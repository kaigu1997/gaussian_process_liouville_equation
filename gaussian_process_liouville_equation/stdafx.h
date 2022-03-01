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
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
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
constexpr double hbar = 1.0; ///< Reduced Planck constant in a.u.

// DOF constants
constexpr std::size_t NumPES = 2;					 ///< The number of quantum degree (potential energy surfaces)
constexpr std::size_t NumElements = NumPES * NumPES; ///< The number of elements of matrices in quantum degree (e.g. H, rho)
constexpr std::size_t Dim = 1;						 ///< The dimension of the system, half the dimension of the phase space
constexpr std::size_t PhaseDim = Dim * 2;			 ///< The dimension of the phase space, twice the dimension of the system

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
using PhaseSpacePoint = std::tuple<ClassicalPhaseVector, QuantumMatrix<std::complex<double>>>;
/// Sets of selected phase space points of one density matrix element
using ElementPoints = EigenVector<PhaseSpacePoint>;
/// Sets of selected phase space points of all density matrix elements
using AllPoints = QuantumArray<ElementPoints>;
/// The function prototype to do MC selection
using DistributionFunction = std::function<QuantumMatrix<std::complex<double>>(const ClassicalPhaseVector&)>;

// frequently used inline functions
/// @brief Similar to range(N) in python
/// @param[in] N The number of indices
/// @return A vector, whose element is 0 to N-1
inline std::vector<std::size_t> get_indices(const std::size_t N)
{
	std::vector<std::size_t> result(N);
	std::iota(result.begin(), result.end(), 0);
	return result;
}

/// @brief The weight function for sorting / MC selection
/// @param x The input variable
/// @return The weight of the input, which is the square of it
inline double weight_function(const std::complex<double>& x)
{
	return std::norm(x);
}

/// @brief To get the (editable) x and p of r
/// @param r The editable phase space coordinates 
/// @return The x block and p block
inline std::tuple<Eigen::Block<ClassicalPhaseVector, Dim, 1>, Eigen::Block<ClassicalPhaseVector, Dim, 1>> split_phase_coordinate(ClassicalPhaseVector& r)
{
	return std::make_tuple(r.block<Dim, 1>(0, 0), r.block<Dim, 1>(Dim, 0));
}

/// @brief To get the (constant) x and p of r
/// @param r The constant phase space coordinates 
/// @return The x and p vector
inline std::tuple<ClassicalVector<double>, ClassicalVector<double>> split_phase_coordinate(const ClassicalPhaseVector& r)
{
	return std::make_tuple(r.block<Dim, 1>(0, 0), r.block<Dim, 1>(Dim, 0));
}

#endif // !STDAFX_H
