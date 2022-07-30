/// @file stdafx.h
/// @brief The header file containing all headers and constants. stdafx = STanDard Application Framework eXtensions

#ifndef STDAFX_H
#define STDAFX_H

/// @def EIGEN_USE_MKL_ALL
/// @brief Ask Eigen to use intel MKL as its backend
#ifndef EIGEN_USE_MKL_ALL
#define EIGEN_USE_MKL_ALL
#endif // !EIGEN_USE_MKL_ALL

/// @def SPDLOG_ACTIVE_LEVEL
/// @brief Set the logging level of spdlog library
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
#include <set>
#include <string>
#include <string_view>
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
#include <xtensor.hpp>

template <std::size_t n>
class indents
{
private:
	static constexpr char tabs[] = "                                ";
public:
	static constexpr std::string_view apply(void)
	{
		if constexpr (n << 2 < sizeof(tabs) / sizeof(char))
		{
			return std::string_view(tabs, n << 2);
		}
		else
		{
			return std::string(n << 2, ' ');
		}
	}
};

/// @brief To calculate power of numeric types
/// @tparam n To calculate n-th power
/// @tparam T The input type
/// @param t The value, could be integer or floating point
/// @return Its nth power
template <std::size_t n, typename T>
inline constexpr T power([[maybe_unused]] const T t)
{
	static_assert(std::is_arithmetic_v<std::decay_t<T>>);
	if constexpr (n == 0)
	{
		return 1;
	}
	else
	{
		return t * power<n - 1>(t);
	}
}

// mathmatical/physical constants
constexpr double hbar = 1.0; ///< Reduced Planck constant in a.u.

// DOF constants
constexpr std::size_t NumPES = 2;										  ///< The number of quantum degree (potential energy surfaces)
constexpr std::size_t NumElements = power<2>(NumPES);					  ///< The number of elements of matrices in quantum degree (e.g. H, rho)
constexpr std::size_t NumOffDiagonalElements = NumElements - NumPES;	  ///< The number of off-diagonal elements (including lower and upper)
constexpr std::size_t NumTriangularElements = (NumElements + NumPES) / 2; ///< The number of (lower or upper) triangular elements
constexpr std::size_t Dim = 1;											  ///< The dimension of the system, half the dimension of the phase space
constexpr std::size_t PhaseDim = Dim * 2;								  ///< The dimension of the phase space, twice the dimension of the system

// other constants
constexpr double PurityFactor = power<Dim>(2.0 * M_PI * hbar); ///< The factor for purity. @f$ S=(2\pi\hbar)^D\int\mathrm{d}\Gamma\mathrm{Tr}\rho^2 @f$

// formatters
const Eigen::IOFormat VectorFormatter(Eigen::StreamPrecision, Eigen::DontAlignCols, " ", " ", "", "", "", "");	///< Formatter for output vector
const Eigen::IOFormat MatrixFormatter(Eigen::StreamPrecision, Eigen::DontAlignCols, " ", "\n", "", "", "", ""); ///< Formatter for output matrix

// typedefs, notice Eigen is column-major while xtensor is row-major
/// Matrices used for quantum degree
template <typename T>
using QuantumMatrix = Eigen::Matrix<T, NumPES, NumPES, Eigen::StorageOptions::RowMajor | Eigen::StorageOptions::AutoAlign>;
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
using ClassicalPhaseVector = Eigen::Matrix<double, PhaseDim, 1>;
/// rank-3 tensor type for Force and NAC, based on Matrix with extra dimenstion from classical degree
/// 1st rank is classical dim, 2nd is row in mat, 3rd is column in mat
using Tensor3d = xt::xtensor_fixed<double, xt::xshape<Dim, NumPES, NumPES>>;
/// The type for phase space point, first being x, second being p, third being the partial wigner-transformed density matrix
using PhaseSpacePoint = std::tuple<ClassicalPhaseVector, QuantumMatrix<std::complex<double>>>;
/// The type for bunches of phase space points, coordinates only, without information of density
using PhasePoints = Eigen::Matrix<double, PhaseDim, Eigen::Dynamic>;
/// Sets of selected phase space points of one density matrix element
using ElementPoints = EigenVector<PhaseSpacePoint>;
/// Sets of selected phase space points of all density matrix elements
using AllPoints = QuantumArray<ElementPoints>;
/// The function prototype to do MC selection
using DistributionFunction = std::function<QuantumMatrix<std::complex<double>>(const ClassicalPhaseVector&)>;

/// @brief To get the (constant) x and p of r
/// @param r The constant phase space coordinates
/// @return The x and p vector
inline std::tuple<ClassicalVector<double>, ClassicalVector<double>> split_phase_coordinate(const ClassicalPhaseVector& r)
{
	return std::make_tuple(r.block<Dim, 1>(0, 0), r.block<Dim, 1>(Dim, 0));
}

#endif // !STDAFX_H
