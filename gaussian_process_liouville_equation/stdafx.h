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
#include <memory>
#include <numbers>
#include <numeric>
#include <optional>
#include <random>
#include <ranges>
#include <string>
#include <string_view>
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

/// @brief To provide indentations for output
/// @tparam N The number of indentations, each equal to 4 spaces
template <std::size_t N>
class indents
{
private:
	/// @brief The pre-setup string of indentations
	static constexpr char tabs[] = "                                ";

public:
	/// @brief To really give the indentations
	static constexpr std::string_view apply(void)
	{
		if constexpr (N * 4 < sizeof(tabs) / sizeof(char))
		{
			return std::string_view(tabs, N * 4);
		}
		else
		{
			return std::string(N * 4, ' ');
		}
	}
};

/// @brief To calculate power of numeric types
/// @tparam N To calculate n-th power
/// @tparam T The input type
/// @param t The value, could be integer or floating point
/// @return Its nth power
template <std::size_t N, typename T>
	requires std::is_arithmetic_v<std::decay_t<T>>
inline constexpr T power([[maybe_unused]] const T t)
{
	if constexpr (N == 0)
	{
		return 1;
	}
	else if constexpr (N == 1)
	{
		return t;
	}
	else
	{
		return power<N / 2>(t) * power<N - N / 2>(t);
	}
}

// mathmatical/physical constants
/// @brief Reduced Planck constant in a.u.
constexpr double hbar = 1.0;

// DOF constants
/// @brief The number of quantum degree (potential energy surfaces)
constexpr std::size_t NumPES = 2;
/// @brief The number of elements of matrices in quantum degree (e.g. H, rho)
constexpr std::size_t NumElements = power<2>(NumPES);
/// @brief The number of off-diagonal elements (including lower and upper)
constexpr std::size_t NumOffDiagonalElements = (NumElements - NumPES) / 2;
/// @brief The number of (lower or upper) triangular elements
constexpr std::size_t NumTriangularElements = (NumElements + NumPES) / 2;
/// @brief The dimension of the system, half the dimension of the phase space
constexpr std::size_t Dim = 1;
/// @brief The dimension of the phase space, twice the dimension of the system
constexpr std::size_t PhaseDim = Dim * 2;

// other constants
/// @brief The factor for purity. @f$ S=(2\pi\hbar)^D\int\mathrm{d}\Gamma\ \mathrm{Tr}\rho^2 @f$
constexpr double PurityFactor = power<Dim>(2.0 * std::numbers::pi * hbar);

// formatters
/// @brief Formatter for output vector
const Eigen::IOFormat VectorFormatter(Eigen::StreamPrecision, Eigen::DontAlignCols, " ", " ", "", "", "", "");
/// @brief Formatter for output matrix
const Eigen::IOFormat MatrixFormatter(Eigen::StreamPrecision, Eigen::DontAlignCols, " ", "\n", "", "", "", "");

// typedefs, notice Eigen is column-major while xtensor is row-major
/// @brief Matrices used for quantum degree
template <typename T>
using QuantumMatrix = Eigen::Matrix<T, NumPES, NumPES, Eigen::StorageOptions::RowMajor | Eigen::StorageOptions::AutoAlign>;
/// @brief Vectors used for quantum degree
template <typename T>
using QuantumVector = Eigen::Matrix<T, NumPES, 1>;
/// @brief Vectors used for classical degree
template <typename T>
using ClassicalVector = Eigen::Matrix<T, Dim, 1>;
/// @brief std::vector with eigen allocator
template <typename T>
using EigenVector = std::vector<T, Eigen::aligned_allocator<T>>;
/// @brief The vector to depict a phase space point (i.e. coordinate)
using ClassicalPhaseVector = Eigen::Matrix<double, PhaseDim, 1>;
/// @brief Rank-3 tensor type for Force and NAC,
/// based on Matrix with extra dimenstion from classical degree. @n
/// 1st rank is classical dim, 2nd is row in mat, 3rd is column in mat
using Tensor3d = xt::xtensor_fixed<double, xt::xshape<Dim, NumPES, NumPES>>;
/// @brief The type for bunches of phase space points, coordinates only, without information of density
using PhasePoints = Eigen::Matrix<double, PhaseDim, Eigen::Dynamic>;
/// @brief The function prototype to do MC selection
using DistributionFunction = std::function<std::complex<double>(const ClassicalPhaseVector&, std::size_t, std::size_t)>;

/// @brief To get the (constant) x and p of r
/// @param r The constant phase space coordinates
/// @return The x and p vector
inline std::tuple<ClassicalVector<double>, ClassicalVector<double>> split_phase_coordinate(const ClassicalPhaseVector& r)
{
	return std::make_tuple(r.block<Dim, 1>(0, 0), r.block<Dim, 1>(Dim, 0));
}

#endif // !STDAFX_H
