/// @file input.cpp
/// @brief Implementation of input.h

#include "stdafx.h"

#include "input.h"

/// @brief Read a parameter from input stream
/// @param[inout] is The input stream (could be input file stream)
/// @details The content would be: "[Descriptor]:\n[Value]\n", so need buffer to read descriptor and newline
template <typename T>
static T read_param(std::istream& is)
{
	std::string buffer;
	T result;
	std::getline(is, buffer);
	is >> result;
	std::getline(is, buffer);
	return result;
}

/// @brief Read a vector in Parameter from input stream
/// @param[inout] is The input stream (could be input file stream)
/// @details This is the specialization of read_param() function for ClassicalVector<double>.
template <>
ClassicalVector<double> read_param<ClassicalVector<double>>(std::istream& is)
{
	std::string buffer;
	ClassicalVector<double> result;
	std::getline(is, buffer);
	for (std::size_t i = 0; i < Dim; i++)
	{
		is >> result[i];
	}
	std::getline(is, buffer);
	return result;
}

InitialParameters::InitialParameters(const std::string& input_file_name)
{
	// read input
	std::ifstream in(input_file_name);
	assert(in.is_open());
	mass = read_param<ClassicalVector<double>>(in);
	const ClassicalVector<double> x0 = read_param<ClassicalVector<double>>(in);
	const ClassicalVector<double> p0 = read_param<ClassicalVector<double>>(in);
	const ClassicalVector<double> SigmaP0 = read_param<ClassicalVector<double>>(in);
	OutputTime = read_param<double>(in);
	ReoptimizationTime = read_param<double>(in);
	dt = read_param<double>(in);
	NumberOfSelectedPoints = read_param<int>(in);
	in.close();
	// deal with input
	const ClassicalVector<double> xmax = x0.array().abs() * 2, xmin = -xmax;
	const ClassicalVector<double> SigmaX0 = hbar / 2.0 * SigmaP0.array().inverse();			 // follow the minimal uncertainty principle
	ClassicalVector<double> dx = M_PI * hbar / 2.0 * (p0 + 3.0 * SigmaP0).array().inverse(); // 4 grids per de Broglie wavelength
	const ClassicalVector<std::size_t> DirectionNumGrids = ((xmax - xmin).array() / dx.array()).cast<std::size_t>();
	dx = (xmax - xmin).array() / DirectionNumGrids.cast<double>().array();
	// in grid solution, dx = pi*hbar/(p0+3sigma_p), p in p0+pi*hbar/2/dx*[-1, 1]
	const ClassicalVector<double> pmax = p0.array() + M_PI * hbar / 2.0 * dx.array().inverse();
	const ClassicalVector<double> pmin = p0.array() - M_PI * hbar / 2.0 * dx.array().inverse();
	const ClassicalVector<double> dp = (pmax - pmin).array() / DirectionNumGrids.cast<double>().array();
	// get r version
	r0 << x0, p0;
	rmax << xmax, pmax;
	rmin << xmin, pmin;
	dr << dx, dp;
	SigmaR0 << SigmaX0, SigmaP0;
	// whole phase space grids
	const std::size_t NumGrids = DirectionNumGrids.prod() * DirectionNumGrids.prod();
	PhaseGrids.resize(Eigen::NoChange, NumGrids);
	const std::vector<std::size_t> indices = get_indices(NumGrids);
	std::for_each(
		std::execution::par_unseq,
		indices.cbegin(),
		indices.cend(),
		[&PhaseGrids = PhaseGrids, &DirectionNumGrids, &xmin, &pmin, &dx, &dp](std::size_t iPoint) -> void
		{
			ClassicalVector<double> xPoint = xmin, pPoint = pmin;
			std::size_t NextIndex = iPoint;
			for (std::size_t idx = Dim - 1; idx < Dim; idx--)
			{
				pPoint[idx] += dp[idx] * (NextIndex % DirectionNumGrids[idx]);
				NextIndex /= DirectionNumGrids[idx];
			}
			for (std::size_t idx = Dim - 1; idx < Dim; idx--)
			{
				xPoint[idx] += dx[idx] * (NextIndex % DirectionNumGrids[idx]);
				NextIndex /= DirectionNumGrids[idx];
			}
			PhaseGrids.col(iPoint) << xPoint, pPoint;
		});
	// limitations on time intervals
	if (ReoptimizationTime < dt)
	{
		ReoptimizationTime = dt;
	}
	if (OutputTime < ReoptimizationTime)
	{
		OutputTime = ReoptimizationTime;
	}
	const auto Distance = 2.0 * x0.array().abs();
	const auto Speed = (p0.array() / mass.array()).abs();
	const double TotalTime = (Distance / Speed).maxCoeff();
	TotalTicks = static_cast<std::size_t>(2.0 * TotalTime / dt);
}
