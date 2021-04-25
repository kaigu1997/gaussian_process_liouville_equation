/// @file io.cpp
/// @brief definition of input-output functions

#include "stdafx.h"

#include "io.h"

/// @brief Read a parameter from input stream
/// @param[inout] is The input stream (could be input file stream)
/// @param[inout] param The variable waiting to be written
/// @details The content would be: "[Descriptor]:\n[Value]\n", so need buffer to read descriptor and newline
template<typename T>
static void read_param(std::istream& is, T& param)
{
	std::string buffer;
	std::getline(is, buffer);
	is >> param;
	std::getline(is, buffer);
}

/// @brief Read a vector in Parameter from input stream
/// @param[inout] is The input stream (could be input file stream)
/// @param[inout] vec The vector waiting to be written
/// @details This is the specialization of read_param() function for ClassicalDoubleVector.
/// The content would be: "[Descriptor]:\n[Values]\n", so need buffer to read descriptor and newline
template<>
void read_param<ClassicalDoubleVector>(std::istream& is, ClassicalDoubleVector& vec)
{
	std::string buffer;
	std::getline(is, buffer);
	for (int i = 0; i < Dim; i++)
	{
		is >> vec[i];
	}
	std::getline(is, buffer);
}

Parameters::Parameters(const std::string& input_file_name)
{
	// read input
	std::ifstream in(input_file_name);
	read_param(in, mass);
	read_param(in, x0);
	read_param(in, p0);
	read_param(in, SigmaP0);
	read_param(in, OutputTime);
	read_param(in, ReoptimizationTime);
	read_param(in, dt);
	read_param(in, NumberOfSelectedPoints);
	in.close();
	// deal with input
	xmax = x0.array().abs() * 2;
	xmin = -xmax;
	SigmaX0 = hbar / 2.0 * SigmaP0.array().inverse(); // follow the minimal uncertainty principle
	dx = M_PI * hbar * (p0 + 3.0 * SigmaP0).array().inverse(); // 4-5 grids per de Broglie wavelength
	xNumGrids = ((xmax - xmin).array() / dx.array()).cast<int>();
	dx = (xmax - xmin).array() / xNumGrids.cast<double>().array();
	// in grid solution, dx = pi*hbar/(p0+3sigma_p), p in p0+pi*hbar/2/dx*[-1, 1]
	pmax = p0.array() + M_PI * hbar / 2.0 * dx.array().inverse();
	pmin = p0.array() - M_PI * hbar / 2.0 * dx.array().inverse();
	pNumGrids = xNumGrids;
	dp = (pmax - pmin).array() / pNumGrids.cast<double>().array();
	// whole phase space grids
	PhasePoints.resize(PhaseDim, xNumGrids.prod() * pNumGrids.prod());
	for (int iPoint = 0; iPoint < PhasePoints.cols(); iPoint++)
	{
		ClassicalDoubleVector xPoint = xmin, pPoint = pmin;
		int NextIndex = iPoint;
		for (int idx = Dim - 1; idx >= 0; idx--)
		{
			pPoint[idx] += dp[idx] * (NextIndex % pNumGrids[idx]);
			NextIndex /= pNumGrids[idx];
		}
		for (int idx = Dim - 1; idx >= 0; idx--)
		{
			xPoint[idx] += dx[idx] * (NextIndex % xNumGrids[idx]);
			NextIndex /= xNumGrids[idx];
		}
		PhasePoints.col(iPoint) << xPoint, pPoint;
	}
}

std::ostream& print_time(std::ostream& os)
{
	const std::time_t CurrentTime = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
	return os << std::put_time(std::localtime(&CurrentTime), "%F %T %Z");
}

Eigen::MatrixXd print_point(const EvolvingDensity& density)
{
	const int NPoint = density.size();
	Eigen::MatrixXd result(PhaseDim, NPoint);
	for (int iPoint = 0; iPoint < NPoint; iPoint++)
	{
		const PhaseSpacePoint& psp = density[iPoint];
		result.col(iPoint) << std::get<0>(psp), std::get<1>(psp);
	}
	return result;
}
