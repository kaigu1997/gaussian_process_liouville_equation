/// @file input.cpp
/// @brief Implementation of input.h

#include "stdafx.h"

#include "input.h"

/// @brief The number of grids should be no more than that, to prevent too big output files (.txt and .gif)
static constexpr std::size_t MaximumGridsForOneDimension = 200;

/// @brief Read a parameter from input stream
/// @tparam T The input data type
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
/// It deals with 2 cases: @p Dim numbers of input, or 1 input (that fills the vector)
template <>
ClassicalVector<double> read_param<ClassicalVector<double>>(std::istream& is)
{
	std::string buffer;
	ClassicalVector<double> result;
	std::getline(is, buffer);
	std::getline(is, buffer);
	std::size_t start = 0, end = buffer.find(' '), idx = 0;
	do
	{
		result[idx] = std::stod(buffer.substr(start, end - start));
		idx++;
		start = end + 1;
		end = buffer.find(' ', start);
	}
	while (idx < Dim && end != std::string::npos);
	assert(Dim % idx == 0);
	if (idx != Dim)
	{
		const std::size_t div = Dim / idx;
		for (std::size_t i = 1; i < div; i++)
		{
			result(Eigen::seqN(i * idx, idx)) = result(Eigen::seqN(0, idx));
		}
	}
	return result;
}

/// @brief To combine x and p into a phase vector
/// @param[in] x The position part
/// @param[in] p The momentum part
/// @return A phase vector, whose first @p Dim elements are the same as @p x, last as @p p
static inline ClassicalPhaseVector combine_into_phase_coordinates(const Eigen::DenseBase<ClassicalVector<double>>& x, const Eigen::DenseBase<ClassicalVector<double>>& p)
{
	ClassicalPhaseVector result;
	result << x, p;
	return result;
}

InitialParameters::InitialParameters(
	const ClassicalVector<double>& Mass,
	const ClassicalVector<double>& x0,
	const ClassicalVector<double>& p0,
	const ClassicalVector<double>& sigma_p0,
	const double output_time,
	const double re_optimization_time,
	const double dt_,
	const std::size_t num_points
):
	mass(Mass),
	r0(combine_into_phase_coordinates(x0, p0)),
	xmin(-2.0 * x0.array().abs()),
	xmax(-xmin),
	NumGridsForOneDim(std::max(MaximumGridsForOneDimension, ((xmax - xmin).array() / (M_PI_2 * hbar * (p0 + 3.0 * sigma_p0).array().inverse())).cast<std::size_t>().maxCoeff() + 1)),
	NumGridsTotal(power<PhaseDim>(NumGridsForOneDim)),
	dx((xmax - xmin) / NumGridsForOneDim),
	pmin(p0.array() - M_PI_2 * hbar * dx.array().inverse()),
	pmax(p0.array() + M_PI_2 * hbar * dx.array().inverse()),
	dp((pmax - pmin) / NumGridsForOneDim),
	rmin(combine_into_phase_coordinates(xmin, pmin)),
	rmax(combine_into_phase_coordinates(xmax, pmax)),
	dr(combine_into_phase_coordinates(dx, dp)),
	SigmaR0(combine_into_phase_coordinates(ClassicalVector<double>(hbar / 2.0 * sigma_p0.array().inverse()), sigma_p0)),
	PhaseGrids(
		[NumGrids = NumGridsTotal, NumGridsForOneDim = NumGridsForOneDim, &rmin = rmin, &dr = dr](void) -> PhasePoints
		{
			PhasePoints result(PhasePoints::RowsAtCompileTime, NumGrids);
			const auto indices = xt::arange(NumGrids);
			std::for_each(
				std::execution::par_unseq,
				indices.cbegin(),
				indices.cend(),
				[&result, NumGridsForOneDim, &rmin, &dr](std::size_t iPoint) -> void
				{
					result.col(iPoint) = rmin;
					std::size_t NextIndex = iPoint;
					for (std::size_t idx = PhaseDim - 1; idx < PhaseDim; idx--)
					{
						result(idx, iPoint) += dr[idx] * (NextIndex % NumGridsForOneDim);
						NextIndex /= NumGridsForOneDim;
					}
				}
			);
			return result;
		}()
	),
	dt(dt_),
	ReoptimizationTime(std::max(re_optimization_time, dt)),
	OutputTime(std::max(output_time, ReoptimizationTime)),
	NumberOfSelectedPoints(num_points),
	TotalTicks(static_cast<std::size_t>(2.0 * (2.0 * x0.array() * mass.array() / p0.array()).abs().maxCoeff() / dt))
{
}

InitialParameters read_input(const std::string_view& input_file_name)
{
	std::ifstream in(input_file_name.data());
	// 1. mass
	const ClassicalVector<double> mass = read_param<ClassicalVector<double>>(in);
	// 2-3. x0 and p0
	const ClassicalVector<double> x0 = read_param<ClassicalVector<double>>(in);
	const ClassicalVector<double> p0 = read_param<ClassicalVector<double>>(in);
	// 4. sigma p
	const ClassicalVector<double> sigma_p0 = read_param<ClassicalVector<double>>(in);
	// 5. output time
	const double OutputTime = read_param<double>(in);
	// 6. re-optimization time
	const double ReOptimizationTime = read_param<double>(in);
	// 7. dt (that is the last one)
	const double dt = read_param<double>(in);
	// 8. number of points selected
	const std::size_t NumPoints = read_param<std::size_t>(in);
	// close the input file
	in.close();
	// then construct the parameters
	return InitialParameters(mass, x0, p0, sigma_p0, OutputTime, ReOptimizationTime, dt, NumPoints);
}
