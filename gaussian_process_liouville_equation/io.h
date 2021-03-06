/// @file io.h
/// @brief Interface of input/output functions

#ifndef IO_H
#define IO_H

#include "stdafx.h"

/// The vector used in classical degree (e.g. number of grids on x/p) with int values
using ClassicalIntVector = Eigen::Matrix<int, Dim, 1>;

/// parameters from input file or calculated directly
class Parameters final
{
private:
	ClassicalDoubleVector mass, x0, xmin, xmax, dx, p0, pmin, pmax, dp, SigmaX0, SigmaP0;
	ClassicalIntVector xNumGrids, pNumGrids;
	Eigen::MatrixXd PhasePoints;
	double OutputTime, ReoptimizationTime, dt;
	int NumberOfSelectedPoints;

public:
	/// @brief Constructor, read input from file
	/// @param[in] input_file_name The input file name
	Parameters(const std::string& input_file_name);

	// interface for data accessing, inline will make them faster
	/// @brief To get classical mass(es)
	/// @return Classical mass(es)
	const ClassicalDoubleVector& get_mass(void) const
	{
		return mass;
	}

	/// @brief To get initial position(s)
	/// @return Initial position(s)
	const ClassicalDoubleVector& get_x0(void) const
	{
		return x0;
	}

	/// @brief To get minimum position(s) possible
	/// @return Minimum position(s) possible
	const ClassicalDoubleVector& get_xmin(void) const
	{
		return xmin;
	}

	/// @brief To get maximum position(s) possible
	/// @return Maximum position(s) possible
	const ClassicalDoubleVector& get_xmax(void) const
	{
		return xmax;
	}

	/// @brief To get maximum replacement in position(s) in MC
	/// @return Maximum repalcement in position(s) in MC
	const ClassicalDoubleVector& get_dx(void) const
	{
		return dx;
	}

	/// @brief To get initial position variance(s)
	/// @return Initial position variance(s)
	const ClassicalDoubleVector& get_sigma_x0(void) const
	{
		return SigmaX0;
	}

	/// @brief To get the number of grids of each dimension of x
	/// @return The number of grids of each x dimension
	const ClassicalIntVector& get_num_grids_on_x(void) const
	{
		return xNumGrids;
	}

	/// @brief To get initial momentum/momenta
	/// @return Initial momentum/momenta
	const ClassicalDoubleVector& get_p0(void) const
	{
		return p0;
	}

	/// @brief To get minimum momentum/momenta possible
	/// @return Minimum momentum/momenta possible
	const ClassicalDoubleVector& get_pmin(void) const
	{
		return pmin;
	}

	/// @brief To get maximum momentum/momenta possible
	/// @return Maximum momentum/momenta possible
	const ClassicalDoubleVector& get_pmax(void) const
	{
		return pmax;
	}

	/// @brief To get maximum replacement in momentum/momenta in MC
	/// @return Maximum repalcement in momentum/momenta in MC
	const ClassicalDoubleVector& get_dp(void) const
	{
		return dp;
	}

	/// @brief To get initial momentum variance(s)
	/// @return Initial momentum variance(s)
	const ClassicalDoubleVector& get_sigma_p0(void) const
	{
		return SigmaP0;
	}

	/// @brief To get the number of grids of each dimension of p
	/// @return The number of grids of each p dimension
	const ClassicalIntVector& get_num_grids_on_p(void) const
	{
		return pNumGrids;
	}

	/// @brief To get all the phase space grids
	/// @return All the phase space points, one in a column
	const Eigen::MatrixXd& get_phase_space_points(void) const
	{
		return PhasePoints;
	}

	/// @brief To get re-selection frequency
	/// @return Re-selection frequency, i.e. to re-select point by fitting current distribution every how many dt
	int get_reoptimize_freq(void) const
	{
		return static_cast<int>(ReoptimizationTime / dt);
	}

	/// @brief To get output frequency
	/// @return Output frequency, i.e. to output every how many dt
	/// @details To do less model-training, the program guarantees that
	/// there will be reselection at each output time
	int get_output_freq(void) const
	{
		return static_cast<int>(OutputTime / ReoptimizationTime) * static_cast<int>(ReoptimizationTime / dt);
	}

	/// @brief To get dt
	/// @return dt, or the time step
	double get_dt(void) const
	{
		return dt;
	}

	/// @brief To get the number of points to select
	/// @return The number of points selected each time
	int get_number_of_selected_points(void) const
	{
		return NumberOfSelectedPoints;
	}

	/// @brief To estimate the maximum time to finish and how many dt it corresponds to
	/// @return Estimated total time in unit of dt
	/// @details Estimate T=(|x0|-(-|x0|))/(p0/m) on each direction, choose the max one, then times 2.0 and divide dt
	int calc_total_ticks(void) const
	{
		const auto Distance = 2.0 * x0.array().abs();
		const auto Speed = (p0.array() / mass.array()).abs();
		const double TotalTime = (Distance / Speed).maxCoeff();
		return static_cast<int>(2.0 * TotalTime / dt);
	}
};

/// @brief To print current time in "yyyy-mm-dd hh:mm:ss time_zone" mode
/// @param[inout] os The output stream
/// @return The same output stream
std::ostream& print_time(std::ostream& os);

/// @brief To format the selected phase points for printing
/// @param[in] density The vector of selected points with its density matrix
/// @return A PhaseDim-by-N matrix, N the number of points
/// @param NumPoints The number of points for each element
Eigen::MatrixXd print_point(const EvolvingDensity& density, const int NumPoints);

#endif // !IO_H
