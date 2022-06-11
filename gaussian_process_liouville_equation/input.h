/// @file input.h
/// @brief Interface of input functions

#ifndef INPUT_H
#define INPUT_H

#include "stdafx.h"

/// @brief Initial parameters from or calculated directly from input file
class InitialParameters final
{
private:
	ClassicalVector<double> mass;		///< Mass of classical degree of freedom
	ClassicalPhaseVector r0;			///< Initial position and momentum of classical degree of freedom
	ClassicalPhaseVector rmin;			///< Lower bound of position and momentum of classical degree of freedom for the grids
	ClassicalPhaseVector rmax;			///< Lower bound of position and momentum of classical degree of freedom for the grids
	ClassicalPhaseVector dr;			///< Grid size of position and momentum of classical degree of freedom for the grids
	ClassicalPhaseVector SigmaR0;		///< Initial standard deviation of classical degree of freedom
	PhasePoints PhaseGrids;				///< Grids for plot phase space distribution
	double OutputTime;					///< The time interval to give outputs (in a.u.)
	double ReoptimizationTime;			///< The time interval to redo the paramter optimization (in a.u.)
	double dt;							///< The time interval for evolution
	std::size_t NumberOfSelectedPoints; ///< The size of training set for parameter optimization
	std::size_t TotalTicks;				///< The maximum number of dt to finish the evolution

public:
	/// @brief Constructor, read input from file
	/// @param[in] input_file_name The input file name
	InitialParameters(const std::string& input_file_name);

	// The interface for data accessing, inline will make them faster
	/// @brief To get classical mass(es)
	/// @return Classical mass(es)
	const ClassicalVector<double>& get_mass(void) const
	{
		return mass;
	}

	/// @brief To get initial position(s)
	/// @return Initial position(s)
	const ClassicalPhaseVector& get_r0(void) const
	{
		return r0;
	}

	/// @brief To get minimum position(s) possible
	/// @return Minimum position(s) possible
	const ClassicalPhaseVector& get_rmin(void) const
	{
		return rmin;
	}

	/// @brief To get maximum position(s) possible
	/// @return Maximum position(s) possible
	const ClassicalPhaseVector& get_rmax(void) const
	{
		return rmax;
	}

	/// @brief To get maximum replacement in position(s) in MC
	/// @return Maximum repalcement in position(s) in MC
	const ClassicalPhaseVector& get_dr(void) const
	{
		return dr;
	}

	/// @brief To get initial position variance(s)
	/// @return Initial position variance(s)
	const ClassicalPhaseVector& get_sigma_r0(void) const
	{
		return SigmaR0;
	}

	/// @brief To get all the phase space grids
	/// @return All the phase space points, one in a column
	const PhasePoints& get_phase_space_points(void) const
	{
		return PhaseGrids;
	}

	/// @brief To get re-selection frequency
	/// @return Re-selection frequency, i.e. to re-select point by fitting current distribution every how many dt
	std::size_t get_reoptimize_freq(void) const
	{
		return static_cast<std::size_t>(ReoptimizationTime / dt);
	}

	/// @brief To get output frequency
	/// @return Output frequency, i.e. to output every how many dt
	/// @details To do less model-training, the program guarantees that
	/// there will be reselection at each output time
	std::size_t get_output_freq(void) const
	{
		return static_cast<std::size_t>(OutputTime / ReoptimizationTime) * static_cast<std::size_t>(ReoptimizationTime / dt);
	}

	/// @brief To get dt
	/// @return dt, or the time step
	double get_dt(void) const
	{
		return dt;
	}

	/// @brief To get the number of points to select
	/// @return The number of points selected each time
	std::size_t get_number_of_selected_points(void) const
	{
		return NumberOfSelectedPoints;
	}

	/// @brief To estimate the maximum time to finish and how many dt it corresponds to
	/// @return Estimated total time in unit of dt
	/// @details Estimate @f$ T=\frac{|x_0|-(-|x_0|)}{p_0/m} @f$ on each direction, choose the max one, then times 2.0 and divide dt
	std::size_t calc_total_ticks(void) const
	{
		return TotalTicks;
	}
};

#endif // !INPUT_H
