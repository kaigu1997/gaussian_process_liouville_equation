/// @file input.h
/// @brief Interface of input functions

#ifndef INPUT_H
#define INPUT_H

#include "stdafx.h"

/// @brief Initial parameters from or calculated directly from input file
class InitialParameters final
{
private:
	const ClassicalVector<double> mass;		  ///< Mass of classical degree of freedom
	const ClassicalPhaseVector r0;			  ///< Initial position and momentum of classical degree of freedom
	const ClassicalVector<double> xmin;		  ///< Lower bound of position
	const ClassicalVector<double> xmax;		  ///< Upper bound of position
	const std::size_t NumGridsForOneDim;	  ///< Number of grids for one of the (position/momentum) dimensions
	const std::size_t NumGridsTotal;		  ///< Number of grids for all phase space dimensions
	const ClassicalVector<double> dx;		  ///< Grid spacing of position
	const ClassicalVector<double> pmin;		  ///< Lower bound of momentum
	const ClassicalVector<double> pmax;		  ///< Upper bound of momentum
	const ClassicalVector<double> dp;		  ///< Grid spacing of momentum
	const ClassicalPhaseVector rmin;		  ///< Lower bound of position and momentum of classical degree of freedom for the grids
	const ClassicalPhaseVector rmax;		  ///< Lower bound of position and momentum of classical degree of freedom for the grids
	const ClassicalPhaseVector dr;			  ///< Grid size of position and momentum of classical degree of freedom for the grids
	const ClassicalPhaseVector SigmaR0;		  ///< Initial standard deviation of classical degree of freedom
	const PhasePoints PhaseGrids;			  ///< Grids for plot phase space distribution
	const double dt;						  ///< The time interval for evolution
	const double ReoptimizationTime;		  ///< The time interval to redo the paramter optimization (in a.u.)
	const double OutputTime;				  ///< The time interval to give outputs (in a.u.)
	const std::size_t NumberOfSelectedPoints; ///< The size of training set for parameter optimization
	const std::size_t TotalTicks;			  ///< The maximum number of dt to finish the evolution

public:
	InitialParameters(void) = delete;

	/// @brief Constructor
	/// @param[in] mass Mass of classical degree of freedom
	/// @param[in] x0 Initial position
	/// @param[in] p0 Initial momentum
	/// @param[in] sigma_p0 Initial standard deviation of momentum
	/// @param[in] output_time The interval of output in unit of atomic unit time
	/// @param[in] re_optimization_time The inverval of optimizations in unit of atomic unit time
	/// @param[in] dt_ The interval of evolution in unit of atomic unit time
	/// @param[in] num_points The number of points selected for each elements
	InitialParameters(
		const ClassicalVector<double>& mass_,
		const ClassicalVector<double>& x0,
		const ClassicalVector<double>& p0,
		const ClassicalVector<double>& sigma_p0,
		const double output_time,
		const double re_optimization_time,
		const double dt_,
		const std::size_t num_points);

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

/// @brief To return the instance of @p InitialParameters
/// @param[in] input_file_name The input file name
/// @return Initial parameters
InitialParameters read_input(const std::string_view& input_file_name);

#endif // !INPUT_H
