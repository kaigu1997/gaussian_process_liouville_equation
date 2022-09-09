/// @file output.h
/// @brief Interface of output functions

#ifndef OUTPUT_H
#define OUTPUT_H

#include "stdafx.h"

#include "input.h"
#include "kernel.h"
#include "mc.h"
#include "opt.h"
#include "predict.h"

/// @brief To output averages
/// @param[inout] os The output stream
/// @param[in] AllKernels Kernels of all elements for prediction
/// @param[in] density The selected points in phase space for each element of density matrices
/// @param[in] mass Mass of classical degree of freedom
void output_average(
	std::ostream& os,
	const Kernels& AllKernels,
	const AllPoints& density,
	const ClassicalVector<double>& mass
);

/// @brief To output parameters
/// @param[inout] os The output stream
/// @param[in] Optimizer The object of @p Opitimization class
void output_param(std::ostream& os, const Optimization& Optimizer);

/// @brief To output the selected points
/// @param[inout] coord The output stream for phase space coordinates of the points
/// @param[inout] value The output stream for the density of the points
/// @param[in] density The selected points in phase space for each element of density matrices
/// @param[in] extra_points The extra selected points to reduce overfitting
void output_point(std::ostream& coord, std::ostream& value, const AllPoints& density, const AllPoints& extra_points);

/// @brief To output the gridded phase space distribution, and variance on each point
/// @param[inout] phase The output stream for phase space distribution without adiabatic factor
/// @param[inout] phasefactor The output stream for phase factor
/// @param[inout] variance The output stream for variance
/// @param[in] mass Mass of classical degree of freedom
/// @param[in] dt Time interval
/// @param[in] NumTicks Number of @p dt since T=0; i.e., now is @p NumTicks * @p dt
/// @param[in] AllKernels Kernels of all elements for prediction
/// @param[in] PhaseGrids The phase space coordinates of the grids
void output_phase(
	std::ostream& phase,
	std::ostream& phasefactor,
	std::ostream& variance,
	const ClassicalVector<double>& mass,
	const double dt,
	const std::size_t NumTicks,
	const Kernels& AllKernels,
	const PhasePoints& PhaseGrids
);

/// @brief To output the process of setting monte carlo steps by calculating autocorrelation
/// @param[inout] os The output stream
/// @param[inout] MCParams Monte carlo parameters (number of steps, maximum displacement, etc)
/// @param[in] distribution The distribution function of the current partial Wigner-transformed density matrix
/// @param[in] density The selected points in phase space for each element of density matrices
void output_autocor(
	std::ostream& os,
	MCParameters& MCParams,
	const DistributionFunction& distribution,
	const AllPoints& density
);

/// @brief To output some basic information
/// @param[inout] os The output stream
/// @param[in] time Current evolution time
/// @param[in] OptResult The result of optimization, including total error and number of steps
/// @param[in] MCParams Monte carlo parameters (number of steps, maximum displacement, etc)
/// @param[in] CPUTime The time cost from last step to the current step
void output_logging(
	std::ostream& os,
	const double time,
	const Optimization::Result& OptResult,
	const MCParameters& MCParams,
	const std::chrono::duration<double>& CPUTime
);

#endif // !OUTPUT_H
