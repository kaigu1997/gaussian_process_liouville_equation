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
/// @param[in] Kernels An array of kernels for prediction, whose size is NumElements
/// @param[in] density The selected points in phase space for each element of density matrices
/// @param[in] mc_points The selected points for calculating mc integration
/// @param[in] mass Mass of classical degree of freedom
void output_average(
	std::ostream& os,
	const OptionalKernels& Kernels,
	const AllPoints& density,
	const AllPoints& mc_points,
	const ClassicalVector<double>& mass);

/// @brief To output parameters
/// @param[inout] os The output stream
/// @param[in] Optimizer The object of @p Opitimization class
void output_param(std::ostream& os, const Optimization& Optimizer);

/// @brief To output the selected points
/// @param[inout] os The output stream
/// @param[in] Points The selected density matrices
void output_point(std::ostream& os, const AllPoints& Points);

/// @brief To output the gridded phase space distribution, and variance on each point
/// @param[inout] phase The output stream for phase space distribution
/// @param[inout] variance The output stream for variance
/// @param[in] Kernels An array of kernels for prediction, whose size is NumElements
/// @param[in] PhaseGrids The phase space coordinates of the grids
void output_phase(std::ostream& phase, std::ostream& variance, const OptionalKernels& Kernels, const PhasePoints& PhaseGrids);

/// @brief To output the process of setting monte carlo steps by calculating autocorrelation
/// @param[inout] os The output stream
/// @param[inout] MCParams Monte carlo parameters (number of steps, maximum displacement, etc)
/// @param[in] dist The distribution function of the current partial Wigner-transformed density matrix
/// @param[in] density The selected points in phase space for each element of density matrices
void output_autocor(
	std::ostream& os,
	MCParameters& MCParams,
	const DistributionFunction& dist,
	const AllPoints& density);

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
	const std::chrono::duration<double>& CPUTime);

#endif // !OUTPUT_H