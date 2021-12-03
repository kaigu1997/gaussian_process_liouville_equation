/// @file output.h
/// @brief Interface of output functions

#ifndef OUTPUT_H
#define OUTPUT_H

#include "stdafx.h"

#include "input.h"
#include "mc.h"
#include "opt.h"
#include "predict.h"

/// @brief To output averages
/// @param[inout] os The output stream
/// @param[in] time Current evolution time
/// @param[in] Predictors An array of predictors for prediction, whose size is NumElements
/// @param[in] density The selected density matrices
/// @param[in] mass Mass of classical degree of freedom
void output_average(
	std::ostream& os,
	const double time,
	const Predictions& Predictors,
	const EigenVector<PhaseSpacePoint>& density,
	const ClassicalVector<double>& mass);

/// @brief To output parameters
/// @param[inout] os The output stream
/// @param[in] Optimizer It contains parameters
void output_param(std::ostream& os, const Optimization& Optimizer);

/// @brief To output the selected points
/// @param[inout] os The output stream
/// @param[in] density The selected density matrices
void output_point(std::ostream& os, const EigenVector<PhaseSpacePoint>& density);

/// @brief To output the gridded phase space distribution
/// @param[inout] os The output stream
/// @param[in] Predictors An array of predictors for prediction, whose size is NumElements
/// @param[in] PhaseGrids The phase space coordinates of the grids
void output_phase(std::ostream& os, const Predictions& Predictors, const Eigen::MatrixXd& PhaseGrids);

/// @brief To output the process of setting monte carlo steps by calculating autocorrelation
/// @param[inout] os The output stream
/// @param[inout] MCParams Monte carlo parameters (number of steps, maximum displacement, etc)
/// @param[in] dist The distribution function of the current partial Wigner-transformed density matrix
/// @param[in] IsSmall The matrix that saves whether each element is small or not
/// @param[in] density The selected density matrices
void output_autocor(
	std::ostream& os,
	MCParameters& MCParams,
	const DistributionFunction& dist,
	const QuantumMatrix<bool>& IsSmall,
	const EigenVector<PhaseSpacePoint>& density);

/// @brief To output some basic information
/// @param[inout] logging The output stream
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