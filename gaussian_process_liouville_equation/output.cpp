/// @file output.cpp
/// @brief Implementation of output.h

#include "stdafx.h"

#include "output.h"

#include "input.h"
#include "mc.h"
#include "mc_ave.h"
#include "opt.h"
#include "predict.h"

static const Eigen::IOFormat VectorFormatter(Eigen::StreamPrecision, Eigen::DontAlignCols, " ", " ", "", "", " ", "");  ///< Formatter for output vector
static const Eigen::IOFormat MatrixFormatter(Eigen::StreamPrecision, Eigen::DontAlignCols, " ", "\n", " ", "", "", ""); ///< Formatter for output matrix

/// @brief To print current time in "yyyy-mm-dd hh:mm:ss time_zone" mode
/// @param[inout] os The output stream
/// @return The same output stream
static std::ostream& print_time(std::ostream& os)
{
	const std::time_t CurrentTime = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
	return os << std::put_time(std::localtime(&CurrentTime), "%F %T %Z");
}

void output_average(
	std::ostream& os,
	const double time,
	const Predictions& Predictors,
	const EigenVector<PhaseSpacePoint>& density,
	const ClassicalVector<double>& mass)
{
	// average: time, population, x, p, V, T, E of each PES
	double ppl_all = 0.;
	ClassicalPhaseVector r_prm_all = ClassicalPhaseVector::Zero();
	ClassicalPhaseVector r_mc_all = ClassicalPhaseVector::Zero();
	double T_all = 0.;
	double V_all = 0.;
	os << time;
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		const int ElementIndex = iPES * (NumPES + 1);
		const double ppl_prm = Predictors[ElementIndex].has_value() ? Predictors[ElementIndex]->calculate_population() : 0.0; // prm = parameter
		const ClassicalPhaseVector r_prm = Predictors[ElementIndex].has_value() ? Predictors[ElementIndex]->calculate_1st_order_average() : ClassicalPhaseVector::Zero();
		const ClassicalPhaseVector r_mc = calculate_1st_order_average(density, iPES);
		const double T_mc = calculate_kinetic_energy_average(density, iPES, mass), V_mc = calculate_potential_energy_average(density, iPES);
		os << ' ' << ppl_prm << r_prm.format(VectorFormatter) << r_mc.format(VectorFormatter) << ' ' << T_mc << ' ' << V_mc << ' ' << V_mc + T_mc;
		ppl_all += ppl_prm;
		r_prm_all += ppl_prm * r_prm;
		r_mc_all += ppl_prm * r_mc;
		T_all += ppl_prm * T_mc;
		V_all += ppl_prm * V_mc;
	}
	os << ' ' << ppl_all << r_prm_all.format(VectorFormatter) << r_mc_all.format(VectorFormatter) << ' ' << T_all << ' ' << V_all << ' ' << V_all + T_all;
	os << ' ' << calculate_purity(Predictors) << std::endl;
}

void output_param(std::ostream& os, const Optimization& Optimizer)
{
	for (int iElement = 0; iElement < NumElements; iElement++)
	{
		for (double d : Optimizer.get_parameter(iElement))
		{
			os << ' ' << d;
		}
		os << '\n';
	}
	os << '\n';
}

void output_point(std::ostream& os, const EigenVector<PhaseSpacePoint>& density)
{
	assert(density.size() % NumElements == 0);
	const int NumPoints = density.size() / NumElements;
	Eigen::MatrixXd result(PhaseDim * NumElements, NumPoints);
	for (int iElement = 0; iElement < NumElements; iElement++)
	{
		const int StartRow = iElement * PhaseDim;
		const int StartPoint = iElement * NumPoints;
		for (int iPoint = 0; iPoint < NumPoints; iPoint++)
		{
			const auto& [x, p, rho] = density[iPoint + StartPoint];
			result.block<PhaseDim, 1>(StartRow, iPoint) << x, p;
		}
	}
	os << result.format(MatrixFormatter) << "\n\n";
}

void output_phase(std::ostream& os, const Predictions& Predictors, const Eigen::MatrixXd& PhaseGrids)
{
	for (int iElement = 0; iElement < NumElements; iElement++)
	{
		if (Predictors[iElement].has_value())
		{
			os << Predictors[iElement]->predict_elements(PhaseGrids).format(VectorFormatter);
		}
		else
		{
			os << Eigen::VectorXd::Zero(PhaseGrids.cols()).format(VectorFormatter);
		}
		os << '\n';
	}
	os << std::endl;
}

void output_autocor(
	std::ostream& os,
	MCParameters& MCParams,
	const DistributionFunction& dist,
	const QuantumMatrix<bool>& IsSmall,
	const EigenVector<PhaseSpacePoint>& density)
{
	const AutoCorrelations step_autocor = autocorrelation_optimize_steps(MCParams, dist, IsSmall, density);
	const AutoCorrelations displ_autocor = autocorrelation_optimize_displacement(MCParams, dist, IsSmall, density);
	for (int iElement = 0; iElement < NumElements; iElement++)
	{
		os << step_autocor[iElement].format(VectorFormatter) << displ_autocor[iElement].format(VectorFormatter) << '\n';
	}
	os << std::endl;
}

void output_logging(
	std::ostream& os,
	const double time,
	const Optimization::Result& OptResult,
	const MCParameters& MCParams,
	const std::chrono::duration<double>& CPUTime)
{
	const auto& [error, Steps] = OptResult;
	os << time << ' ' << error << ' ' << MCParams.get_num_MC_steps() << ' ' << MCParams.get_max_dispalcement();
	for (auto step : Steps)
	{
		os << ' ' << step;
	}
	os << ' ' << CPUTime.count() << ' ' << print_time << std::endl;
}