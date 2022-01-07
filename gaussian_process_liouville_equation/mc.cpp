/// @file mc.cpp
/// @brief Implementation of mc.h

#include "stdafx.h"

#include "mc.h"

#include "input.h"
#include "pes.h"

/// All the distributions of displacements of position OR momentum
using Displacements = std::array<std::uniform_real_distribution<double>, PhaseDim>;

QuantumMatrix<bool> is_very_small(const AllPoints& density)
{
	// squared value below this are regarded as 0
	static const double epsilon = 1e-4;
	QuantumMatrix<bool> result = QuantumMatrix<bool>::Ones();
	// as density is symmetric, only lower-triangular elements needs evaluating
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		for (int jPES = 0; jPES <= iPES; jPES++)
		{
			if (result(iPES, jPES))
			{
				const int ElementIndex = iPES * NumPES + jPES;
				result(iPES, jPES) &= std::all_of(
					std::execution::par_unseq,
					density[ElementIndex].cbegin(),
					density[ElementIndex].cend(),
					[iPES, jPES](const PhaseSpacePoint& psp) -> bool
					{
						return weight_function(std::get<2>(psp)(iPES, jPES)) < epsilon;
					});
			}
		}
	}
	return result.selfadjointView<Eigen::Lower>();
}

/// @details Under the adiabatic basis, only @f$ rho_{00} @f$ is non-zero initially,
/// i.e. all the population lies on the ground state. For @f$ rho_{00} @f$,
///
/// @f[
/// P(\bf{x},\bf{p})=\prod_i{P(x_i,p_i)}=\prod_i{\left(\frac{1}{2\pi\sigma_{x_i}\sigma_{p_i}}\exp\left[-\frac{(x_i-x_{i0})^2}{2\sigma_{x_i}^2}-\frac{(p_i-p_{i0})^2}{2\sigma_{p_i}^2}\right]\right)}
/// @f]
///
/// where i goes over all direction in the classical DOF
QuantumMatrix<std::complex<double>> initial_distribution(
	const Parameters& Params,
	const std::array<double, NumPES>& InitialPopulation,
	const ClassicalVector<double>& x,
	const ClassicalVector<double>& p)
{
	QuantumMatrix<std::complex<double>> result = QuantumMatrix<std::complex<double>>::Zero(NumPES, NumPES);
	const ClassicalVector<double>& x0 = Params.get_x0();
	const ClassicalVector<double>& p0 = Params.get_p0();
	const ClassicalVector<double>& SigmaX0 = Params.get_sigma_x0();
	const ClassicalVector<double>& SigmaP0 = Params.get_sigma_p0();
	const double GaussWeight = std::exp(-(((x - x0).array() / SigmaX0.array()).square().sum() + ((p - p0).array() / SigmaP0.array()).square().sum()) / 2.0)
		/ (std::pow(2.0 * M_PI, Dim) * SigmaX0.prod() * SigmaP0.prod());
	const double SumWeight = std::reduce(InitialPopulation.begin(), InitialPopulation.end());
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		result(iPES, iPES) = GaussWeight * InitialPopulation[iPES] / SumWeight;
	}
	return result;
}

const double MCParameters::AboveMinFactor = 1.1; ///< choose the minimal step or max displacement whose autocor < factor*min{autocor}

MCParameters::MCParameters(const Parameters& Params, const int NOMC_) :
	NumPoints(Params.get_number_of_selected_points()),
	NOMC(NOMC_),
	displacement(std::min(Params.get_sigma_x0().minCoeff(), Params.get_sigma_p0().minCoeff()))
{
}

/// @brief The weight function for sorting / MC selection
/// @param x The input variable
/// @return The weight of the input, which is the square of it
static inline double weight_function(const double& x)
{
	return x * x;
}

/// @brief To combine density into one array
/// @param[in] density The selected points in phase space for each element of density matrices
/// @return The combination of the paramere
static std::tuple<EigenVector<PhaseSpacePoint>, QuantumMatrix<bool>> pack_density(const AllPoints& density)
{
	EigenVector<PhaseSpacePoint> combination;
	QuantumMatrix<bool> is_small = QuantumMatrix<bool>::Zero();
	// as density is symmetric, only lower-triangular elements needs evaluating
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		for (int jPES = 0; jPES <= iPES; jPES++)
		{
			const int ElementIndex = iPES * NumPES + jPES;
			combination.insert(combination.end(), density[ElementIndex].cbegin(), density[ElementIndex].cend());
			is_small(iPES, jPES) = density[ElementIndex].empty();
		}
	}
	return std::make_tuple(combination, is_small.selfadjointView<Eigen::Lower>());
}

/// @brief To select phase points as the initial phase coordinates for monte carlo
/// @param[in] MCParams Monte carlo parameters (number of steps, maximum displacement, etc)
/// @param[in] IsSmall The matrix that saves whether each element is small or not
/// @param[in] density The combination of selected points in phase space for each element of density matrices
/// @return The phase points with their density matrices
/// @details For each element, this function chooses the n points with largest weight of that element.
///
/// For off-diagonal element, the weight is its squared norm, so the return will be @"symmetric@".
static AllPoints get_maximums(
	const MCParameters& MCParams,
	const QuantumMatrix<bool>& IsSmall,
	const EigenVector<PhaseSpacePoint>& density)
{
	const int NumPoints = MCParams.get_num_points();
	AllPoints maxs;
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		for (int jPES = 0; jPES < NumPES; jPES++)
		{
			if (iPES <= jPES)
			{
				// for row-major, visit upper triangular elements earlier
				// so only deal with upper-triangular
				if (!IsSmall(iPES, jPES))
				{
					const int ElementIndex = iPES * NumPES + jPES;
					maxs[ElementIndex] = density;
					// find the max n elements
					std::nth_element(
						std::execution::par_unseq,
						maxs[ElementIndex].begin(),
						maxs[ElementIndex].begin() + NumPoints,
						maxs[ElementIndex].end(),
						[iPES, jPES](const PhaseSpacePoint& psp1, const PhaseSpacePoint& psp2) -> bool
						{
							return weight_function(std::get<2>(psp1)(iPES, jPES)) > weight_function(std::get<2>(psp2)(iPES, jPES));
						});
					maxs[ElementIndex].erase(maxs[ElementIndex].cbegin() + NumPoints, maxs[ElementIndex].cend());
				}
			}
			else
			{
				// for strict lower triangular elements, copy the upper part
				maxs[iPES * NumPES + jPES] = maxs[jPES * NumPES + iPES];
			}
		}
	}
	return maxs;
}

/// @brief To select phase points as the initial phase coordinates for monte carlo
/// @param[in] Params The input parameters (mass, dt, etc)
/// @param[in] MCParams Monte carlo parameters (number of steps, maximum displacement, etc)
/// @param[in] distribution The function object to generate the distribution
/// @param[in] IsSmall The matrix that saves whether each element is small or not
/// @param[in] density The combination of selected points in phase space for each element of density matrices
/// @return The phase points with their density matrices
/// @details The points are at an estimated place that they will be at next time step.
///
/// Similar to the other get_maximums(), this function will evolve the largest and give a @"symmetric@" return.
static AllPoints get_maximums(
	const Parameters& Params,
	const MCParameters& MCParams,
	const DistributionFunction& distribution,
	const QuantumMatrix<bool>& IsSmall,
	const EigenVector<PhaseSpacePoint>& density)
{
	// alias names
	const ClassicalVector<double>& mass = Params.get_mass();
	const double dt = Params.get_dt();
	AllPoints maxs = get_maximums(MCParams, IsSmall, density);
	// only deal with upper-triangular
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		for (int jPES = 0; jPES < NumPES; jPES++)
		{
			if (iPES <= jPES)
			{
				// for row-major, visit upper triangular elements earlier
				// so only deal with upper-triangular
				if (!IsSmall(iPES, jPES))
				{
					const int ElementIndex = iPES * NumPES + jPES;
					// evolve them
					std::for_each(
						std::execution::par_unseq,
						maxs[ElementIndex].begin(),
						maxs[ElementIndex].end(),
						[&distribution, &mass, dt, iPES, jPES](PhaseSpacePoint& psp) -> void
						{
							auto& [x, p, rho] = psp;
							const Tensor3d f = adiabatic_force(x);
							x = x.array() + p.array() / mass.array() * dt;
							for (int iDim = 0; iDim < Dim; iDim++)
							{
								p[iDim] += (f[iDim](iPES, iPES) + f[iDim](jPES, jPES)) / 2.0 * dt;
							}
							rho = distribution(x, p);
						});
				}
			}
			else
			{
				// for strict lower triangular elements, copy the upper part
				maxs[iPES * NumPES + jPES] = maxs[jPES * NumPES + iPES];
			}
		}
	}
	return maxs;
}

/// @brief To calculate all the displacements
/// @param[in] MaxDisplacement The possible maximum displacement
/// @return The displacement distribution of positions and momenta
static Displacements generate_displacements(const double MaxDisplacement)
{
	Displacements result;
	for (int iDim = 0; iDim < PhaseDim; iDim++)
	{
		result[iDim] = std::uniform_real_distribution<double>(-MaxDisplacement, MaxDisplacement);
	}
	return result;
}

/// @brief To generate the Markov chain monte carlo of Metropolis scheme
/// @param[in] NumSteps Number of steps for the monte carlo chain
/// @param[in] distribution The function object to generate the distribution
/// @param[in] XDispl The displacement distribution of positions
/// @param[in] PDispl The displacement distribution of momenta
/// @param[in] iPES The index of the row of the density matrix element
/// @param[in] jPES The index of the column of the density matrix element
/// @param[in] x The initial position
/// @param[in] p The initial momentum
/// @return A vector containing the phase space coordinates of all grids
static EigenVector<ClassicalPhaseVector> monte_carlo_chain(
	const int NumSteps,
	const DistributionFunction& distribution,
	Displacements& Displ,
	const int iPES,
	const int jPES,
	const ClassicalVector<double>& x,
	const ClassicalVector<double>& p)
{
	// set parameters
	static std::mt19937 engine(std::chrono::system_clock::now().time_since_epoch().count()); // Random number generator, using merseen twister with seed from time
	static std::uniform_real_distribution<double> mc_selection(0.0, 1.0);					 // for whether pick or not
	ClassicalVector<double> x_old = x, p_old = p;
	double weight_old = weight_function(distribution(x_old, p_old)(iPES, jPES));
	EigenVector<ClassicalPhaseVector> r_all(NumSteps + 1);
	r_all[0] << x_old, p_old;
	// going the chain
	for (int iIter = 1; iIter <= NumSteps; iIter++)
	{
		ClassicalVector<double> x_new = x_old, p_new = p_old;
		for (int iDim = 0; iDim < Dim; iDim++)
		{
			x_new[iDim] += Displ[iDim](engine);
			p_new[iDim] += Displ[iDim + Dim](engine);
		}
		const double weight_new = weight_function(distribution(x_new, p_new)(iPES, jPES));
		if (weight_new > weight_old || weight_new / weight_old > weight_function(mc_selection(engine)))
		{
			// new weight is larger than the random number, accept
			// otherwise, reject
			x_old = x_new;
			p_old = p_new;
			weight_old = weight_new;
		}
		r_all[iIter] << x_old, p_old;
	}
	return r_all;
}

/// @details Assuming that the @p distribution function containing all the required information.
/// If it is a distribution from some parameters, those parameters should have already been
/// bound; if it is some learning method, the parameters should have already been optimized.
///
/// This function uses Metropolis algorithm of Markov chain monte carlo.
void monte_carlo_selection(
	const Parameters& Params,
	const MCParameters& MCParams,
	const DistributionFunction& distribution,
	AllPoints& density,
	const bool IsToBeEvolved)
{
	// get maximum, then evolve the points adiabatically
	auto [combined_density, IsSmall] = pack_density(density);
	AllPoints maxs = IsToBeEvolved
		? get_maximums(Params, MCParams, distribution, IsSmall, combined_density)
		: get_maximums(MCParams, IsSmall, combined_density);

	// moving random number generator and MC random number generator
	Displacements displacements = generate_displacements(MCParams.get_max_displacement());

	// Monte Carlo selection
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		for (int jPES = 0; jPES < NumPES; jPES++)
		{
			if (iPES <= jPES)
			{
				const int ElementIndex = iPES * NumPES + jPES;
				// for row-major, visit upper triangular elements earlier
				// so selection for the upper triangular elements
				if (!IsSmall(iPES, jPES))
				{
					// begin from an old point
					std::for_each(
						std::execution::par_unseq,
						maxs[ElementIndex].begin(),
						maxs[ElementIndex].end(),
						[NumSteps = MCParams.get_num_MC_steps(), &distribution, &displacements, iPES, jPES](PhaseSpacePoint& psp) -> void
						{
							auto& [x, p, rho] = psp;
							const EigenVector<ClassicalPhaseVector> WholeChain = monte_carlo_chain(
								NumSteps,
								distribution,
								displacements,
								iPES,
								jPES,
								x,
								p);
							x = WholeChain.back().block<Dim, 1>(0, 0);
							p = WholeChain.back().block<Dim, 1>(Dim, 0);
							rho = distribution(x, p);
						});
					// after MC, the new point is at the original place
					// TODO: then update the number of MC steps
					// after all points updated, insert into the EigenVector<PhaseSpacePoint>
					density[ElementIndex] = maxs[ElementIndex];
				}
				else
				{
					density[ElementIndex].clear();
				}
			}
			else
			{
				// for strict lower triangular elements, copy the upper part
				density[iPES * NumPES + jPES] = density[jPES * NumPES + iPES];
			}
		}
	}
}

/// @brief To calculate the autocorrelation of one thread
/// @param[in] r The phase space coordinates on the thread
/// @return The autocorrelation of different number of steps
/// @details For j steps, the autocorrelation is
/// @f[
/// \langle R_{i+j} R_i\rangle=\frac{1}{N-j}\sum_{i=1}^{N-j}R_{i+j}R_i-|\overline{r}|^2
/// @f]
static Eigen::VectorXd calculate_autocorrelation(const EigenVector<ClassicalPhaseVector>& r)
{
	const int NSteps = r.size();
	Eigen::VectorXd result(NSteps / 2);
	// first, calculate average (aka sum / N)
	const ClassicalPhaseVector Ave = std::reduce(r.begin(), r.end(), ClassicalPhaseVector(ClassicalPhaseVector::Zero())) / NSteps;
	// then, calculate for different j
	for (int j = 0; j < NSteps / 2; j++)
	{
		double sum = 0;
		for (int i = 0; i + j < NSteps; i++)
		{
			sum += (r[i] - Ave).dot(r[i + j] - Ave);
		}
		result[j] = sum / (NSteps - j);
	}
	return result;
}

/// @brief To calculate the autocorrelation of one thread of given autocorrelation steps
/// @param[in] r The phase space coordinates on the thread
/// @param[in] AutoCorSteps The given autocorrelation steps
/// @return The autocorrelation
static double calculate_autocorrelation(const EigenVector<ClassicalPhaseVector>& r, const int AutoCorSteps)
{
	const int NSteps = r.size();
	const ClassicalPhaseVector Ave = std::reduce(r.begin(), r.end(), ClassicalPhaseVector(ClassicalPhaseVector::Zero())) / NSteps;
	double sum = 0;
	for (int i = 0; i + AutoCorSteps < NSteps; i++)
	{
		sum += (r[i] - Ave).dot(r[i + AutoCorSteps] - Ave);
	}
	return sum / (NSteps - AutoCorSteps);
}

AutoCorrelations autocorrelation_optimize_steps(
	MCParameters& MCParams,
	const DistributionFunction& distribution,
	const AllPoints& density)
{
	// parameters needed for MC
	static const int MaxNOMC = PhaseDim * 400; // The number of monte carlo steps (including head and tail)
	static const int MaxAutoCorStep = (MaxNOMC + 1) / 2;
	const auto& [CombinedDensity, IsSmall] = pack_density(density);
	const AllPoints maxs = get_maximums(MCParams, IsSmall, CombinedDensity);

	// moving random number generator and MC random number generator
	Displacements displacements = generate_displacements(MCParams.get_max_displacement());

	// calculation
	AutoCorrelations result; // save the return value, the autocorrelations
	result.fill(Eigen::VectorXd::Zero(MaxAutoCorStep));
	int best_MC_steps = std::numeric_limits<int>::max();
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		for (int jPES = 0; jPES < NumPES; jPES++)
		{
			if (iPES <= jPES)
			{
				// for row-major, visit upper triangular elements earlier
				// so only deal with upper-triangular
				if (!IsSmall(iPES, jPES))
				{
					const int ElementIndex = iPES * NumPES + jPES;
					result[ElementIndex] =
						std::transform_reduce(
							std::execution::par_unseq,
							maxs[ElementIndex].cbegin(),
							maxs[ElementIndex].cend(),
							Eigen::VectorXd(Eigen::VectorXd::Zero(MaxAutoCorStep)),
							std::plus<Eigen::VectorXd>(),
							[&distribution, &displacements, iPES, jPES](const PhaseSpacePoint& psp) -> Eigen::VectorXd
							{
								const auto& [x, p, rho] = psp;
								return calculate_autocorrelation(monte_carlo_chain(
									MaxNOMC,
									distribution,
									displacements,
									iPES,
									jPES,
									x,
									p));
							})
						/ maxs[ElementIndex].size();
					// find the best step
					int min_step = std::numeric_limits<int>::max();
					const double MinAutoCor = result[ElementIndex].array().abs().minCoeff(&min_step);
					for (int iStep = 1; iStep < min_step; iStep++)
					{
						if (std::abs(result[ElementIndex][iStep]) <= MCParameters::AboveMinFactor * MinAutoCor)
						{
							min_step = iStep;
							break;
						}
					}
					best_MC_steps = std::min(min_step, best_MC_steps);
				}
			}
			else
			{
				// for strict lower triangular elements, copy the upper part
				result[iPES * NumPES + jPES] = result[jPES * NumPES + iPES];
			}
		}
	}
	// after all elements, using the max stepsize and max grid size
	MCParams.set_num_MC_steps(best_MC_steps);
	return result;
}

AutoCorrelations autocorrelation_optimize_displacement(
	MCParameters& MCParams,
	const DistributionFunction& distribution,
	const AllPoints& density)
{
	// parameters needed for MC
	static const std::vector<double> PossibleDisplacement = {1e-3, 2e-3, 5e-3, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5};
	static const int NPossibleDisplacement = PossibleDisplacement.size();
	const auto& [CombinedDensity, IsSmall] = pack_density(density);
	const AllPoints maxs = get_maximums(MCParams, IsSmall, CombinedDensity);

	// calculation
	AutoCorrelations result; // save the return value, the autocorrelations
	result.fill(Eigen::VectorXd::Zero(NPossibleDisplacement));
	for (int iDisp = 0; iDisp < NPossibleDisplacement; iDisp++)
	{
		Displacements displacements = generate_displacements(PossibleDisplacement[iDisp]);
		for (int iPES = 0; iPES < NumPES; iPES++)
		{
			for (int jPES = 0; jPES < NumPES; jPES++)
			{
				if (iPES <= jPES)
				{
					// for row-major, visit upper triangular elements earlier
					// so only deal with upper-triangular
					if (!IsSmall(iPES, jPES))
					{
						const int ElementIndex = iPES * NumPES + jPES;
						result[ElementIndex][iDisp] =
							std::transform_reduce(
								std::execution::par_unseq,
								maxs[ElementIndex].cbegin(),
								maxs[ElementIndex].cend(),
								0.0,
								std::plus<double>(),
								[NumSteps = MCParams.get_num_MC_steps(), &distribution, &displacements, iPES, jPES](const PhaseSpacePoint& psp) -> double
								{
									const auto& [x, p, rho] = psp;
									return calculate_autocorrelation(
										monte_carlo_chain(
											NumSteps,
											distribution,
											displacements,
											iPES,
											jPES,
											x,
											p),
										NumSteps);
								})
							/ maxs[ElementIndex].size();
					}
				}
				else
				{
					// for strict lower triangular elements, copy the upper part
					result[iPES * NumPES + jPES] = result[jPES * NumPES + iPES];
				}
			}
		}
	}
	// find the best step
	int best_displacement_index = 0;
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		for (int jPES = 0; jPES <= iPES; jPES++)
		{
			// only deal with lower-triangular
			const int ElementIndex = iPES * NumPES + jPES;
			int min_dist_index = std::numeric_limits<int>::max();
			const double MinAutoCor = result[ElementIndex].array().abs().minCoeff(&min_dist_index);
			for (int iDispl = NPossibleDisplacement - 1; iDispl > min_dist_index; iDispl--)
			{
				if (std::abs(result[ElementIndex][iDispl]) <= MCParameters::AboveMinFactor * MinAutoCor)
				{
					min_dist_index = iDispl;
					break;
				}
			}
			best_displacement_index = std::min(min_dist_index, best_displacement_index);
		}
	}
	MCParams.set_displacement(PossibleDisplacement[best_displacement_index]);
	return result;
}

/// @details There should be at least one diagonal element that is non-zero.
int get_num_points(const AllPoints& Points)
{
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		const int ElementIndex = iPES * NumPES + iPES;
		if (!Points[ElementIndex].empty())
		{
			return Points[ElementIndex].size();
		}
	}
	assert(!"NO DIAGONAL ELEMENT IS NON-ZERO!\n");
}

/// @details This function fills the newly populated density matrix element
/// with density matrices of the greatest weight.
void new_element_point_selection(AllPoints& density, const QuantumMatrix<bool>& IsNew)
{
	if (!IsNew.all())
	{
		return;
	}
	// find the number of points for each elements
	// after all, at least one of the diagonal elements is non-zero
	const int NumPoints = get_num_points(density);
	[[maybe_unused]] auto [combined_density, is_small] = pack_density(density);
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		for (int jPES = 0; jPES < NumPES; jPES++)
		{
			if (iPES <= jPES)
			{
				// for row-major, visit upper triangular elements earlier
				// so selection for the upper triangular elements
				if (IsNew(iPES, jPES))
				{
					std::nth_element(
						std::execution::par_unseq,
						combined_density.begin(),
						combined_density.begin() + NumPoints,
						combined_density.end(),
						[iPES, jPES](const PhaseSpacePoint& psp1, const PhaseSpacePoint& psp2) -> bool
						{
							return weight_function(std::get<2>(psp1)(iPES, jPES)) > weight_function(std::get<2>(psp2)(iPES, jPES));
						});
				}
				// then copy to the right place
				density[iPES * NumPES + jPES] = EigenVector<PhaseSpacePoint>(combined_density.cbegin(), combined_density.cbegin() + NumPoints);
			}
			else
			{
				// for strict lower triangular elements, copy the upper part
				density[iPES * NumPES + jPES] = density[jPES * NumPES + iPES];
			}
		}
	}
}
