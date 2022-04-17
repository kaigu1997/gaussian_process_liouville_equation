/// @file mc.cpp
/// @brief Implementation of mc.h

#include "stdafx.h"

#include "mc.h"

#include "input.h"
#include "pes.h"

/// All the distributions of displacements of position OR momentum
using Displacements = std::array<std::uniform_real_distribution<double>, PhaseDim>;

static std::mt19937 engine(std::chrono::system_clock::now().time_since_epoch().count()); ///< The random number generator, using Mersenen twister algorithm with seed from time

QuantumMatrix<bool> is_very_small(const AllPoints& density)
{
	// squared value below this are regarded as 0
	QuantumMatrix<bool> result = QuantumMatrix<bool>::Ones();
	// as density is symmetric, only lower-triangular elements needs evaluating
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES <= iPES; jPES++)
		{
			if (result(iPES, jPES))
			{
				const std::size_t ElementIndex = iPES * NumPES + jPES;
				result(iPES, jPES) &= std::all_of(
					std::execution::par_unseq,
					density[ElementIndex].cbegin(),
					density[ElementIndex].cend(),
					[iPES, jPES](const PhaseSpacePoint& psp) -> bool
					{
						return std::abs(std::get<1>(psp)(iPES, jPES)) < epsilon;
					});
			}
		}
	}
	return result.selfadjointView<Eigen::Lower>();
}

/// @details Under the adiabatic basis, only @f$ rho_{00} @f$ is non-zero initially,
/// i.e. all the population lies on the ground state. For @f$ rho_{00} @f$, @n
/// @f[
/// P(\mathbf{r})=\prod_i{P(r_i)}=\prod_i{\left(\frac{1}{2\pi\sigma_{r_i}}\mathrm{exp}\left[-\frac{(r_i-r_{i,0})^2}{2\sigma_{r_i}^2}\right]\right)}
/// @f] @n
/// where i goes over all direction in the classical DOF
QuantumMatrix<std::complex<double>> initial_distribution(
	const ClassicalPhaseVector& r0,
	const ClassicalPhaseVector& SigmaR0,
	const std::array<double, NumPES>& InitialPopulation,
	const ClassicalPhaseVector& r)
{
	QuantumMatrix<std::complex<double>> result = QuantumMatrix<std::complex<double>>::Zero(NumPES, NumPES);
	const double GaussWeight = std::exp(-((r - r0).array() / SigmaR0.array()).square().sum() / 2.0) / (power<Dim>(2.0 * M_PI) * SigmaR0.prod());
	const double SumWeight = std::reduce(InitialPopulation.begin(), InitialPopulation.end());
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		result(iPES, iPES) = GaussWeight * InitialPopulation[iPES] / SumWeight;
	}
	result(0, 1) = result(1, 0) = GaussWeight * std::sqrt(QuantumVector<double>::Map(InitialPopulation.data()).prod());
	return result;
}

MCParameters::MCParameters(const double InitialDisplacement, const std::size_t InitialSteps) :
	NOMC(InitialSteps),
	displacement(InitialDisplacement)
{
}

/// @details There should be at least one diagonal element that is non-zero.
std::size_t get_num_points(const AllPoints& Points)
{
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		const std::size_t ElementIndex = iPES * NumPES + iPES;
		if (!Points[ElementIndex].empty())
		{
			return Points[ElementIndex].size();
		}
	}
	assert(!"NO DIAGONAL ELEMENT IS NON-ZERO!\n");
}

/// @brief To combine density into one array
/// @param[in] density The selected points in phase space for each element of density matrices
/// @return The combination of the paramere
static std::tuple<ElementPoints, QuantumMatrix<bool>> pack_density(const AllPoints& density)
{
	ElementPoints combination;
	QuantumMatrix<bool> is_small = QuantumMatrix<bool>::Zero();
	// as density is symmetric, only lower-triangular elements needs evaluating
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES <= iPES; jPES++)
		{
			const std::size_t ElementIndex = iPES * NumPES + jPES;
			combination.insert(combination.cend(), density[ElementIndex].cbegin(), density[ElementIndex].cend());
			is_small(iPES, jPES) = density[ElementIndex].empty();
		}
	}
	return std::make_tuple(combination, is_small.selfadjointView<Eigen::Lower>());
}

/// @brief To select phase points as the initial phase coordinates for monte carlo
/// @param[in] NumPoints Number of points for each element
/// @param[in] IsSmall The matrix that saves whether each element is small or not
/// @param[in] density The combination of selected points in phase space for each element of density matrices
/// @return The phase points with their density matrices
/// @details For each element, this function chooses the n points with largest weight of that element. @n
/// For off-diagonal element, the weight is its squared norm, so the return will be @"symmetric@".
static AllPoints get_maximums(
	const std::size_t NumPoints,
	const QuantumMatrix<bool>& IsSmall,
	const ElementPoints& density)
{
	AllPoints maxs;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES < NumPES; jPES++)
		{
			if (iPES <= jPES)
			{
				// for row-major, visit upper triangular elements earlier
				// so only deal with upper-triangular
				if (!IsSmall(iPES, jPES))
				{
					const std::size_t ElementIndex = iPES * NumPES + jPES;
					maxs[ElementIndex] = density;
					// find the max n elements
					std::nth_element(
						std::execution::par_unseq,
						maxs[ElementIndex].begin(),
						maxs[ElementIndex].begin() + NumPoints,
						maxs[ElementIndex].end(),
						[iPES, jPES](const PhaseSpacePoint& psp1, const PhaseSpacePoint& psp2) -> bool
						{
							return std::abs(std::get<1>(psp1)(iPES, jPES)) > std::abs(std::get<1>(psp2)(iPES, jPES));
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

/// @brief To calculate all the displacements
/// @param[in] MaxDisplacement The possible maximum displacement
/// @return The displacement distribution of positions and momenta
static Displacements generate_displacements(const double MaxDisplacement)
{
	Displacements result;
	for (std::size_t iDim = 0; iDim < PhaseDim; iDim++)
	{
		result[iDim] = std::uniform_real_distribution<double>(-MaxDisplacement, MaxDisplacement);
	}
	return result;
}

/// @brief To generate the Markov chain monte carlo of Metropolis scheme
/// @param[in] NumSteps Number of steps for the monte carlo chain
/// @param[in] distribution The function object to generate the distribution
/// @param[in] Displ The displacement distribution of x and p
/// @param[in] iPES The index of the row of the density matrix element
/// @param[in] jPES The index of the column of the density matrix element
/// @param[in] r The phase space coordinates as the starting point of random walk
/// @return A vector containing the phase space coordinates of all grids
static EigenVector<ClassicalPhaseVector> generate_markov_chain(
	const std::size_t NumSteps,
	const DistributionFunction& distribution,
	Displacements& Displ,
	const std::size_t iPES,
	const std::size_t jPES,
	const ClassicalPhaseVector& r)
{
	// set parameters
	static std::uniform_real_distribution<double> mc_selection(0.0, 1.0); // for whether pick or not
	EigenVector<ClassicalPhaseVector> r_all;
	r_all.reserve(NumSteps + 1);
	r_all.push_back(r);
	double weight_old = std::abs(distribution(r_all.back())(iPES, jPES));
	// going the chain
	for (std::size_t iIter = 1; iIter <= NumSteps; iIter++)
	{
		const ClassicalPhaseVector r_new = [&Displ](ClassicalPhaseVector r_init) -> ClassicalPhaseVector
		{
			for (std::size_t iDim = 0; iDim < PhaseDim; iDim++)
			{
				r_init[iDim] += Displ[iDim](engine);
			}
			return r_init;
		}(r_all.back());
		const double weight_new = std::abs(distribution(r_new)(iPES, jPES));
		if (weight_new > weight_old || weight_new / weight_old > mc_selection(engine))
		{
			// new weight is larger than the random number, accept
			r_all.push_back(r_new);
			weight_old = weight_new;
		}
		else
		{
			// otherwise, reject, use last step
			r_all.push_back(r_all.back());
		}
	}
	return r_all;
}

/// @details Assuming that the @p distribution function containing all the required information.
/// If it is a distribution from some parameters, those parameters should have already been
/// bound; if it is some fitting method, the parameters should have already been optimized. @n
/// This function uses Metropolis algorithm of Markov chain monte carlo.
void monte_carlo_selection(
	const MCParameters& MCParams,
	const DistributionFunction& distribution,
	AllPoints& density)
{
	// get maximum, then evolve the points adiabatically
	auto [combined_density, IsSmall] = pack_density(density);
	AllPoints maxs = get_maximums(get_num_points(density), IsSmall, combined_density);

	// moving random number generator and MC random number generator
	Displacements displacements = generate_displacements(MCParams.get_max_displacement());

	// Monte Carlo selection
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES < NumPES; jPES++)
		{
			if (iPES <= jPES)
			{
				const std::size_t ElementIndex = iPES * NumPES + jPES;
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
							auto& [r, rho] = psp;
							const EigenVector<ClassicalPhaseVector> WholeChain = generate_markov_chain(
								NumSteps,
								distribution,
								displacements,
								iPES,
								jPES,
								r);
							r = WholeChain.back();
							rho = distribution(r);
						});
					// after MC, the new point is at the original place
					// TODO: then update the number of MC steps
					// after all points updated, insert into the ElementPoints
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
/// @param[in] Rs The phase space coordinates on the thread, stands for multiple r
/// @return The autocorrelation of different number of steps
/// @details For j steps, the autocorrelation is
/// @f[
/// \langle R_{i+j} R_i\rangle=\frac{1}{N-j}\sum_{i=1}^{N-j}R_{i+j}R_i-|\overline{r}|^2
/// @f]
static Eigen::VectorXd calculate_autocorrelation(const EigenVector<ClassicalPhaseVector>& Rs)
{
	const std::size_t NSteps = Rs.size();
	Eigen::VectorXd result(NSteps / 2);
	// first, calculate average (aka sum / N)
	const ClassicalPhaseVector Ave = std::reduce(Rs.begin(), Rs.end(), ClassicalPhaseVector(ClassicalPhaseVector::Zero())) / NSteps;
	// then, calculate for different j
	for (std::size_t j = 0; j < NSteps / 2; j++)
	{
		double sum = 0;
		for (std::size_t i = 0; i + j < NSteps; i++)
		{
			sum += (Rs[i] - Ave).dot(Rs[i + j] - Ave);
		}
		result[j] = sum / (NSteps - j);
	}
	return result;
}

/// @brief To calculate the autocorrelation of one thread of given autocorrelation steps
/// @param[in] Rs The phase space coordinates on the thread, stands for multiple r
/// @param[in] AutoCorSteps The given autocorrelation steps
/// @return The autocorrelation
static double calculate_autocorrelation(const EigenVector<ClassicalPhaseVector>& Rs, const std::size_t AutoCorSteps)
{
	const std::size_t NSteps = Rs.size();
	const ClassicalPhaseVector Ave = std::reduce(Rs.begin(), Rs.end(), ClassicalPhaseVector(ClassicalPhaseVector::Zero())) / NSteps;
	double sum = 0;
	for (std::size_t i = 0; i + AutoCorSteps < NSteps; i++)
	{
		sum += (Rs[i] - Ave).dot(Rs[i + AutoCorSteps] - Ave);
	}
	return sum / (NSteps - AutoCorSteps);
}

AutoCorrelations autocorrelation_optimize_steps(
	MCParameters& MCParams,
	const DistributionFunction& distribution,
	const AllPoints& density)
{
	// parameters needed for MC
	static const std::size_t MaxNOMC = PhaseDim * 400; // The number of monte carlo steps (including head and tail)
	static const std::size_t MaxAutoCorStep = (MaxNOMC + 1) / 2;
	const auto& [CombinedDensity, IsSmall] = pack_density(density);
	const AllPoints maxs = get_maximums(get_num_points(density), IsSmall, CombinedDensity);

	// moving random number generator and MC random number generator
	Displacements displacements = generate_displacements(MCParams.get_max_displacement());

	// calculation
	AutoCorrelations result; // save the return value, the autocorrelations
	result.fill(Eigen::VectorXd::Zero(MaxAutoCorStep));
	std::size_t best_MC_steps = 0;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES < NumPES; jPES++)
		{
			if (iPES <= jPES)
			{
				// for row-major, visit upper triangular elements earlier
				// so only deal with upper-triangular
				if (!IsSmall(iPES, jPES))
				{
					const std::size_t ElementIndex = iPES * NumPES + jPES;
					result[ElementIndex] =
						std::transform_reduce(
							std::execution::par_unseq,
							maxs[ElementIndex].cbegin(),
							maxs[ElementIndex].cend(),
							Eigen::VectorXd(Eigen::VectorXd::Zero(MaxAutoCorStep)),
							std::plus<Eigen::VectorXd>(),
							[&distribution, &displacements, iPES, jPES](const PhaseSpacePoint& psp) -> Eigen::VectorXd
							{
								const auto& [r, rho] = psp;
								return calculate_autocorrelation(generate_markov_chain(
									MaxNOMC,
									distribution,
									displacements,
									iPES,
									jPES,
									r));
							})
						/ maxs[ElementIndex].size();
					// find the best step
					std::size_t min_step = result[ElementIndex].size();
					const double MinAutoCor = result[ElementIndex].array().abs().minCoeff(&min_step);
					for (std::size_t iStep = 1; iStep < min_step; iStep++)
					{
						if (std::abs(result[ElementIndex][iStep]) <= MCParameters::AboveMinFactor * MinAutoCor)
						{
							min_step = iStep;
							break;
						}
					}
					best_MC_steps = std::max(min_step, best_MC_steps);
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
	static constexpr std::array<double, 9> PossibleDisplacement = {1e-3, 2e-3, 5e-3, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5};
	static constexpr std::size_t NPossibleDisplacement = PossibleDisplacement.size();
	const auto& [CombinedDensity, IsSmall] = pack_density(density);
	const AllPoints maxs = get_maximums(get_num_points(density), IsSmall, CombinedDensity);

	// calculation
	AutoCorrelations result; // save the return value, the autocorrelations
	result.fill(Eigen::VectorXd::Zero(NPossibleDisplacement));
	for (std::size_t iDisp = 0; iDisp < NPossibleDisplacement; iDisp++)
	{
		Displacements displacements = generate_displacements(PossibleDisplacement[iDisp]);
		for (std::size_t iPES = 0; iPES < NumPES; iPES++)
		{
			for (std::size_t jPES = 0; jPES < NumPES; jPES++)
			{
				if (iPES <= jPES)
				{
					// for row-major, visit upper triangular elements earlier
					// so only deal with upper-triangular
					if (!IsSmall(iPES, jPES))
					{
						const std::size_t ElementIndex = iPES * NumPES + jPES;
						result[ElementIndex][iDisp] =
							std::transform_reduce(
								std::execution::par_unseq,
								maxs[ElementIndex].cbegin(),
								maxs[ElementIndex].cend(),
								0.0,
								std::plus<double>(),
								[NumSteps = MCParams.get_num_MC_steps(), &distribution, &displacements, iPES, jPES](const PhaseSpacePoint& psp) -> double
								{
									const auto& [r, rho] = psp;
									return calculate_autocorrelation(
										generate_markov_chain(
											NumSteps,
											distribution,
											displacements,
											iPES,
											jPES,
											r),
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
	std::size_t best_displacement_index = 0;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES <= iPES; jPES++)
		{
			// only deal with lower-triangular
			if (!IsSmall(iPES, jPES))
			{
				const std::size_t ElementIndex = iPES * NumPES + jPES;
				std::size_t min_dist_index = result[ElementIndex].size();
				const double MinAutoCor = result[ElementIndex].array().abs().minCoeff(&min_dist_index);
				for (std::size_t iDispl = NPossibleDisplacement - 1; iDispl > min_dist_index; iDispl--)
				{
					if (std::abs(result[ElementIndex][iDispl]) <= MCParameters::AboveMinFactor * MinAutoCor)
					{
						min_dist_index = iDispl;
						break;
					}
				}
				best_displacement_index = std::max(min_dist_index, best_displacement_index);
			}
		}
	}
	MCParams.set_displacement(PossibleDisplacement[best_displacement_index]);
	return result;
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
	const std::size_t NumPoints = get_num_points(density);
	[[maybe_unused]] auto [combined_density, is_small] = pack_density(density);
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES < NumPES; jPES++)
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
							return std::abs(std::get<1>(psp1)(iPES, jPES)) > std::abs(std::get<1>(psp2)(iPES, jPES));
						});
				}
				// then copy to the right place
				density[iPES * NumPES + jPES] = ElementPoints(combined_density.cbegin(), combined_density.cbegin() + NumPoints);
			}
			else
			{
				// for strict lower triangular elements, copy the upper part
				density[iPES * NumPES + jPES] = density[jPES * NumPES + iPES];
			}
		}
	}
}
