/// @file mc.cpp
/// @brief Implementation of mc.h

#include "stdafx.h"

#include "mc.h"

#include "evolve.h"
#include "pes.h"
#include "predict.h"
#include "storage.h"

/// @brief All the distributions of displacements of position OR momentum
using Displacements = std::array<std::uniform_real_distribution<double>, PhaseDim>;

/// @brief The random number generator, using Mersenen twister algorithm with seed from time
static std::mt19937 engine(std::chrono::system_clock::now().time_since_epoch().count());
/// @brief The upper limit of Monte Carlo acceptance ratio
static constexpr double MaxAcceptRatio = 0.5;
/// @brief The lower limit of Monte Carlo acceptance ratio
static constexpr double MinAcceptRatio = 0.15;

/// @details Under the adiabatic basis, only @f$ rho_{00} @f$ is non-zero initially,
/// i.e. all the population lies on the ground state. For @f$ rho_{00} @f$, @n
/// @f[
/// P_{ij}(\mathbf{r})=\prod_i{P(r_i)}=
/// \prod_i\left(\frac{1}{2\pi\sigma_{r_i}}\mathrm{exp}\left[-\frac{(r_i-r_{i,0})^2}{2\sigma_{r_i}^2}\right]\right)
/// @f] @n
/// where i goes over all direction in the classical DOF
std::complex<double> initial_distribution(
	const ClassicalPhaseVector& r0,
	const ClassicalPhaseVector& SigmaR0,
	const ClassicalPhaseVector& r,
	const std::size_t RowIndex,
	const std::size_t ColIndex,
	const std::array<double, NumPES>& InitialPopulation,
	const std::array<double, NumPES>& InitialPhaseFactor
)
{
	using namespace std::literals::complex_literals;
	const double GaussWeight = std::exp(-((r - r0).array() / SigmaR0.array()).square().sum() / 2.0) / (power<Dim>(2.0 * std::numbers::pi) * SigmaR0.prod());
	const double SumWeight = std::transform_reduce(
		InitialPopulation.begin(),
		InitialPopulation.end(),
		0.0,
		std::plus<double>{},
		power<2, double>
	);
	return GaussWeight * InitialPopulation[RowIndex] * InitialPopulation[ColIndex] / SumWeight * std::exp(1.0i * (InitialPhaseFactor[RowIndex] - InitialPhaseFactor[ColIndex]));
}

/// @brief To generate the extra points for the density fitting of one element of density matrix
/// @param[in] density The exact density of the given element at the given point
/// @param[in] NumExtraPoints The number of extra selected points for each elements
/// @param[in] distribution Phase space distribution function
/// @param[in] RowIndex RowIndex Index of row of the element in density matrix. Must be a valid index (< @p NumPES )
/// @param[in] ColIndex RowIndex Index of row of the element in density matrix. Must be a valid index and no more than @p RowIndex
/// @return The newly selected points with its corresponding distribution of one element of density matrix
static ElementPoints generate_element_extra_points(
	const ElementPoints& density,
	const std::size_t NumExtraPoints,
	const DistributionFunction& distribution,
	const std::size_t RowIndex,
	const std::size_t ColIndex
)
{
	static std::array<std::normal_distribution<double>, PhaseDim> normdists;
	const ClassicalPhaseVector stddev = calculate_standard_deviation_one_surface(density);
	for (const std::size_t iDim : std::ranges::iota_view{0ul, PhaseDim})
	{
		normdists[iDim] = std::normal_distribution<double>(0.0, stddev[iDim]);
	}
	// generation
	ElementPoints result = ElementPoints(NumExtraPoints);
	const auto indices = xt::arange(result.size());
	std::for_each(
		std::execution::par_unseq,
		indices.cbegin(),
		indices.cend(),
		[&result, &density, &distribution, RowIndex, ColIndex](const std::size_t iPoint) -> void
		{
			auto& [r, rho] = result[iPoint];
			r = density[iPoint % density.size()].get<0>();
			for (const std::size_t iDim : std::ranges::iota_view{0ul, PhaseDim})
			{
				// a deviation from the center
				r[iDim] += normdists[iDim](engine);
			}
			// current distribution
			rho = distribution(r, RowIndex, ColIndex);
		}
	);
	return result;
}

AllPoints generate_extra_points(
	const AllPoints& density,
	const std::size_t NumExtraPoints,
	const DistributionFunction& distribution
)
{
	AllPoints result;
	for (const std::size_t iPES : std::ranges::iota_view{0ul, NumPES})
	{
		for (const std::size_t jPES : std::ranges::iota_view{0ul, iPES + 1})
		{
			if (!density(iPES, jPES).empty())
			{
				result(iPES, jPES) = generate_element_extra_points(
					density(iPES, jPES),
					NumExtraPoints,
					distribution,
					iPES,
					jPES
				);
			}
		}
	}
	return result;
}

/// @brief To calculate all the displacements
/// @param[in] MaxDisplacement The possible maximum displacement
/// @return The displacement distribution of positions and momenta
static Displacements generate_displacements(const double MaxDisplacement)
{
	Displacements result;
	for (const std::size_t iDim : std::ranges::iota_view{0ul, PhaseDim})
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
/// @return A vector containing the phase space coordinates of all grids, and the acceptance ratio
static std::tuple<EigenVector<ClassicalPhaseVector>, double> generate_markov_chain(
	const std::size_t NumSteps,
	const DistributionFunction& distribution,
	Displacements& Displ,
	const std::size_t iPES,
	const std::size_t jPES,
	const ClassicalPhaseVector& r
)
{
	// set parameters
	static std::uniform_real_distribution<double> mc_selection(0.0, 1.0); // for whether pick or not
	std::tuple<EigenVector<ClassicalPhaseVector>, double> result;
	EigenVector<ClassicalPhaseVector>& r_all = std::get<0>(result);
	r_all.reserve(NumSteps + 1);
	r_all.push_back(r);
	double weight_old = std::abs(distribution(r_all.back(), iPES, jPES));
	// going the chain
	std::size_t acc = 0;
	for ([[maybe_unused]] const std::size_t iIter : std::ranges::iota_view{0ul, NumSteps})
	{
		const ClassicalPhaseVector r_new = [&Displ](const ClassicalPhaseVector& r_init) -> ClassicalPhaseVector
		{
			ClassicalPhaseVector result = r_init;
			for (const std::size_t iDim : std::ranges::iota_view{0ul, PhaseDim})
			{
				result[iDim] += Displ[iDim](engine);
			}
			return result;
		}(r_all.back());
		const double weight_new = std::abs(distribution(r_new, iPES, jPES));
		if (weight_new > weight_old || weight_new / weight_old > mc_selection(engine))
		{
			// new weight is larger than the random number, accept
			r_all.push_back(r_new);
			weight_old = weight_new;
			acc++;
		}
		else
		{
			// otherwise, reject, use last step
			r_all.push_back(r_all.back());
		}
	}
	std::get<1>(result) = acc * 1. / NumSteps;
	return result;
}

/// @brief To optimize the number of the monte carlo steps by autocorrelation
/// @param[inout] MCParams Monte carlo parameters (number of steps, maximum displacement, etc)
/// @param[in] distribution The function object to generate the distribution
/// @param[in] density The selected points in phase space for each element of density matrices
/// @param[in] RowIndex RowIndex Index of row of the element in density matrix. Must be a valid index (< @p NumPES )
/// @param[in] ColIndex RowIndex Index of row of the element in density matrix. Must be a valid index (< @p NumPES )
/// @return The autocorrelation of all tested number of steps
static void autocorrelation_optimize_steps(
	MCParameters& MCParams,
	const DistributionFunction& distribution,
	const ElementPoints& density,
	const std::size_t RowIndex,
	const std::size_t ColIndex
)
{
	// parameters needed for MC
	static constexpr std::size_t MaxNOMC = PhaseDim * 1000; // The number of monte carlo steps (including head and tail)
	static constexpr std::size_t MaxAutoCorStep = (MaxNOMC + 1) / 2;

	// moving random number generator and MC random number generator
	Displacements displacements = generate_displacements(MCParams.get_max_displacement());

	// calculation
	const Eigen::VectorXd AutoCors =
		std::transform_reduce(
			std::execution::par_unseq,
			density.cbegin(),
			density.cend(),
			Eigen::VectorXd(Eigen::VectorXd::Zero(MaxAutoCorStep)),
			std::plus<Eigen::VectorXd>(),
			[&distribution, &displacements, RowIndex, ColIndex](const PhaseSpacePoint& psp) -> Eigen::VectorXd
			{
				[[maybe_unused]] const auto [WholeChain, AcceptRatio] = generate_markov_chain(
					MaxNOMC,
					distribution,
					displacements,
					RowIndex,
					ColIndex,
					psp.get<0>()
				);
				const std::size_t NSteps = WholeChain.size();
				Eigen::VectorXd result(NSteps / 2);
				// first, calculate average (aka sum / N)
				const ClassicalPhaseVector Ave = std::reduce(WholeChain.cbegin(), WholeChain.cend(), ClassicalPhaseVector(ClassicalPhaseVector::Zero())) / NSteps;
				// then, calculate for different j
				for (const std::size_t j : std::ranges::iota_view{0ul, NSteps / 2})
				{
					double sum = 0;
					for (const std::size_t i : std::ranges::iota_view{0ul, NSteps - j})
					{
						sum += (WholeChain[i] - Ave).dot(WholeChain[i + j] - Ave);
					}
					result[j] = sum / (NSteps - j);
				}
				return result;
			}
		)
		/ density.size();

	// find the best step
	const std::size_t len = AutoCors.size();
	std::size_t min_start_step = 0, min_autocor_step = 0;
	double acc = 0, min_auto_cor = 0;
	do
	{
		min_start_step = min_autocor_step + 1;
		if (min_start_step >= len)
		{
			// no result satisfy this
			min_start_step = 1;
			min_auto_cor = AutoCors.array().abs().minCoeff(&min_autocor_step);
			break;
		}
		min_auto_cor = AutoCors(Eigen::seqN(min_start_step, len - min_start_step)).array().abs().minCoeff(&min_autocor_step);
		min_autocor_step += min_start_step;
		acc = std::get<1>(generate_markov_chain(min_autocor_step, distribution, displacements, RowIndex, ColIndex, density[0].get<0>()));
	}
	while (acc > MaxAcceptRatio || acc < MinAcceptRatio);
	for (const std::size_t iStep : std::ranges::iota_view{min_start_step, min_autocor_step})
	{
		if (std::abs(AutoCors[iStep]) <= MCParameters::AboveMinFactor * min_auto_cor)
		{
			min_autocor_step = iStep;
			break;
		}
	}

	// after all elements, using the max stepsize and max grid size
	MCParams.set_num_MC_steps(min_autocor_step);
}

/// @brief To optimize the number of the monte carlo steps by autocorrelation
/// @param[inout] MCParams Monte carlo parameters (number of steps, maximum displacement, etc)
/// @param[in] distribution The function object to generate the distribution
/// @param[in] density The selected points in phase space for each element of density matrices
/// @param[in] RowIndex RowIndex Index of row of the element in density matrix. Must be a valid index (< @p NumPES )
/// @param[in] ColIndex RowIndex Index of row of the element in density matrix. Must be a valid index (< @p NumPES )
/// @return The autocorrelation of all tested displacements
static void acceptance_optimize_displacement(
	MCParameters& MCParams,
	const DistributionFunction& distribution,
	const ElementPoints& density,
	const std::size_t RowIndex,
	const std::size_t ColIndex
)
{
	// parameters needed for MC
	static constexpr std::size_t MaxNOMC = PhaseDim * 500; // The number of monte carlo steps (including head and tail)
	static constexpr std::array PossibleDisplacement{1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0};
	static constexpr std::size_t NPossibleDisplacement = PossibleDisplacement.size();

	// calculation
	for (const std::size_t iDisp : std::ranges::iota_view{0ul, NPossibleDisplacement} | std::views::reverse)
	{
		Displacements displacements = generate_displacements(PossibleDisplacement[iDisp]);
		const double AcceptRatio =
			std::transform_reduce(
				std::execution::par_unseq,
				density.cbegin(),
				density.cend(),
				0.0,
				std::plus<double>(),
				[&distribution, &displacements, RowIndex, ColIndex](const PhaseSpacePoint& psp) -> double
				{
					return std::get<1>(generate_markov_chain(
						MaxNOMC,
						distribution,
						displacements,
						RowIndex,
						ColIndex,
						psp.get<0>()
					));
				}
			)
			/ density.size();
		if (AcceptRatio < MaxAcceptRatio && AcceptRatio > MinAcceptRatio)
		{
			MCParams.set_displacement(PossibleDisplacement[iDisp]);
			return;
		}
	}
}

/// @brief Doing Random walk for a element in density matrix
/// @param[inout] density The selected points in phase space for the element of density matrix
/// @param[in] MCParams Monte carlo parameters (number of steps, maximum displacement, etc)
/// @param[in] distribution The function object to generate the distribution
/// @param[in] RowIndex RowIndex Index of row of the element in density matrix. Must be a valid index (< @p NumPES )
/// @param[in] ColIndex RowIndex Index of row of the element in density matrix. Must be a valid index and no more than @p RowIndex
static void element_monte_carlo(
	ElementPoints& density,
	MCParameters& MCParams,
	const DistributionFunction& distribution,
	const std::size_t RowIndex,
	const std::size_t ColIndex
)
{
	// optimize the steps and displacement
	acceptance_optimize_displacement(MCParams, distribution, density, RowIndex, ColIndex);
	autocorrelation_optimize_steps(MCParams, distribution, density, RowIndex, ColIndex);
	spdlog::info("For element ({}, {}), after optimization, MC takes {} steps with maximum displacement of {}.", RowIndex, ColIndex, MCParams.get_num_MC_steps(), MCParams.get_max_displacement());
	// do selection
	Displacements displacements = generate_displacements(MCParams.get_max_displacement());
	std::for_each(
		std::execution::par_unseq,
		density.begin(),
		density.end(),
		[NumSteps = MCParams.get_num_MC_steps(), &displacements, &distribution, RowIndex, ColIndex](PhaseSpacePoint& psp) -> void
		{
			auto& [r, rho] = psp;
			[[maybe_unused]] const auto [WholeChain, AccRatio] = generate_markov_chain(
				NumSteps,
				distribution,
				displacements,
				RowIndex,
				ColIndex,
				r
			);
			r = WholeChain.back();
			rho = distribution(r, RowIndex, ColIndex);
		}
	);
	// after MC, the new point is at the original place
	// TODO: then update the number of MC steps
}

/// @details Assuming that the @p distribution function containing all the required information. @n
/// If it is a distribution from some parameters, those parameters should have already been
/// bound; if it is some fitting method, the parameters should have already been optimized. @n
/// This function uses Metropolis algorithm of Markov chain monte carlo.
void monte_carlo_selection(
	AllPoints& density,
	QuantumStorage<MCParameters>& MCParams,
	const DistributionFunction& distribution
)
{
	// Monte Carlo selection
	for (const std::size_t iPES : std::ranges::iota_view{0ul, NumPES})
	{
		for (const std::size_t jPES : std::ranges::iota_view{0ul, iPES + 1})
		{
			if (!density(iPES, jPES).empty())
			{
				element_monte_carlo(
					density(iPES, jPES),
					MCParams(iPES, jPES),
					distribution,
					iPES,
					jPES
				);
			}
		}
	}
}

/// @details This function fills the newly populated density matrix element
/// with density matrices of the greatest weight.
void new_element_point_selection(
	AllPoints& density,
	AllPoints& extra_points,
	const QuantumStorage<bool>& IsSmallOld,
	const QuantumStorage<bool>& IsSmall,
	QuantumStorage<MCParameters>& MCParams,
	const DistributionFunction& distribution
)
{
	// In ElementChange, 1 for newly populated, -1 for newly unpopulated, 0 for unchanged
	if (IsSmallOld != IsSmall)
	{
		// find the number of points for each elements
		// after all, at least one of the diagonal elements is non-zero
		const std::size_t NumPoints = density(0).size();
		const std::size_t NumExtraPoints = extra_points(0).size();
		const auto PossibleCoordinates = [&density, &extra_points, NumPoints, NumExtraPoints](void)
		{
			std::size_t total_pts = 0;
			for (const std::size_t iPES : std::ranges::iota_view{0ul, NumPES})
			{
				for (const std::size_t jPES : std::ranges::iota_view{0ul, iPES + 1})
				{
					total_pts += density(iPES, jPES).size() + extra_points(iPES, jPES).size();
				}
			}
			EigenVector<ClassicalPhaseVector> result(total_pts, ClassicalPhaseVector::Zero());
			std::size_t idx = 0;
			for (const std::size_t iPES : std::ranges::iota_view{0ul, NumPES})
			{
				for (const std::size_t jPES : std::ranges::iota_view{0ul, iPES + 1})
				{
					std::transform(
						std::execution::par_unseq,
						density(iPES, jPES).cbegin(),
						density(iPES, jPES).cend(),
						result.begin() + idx,
						[](const PhaseSpacePoint& psp) -> ClassicalPhaseVector
						{
							return psp.get<0>();
						}
					);
					idx += density(iPES, jPES).size();
					std::transform(
						std::execution::par_unseq,
						extra_points(iPES, jPES).cbegin(),
						extra_points(iPES, jPES).cend(),
						result.begin() + idx,
						[](const PhaseSpacePoint& psp) -> ClassicalPhaseVector
						{
							return psp.get<0>();
						}
					);
					idx += extra_points(iPES, jPES).size();
				}
			}
			return result;
		}();
		for (const std::size_t iPES : std::ranges::iota_view{0ul, NumPES})
		{
			for (const std::size_t jPES : std::ranges::iota_view{0ul, iPES + 1})
			{
				if (IsSmallOld(iPES, jPES) == true && IsSmall(iPES, jPES) == false)
				{
					spdlog::info("New element appeares at ({}, {})", iPES, jPES);
					// new element appears
					// calculate their density
					ElementPoints& element_density = density(iPES, jPES);
					element_density = ElementPoints(PossibleCoordinates.size());
					std::transform(
						std::execution::par_unseq,
						PossibleCoordinates.cbegin(),
						PossibleCoordinates.cend(),
						element_density.begin(),
						[&distribution, iPES, jPES](const ClassicalPhaseVector& r) -> PhaseSpacePoint
						{
							return PhaseSpacePoint(r, distribution(r, iPES, jPES));
						}
					);
					// see how many point could be used in random walk
					const std::size_t NumNonZeroPoints = std::count_if(
						std::execution::par_unseq,
						element_density.cbegin(),
						element_density.cend(),
						[](const PhaseSpacePoint& psp)
						{
							return psp.get<1>() != 0.0;
						}
					);
					// choose the most N important points
					std::nth_element(
						std::execution::par_unseq,
						element_density.begin(),
						element_density.begin() + std::min(NumPoints, NumNonZeroPoints),
						element_density.end(),
						[](const PhaseSpacePoint& psp1, const PhaseSpacePoint& psp2) -> bool
						{
							return std::norm(psp1.get<1>()) > std::norm(psp2.get<1>());
						}
					);
					// erase the 0 (or small) points
					element_density.erase(element_density.cbegin() + std::min(NumPoints, NumNonZeroPoints), element_density.cend());
					// fill up to N points
					while (NumPoints >= 2 * element_density.size())
					{
						element_density.insert(element_density.cend(), element_density.cbegin(), element_density.cend());
					}
					if (element_density.size() < NumPoints)
					{
						element_density.insert(
							element_density.cend(),
							element_density.cbegin(),
							element_density.cbegin() + (NumPoints - element_density.size())
						);
					}
					// and do random walk to be better distributed
					element_monte_carlo(element_density, MCParams(iPES, jPES), distribution, iPES, jPES);
					// .. and extra points
					extra_points(iPES, jPES) = generate_element_extra_points(density(iPES, jPES), NumExtraPoints, distribution, iPES, jPES);
				}
				else if (IsSmallOld(iPES, jPES) == false && IsSmall(iPES, jPES) == true)
				{
					spdlog::info("Element disappeares at ({}, {})", iPES, jPES);
					// delete the element
					density(iPES, jPES).clear();
					extra_points(iPES, jPES).clear();
				}
			}
		}
	}
}
