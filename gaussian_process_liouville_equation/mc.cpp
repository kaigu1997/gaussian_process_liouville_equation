/// @file mc.cpp
/// @brief Implementation of mc.h

#include "stdafx.h"

#include "mc.h"

#include "evolve.h"
#include "pes.h"
#include "storage.h"

/// @brief All the distributions of displacements of position OR momentum
using Displacements = std::array<std::uniform_real_distribution<double>, PhaseDim>;

/// @brief The random number generator, using Mersenen twister algorithm with seed from time
static std::mt19937 engine(std::chrono::system_clock::now().time_since_epoch().count());

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
	const double GaussWeight = std::exp(-((r - r0).array() / SigmaR0.array()).square().sum() / 2.0) / (power<Dim>(2.0 * M_PI) * SigmaR0.prod());
	const double SumWeight = std::transform_reduce(
		InitialPopulation.begin(),
		InitialPopulation.end(),
		0.0,
		std::plus<double>{},
		[](double d) -> double
		{
			return d * d;
		}
	);
	return GaussWeight * InitialPopulation[RowIndex] * InitialPopulation[ColIndex] / SumWeight * std::exp(1.0i * (InitialPhaseFactor[RowIndex] - InitialPhaseFactor[ColIndex]));
}

AllPoints generate_extra_points(
	const AllPoints& density,
	const std::size_t NumPoints,
	const ClassicalVector<double>& mass,
	const double dt,
	const std::size_t NumTicks,
	const DistributionFunction& distribution
)
{
	using namespace std::literals::complex_literals;
	AllPoints result;
	static std::mt19937 engine(std::chrono::system_clock::now().time_since_epoch().count()); // Random number generator, using merseen twister with seed from time
	std::array<std::normal_distribution<double>, PhaseDim> normdists;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES <= iPES; jPES++)
		{
			const ElementPoints& element_density = density(iPES, jPES);
			ElementPoints& element_result = result(iPES, jPES);
			const ClassicalPhaseVector stddev = calculate_standard_deviation_one_surface(element_density);
			for (std::size_t iDim = 0; iDim < PhaseDim; iDim++)
			{
				normdists[iDim] = std::normal_distribution<double>(0.0, stddev[iDim]);
			}
			// generation
			element_result = ElementPoints(NumPoints);
			const auto indices = xt::arange(element_result.size());
			std::for_each(
				std::execution::par_unseq,
				indices.cbegin(),
				indices.cend(),
				[&element_density, &element_result, &normdists, &mass, dt, NumTicks, &distribution, iPES, jPES](std::size_t iPoint) -> void
				{
					[[maybe_unused]] auto& [r, rho, theta] = element_result[iPoint];
					r = element_density[iPoint % element_density.size()].get<0>();
					for (std::size_t iDim = 0; iDim < PhaseDim; iDim++)
					{
						// a deviation from the center
						r[iDim] += normdists[iDim](engine);
					}
					// calculate phase factor
					theta = get_phase_factor(r, mass, dt, NumTicks, iPES, jPES);
					// current distribution
					element_result[iPoint].set_density(distribution(r, iPES, jPES));
				}
			);
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
	const ClassicalPhaseVector& r
)
{
	// set parameters
	static std::uniform_real_distribution<double> mc_selection(0.0, 1.0); // for whether pick or not
	EigenVector<ClassicalPhaseVector> r_all;
	r_all.reserve(NumSteps + 1);
	r_all.push_back(r);
	double weight_old = std::abs(distribution(r_all.back(), iPES, jPES));
	// going the chain
	for (std::size_t iIter = 1; iIter <= NumSteps; iIter++)
	{
		const ClassicalPhaseVector r_new = [&Displ](const ClassicalPhaseVector& r_init) -> ClassicalPhaseVector
		{
			ClassicalPhaseVector result = r_init;
			for (std::size_t iDim = 0; iDim < PhaseDim; iDim++)
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
	const ClassicalVector<double>& mass,
	const double dt,
	const std::size_t NumTicks,
	const DistributionFunction& distribution,
	AllPoints& density
)
{
	// moving random number generator and MC random number generator
	Displacements displacements = generate_displacements(MCParams.get_max_displacement());

	// Monte Carlo selection
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES <= iPES; jPES++)
		{
			std::for_each(
				std::execution::par_unseq,
				density(iPES, jPES).begin(),
				density(iPES, jPES).end(),
				[NumSteps = MCParams.get_num_MC_steps(), &displacements, &mass, dt, NumTicks, &distribution, iPES, jPES](PhaseSpacePoint& psp) -> void
				{
					[[maybe_unused]] auto& [r, rho, theta] = psp;
					const EigenVector<ClassicalPhaseVector> WholeChain = generate_markov_chain(
						NumSteps,
						distribution,
						displacements,
						iPES,
						jPES,
						r
					);
					r = WholeChain.back();
					theta = get_phase_factor(r, mass, dt, NumTicks, iPES, jPES);
					psp.set_density(distribution(r, iPES, jPES));
				}
			);
			// after MC, the new point is at the original place
			// TODO: then update the number of MC steps
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
	const AllPoints& density
)
{
	// parameters needed for MC
	static constexpr std::size_t MaxNOMC = PhaseDim * 400; // The number of monte carlo steps (including head and tail)
	static constexpr std::size_t MaxAutoCorStep = (MaxNOMC + 1) / 2;

	// moving random number generator and MC random number generator
	Displacements displacements = generate_displacements(MCParams.get_max_displacement());

	// calculation
	AutoCorrelations result(Eigen::VectorXd::Zero(MaxAutoCorStep), Eigen::VectorXd::Zero(MaxAutoCorStep)); // save the return value, the autocorrelations
	std::size_t best_MC_steps = 0;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES <= iPES; jPES++)
		{
			result(iPES, jPES) =
				std::transform_reduce(
					std::execution::par_unseq,
					density(iPES, jPES).cbegin(),
					density(iPES, jPES).cend(),
					Eigen::VectorXd(Eigen::VectorXd::Zero(MaxAutoCorStep)),
					std::plus<Eigen::VectorXd>(),
					[&distribution, &displacements, iPES, jPES](const PhaseSpacePoint& psp) -> Eigen::VectorXd
					{
						return calculate_autocorrelation(generate_markov_chain(
							MaxNOMC,
							distribution,
							displacements,
							iPES,
							jPES,
							psp.get<0>()
						));
					}
				)
				/ density(iPES, jPES).size();
			// find the best step
			std::size_t min_step = result(iPES, jPES).size();
			const double MinAutoCor = result(iPES, jPES).array().abs().minCoeff(&min_step);
			for (std::size_t iStep = 1; iStep < min_step; iStep++)
			{
				if (std::abs(result(iPES, jPES)[iStep]) <= MCParameters::AboveMinFactor * MinAutoCor)
				{
					min_step = iStep;
					break;
				}
			}
			best_MC_steps = std::max(min_step, best_MC_steps);
		}
	}
	// after all elements, using the max stepsize and max grid size
	MCParams.set_num_MC_steps(best_MC_steps);
	return result;
}

AutoCorrelations autocorrelation_optimize_displacement(
	MCParameters& MCParams,
	const DistributionFunction& distribution,
	const AllPoints& density
)
{
	// parameters needed for MC
	static constexpr std::array<double, 15> PossibleDisplacement = {1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5};
	static constexpr std::size_t NPossibleDisplacement = PossibleDisplacement.size();

	// calculation
	AutoCorrelations result(Eigen::VectorXd::Zero(NPossibleDisplacement), Eigen::VectorXd::Zero(NPossibleDisplacement)); // save the return value, the autocorrelations
	for (std::size_t iDisp = 0; iDisp < NPossibleDisplacement; iDisp++)
	{
		Displacements displacements = generate_displacements(PossibleDisplacement[iDisp]);
		for (std::size_t iPES = 0; iPES < NumPES; iPES++)
		{
			for (std::size_t jPES = 0; jPES <= iPES; jPES++)
			{
				result(iPES, jPES)[iDisp] =
					std::transform_reduce(
						std::execution::par_unseq,
						density(iPES, jPES).cbegin(),
						density(iPES, jPES).cend(),
						0.0,
						std::plus<double>(),
						[NumSteps = MCParams.get_num_MC_steps(), &distribution, &displacements, iPES, jPES](const PhaseSpacePoint& psp) -> double
						{
							return calculate_autocorrelation(
								generate_markov_chain(
									NumSteps,
									distribution,
									displacements,
									iPES,
									jPES,
									psp.get<0>()
								),
								NumSteps
							);
						}
					)
					/ density(iPES, jPES).size();
			}
		}
	}
	// find the best step
	std::size_t best_displacement_index = 0;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES <= iPES; jPES++)
		{
			std::size_t min_dist_index = result(iPES, jPES).size();
			const double MinAutoCor = result(iPES, jPES).array().abs().minCoeff(&min_dist_index);
			for (std::size_t iDispl = NPossibleDisplacement - 1; iDispl > min_dist_index; iDispl--)
			{
				if (std::abs(result(iPES, jPES)[iDispl]) <= MCParameters::AboveMinFactor * MinAutoCor)
				{
					min_dist_index = iDispl;
					break;
				}
			}
			best_displacement_index = std::max(min_dist_index, best_displacement_index);
		}
	}
	MCParams.set_displacement(PossibleDisplacement[best_displacement_index]);
	return result;
}

/// @details This function fills the newly populated density matrix element
/// with density matrices of the greatest weight.
void new_element_point_selection(
	AllPoints& density,
	const AllPoints& extra_points,
	const QuantumMatrix<bool>& IsNew,
	const ClassicalVector<double>& mass,
	const double dt,
	const std::size_t NumTicks,
	const DistributionFunction& distribution
)
{
	// In ElementChange, 1 for newly populated, -1 for newly unpopulated, 0 for unchanged
	if (IsNew.any())
	{
		// find the number of points for each elements
		// after all, at least one of the diagonal elements is non-zero
		const std::size_t NumPoints = density(0).size();
		const ElementPoints CombinedDensity = [&density, &extra_points](void) -> ElementPoints
		{
			ElementPoints result;
			for (std::size_t iPES = 0; iPES < NumPES; iPES++)
			{
				for (std::size_t jPES = 0; jPES <= iPES; jPES++)
				{
					result.insert(result.cend(), density(iPES, jPES).cbegin(), density(iPES, jPES).cend());
					result.insert(result.cend(), extra_points(iPES, jPES).cbegin(), extra_points(iPES, jPES).cend());
				}
			}
			return result;
		}();
		std::uniform_int_distribution<std::size_t> dist(0, CombinedDensity.size());
		for (std::size_t iPES = 0; iPES < NumPES; iPES++)
		{
			for (std::size_t jPES = 0; jPES <= iPES; jPES++)
			{
				if (IsNew(iPES, jPES))
				{
					// new element appears
					// sample N different points into it
					std::unordered_set<std::size_t> indices;
					while (density(iPES, jPES).size() < NumPoints)
					{
						const std::size_t val = dist(engine);
						[[maybe_unused]] const auto [iter, success] = indices.insert(val);
						if (success)
						{
							const ClassicalPhaseVector& r = CombinedDensity[val].get<0>();
							density(iPES, jPES).emplace_back(r, distribution(r, iPES, jPES), get_phase_factor(r, mass, dt, NumTicks, iPES, jPES));
						}
					}
				}
			}
		}
	}
}
