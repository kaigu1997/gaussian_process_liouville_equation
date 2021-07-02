/// @file mc.cpp
/// @brief Implementation of mc.h

#include "stdafx.h"

#include "mc.h"

#include "pes.h"

/// The array containing the maximum elements of each of the density matrix element
using Maximums = std::array<EvolvingDensity, NumElements>;
/// All the distributions of displacements of position OR momentum
using Displacements = std::array<std::uniform_real_distribution<double>, Dim>;
/// The vector to depict a phase space point (or coordinate)
using ClassicalPhaseVector = Eigen::Matrix<double, 2 * Dim, 1>;
/// A collection of phase space coordinates vectors
using ClassicalPhaseVectors = std::vector<ClassicalPhaseVector, Eigen::aligned_allocator<ClassicalPhaseVector>>;

QuantumBoolMatrix is_very_small(const EvolvingDensity& density)
{
	// value below this are regarded as 0
	static const double epsilon = 1e-2;
	QuantumBoolMatrix result;
	for (int iElement = 0; iElement < NumElements; iElement++)
	{
		result(iElement / NumPES, iElement % NumPES) = std::all_of(density.begin(), density.end(), [=](const PhaseSpacePoint& psp) -> bool { return get_density_matrix_element(std::get<2>(psp), iElement) < epsilon; });
	}
	return result;
}

/// @details Under the adiabatic basis, only rho[0][0] is non-zero initially,
/// i.e. all the population lies on the ground state. For rho[0][0],
///
/// \f[
/// P(\bf{x},\bf{p})=\prod_i{P(x_i,p_i)}=\prod_i{\left(\frac{1}{2\pi\sigma_{x_i}\sigma_{p_i}}\exp\left[-\frac{(x_i-x_{i0})^2}{2\sigma_{x_i}^2}-\frac{(p_i-p_{i0})^2}{2\sigma_{p_i}^2}\right]\right)}
/// \f]
///
/// where i goes over all direction in the classical DOF
QuantumComplexMatrix initial_distribution(
	const Parameters& Params,
	const ClassicalDoubleVector& x,
	const ClassicalDoubleVector& p)
{
	QuantumComplexMatrix result = QuantumComplexMatrix::Zero(NumPES, NumPES);
	const ClassicalDoubleVector& x0 = Params.get_x0();
	const ClassicalDoubleVector& p0 = Params.get_p0();
	const ClassicalDoubleVector& SigmaX0 = Params.get_sigma_x0();
	const ClassicalDoubleVector& SigmaP0 = Params.get_sigma_p0();
	result(0, 0) = std::exp(-(((x - x0).array() / SigmaX0.array()).abs2().sum() + ((p - p0).array() / SigmaP0.array()).abs2().sum()) / 2.0)
		/ (std::pow(2.0 * M_PI, Dim) * SigmaX0.prod() * SigmaP0.prod());
	return result;
}

const double MCParameters::AboveMinFactor = 2.5; ///< choose the minimal step or max displacement whose autocor < factor*min{autocor}

MCParameters::MCParameters(const Parameters& Params, const int NOMC_):
	NumPoints(Params.get_number_of_selected_points()), NOMC(NOMC_),
	displacement(std::min(Params.get_sigma_x0().minCoeff(), Params.get_sigma_p0().minCoeff()))
{
}

/// @brief To select phase points as the initial phase coordinates for monte carlo
/// @param[in] Params The input parameters (mass, dt, etc)
/// @param[in] MCParams Monte carlo parameters (number of steps, maximum displacement, etc)
/// @param[in] distribution The function object to generate the distribution
/// @param[in] IsSmall The matrix that saves whether each element is small or not
/// @param[in] density The selected density matrices
/// @return The phase points with their density matrices
/// @details The points are at an estimated place that they will be at next time step.
static Maximums get_maximums(
	const Parameters& Params,
	const MCParameters& MCParams,
	const DistributionFunction& distribution,
	const QuantumBoolMatrix& IsSmall,
	const EvolvingDensity& density)
{
	// alias names
	static const ClassicalDoubleVector& mass = Params.get_mass();
	static const double dt = Params.get_dt();
	const int NumPoints = MCParams.get_num_points();
	Maximums maxs;
	for (int iElement = 0; iElement < NumElements; iElement++)
	{
		const int iPES = iElement / NumPES, jPES = iElement % NumPES;
		if (IsSmall(iPES, jPES) == false)
		{
			const int BeginIndex = iElement * NumPoints;
			// with evolution
			for (int iPoint = 0; iPoint < NumPoints; iPoint++)
			{
				auto [x, p, rho] = density[BeginIndex + iPoint];
				const Tensor3d f = adiabatic_force(x);
				x = x.array() + p.array() / mass.array() * dt;
				for (int iDim = 0; iDim < Dim; iDim++)
				{
					p[iDim] += (f[iDim](iPES, iPES) + f[iDim](jPES, jPES)) / 2.0 * dt;
				}
				rho = distribution(x, p);
				maxs[iElement].push_back(std::make_tuple(x, p, rho));
			}
		}
	}
	return maxs;
}

/// @brief To select phase points as the initial phase coordinates for monte carlo
/// @param[in] MCParams Monte carlo parameters (number of steps, maximum displacement, etc)
/// @param[in] distribution The function object to generate the distribution
/// @param[in] IsSmall The matrix that saves whether each element is small or not
/// @param[in] density The selected density matrices
/// @return The phase points with their density matrices
/// @details The points are at their original places.
static Maximums get_maximums(
	const MCParameters& MCParams,
	const DistributionFunction& distribution,
	const QuantumBoolMatrix& IsSmall,
	const EvolvingDensity& density)
{
	const int NumPoints = MCParams.get_num_points();
	Maximums maxs;
	for (int iElement = 0; iElement < NumElements; iElement++)
	{
		if (IsSmall(iElement / NumPES, iElement % NumPES) == false)
		{
			// no evolution, just the original ones
			maxs[iElement].insert(
				maxs[iElement].end(),
				density.begin() + iElement * NumPoints,
				density.begin() + (iElement + 1) * NumPoints);
		}
	}
	return maxs;
}

/// @brief To calculate all the displacements
/// @param[in] MCParams Parameters objects containing all the required information (dmax, steps, etc)
/// @return The displacement distribution of positions and momenta
static std::pair<Displacements, Displacements>& generate_displacements(const MCParameters& MCParams)
{
	static std::pair<Displacements, Displacements> result;
	for (int iDim = 0; iDim < Dim; iDim++)
	{
		result.first[iDim] = std::uniform_real_distribution<double>(-MCParams.get_max_dispalcement(), MCParams.get_max_dispalcement());
		result.second[iDim] = std::uniform_real_distribution<double>(-MCParams.get_max_dispalcement(), MCParams.get_max_dispalcement());
	}
	return result;
}

/// @brief To calculate all the displacements based on different number of grids
/// @param[inout] MCParams Parameters objects containing all the required information (dmax, steps, etc)
/// @param[in] NGrids The number of grids used for generation of this displacement distributions
/// @return The displacement distribution of positions and momenta
static std::pair<Displacements, Displacements>& generate_displacements(MCParameters& MCParams, const double displacement)
{
	// different grids, reset distributions
	MCParams.set_displacement(displacement);
	return generate_displacements(MCParams);
}

/// @brief To generate the Markov chain monte carlo
/// @param[in] MCParams Parameters objects containing all the required information (dmax, steps, etc)
/// @param[in] distribution The function object to generate the distribution
/// @param[in] XDispl The displacement distribution of positions
/// @param[in] PDispl The displacement distribution of momenta
/// @param[in] ElementIndex The index of the density matrix element
/// @param[in] x The initial position
/// @param[in] p The initial momentum
/// @param[in] weight The initial weight
/// @return A vector containing the phase space coordinates of all grids
static ClassicalPhaseVectors monte_carlo_chain(
	const MCParameters& MCParams,
	const DistributionFunction& distribution,
	Displacements& XDispl,
	Displacements& PDispl,
	const int ElementIndex,
	const ClassicalDoubleVector& x,
	const ClassicalDoubleVector& p,
	const double weight)
{
	// set parameters
	static std::mt19937 engine(std::chrono::system_clock::now().time_since_epoch().count()); // Random number generator, using merseen twister with seed from time
	static std::uniform_real_distribution<double> mc_selection(0.0, 1.0);					 // for whether pick or not
	ClassicalDoubleVector x_old, x_new, p_old, p_new;
	double weight_old = weight, weight_new;
	x_old = x;
	p_old = p;
	ClassicalPhaseVectors r_all(MCParams.get_num_MC_steps() + 1);
	r_all[0] << x_old, p_old;
	// going the chain
	for (int iIter = 1; iIter <= MCParams.get_num_MC_steps(); iIter++)
	{
		x_new = x_old, p_new = p_old;
		for (int iDim = 0; iDim < Dim; iDim++)
		{
			x_new[iDim] += XDispl[iDim](engine);
			p_new[iDim] += PDispl[iDim](engine);
		}
		const double weight_new = weight_function(get_density_matrix_element(distribution(x_new, p_new), ElementIndex));
		if (weight_new > weight_old || weight_new / weight_old > mc_selection(engine))
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

/// @details Assuming that the distribution function containing all the required information. If
/// it is a distribution from some parameters, those parameters should have already been
/// bound; if it is some learning method, the parameters should have already been optimized.
/// This MC function gives a random displacement in the phase space each time, then simply
/// input the phase space coordinate and hope to get the predicted density matrix, calculate
/// its weight by the weight function, and finally do the MC selection based on the weight.
void monte_carlo_selection(
	const Parameters& Params,
	const MCParameters& MCParams,
	const DistributionFunction& distribution,
	const QuantumBoolMatrix& IsSmall,
	EvolvingDensity& density,
	const bool IsToBeEvolved)
{
	// parameters needed for MC
	const int NumPoints = MCParams.get_num_points(); // The number of threads in MC
	// get maximum, then evolve the points adiabatically
	Maximums maxs = IsToBeEvolved == true ? get_maximums(Params, MCParams, distribution, IsSmall, density)
										  : get_maximums(MCParams, distribution, IsSmall, density);
	density.clear();

	// moving random number generator and MC random number generator
	std::pair<Displacements, Displacements> displacements = generate_displacements(MCParams);

	// Monte Carlo selection
	std::array<Eigen::VectorXd, NumElements> result;
	for (int iElement = 0; iElement < NumElements; iElement++)
	{
		if (IsSmall(iElement / NumPES, iElement % NumPES) == false)
		{
			// begin from an old point
#pragma omp parallel for
			for (auto& [x, p, rho] : maxs[iElement])
			{
				// do MC
				const ClassicalPhaseVectors WholeChain = monte_carlo_chain(
					MCParams,
					distribution,
					displacements.first,
					displacements.second,
					iElement,
					x,
					p,
					weight_function(get_density_matrix_element(rho, iElement)));
				x = WholeChain.crbegin()->block<Dim, 1>(0, 0);
				p = WholeChain.crbegin()->block<Dim, 1>(Dim, 0);
				rho = distribution(x, p);
			}
			// after MC, the new point is at the original place
			// TODO: then update the number of MC steps
			// after all points updated, insert into the EvolvingDensity
			density.insert(density.end(), maxs[iElement].cbegin(), maxs[iElement].cend());
		}
		else
		{
			density.insert(density.end(), NumPoints, std::make_tuple(ClassicalDoubleVector::Zero(), ClassicalDoubleVector::Zero(), QuantumComplexMatrix::Zero()));
		}
	}
}

/// @brief To calculate the autocorrelation of one thread
/// @param[in] r The phase space coordinates on the thread
/// @return The autocorrelation of different number of steps
/// @details For j steps, the autocorrelation is
/// \f[
/// \langle R_{i+j} R_i\rangle=\frac{1}{N-j}\sum_{i=1}^{N-j}R_{i+j}R_i-|\overline{r}|^2
/// \f]
static Eigen::VectorXd calculate_autocorrelation(const ClassicalPhaseVectors& r)
{
	const int NSteps = r.size();
	Eigen::VectorXd result(NSteps / 2);
	// first, calculate average (aka sum / N)
	const ClassicalPhaseVector Ave = std::accumulate(r.begin(), r.end(), ClassicalPhaseVector(ClassicalPhaseVector::Zero())) / NSteps;
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
static double calculate_autocorrelation(const ClassicalPhaseVectors& r, const int AutoCorSteps)
{
	const int NSteps = r.size();
	const ClassicalPhaseVector Ave = std::accumulate(r.begin(), r.end(), ClassicalPhaseVector(ClassicalPhaseVector::Zero())) / NSteps;
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
	const QuantumBoolMatrix& IsSmall,
	const EvolvingDensity& density)
{
	// parameters needed for MC
	static const int MaxNOMC = PhaseDim * 400; // The number of monte carlo steps (including head and tail)
	static const int MaxAutoCorStep = (MaxNOMC + 1) / 2;
	const Maximums maxs = get_maximums(MCParams, distribution, IsSmall, density);

	// moving random number generator and MC random number generator
	std::pair<Displacements, Displacements> displacements = generate_displacements(MCParams);

	AutoCorrelations result; // save the return value, the autocorrelations
	result.fill(Eigen::VectorXd::Zero(MaxAutoCorStep));
	std::array<int, NumElements> BestMCSteps = { 0 };
	for (int iElement = 0; iElement < NumElements; iElement++)
	{
		if (IsSmall(iElement / NumPES, iElement % NumPES) == false)
		{
#pragma omp parallel
			{
				Eigen::VectorXd localsum = Eigen::VectorXd::Zero(MaxAutoCorStep);
#pragma omp single
				{
					MCParams.set_num_MC_steps(MaxNOMC);
				}
#pragma omp for
				// calculate for different number of steps, up to the maximum available number
				for (auto [x, p, rho] : maxs[iElement])
				{
					localsum += calculate_autocorrelation(monte_carlo_chain(
						MCParams,
						distribution,
						displacements.first,
						displacements.second,
						iElement,
						x,
						p,
						weight_function(get_density_matrix_element(rho, iElement))));
				}
#pragma omp critical
				{
					result[iElement] += localsum;
				}
			}
			result[iElement] /= maxs[iElement].size();
			// find the best step
			const double MinAutoCor = std::abs(result[iElement].array().abs().minCoeff(&BestMCSteps[iElement]));
			for (int iStep = 1; iStep < BestMCSteps[iElement]; iStep++)
			{
				if (std::abs(result[iElement][iStep]) <= MCParameters::AboveMinFactor * MinAutoCor)
				{
					BestMCSteps[iElement] = iStep;
					break;
				}
			}
		}
		// else fill 0 for small elements
	}
	// after all elements, using the max stepsize and max grid size
	MCParams.set_num_MC_steps(*std::max_element(BestMCSteps.cbegin(), BestMCSteps.cend()));
	return result;
}

AutoCorrelations autocorrelation_optimize_displacement(
	MCParameters& MCParams,
	const DistributionFunction& distribution,
	const QuantumBoolMatrix& IsSmall,
	const EvolvingDensity& density)
{
	// parameters needed for MC
	static const std::vector<double> PossibleDisplacement = { 1e-3, 2e-3, 5e-3, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5 };
	static const int NPossibleDisplacement = PossibleDisplacement.size();
	const Maximums maxs = get_maximums(MCParams, distribution, IsSmall, density);

	// moving random number generator and MC random number generator
	std::pair<Displacements, Displacements> displacements = generate_displacements(MCParams);

	AutoCorrelations result; // save the return value, the autocorrelations
	result.fill(Eigen::VectorXd::Zero(NPossibleDisplacement));
	std::array<int, NumElements> BestDisplacement;
	BestDisplacement.fill(std::numeric_limits<int>::max());
	for (int iElement = 0; iElement < NumElements; iElement++)
	{
		if (IsSmall(iElement / NumPES, iElement % NumPES) == false)
		{
#pragma omp parallel
			{
				Eigen::VectorXd localsum = Eigen::VectorXd::Zero(NPossibleDisplacement);
				for (int iDispl = 0; iDispl < NPossibleDisplacement; iDispl++)
				{
#pragma omp single
					{
						MCParams.set_displacement(PossibleDisplacement[iDispl]);
					}
#pragma omp for
					for (auto [x, p, rho] : maxs[iElement])
					{
						localsum[iDispl] += calculate_autocorrelation(
							monte_carlo_chain(
								MCParams,
								distribution,
								displacements.first,
								displacements.second,
								iElement,
								x,
								p,
								weight_function(get_density_matrix_element(rho, iElement))),
							MCParams.get_num_MC_steps());
					}
				}
#pragma omp critical
				{
					result[iElement] += localsum;
				}
			}
			result[iElement] /= maxs[iElement].size();
			// find the best displacement
			const double MinAutoCor = std::abs(result[iElement].array().abs().minCoeff(&BestDisplacement[iElement]));
			for (int iDispl = NPossibleDisplacement - 1; iDispl > BestDisplacement[iElement]; iDispl--)
			{
				if (std::abs(result[iElement][iDispl]) <= MCParameters::AboveMinFactor * MinAutoCor)
				{
					BestDisplacement[iElement] = iDispl;
					break;
				}
			}
		}
		// else fill 0 for small elements
	}
	// after all elements, using the max stepsize and max grid size
	MCParams.set_displacement(PossibleDisplacement[*std::min_element(BestDisplacement.cbegin(), BestDisplacement.cend())]);
	return result;
}

/// @details This function fills the newly populated density matrix element
/// with density matrices of the greatest weight.
void new_element_point_selection(EvolvingDensity& density, const QuantumBoolMatrix& IsNew, const int NumPoints)
{
	if (IsNew.all() == false)
	{
		return;
	}
	assert(NumElements * NumPoints == density.size());
	EvolvingDensity density_copy = density;
	for (int iElement = 0; iElement < NumElements; iElement++)
	{
		if (IsNew(iElement / NumPES, iElement % NumPES) == true)
		{
			// rearrange to find the points with the largest weight
			std::nth_element(
				std::execution::par_unseq,
				density_copy.begin(),
				density_copy.begin() + NumPoints,
				density_copy.end(),
				[ElementIndex = iElement](const PhaseSpacePoint& psp1, const PhaseSpacePoint& psp2) -> bool { return weight_function(get_density_matrix_element(std::get<2>(psp1), ElementIndex)) > weight_function(get_density_matrix_element(std::get<2>(psp2), ElementIndex)); });
			// then copy to the right place
			std::copy(
				std::execution::par_unseq,
				density_copy.begin(),
				density_copy.begin() + NumPoints,
				density.begin() + iElement * NumPoints);
		}
	}
}
