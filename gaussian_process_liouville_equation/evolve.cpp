/// @file evolve.cpp
/// @brief Implementation of evolve.h

#include "stdafx.h"

#include "evolve.h"

#include "mc.h"
#include "pes.h"
#include "predict.h"

/// The direction of dynamics evolving
enum Direction
{
	Backward = -1, ///< Going back, for density prediction
	Forward = 1	   ///< Going forward, for point evolution
};

/// @brief To judge if current point have large coupling in any of the given directions
/// @param[in] x Position of classical degree of freedom
/// @param[in] p Momentum of classical degree of freedom
/// @param[in] mass Mass of classical degree of freedom
/// @return Whether each classical direction has coupling or not
static ClassicalVector<bool> is_coupling(
	[[maybe_unused]] const ClassicalVector<double>& x,
	[[maybe_unused]] const ClassicalVector<double>& p,
	[[maybe_unused]] const ClassicalVector<double>& mass)
{
	[[maybe_unused]] static constexpr double CouplingCriterion = 0.01; // Criteria of whether have coupling or not
#define ADIA
#ifdef ADIA
	return ClassicalVector<bool>::Zero();
#else
	if (NumPES == 2)
	{
		const ClassicalVector<double> nac_01 = tensor_slice(adiabatic_coupling(x), 0, 1), f_01 = tensor_slice(adiabatic_force(x), 0, 1);
		return nac_01.array() * p.array() / mass.array() > CouplingCriterion || f_01.array() > CouplingCriterion;
	}
	else
	{
		ClassicalVector<bool> result = ClassicalVector<bool>::Zero();
		for (std::size_t iPES = 1; iPES < NumPES; iPES++)
		{
			for (std::size_t jPES = 0; jPES < iPES; jPES++)
			{
				const ClassicalVector<double> nac = tensor_slice(adiabatic_coupling(x), iPES, jPES), f = tensor_slice(adiabatic_force(x), iPES, jPES);
				result = result.array() || nac.array() * p.array() / mass.array() > CouplingCriterion || f.array() > CouplingCriterion;
			}
		}
		return result;
	}
#endif
}

bool is_coupling(const AllPoints& density, const ClassicalVector<double>& mass)
{
	for (std::size_t iElement = 0; iElement < NumElements; iElement++)
	{
		const bool IsElementCoupling =
			std::any_of(
				std::execution::par_unseq,
				density[iElement].cbegin(),
				density[iElement].cend(),
				[&mass](const PhaseSpacePoint& psp) -> bool
				{
					const auto& [r, rho] = psp;
					const auto [x, p] = split_phase_coordinate(r);
					return is_coupling(x, p, mass).any();
				})
			&& !density[iElement].empty();
		if (IsElementCoupling)
		{
			return true;
		}
	}
	return false;
}

/// @brief To evolve all the positions
/// @param[in] x Position of classical degree of freedom
/// @param[in] p Momentum of classical degree of freedom
/// @param[in] mass Mass of classical degree of freedom
/// @param[in] dt The time interval of evolution
/// @param[in] drc The direction of evolution
/// @return All branches of momentum vectors
/// @details @f$ x_{i,j} = x_i - \frac{p_{i,j}}{m} dt @f$
static EigenVector<ClassicalVector<double>> position_evolve(
	const EigenVector<ClassicalVector<double>>& x,
	const EigenVector<ClassicalVector<double>>& p,
	const ClassicalVector<double>& mass,
	const double dt,
	const Direction drc)
{
	assert(p.size() % x.size() == 0);
	const std::size_t NumPoints = x.size(), NumBranch = p.size() / x.size();
	EigenVector<ClassicalVector<double>> result;
	for (std::size_t i = 0; i < NumPoints; i++)
	{
		for (std::size_t j = 0; j < NumBranch; j++)
		{
			result.push_back(x[i].array() + drc * dt * p[i * NumBranch + j].array() / mass.array());
		}
	}
	return result;
}

/// @brief To evolve one position
/// @param[in] x Position of classical degree of freedom
/// @param[in] p Momentum of classical degree of freedom
/// @param[in] mass Mass of classical degree of freedom
/// @param[in] dt The time interval of evolution
/// @param[in] drc The direction of evolution
/// @return All branches of momentum vectors
/// @details @f$ x_{i,j} = x_i - \frac{p_{i,j}}{m} dt @f$
static inline ClassicalVector<double> position_evolve(
	const ClassicalVector<double>& x,
	const ClassicalVector<double>& p,
	const ClassicalVector<double>& mass,
	const double dt,
	const Direction drc)
{
	return x.array() + drc * dt * p.array() / mass.array();
}

/// @brief To get the diagonal forces under adiabatic basis at given position
/// @param[in] x Position of classical degree of freedom
/// @return Bunches of force vectors corresponding to diagonal elements of force matrices of the whole force tensor
static EigenVector<ClassicalVector<double>> adiabatic_diagonal_forces(const ClassicalVector<double>& x)
{
	const Tensor3d& force = adiabatic_force(x);
	EigenVector<ClassicalVector<double>> result;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		result.push_back(tensor_slice(force, iPES, iPES));
	}
	return result;
}

/// @brief To evolve the momentum under the adiabatic diagonal forces
/// @param[in] x Position of classical degree of freedom
/// @param[in] p Momentum of classical degree of freedom
/// @param[in] dt The time interval of evolution
/// @param[in] drc The direction of evolution
/// @return All branches of momentum vectors
/// @details @f$ p_{i,j} = p_i - f_j(x_i) dt @f$, where @f$ f_j = \frac{f_{aa}+f_{bb}}{2} @f$
static EigenVector<ClassicalVector<double>> momentum_diagonal_branching_evolve(
	const EigenVector<ClassicalVector<double>>& x,
	const EigenVector<ClassicalVector<double>>& p,
	const double dt,
	const Direction drc)
{
	assert(x.size() == p.size());
	const std::size_t NumPoints = x.size();
	EigenVector<ClassicalVector<double>> result;
	for (std::size_t i = 0; i < NumPoints; i++)
	{
		const EigenVector<ClassicalVector<double>> f = adiabatic_diagonal_forces(x[i]);
		for (std::size_t iPES = 0; iPES < NumPES; iPES++)
		{
			for (std::size_t jPES = 0; jPES <= iPES; jPES++)
			{
				result.push_back(p[i] + drc * dt / 2.0 * (f[iPES] + f[jPES]));
			}
		}
	}
	return result;
}

/// @brief To evolve the momentum under the adiabatic diagonal forces
/// @param[in] x Position of classical degree of freedom
/// @param[in] p Momentum of classical degree of freedom
/// @param[in] dt The time interval of evolution
/// @param[in] drc The direction of evolution
/// @param[in] iPES The first index of density matrix
/// @param[in] jPES The second index of density matrix
/// @return All branches of momentum vectors
/// @details @f$ p_{i,j} = p_i - f_j(x_i) dt @f$, where @f$ f_j = \frac{f_{aa}+f_{bb}}{2} @f$
static ClassicalVector<double> momentum_diagonal_nonbranch_evolve(
	const ClassicalVector<double>& x,
	const ClassicalVector<double>& p,
	const double dt,
	const Direction drc,
	const std::size_t iPES,
	const std::size_t jPES)
{
	const EigenVector<ClassicalVector<double>> f = adiabatic_diagonal_forces(x);
	return p + drc * dt / 2.0 * (f[iPES] + f[jPES]);
}


/// @brief To evolve the momentum under the off-diagonal forces
/// @param[in] x Position of classical degree of freedom
/// @param[in] p Momentum of classical degree of freedom
/// @param[in] dt The time interval of evolution
/// @param[in] Couple A vector of 0 or 1, 1 for evolving dimensions and 0 for non-evolving
/// @return All branches of momentum vectors
/// @details @f$ p_{i,j} = p_i + n f_{01}(x_i) dt @f$, where @f$ n = -1, 0, 1 @f$ and @f$ f_{01}=(E_0-E_1)d_{01} @f$
static EigenVector<ClassicalVector<double>> momentum_offdiagonal_evolve(
	const EigenVector<ClassicalVector<double>>& x,
	const EigenVector<ClassicalVector<double>>& p,
	const double dt,
	const ClassicalVector<double>& Couple)
{
	assert(x.size() == p.size());
	if (Couple.cast<bool>().any())
	{
		const std::size_t NumPoints = x.size();
		EigenVector<ClassicalVector<double>> result;
		for (std::size_t i = 0; i < NumPoints; i++)
		{
			const ClassicalVector<double> f_01 = tensor_slice(adiabatic_force(x[i]), 0, 1);
			result.push_back(p[i].array() - f_01.array() * dt * Couple.array());
			result.push_back(p[i]);
			result.push_back(p[i].array() + f_01.array() * dt * Couple.array());
		}
		return result;
	}
	else
	{
		return p;
	}
}

/// @brief To calculate sin and cos of @f$ dt(\omega(x_0)+2\omega(x_1)+\omega(x_2))/4 @f$,
/// where @f$ \omega(x) =(E_i(x)-E_j(x))/\hbar @f$
/// @param[in] x0 First position
/// @param[in] x1 Second position
/// @param[in] x2 Third position
/// @param[in] dt Time interval
/// @param[in] drc The direction of evolution
/// @param[in] SmallIndex The smaller index of density matrix
/// @param[in] LargeIndex The larger index of density matrix
/// @return sine and cosine of @f$ dt(\omega(x_0)+2\omega(x_1)+\omega(x_2))/4 @f$
static inline Eigen::Vector2d calculate_omega0(
	const ClassicalVector<double>& x0,
	const ClassicalVector<double>& x1,
	const ClassicalVector<double>& x2,
	const double dt,
	const Direction drc,
	const std::size_t SmallIndex,
	const std::size_t LargeIndex)
{
	assert(SmallIndex < LargeIndex && LargeIndex < NumPES);
	const QuantumVector<double> E0 = adiabatic_potential(x0), E1 = adiabatic_potential(x1), E2 = adiabatic_potential(x2);
	const double omega = -drc * dt / 4.0 / hbar
		* (E0[SmallIndex] - E0[LargeIndex] + 2.0 * (E1[SmallIndex] - E1[LargeIndex]) + E2[SmallIndex] - E2[LargeIndex]);
	Eigen::Vector2d result;
	result << std::sin(omega), std::cos(omega);
	return result;
}

/// @brief To expand x so that x and p have same number of vectors
/// @param[inout] x Positions
/// @param[in] p Momenta
void expand(EigenVector<ClassicalVector<double>>& x, const EigenVector<ClassicalVector<double>>& p)
{
	assert(p.size() % x.size() == 0);
	if (x.size() != p.size())
	{
		const std::size_t repeat = p.size() / x.size() - 1;
		x.reserve(p.size());
		for (auto iter = x.cbegin(); iter != x.cend(); ++iter)
		{
			iter = x.insert(iter, repeat, *iter) + repeat;
		}
	}
};

/// @brief To evolve all dimensions adiabatically
/// @param[inout] x The positions of the evolving dimensions
/// @param[inout] p The momenta of the evolving dimensions
/// @param[in] mass Mass of classical degree of freedom
/// @param[in] dt Time interval
/// @param[in] drc The direction of evolution
/// @param[in] iPES The first index of density matrix
/// @param[in] jPES The second index of density matrix
static inline void adiabatic_evolve(
	ClassicalVector<double>& x,
	ClassicalVector<double>& p,
	const ClassicalVector<double>& mass,
	const double dt,
	const Direction drc,
	const std::size_t iPES,
	const std::size_t jPES)
{
	x = position_evolve(x, p, mass, dt / 2.0, drc);
	p = momentum_diagonal_nonbranch_evolve(x, p, dt, drc, iPES, jPES);
	x = position_evolve(x, p, mass, dt / 2.0, drc);
}

/// @brief To calculate sin and cos of @f$ dt(\omega(x_2^{\gamma})+2\omega(x_{3,n}^{\gamma})+\omega(x_{4,n}^{\gamma,\gamma^{\prime}}))/4 @f$
/// @param[in] x2 Third position
/// @param[in] x3 Fourth position
/// @param[in] x4 Final position
/// @param[in] dt Time interval
/// @param[in] drc The direction of evolution
/// @return sin and cos @f$ dt(\omega(x_2^{\gamma})+2\omega(x_{3,n}^{\gamma})+\omega(x_{4,n}^{\gamma,\gamma^{\prime}}))/4 @f$,
/// where @f$ \omega(x) = (E_0(x)-E_1(x))/\hbar @f$
static Eigen::MatrixXd calculate_omega1_of_2_level(
	const EigenVector<ClassicalVector<double>>& x2,
	const EigenVector<ClassicalVector<double>>& x3,
	const EigenVector<ClassicalVector<double>>& x4,
	const double dt,
	const Direction drc)
{
	const std::size_t NumPoints = x2.size(), NumBranch1 = x3.size() / x2.size(), NumBranch2 = x4.size() / x3.size();
	Eigen::MatrixXd result(x3.size(), 2);
	for (std::size_t i = 0; i < NumPoints; i++)
	{
		for (std::size_t j = 0; j < NumBranch1; j++)
		{
			const std::size_t Index = i * NumBranch1 + j;
			result.row(Index) = calculate_omega0(x2[i], x3[Index], x4[Index * NumBranch2 + 1], dt, drc, 0, 1);
		}
	}
	return result;
}

/// @brief To calculate sin and cos of @f$ \frac{\vec{P}\cdot\vec{d}_{01}}{M}dt @f$
/// @param[in] x Position of classical degree of freedom
/// @param[in] p Momentum of classical degree of freedom
/// @param[in] mass Mass of classical degree of freedom
/// @param[in] dt Time interval
/// @param[in] drc The direction of evolution
/// @return sin and cos @f$ \frac{\vec{P}\cdot\vec{d}_{01}}{M}dt @f$ for each input (x,p) pairs
static Eigen::MatrixXd calculate_phi_of_2_level(
	const EigenVector<ClassicalVector<double>>& x,
	const EigenVector<ClassicalVector<double>>& p,
	const ClassicalVector<double>& mass,
	const double dt,
	const Direction drc)
{
	assert(p.size() % x.size() == 0);
	const std::size_t NumPoints = x.size(), NumBranch = p.size() / x.size();
	Eigen::MatrixXd result(p.size(), 2);
	for (std::size_t i = 0; i < NumPoints; i++)
	{
		const ClassicalVector<double>& d01 = tensor_slice(adiabatic_coupling(x[i]), 0, 1);
		for (std::size_t j = 0; j < NumBranch; j++)
		{
			const std::size_t Index = i * NumBranch + j;
			const double phi = -drc * dt * (p[Index].array() / mass.array()).matrix().dot(d01);
			result.row(Index) << std::sin(phi), std::cos(phi);
		}
	}
	return result;
}

/// @brief To construct the training set as a matrix from std::vector of x and p
/// @param[in] x Position of classical degree of freedom
/// @param[in] p Momentum of classical degree of freedom
/// @return The combination of the input
static Eigen::MatrixXd construct_training_feature(const EigenVector<ClassicalVector<double>>& x, const EigenVector<ClassicalVector<double>>& p)
{
	assert(x.size() == p.size());
	const std::size_t size = x.size();
	Eigen::MatrixXd result(PhaseDim, size);
	const std::vector<std::size_t> indices = get_indices(size);
	std::for_each(
		std::execution::par_unseq,
		indices.cbegin(),
		indices.cend(),
		[&result, &x, &p](std::size_t i) -> void
		{
			result.col(i) << x[i], p[i];
		});
	return result;
}

/// @brief To calculate all the required density matrix elements for non-adiabatic back propagation prediction
/// @param[in] x4 Final position
/// @param[in] p3 Final momentum
/// @param[in] Kernels An array of predictors for prediction, whose size is NumElements
/// @return All the density matrix elements required for prediction
static Eigen::MatrixXd calculate_non_adiabatic_rho_elements(
	const EigenVector<ClassicalVector<double>>& x4,
	const EigenVector<ClassicalVector<double>>& p3,
	const OptionalKernels& Kernels)
{
	const std::size_t NBranch = x4.size() / 3;
	Eigen::MatrixXd result = Eigen::MatrixXd::Zero(NBranch, NumElements);
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES <= iPES; jPES++)
		{
			const std::size_t ElmIdx = iPES * NumPES + jPES;
			const Eigen::MatrixXd TrainingFeature = construct_training_feature(
				EigenVector<ClassicalVector<double>>(x4.begin() + ElmIdx * NBranch, x4.begin() + (ElmIdx + 1) * NBranch),
				EigenVector<ClassicalVector<double>>(p3.begin() + ElmIdx * NBranch, p3.begin() + (ElmIdx + 1) * NBranch));
			if (Kernels[ElmIdx].has_value())
			{
				result.col(ElmIdx) = predict_elements_with_variance_comparison(Kernels[ElmIdx].value(), TrainingFeature);
			}
			if (iPES != jPES)
			{
				const std::size_t SymElmIdx = jPES * NumPES + iPES;
				if (Kernels[SymElmIdx].has_value())
				{
					result.col(SymElmIdx) = predict_elements_with_variance_comparison(Kernels[SymElmIdx].value(), TrainingFeature);
				}
			}
		}
	}
	// now the arrangement is: gamma*(n, gamma'); need to rearrange to gamma*(gamma', n) for prediction
	Eigen::MatrixXd rearrange_result(NBranch, NumElements);
	for (std::size_t iN = 0; iN < 3; iN++)
	{
		for (std::size_t iGammaPrime = 0; iGammaPrime < 3; iGammaPrime++)
		{
			const std::size_t OldRow = iN * 3 + iGammaPrime, NewRow = iGammaPrime * 3 + iN;
			rearrange_result.row(NewRow) = result.row(OldRow);
		}
	}
	return rearrange_result;
}

/**
 * @brief To calculate the element of 2-level system
 * @param[in] phi sin and cos of @f$ dt\phi_{\pm} @f$
 * @param[in] omega1 sin and cos of @f$ dt\omega_{1,\pm}/2 @f$
 * @param[in] rho The predicted density matrix element
 * @param[in] Coe_1 @f$ c_{-1} @f$
 * @param[in] Coe0 @f$ c_0 @f$
 * @param[in] Coe1 @f$ c_1 @f$
 * @return The element
 * @details The element is given by @n
 * @f{eqnarray*}{
 * &&c_{-1}((1+\sin\phi_-)\rho_{00}+2\cos\phi_-\cos\omega_{1,-}\rho_{01}-2\cos\phi_-\sin\omega_{1,-}\rho_{10}+(1-\sin\phi_-)\rho_{11}) \\
 * +&&c_0(\cos\phi\rho_{00}-2\sin\phi\cos\omega_1\rho_{01}+2\sin\phi\sin\omega_1\rho_{10}-\cos\phi\rho_{11}) \\
 * +&&c_1((\sin\phi_+-1)\rho_{00}+2\cos\phi_+\cos\omega_{1,+}\rho_{01}-2\cos\phi_+\sin\omega_{1,+}\rho_{10}-(\sin\phi_++1)\rho_{11})
 * @f} @n
 * where @f$ \rho_{01}=\Re\rho_{10} @f$ and @f$ \rho_{10}=\Im\rho_{10} @f$.
 */
static inline double calculate_2_level_element(
	const Eigen::MatrixXd& phi,
	const Eigen::MatrixXd& omega1,
	const Eigen::MatrixXd& rho,
	const double Coe_1,
	const double Coe0,
	const double Coe1)
{
	return Coe_1 * ((1.0 + phi(0, 0)) * rho(0, 0) + 2.0 * phi(0, 1) * omega1(0, 1) * rho(0, 1) - 2.0 * phi(0, 1) * omega1(0, 0) * rho(0, 2) + (1.0 - phi(0, 0)) * rho(0, 3))
		+ Coe0 * (phi(1, 1) * rho(1, 0) - 2.0 * phi(1, 0) * omega1(1, 1) * rho(1, 1) + 2.0 * phi(1, 0) * omega1(1, 0) * rho(1, 2) - phi(1, 1) * rho(1, 3))
		+ Coe1 * ((1.0 - phi(2, 0)) * rho(2, 0) - 2.0 * phi(2, 1) * omega1(2, 1) * rho(2, 1) + 2.0 * phi(2, 1) * omega1(2, 0) * rho(2, 2) + (1.0 + phi(2, 0)) * rho(2, 3));
}

/// @brief To predict the density matrix of the given point after evolving 1 time step back
/// @param[in] r Phase space coordinates of classical degree of freedom
/// @param[in] mass Mass of classical degree of freedom
/// @param[in] dt Time interval
/// @param[in] Kernels An array of kernels for prediction, whose size is NumElements
/// @return The density matrix at the given point after evolving
static QuantumMatrix<std::complex<double>> non_adiabatic_evolve_predict(
	const ClassicalPhaseVector& r,
	const ClassicalVector<double>& mass,
	const double dt,
	const OptionalKernels& Kernels)
{
	const auto [x, p] = split_phase_coordinate(r);
	using namespace std::literals::complex_literals;
	static constexpr Direction drc = Direction::Backward;
	QuantumMatrix<std::complex<double>> result = QuantumMatrix<std::complex<double>>::Zero();
	if constexpr (NumPES == 2)
	{
		// 2-level system
		const ClassicalVector<bool> IsCouple = is_coupling(x, p, mass);
		const ClassicalVector<double> Couple = IsCouple.cast<double>();
		// x_i and p_{i-1} have same number of elements, and p_i is branching
		if (IsCouple.any())
		{
			// 17 steps
			const ClassicalVector<double> x1 = position_evolve(x, p, mass, dt / 4.0, drc);
			// index: {00, 10, 11}
			const EigenVector<ClassicalVector<double>> p1 = momentum_diagonal_branching_evolve({x1}, {p}, dt / 2.0, drc);
			const EigenVector<ClassicalVector<double>> x2 = position_evolve({x1}, p1, mass, dt / 4.0, drc);
			// index: {00, 10, 11}, {-1, 0, 1}
			const EigenVector<ClassicalVector<double>> p2 = momentum_offdiagonal_evolve(x2, p1, dt, Couple);
			const EigenVector<ClassicalVector<double>> x3 = position_evolve(x2, p2, mass, dt / 4.0, drc);
			// index: {00, 10, 11}, {-1, 0, 1}, {00, 10, 11}
			const EigenVector<ClassicalVector<double>> p3 = momentum_diagonal_branching_evolve(x3, p2, dt / 2.0, drc);
			const EigenVector<ClassicalVector<double>> x4 = position_evolve(x3, p3, mass, dt / 4.0, drc);
			// prepare: evolved elements, phi, omega0, omega1
			const Eigen::Vector2d omega0 = calculate_omega0(x, x1, x2[1], dt / 2.0, drc, 0, 1);
			const Eigen::MatrixXd omega1 = calculate_omega1_of_2_level(x2, x3, x4, dt / 2.0, drc);
			const Eigen::MatrixXd phi = calculate_phi_of_2_level(x2, p2, mass, dt, drc);
			const Eigen::MatrixXd rho_predict = calculate_non_adiabatic_rho_elements(x4, p3, Kernels);
			// calculate
			const std::size_t NBranch = 3; // the availability of n (off-diagonal momentum evolution branches)
			result(0, 0) = calculate_2_level_element(
				phi.block(0, 0, NBranch, phi.cols()),
				omega1.block(0, 0, NBranch, omega1.cols()),
				rho_predict.block(0, 0, NBranch, rho_predict.cols()),
				(1.0 - phi(1, 0)) / 4.0,
				phi(1, 1) / 2.0,
				(1.0 + phi(1, 0)) / 4.0);
			result(1, 0) = (omega0[1] + 1.0i * omega0[0])
				* (1.0i * (omega1(1 + NBranch, 0) * rho_predict(1 + NBranch, 1) + omega1(1 + NBranch, 1) * rho_predict(1 + NBranch, 2))
					+ calculate_2_level_element(
						phi.block(NBranch, 0, NBranch, phi.cols()),
						omega1.block(NBranch, 0, NBranch, omega1.cols()),
						rho_predict.block(NBranch, 0, NBranch, rho_predict.cols()),
						phi(1 + NBranch, 1) / 4.0,
						phi(1 + NBranch, 0) / 2.0,
						-phi(1 + NBranch, 1) / 4.0));
			result(1, 1) = calculate_2_level_element(
				phi.block(2.0 * NBranch, 0, NBranch, phi.cols()),
				omega1.block(2.0 * NBranch, 0, NBranch, omega1.cols()),
				rho_predict.block(2.0 * NBranch, 0, NBranch, rho_predict.cols()),
				(1.0 + phi(1 + 2.0 * NBranch, 0)) / 4.0,
				-phi(1 + 2.0 * NBranch, 1) / 2.0,
				(1.0 - phi(1 + 2.0 * NBranch, 0)) / 4.0);
		}
		else
		{
			// evolve backward adiabatically
			// 7 steps
			const ClassicalVector<double> x1 = position_evolve(x, p, mass, dt / 2.0, drc);
			const EigenVector<ClassicalVector<double>> p1 = momentum_diagonal_branching_evolve({x1}, {p}, dt, drc);
			const EigenVector<ClassicalVector<double>> x2 = position_evolve({x1}, p1, mass, dt / 2.0, drc);
			const Eigen::Vector2d omega = calculate_omega0(x, x1, x2[1], dt, drc, 0, 1);
			if (Kernels[0].has_value())
			{
				result(0, 0) = predict_elements_with_variance_comparison(Kernels[0].value(), construct_training_feature({x2[0]}, {p1[0]})).value();
			}
			if (Kernels[1].has_value() || Kernels[2].has_value())
			{
				const Eigen::MatrixXd TrainingFeature = construct_training_feature({x2[1]}, {p1[1]});
				const double re = Kernels[1].has_value() ? predict_elements_with_variance_comparison(Kernels[1].value(), TrainingFeature).value() : 0.0;
				const double im = Kernels[2].has_value() ? predict_elements_with_variance_comparison(Kernels[2].value(), TrainingFeature).value() : 0.0;
				result(1, 0) = (re + 1.0i * im) * (omega[1] + 1.0i * omega[0]);
			}
			if (Kernels[3].has_value())
			{
				result(1, 1) = predict_elements_with_variance_comparison(Kernels[3].value(), construct_training_feature({x2[2]}, {p1[2]})).value();
			}
		}
	}
	else
	{
		assert(!"NO INSTANTATION OF MORE THAN TWO LEVEL SYSTEM NOW\n");
		// const ClassicalVector<bool> IsCoupling = is_coupling(x, p, mass);
	}
	return result.selfadjointView<Eigen::Lower>();
}

/// @details If the point is not in the coupling region, evolve it adiabatically;
/// otherwise, branch the point with non-adiabatic dynamics, then
/// calculate exact density at the new point, and select with monte carlo
void evolve(
	AllPoints& density,
	const ClassicalVector<double>& mass,
	const double dt,
	const OptionalKernels& Kernels)
{
	using namespace std::literals::complex_literals;
	static std::mt19937 engine(std::chrono::system_clock::now().time_since_epoch().count()); // Random number generator, using merseen twister with seed from time
	static std::uniform_real_distribution<double> mc_selection(0.0, 1.0);					 // for whether pick or not
	static constexpr Direction drc = Direction::Forward;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES < NumPES; jPES++)
		{
			if (iPES <= jPES)
			{
				// for row-major, visit upper triangular elements earlier
				// so selection for the upper triangular elements
				const std::size_t ElementIndex = iPES * NumPES + jPES;
				if (!density[ElementIndex].empty())
				{
					std::for_each(
						std::execution::par_unseq,
						density[ElementIndex].begin(),
						density[ElementIndex].end(),
						[&mass, dt, &Kernels, iPES, jPES](PhaseSpacePoint& psp) -> void
						{
							auto& [r, rho] = psp;
							auto [x, p] = split_phase_coordinate(const_cast<const ClassicalPhaseVector&>(r));
							// evolve the not-coupled degree adiabatically
							const ClassicalVector<bool> IsCouple = is_coupling(x, p, mass);
							if (IsCouple.any())
							{
								// for coupled cases, the exact density is calculated by back propagate
								// first, evolve adiabatically for dt/2
								adiabatic_evolve(x, p, mass, dt / 2, drc, iPES, jPES);
								// second, branch with the off-diagonal force
								const ClassicalVector<double> Couple = IsCouple.cast<double>();
								EigenVector<ClassicalVector<double>> x_branch = {x}, p_branch = momentum_offdiagonal_evolve(x_branch, {p}, dt, Couple);
								expand(x_branch, p_branch);
								// finally, evolve adiabatically again for dt/2
								// then predict its exact density matrix and corresponding weight
								const std::size_t NBranch = x_branch.size();
								EigenVector<QuantumMatrix<std::complex<double>>> predicts(NBranch, QuantumMatrix<std::complex<double>>::Zero());
								EigenVector<ClassicalPhaseVector> r_branch(NBranch, ClassicalPhaseVector::Zero());
								std::vector<double> weight(NBranch, 0.0);
								for (std::size_t iBranch = 0; iBranch < NBranch; iBranch++)
								{
									adiabatic_evolve(x_branch[iBranch], p_branch[iBranch], mass, dt / 2, drc, iPES, jPES);
									r_branch[iBranch] << x_branch[iBranch], p_branch[iBranch];
									predicts[iBranch] = non_adiabatic_evolve_predict(
										r_branch[iBranch],
										mass,
										dt,
										Kernels);
									weight[iBranch] = std::abs(predicts[iBranch](iPES, jPES));
								}
								// select one of them with monte carlo
								const double rand = mc_selection(engine) * std::reduce(weight.cbegin(), weight.cend());
								double sum = 0;
								for (std::size_t iBranch = 0; iBranch < NBranch; iBranch++)
								{
									sum += weight[iBranch];
									if (sum > rand)
									{
										r = r_branch[iBranch];
										rho = predicts[iBranch];
										break;
									}
								}
							}
							else
							{
								// evolve
								// To calculate phase later, every half-step positions are needed
								const ClassicalVector<double> x1 = position_evolve(x, p, mass, dt / 2.0, drc);
								const ClassicalVector<double> p1 = momentum_diagonal_nonbranch_evolve(x1, p, dt, drc, iPES, jPES);
								const ClassicalVector<double> x2 = position_evolve(x1, p1, mass, dt / 2.0, drc);
								// then change phase
								if (iPES != jPES)
								{
									const Eigen::Vector2d omega0 = calculate_omega0(x, x1, x2, dt, drc, std::min(iPES, jPES), std::max(iPES, jPES));
									rho(std::max(iPES, jPES), std::min(iPES, jPES)) *= (omega0[1] + 1.0i * omega0[0]);
									rho(std::min(iPES, jPES), std::max(iPES, jPES)) = std::conj(rho(std::max(iPES, jPES), std::min(iPES, jPES)));
								}
								// finally, update the phase space coordinates
								r << x2, p1;
							}
						});
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
