/// @file evolve.cpp
/// @brief Implementation of evolve.h

#include "stdafx.h"

#include "evolve.h"

#include "mc.h"
#include "pes.h"
#include "predict.h"

static constexpr std::size_t NumOffDiagonalBranches = 3; ///< The number of branches resulted by one off-diagonal force, including forward, static and backward
static constexpr std::array<std::ptrdiff_t, NumOffDiagonalBranches> OffDiagonalBranches = {-1, 0, 1};

/// The matrix that saves diagonal forces: i-th column is the force on i-th surface
using DiagonalForces = Eigen::Matrix<double, Dim, NumPES>;
/// The tensor form of @p ClassicalVector<double>
using TensorVector = Eigen::TensorFixedSize<double, Eigen::Sizes<Dim>>;
/// The tensor form for @f$ P_1 @f$ and @f$ R_2 @f$
using DiagonalBranched = Eigen::TensorFixedSize<double, Eigen::Sizes<Dim, NumElements>>;
/// The tensor form for @f$ P_2 @f$ and @f$ R_3 @f$
using OffDiagonalBranched = Eigen::TensorFixedSize<double, Eigen::Sizes<Dim, NumOffDiagonalBranches, NumElements>>;
/// The tensor form for @f$ P_3 @f$ and @f$ R_4 @f$
using TotallyBranched = Eigen::TensorFixedSize<double, Eigen::Sizes<Dim, NumElements, NumOffDiagonalBranches, NumElements>>;

/// The direction of dynamics evolving
enum Direction
{
	Backward = -1, ///< Going back, for density prediction
	Forward = 1	   ///< Going forward, for point evolution
};

/// @brief To get a slice of the tensor
/// @param[in] tensor The tensor which the slice is from
/// @param[in] row The row index of the tensor
/// @param[in] col The column index of the tensor
/// @return A vector, that is tensor(0, row, col) to tensor(idx1_max, row, col)
static inline TensorVector tensor_slice(const Tensor3d& tensor, const std::size_t row, const std::size_t col)
{
	return tensor.slice(std::array<std::size_t, 3>{row, col, 0}, std::array<std::size_t, 3>{1, 1, Dim}).reshape(std::array<std::size_t, 1>{Dim});
}

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
	if constexpr (NumPES == 2)
	{
		const TensorVector nac_01 = tensor_slice(adiabatic_coupling(x), 0, 1), f_01 = tensor_slice(adiabatic_force(x), 0, 1);
		return ClassicalVector<double>::Map(nac_01.data()).array() * p.array() / mass.array() > CouplingCriterion
			|| ClassicalVector<double>::Map(f_01.data()).array() > CouplingCriterion;
	}
	else
	{
		ClassicalVector<bool> result = ClassicalVector<bool>::Zero();
		for (std::size_t iPES = 1; iPES < NumPES; iPES++)
		{
			for (std::size_t jPES = 0; jPES < iPES; jPES++)
			{
				const TensorVector nac = tensor_slice(adiabatic_coupling(x), iPES, jPES), f = tensor_slice(adiabatic_force(x), iPES, jPES);
				result = result.array() || ClassicalVector<double>::Map(nac.data()).array() * p.array() / mass.array() > CouplingCriterion
					|| ClassicalVector<double>::Map(f.data()).array() > CouplingCriterion;
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
template <std::ptrdiff_t... xIndices, std::ptrdiff_t... pIndices>
static Eigen::TensorFixedSize<double, Eigen::Sizes<Dim, pIndices...>> position_evolve(
	const Eigen::TensorFixedSize<double, Eigen::Sizes<Dim, xIndices...>>& x,
	const Eigen::TensorFixedSize<double, Eigen::Sizes<Dim, pIndices...>>& p,
	const ClassicalVector<double>& mass,
	const double dt,
	const Direction drc)
{
	// broadcast mass
	const Eigen::TensorFixedSize<double, Eigen::Sizes<Dim, pIndices...>> MassBroadcast =
		TensorVector(Eigen::TensorMap<const Eigen::Tensor<const double, 1>>(mass.data(), Dim)).broadcast(std::array<std::size_t, 1>{p.size() / Dim}).reshape(p.dimensions());
	if constexpr (sizeof... (xIndices) == sizeof... (pIndices))
	{
		return x + drc * dt * p / MassBroadcast;
	}
	else
	{
		static_assert(sizeof... (xIndices) + 1u == sizeof... (pIndices));
		// reshape x first to add a dim
		auto generate_reshape = [](void) -> std::array<std::ptrdiff_t, sizeof... (xIndices) + 2u>
		{
			std::array<std::ptrdiff_t, sizeof... (xIndices) + 2u> result = {Dim, 1};
			std::size_t i = 2;
			[[maybe_unused]] auto assign = [&i, &result](std::ptrdiff_t val) -> void
			{
				result[i++] = val;
			};
			(assign(xIndices), ...);
			return result;
		};
		auto generate_broadcast_shape = [&x, &p](void) -> std::array<std::ptrdiff_t, sizeof... (xIndices) + 2u>
		{
			std::array<std::ptrdiff_t, sizeof... (xIndices) + 2u> result;
			result.fill(1);
			result[1] = p.size() / x.size();
			return result;
		};
		return x.reshape(generate_reshape()).broadcast(generate_broadcast_shape()) + drc * dt * p / MassBroadcast;
	}
}

/// @brief To get the diagonal forces under adiabatic basis at given position
/// @param[in] x Position of classical degree of freedom
/// @return A matrix, i-th row is the force of i-th surface
static DiagonalForces adiabatic_diagonal_forces(const ClassicalVector<double>& x)
{
	const Tensor3d& force = adiabatic_force(x);
	DiagonalForces result;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		const TensorVector f_i = tensor_slice(force, iPES, iPES);
		result.col(iPES) = ClassicalVector<double>::Map(f_i.data());
	}
	return result;
}

/// @brief To evolve the momentum under the adiabatic diagonal forces (i.e., @f$ P_0 @f$ to @f$ P_1 @f$)
/// @param[in] x Position of classical degree of freedom
/// @param[in] p Momentum of classical degree of freedom
/// @param[in] dt The time interval of evolution
/// @param[in] drc The direction of evolution
/// @return All branches of momentum vectors
/// @details @f$ p_i = p_0 - f_i(x_0) dt @f$, where @f$ f_i = \frac{f_{aa}+f_{bb}}{2} @f$
static DiagonalBranched momentum_diagonal_evolve(
	const TensorVector& x,
	const TensorVector& p,
	const double dt,
	const Direction drc)
{
	DiagonalBranched result;
	// x and p are vectors, result is a Dim * NumElements matrix
	// map the diagonal forces to a matrix
	const DiagonalForces f = adiabatic_diagonal_forces(ClassicalVector<double>::Map(x.data()));
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES < NumPES; jPES++)
		{
			const TensorVector sum_f = Eigen::TensorMap<const Eigen::Tensor<const double, 1>>(f.col(iPES).data(), Dim)
				+ Eigen::TensorMap<const Eigen::Tensor<const double, 1>>(f.col(jPES).data(), Dim);
			result.chip(iPES * NumPES + jPES, 1) = p + drc * dt / 2.0 * sum_f;
		}
	}
	return result;
}

/// @brief To evolve the momentum under the adiabatic diagonal forces (i.e., @f$ P_2 @f$ to @f$ P_3 @f$)
/// @param[in] x Position of classical degree of freedom
/// @param[in] p Momentum of classical degree of freedom
/// @param[in] dt The time interval of evolution
/// @param[in] drc The direction of evolution
/// @return All branches of momentum vectors
/// @details @f$ p_{j,n,i} = p_{n,i} - f_j(x_{n,i}) dt @f$, where @f$ f_j = \frac{f_{aa}+f_{bb}}{2} @f$
static TotallyBranched momentum_diagonal_evolve(
	const OffDiagonalBranched& x,
	const OffDiagonalBranched& p,
	const double dt,
	const Direction drc)
{
	TotallyBranched result;
	// recursively reduce rank, from highest rank to lowest
	// the boundary condition is the last function
	for (std::size_t iElement = 0; iElement < NumElements; iElement++)
	{
		for (std::size_t iOffDiagonalBranch = 0; iOffDiagonalBranch < NumOffDiagonalBranches; iOffDiagonalBranch++)
		{
			const TensorVector x_slice =
				x.slice(std::array<std::size_t, 3>{0, iOffDiagonalBranch, iElement}, std::array<std::size_t, 3>{Dim, 1, 1})
					.reshape(std::array<std::size_t, 1>{Dim});
			const TensorVector p_slice =
				p.slice(std::array<std::size_t, 3>{0, iOffDiagonalBranch, iElement}, std::array<std::size_t, 3>{Dim, 1, 1})
					.reshape(std::array<std::size_t, 1>{Dim});
			result.slice(std::array<std::size_t, 4>{0, 0, iOffDiagonalBranch, iElement}, std::array<std::size_t, 4>{Dim, NumElements, 1, 1}) =
				momentum_diagonal_evolve(x_slice, p_slice, dt, drc).reshape(std::array<std::size_t, 4>{Dim, NumElements, 1, 1});
		}
	}
	return result;
}

/// @brief To evolve the momentum under the off-diagonal forces
/// @param[in] x Position of classical degree of freedom
/// @param[in] p Momentum of classical degree of freedom
/// @param[in] dt The time interval of evolution
/// @param[in] Couple A vector of 0 or 1, 1 for evolving dimensions and 0 for non-evolving
/// @return All branches of momentum vectors
/// @details @f$ p_{i,j} = p_i + n f_{01}(x_i) dt @f$, where @f$ n = -1, 0, 1 @f$ and @f$ f_{01}=(E_0-E_1)d_{01} @f$
static OffDiagonalBranched momentum_offdiagonal_evolve(
	const DiagonalBranched& x,
	const DiagonalBranched& p,
	const double dt,
	const ClassicalVector<bool>& IsCouple)
{
	using OffDiagonalBranchingFromSameElement = Eigen::TensorFixedSize<double, Eigen::Sizes<Dim, NumOffDiagonalBranches>>;
	OffDiagonalBranched result;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES < NumPES; jPES++)
		{
			const std::size_t ElementIndex = iPES * NumPES + jPES;
			const TensorVector x_slice = x.chip(ElementIndex, 1);
			// get the off-diagonal force
			const OffDiagonalBranchingFromSameElement f_01_broadcast =
				(tensor_slice(adiabatic_force(ClassicalVector<double>::Map(x_slice.data())), 0, 1)
					* Eigen::TensorMap<const Eigen::Tensor<const bool, 1>>(IsCouple.data(), Dim).cast<double>())
					.reshape(std::array<std::size_t, 2>{Dim, 1})
					.broadcast(std::array<std::size_t, 2>{1, NumOffDiagonalBranches});
			// have the n factor broadcast
			const OffDiagonalBranchingFromSameElement n_broadcast = [](void) -> OffDiagonalBranchingFromSameElement
			{
				Eigen::TensorFixedSize<double, Eigen::Sizes<1, NumOffDiagonalBranches>> n;
				n.setValues({{-1, 0, 1}});
				return n.broadcast(std::array<std::size_t, 2>{Dim, 1});
			}();
			// cut the p slice, and broadcast it
			const OffDiagonalBranchingFromSameElement p_slice_brodcast =
				p.slice(std::array<std::size_t, 2>{0, ElementIndex}, std::array<std::size_t, 2>{Dim, 1})
					.broadcast(std::array<std::size_t, 2>{1, NumOffDiagonalBranches});
			// then assign to the chip of result
			result.chip(ElementIndex, 2) = p_slice_brodcast + dt * n_broadcast * f_01_broadcast;
		}
	}
	return result;
}

/// @brief To calculate sin and cos of @f$ dt(\omega(x_0)+2\omega(x_1)+\omega(x_2))/4 @f$,
/// where @f$ \omega(x) =(E_i(x)-E_j(x))/\hbar @f$
/// @param[in] x0 First position
/// @param[in] x1 Second position
/// @param[in] x2 Third position
/// @param[in] drc The direction of evolution
/// @param[in] SmallIndex The smaller index of density matrix
/// @param[in] LargeIndex The larger index of density matrix
/// @return sine and cosine of @f$ dt(\omega(x_0)+2\omega(x_1)+\omega(x_2))/4 @f$
static double calculate_omega0(
	const TensorVector& x0,
	const TensorVector& x1,
	const DiagonalBranched& x2,
	const Direction drc,
	const std::size_t SmallIndex,
	const std::size_t LargeIndex)
{
	assert(SmallIndex < LargeIndex && LargeIndex < NumPES);
	const TensorVector x2_slice = x2.chip(LargeIndex * NumPES + SmallIndex, 1);
	const QuantumVector<double> E0 = adiabatic_potential(ClassicalVector<double>::Map(x0.data()));
	const QuantumVector<double> E1 = adiabatic_potential(ClassicalVector<double>::Map(x1.data()));
	const QuantumVector<double> E2 = adiabatic_potential(ClassicalVector<double>::Map(x2_slice.data()));
	return -drc / 4.0 / hbar
		* (E0[SmallIndex] - E0[LargeIndex] + 2.0 * (E1[SmallIndex] - E1[LargeIndex]) + E2[SmallIndex] - E2[LargeIndex]);
}

/// @brief To predict the density matrix of the given point after evolving 1 time step back
/// @param[in] r Phase space coordinates of classical degree of freedom
/// @param[in] mass Mass of classical degree of freedom
/// @param[in] dt Time interval
/// @param[in] Kernels An array of kernels for prediction, whose size is NumElements
/// @return The density matrix at the given point after evolving
/// @details This function only deals with non-adiabatic case;
/// for adiabatic evolution, @sa evolve
static QuantumMatrix<std::complex<double>> non_adiabatic_evolve_predict(
	const ClassicalPhaseVector& r,
	const ClassicalVector<double>& mass,
	const double dt,
	const OptionalKernels& Kernels)
{
	using namespace std::literals::complex_literals;
	static constexpr Direction drc = Direction::Backward;
	static const Eigen::TensorFixedSize<std::size_t, Eigen::Sizes<NumElements>> IndicesAdd =
		[](void) -> Eigen::TensorFixedSize<std::size_t, Eigen::Sizes<NumElements>>
	{
		Eigen::TensorFixedSize<std::size_t, Eigen::Sizes<NumElements>> result;
		for (std::size_t iPES = 0; iPES < NumPES; iPES++)
		{
			for (std::size_t jPES = 0; jPES < NumPES; jPES++)
			{
				const std::size_t ElementIndex = iPES * NumPES + jPES;
				result(ElementIndex) = iPES + jPES;
			}
		}
		return result;
	}();
	static const Eigen::TensorFixedSize<std::ptrdiff_t, Eigen::Sizes<NumElements>> IndicesSub =
		[](void) -> Eigen::TensorFixedSize<std::ptrdiff_t, Eigen::Sizes<NumElements>>
	{
		Eigen::TensorFixedSize<std::ptrdiff_t, Eigen::Sizes<NumElements>> result;
		for (std::size_t iPES = 0; iPES < NumPES; iPES++)
		{
			for (std::size_t jPES = 0; jPES < NumPES; jPES++)
			{
				const std::size_t ElementIndex = iPES * NumPES + jPES;
				result(ElementIndex) = iPES - jPES;
			}
		}
		return result;
	}();
	const auto [x, p] = split_phase_coordinate(r);
	const ClassicalVector<bool> IsCouple = is_coupling(x, p, mass);
	QuantumMatrix<std::complex<double>> result = QuantumMatrix<std::complex<double>>::Zero();
	// index: {classical dimensions}
	const TensorVector x0 = Eigen::TensorMap<const Eigen::Tensor<const double, 1>>(x.data(), Dim);
	const TensorVector p0 = Eigen::TensorMap<const Eigen::Tensor<const double, 1>>(p.data(), Dim);
	if (IsCouple.any())
	{
		if constexpr (NumPES == 2)
		{
			// 2-level system
			const ClassicalVector<bool> IsCouple = is_coupling(x, p, mass);
			// x_i and p_{i-1} have same number of elements, and p_i is branching
			// 17 steps
			// index: {classical dimensions}
			const TensorVector x1 = position_evolve(x0, p0, mass, dt / 4.0, drc);
			// index: {classical dimensions, density matrix element it comes from}
			const DiagonalBranched p1 = momentum_diagonal_evolve(x1, p0, dt / 2.0, drc);
			const DiagonalBranched x2 = position_evolve(x1, p1, mass, dt / 4.0, drc);
			// index: {classical dimensions, offdiagonal branching, density matrix element it comes from}
			const OffDiagonalBranched p2 = momentum_offdiagonal_evolve(x2, p1, dt, IsCouple);
			const OffDiagonalBranched x3 = position_evolve(x2, p2, mass, dt / 4.0, drc);
			// index: {classical dimensions, density matrix element it goes to, offdiagonal branching, density matrix element it comes from}
			const TotallyBranched p3 = momentum_diagonal_evolve(x3, p2, dt / 2.0, drc);
			const TotallyBranched x4 = position_evolve(x3, p3, mass, dt / 4.0, drc);
			// auxiliary terms: evolved elements, phi, omega0, omega1
			const double omega0 = calculate_omega0(x0, x1, x2, drc, 0, 1);
			// omega1(n, gamma) = omega0(R2(., gamma), R3(., n, gamma), R4(., (1,0), n, gamma)) / (4hbar)
			const Eigen::TensorFixedSize<double, Eigen::Sizes<NumOffDiagonalBranches, NumElements>> omega1 =
				[&x2, &x3, &x4](void) -> Eigen::TensorFixedSize<double, Eigen::Sizes<NumOffDiagonalBranches, NumElements>>
			{
				Eigen::TensorFixedSize<double, Eigen::Sizes<NumOffDiagonalBranches, NumElements>> result;
				for (std::size_t iElement = 0; iElement < NumElements; iElement++)
				{
					for (std::size_t iOffDiagonalBranch = 0; iOffDiagonalBranch < NumOffDiagonalBranches; iOffDiagonalBranch++)
					{
						result(iOffDiagonalBranch, iElement) = calculate_omega0(
							x2.chip(iElement, 1),
							x3.chip(iElement, 2).chip(iOffDiagonalBranch, 1),
							x4.chip(iElement, 3).chip(iOffDiagonalBranch, 2),
							drc,
							0,
							1);
					}
				}
				return result;
			}();
			// phi(n, gamma) = p2(., n, gamma).dot(d01(R2(., gamma)))
			const Eigen::TensorFixedSize<double, Eigen::Sizes<NumOffDiagonalBranches, NumElements>> phi =
				[&x2, &p2, &IsCouple](void) -> Eigen::TensorFixedSize<double, Eigen::Sizes<NumOffDiagonalBranches, NumElements>>
			{
				Eigen::TensorFixedSize<double, Eigen::Sizes<NumOffDiagonalBranches, NumElements>> result;
				for (std::size_t iElement = 0; iElement < NumElements; iElement++)
				{
					// slice the position
					const TensorVector x2_slice = x2.chip(iElement, 1);
					// and get the non-adiabatic coupling
					const TensorVector d01 =
						tensor_slice(adiabatic_coupling(ClassicalVector<double>::Map(x2_slice.data())), 0, 1)
						* Eigen::TensorMap<const Eigen::Tensor<const bool, 1>>(IsCouple.data(), Dim).cast<double>();
					result.chip(iElement, 1) =
						(p2.chip(iElement, 2) * d01.reshape(std::array<std::size_t, 2>{Dim, 1}).broadcast(std::array<std::size_t, 2>{1, NumOffDiagonalBranches}))
							.sum(std::array<std::size_t, 1>{0});
				}
				return result;
			}();
			// rho(gamma', n, gamma) = rho^gamma_W(R4(., gamma', n, gamma), P3(., gamma', n, gamma), t)
			const Eigen::TensorFixedSize<std::complex<double>, Eigen::Sizes<NumElements, NumOffDiagonalBranches, NumElements>> rho_predict =
				[&x4, &p3, &Kernels](void) -> Eigen::TensorFixedSize<std::complex<double>, Eigen::Sizes<NumElements, NumOffDiagonalBranches, NumElements>>
			{
				Eigen::TensorFixedSize<std::complex<double>, Eigen::Sizes<NumElements, NumOffDiagonalBranches, NumElements>> result;
				result.setZero();
				for (std::size_t iPES = 0; iPES < NumPES; iPES++)
				{
					for (std::size_t jPES = 0; jPES < NumPES; jPES++)
					{
						const std::size_t ElementIndex = iPES * NumPES + jPES;
						const std::size_t SymmetricElementIndex = jPES * NumPES + iPES;
						// construct training set
						const Eigen::TensorFixedSize<double, Eigen::Sizes<PhaseDim, NumElements, NumOffDiagonalBranches>> r =
							[&x4, &p3, ElementIndex](void) -> Eigen::TensorFixedSize<double, Eigen::Sizes<PhaseDim, NumElements, NumOffDiagonalBranches>>
						{
							Eigen::TensorFixedSize<double, Eigen::Sizes<PhaseDim, NumElements, NumOffDiagonalBranches>> result;
							result.slice(std::array<std::size_t, 3>{0, 0, 0}, std::array<std::size_t, 3>{Dim, NumElements, NumOffDiagonalBranches}) = x4.chip(ElementIndex, 3);
							result.slice(std::array<std::size_t, 3>{Dim, 0, 0}, std::array<std::size_t, 3>{Dim, NumElements, NumOffDiagonalBranches}) = p3.chip(ElementIndex, 3);
							return result;
						}();
						// get prediction
						const Eigen::VectorXd rho = Kernels[ElementIndex].has_value()
							? predict_elements_with_variance_comparison(Kernels[ElementIndex].value(), PhasePoints::Map(r.data(), PhaseDim, NumElements * NumOffDiagonalBranches))
							: Eigen::VectorXd::Zero(NumElements * NumOffDiagonalBranches);
						if (iPES == jPES)
						{
							// add to real part of diagonal elements
							result.chip(ElementIndex, 2) +=
								Eigen::TensorMap<const Eigen::Tensor<const double, 2>>(rho.data(), NumElements, NumOffDiagonalBranches).cast<std::complex<double>>();
						}
						else if (iPES < jPES)
						{
							// add to real part of off-diagonal elements
							result.chip(ElementIndex, 2) +=
								Eigen::TensorMap<const Eigen::Tensor<const double, 2>>(rho.data(), NumElements, NumOffDiagonalBranches).cast<std::complex<double>>();
							result.chip(SymmetricElementIndex, 2) +=
								Eigen::TensorMap<const Eigen::Tensor<const double, 2>>(rho.data(), NumElements, NumOffDiagonalBranches).cast<std::complex<double>>();
						}
						else
						{
							// add to imaginary part of off-diagonal elements
							result.chip(ElementIndex, 2) +=
								1.0i * Eigen::TensorMap<const Eigen::Tensor<const double, 2>>(rho.data(), NumElements, NumOffDiagonalBranches).cast<std::complex<double>>();
							result.chip(SymmetricElementIndex, 2) -=
								1.0i * Eigen::TensorMap<const Eigen::Tensor<const double, 2>>(rho.data(), NumElements, NumOffDiagonalBranches).cast<std::complex<double>>();
						}
					}
				}
				return result;
			}();
			// auxiliary terms: c1, c2, c3, c4
			// C1(gamma', n) = ((1-delta(n, 0))delta(alpha', beta') + cos(phi(0, gamma')dt - (alpha' + beta' + n)pi/2)exp(i(alpha' - beta')omega0 * dt / 2)) / (4 - 2delta(n, 0))
			const Eigen::TensorFixedSize<std::complex<double>, Eigen::Sizes<NumElements, NumOffDiagonalBranches>> C1 =
				[&phi, omega0, dt](void) -> Eigen::TensorFixedSize<std::complex<double>, Eigen::Sizes<NumElements, NumOffDiagonalBranches>>
			{
				Eigen::TensorFixedSize<std::complex<double>, Eigen::Sizes<NumElements, NumOffDiagonalBranches>> result;
				result.setZero();
				const Eigen::TensorFixedSize<double, Eigen::Sizes<NumElements, NumOffDiagonalBranches>> phi0_broadcast =
					phi.slice(std::array<std::size_t, 2>{1, 0}, std::array<std::size_t, 2>{1, NumElements})
						.reshape(std::array<std::size_t, 2>{NumElements, 1})
						.broadcast(std::array<std::size_t, 2>{1, NumOffDiagonalBranches});
				const Eigen::TensorFixedSize<double, Eigen::Sizes<NumElements, NumOffDiagonalBranches>> indices_add_broadcast =
					IndicesAdd.cast<double>().reshape(std::array<std::size_t, 2>{NumElements, 1}).broadcast(std::array<std::size_t, 2>{1, NumOffDiagonalBranches});
				const Eigen::TensorFixedSize<double, Eigen::Sizes<NumElements, NumOffDiagonalBranches>> offdiag_indices_broadcast =
					Eigen::TensorMap<const Eigen::Tensor<const std::ptrdiff_t, 2>>(OffDiagonalBranches.data(), 1, NumOffDiagonalBranches)
						.cast<double>()
						.broadcast(std::array<std::size_t, 2>{NumElements, 1});
				const Eigen::TensorFixedSize<double, Eigen::Sizes<NumElements, NumOffDiagonalBranches>> within_cosine =
					phi0_broadcast * dt - M_PI / 2.0 * (indices_add_broadcast + offdiag_indices_broadcast);
				const Eigen::TensorFixedSize<double, Eigen::Sizes<NumElements, NumOffDiagonalBranches>> cosine =
					within_cosine.unaryExpr([](double d) -> double
						{
							return std::cos(d);
						});
				const Eigen::TensorFixedSize<std::complex<double>, Eigen::Sizes<NumElements, NumOffDiagonalBranches>> indices_sub_broadcast =
					IndicesSub.cast<std::complex<double>>().reshape(std::array<std::size_t, 2>{NumElements, 1}).broadcast(std::array<std::size_t, 2>{1, NumOffDiagonalBranches});
				result += cosine.cast<std::complex<double>>() * (omega0 * dt / 2.0 * 1.0i * indices_sub_broadcast).exp();
				for (std::size_t iPES = 0; iPES < NumPES; iPES++)
				{
					for (const std::ptrdiff_t n : OffDiagonalBranches)
					{
						const std::size_t OffDiagonalIndex = n + 1;
						if (n != 0)
						{
							result(iPES * NumPES + iPES, OffDiagonalIndex) += 1;
						}
					}
				}
				for (const std::ptrdiff_t n : OffDiagonalBranches)
				{
					const std::size_t OffDiagonalIndex = n + 1;
					result.chip(OffDiagonalIndex, 1) /= result.chip(OffDiagonalIndex, 1).constant(4 - 2 * static_cast<int>(n == 0));
				}
				return result;
			}();
			// c2(gamma', n, gamma) = (1-delta(n, 0))delta(alpha, beta) + cos(phi(n, gamma')dt + (alpha + beta + n)pi/2)exp(i(alpha - beta)omega1(n, gamma') * dt / 2)
			const Eigen::TensorFixedSize<std::complex<double>, Eigen::Sizes<NumElements, NumOffDiagonalBranches, NumElements>> C2 =
				[&phi, &omega1, dt](void) -> Eigen::TensorFixedSize<std::complex<double>, Eigen::Sizes<NumElements, NumOffDiagonalBranches, NumElements>>
			{
				Eigen::TensorFixedSize<std::complex<double>, Eigen::Sizes<NumElements, NumOffDiagonalBranches, NumElements>> result;
				const Eigen::TensorFixedSize<double, Eigen::Sizes<NumElements, NumOffDiagonalBranches, NumElements>> phi_broadcast =
					phi.shuffle(std::array<std::size_t, 2>{1, 0})
						.reshape(std::array<std::size_t, 3>{NumElements, NumOffDiagonalBranches, 1})
						.broadcast(std::array<std::size_t, 3>{1, 1, NumElements});
				const Eigen::TensorFixedSize<double, Eigen::Sizes<NumElements, NumOffDiagonalBranches, NumElements>> indices_add_broadcast =
					IndicesAdd.cast<double>()
						.reshape(std::array<std::size_t, 3>{1, 1, NumElements})
						.broadcast(std::array<std::size_t, 3>{NumElements, NumOffDiagonalBranches, 1});
				const Eigen::TensorFixedSize<double, Eigen::Sizes<NumElements, NumOffDiagonalBranches, NumElements>> offdiag_indices_broadcast =
					Eigen::TensorMap<const Eigen::Tensor<const std::ptrdiff_t, 3>>(OffDiagonalBranches.data(), 1, NumOffDiagonalBranches, 1)
						.cast<double>()
						.broadcast(std::array<std::size_t, 3>{NumElements, 1, NumElements});
				const Eigen::TensorFixedSize<double, Eigen::Sizes<NumElements, NumOffDiagonalBranches, NumElements>> within_cosine =
					phi_broadcast * dt + M_PI / 2.0 * (indices_add_broadcast + offdiag_indices_broadcast);
				const Eigen::TensorFixedSize<double, Eigen::Sizes<NumElements, NumOffDiagonalBranches, NumElements>> cosine =
					within_cosine.unaryExpr([](double d) -> double
						{
							return std::cos(d);
						});
				const Eigen::TensorFixedSize<std::complex<double>, Eigen::Sizes<NumElements, NumOffDiagonalBranches, NumElements>> indices_sub_broadcast =
					IndicesSub.cast<std::complex<double>>()
						.reshape(std::array<std::size_t, 3>{1, 1, NumElements})
						.broadcast(std::array<std::size_t, 3>{NumElements, NumOffDiagonalBranches, 1});
				const Eigen::TensorFixedSize<std::complex<double>, Eigen::Sizes<NumElements, NumOffDiagonalBranches, NumElements>> omega1_broadcast =
					omega1.shuffle(std::array<std::size_t, 2>{1, 0})
						.cast<std::complex<double>>()
						.reshape(std::array<std::size_t, 3>{NumElements, NumOffDiagonalBranches, 1})
						.broadcast(std::array<std::size_t, 3>{1, 1, NumElements});
				result += cosine.cast<std::complex<double>>() * (dt / 2.0 * 1.0i * indices_sub_broadcast * omega1_broadcast).exp();
				for (std::size_t iPES = 0; iPES < NumPES; iPES++)
				{
					for (const std::ptrdiff_t n : OffDiagonalBranches)
					{
						const std::size_t OffDiagonalIndex = n + 1;
						if (n != 0)
						{
							result.slice(std::array<std::size_t, 3>{0, OffDiagonalIndex, iPES * NumPES + iPES}, std::array<std::size_t, 3>{NumElements, 1, 1}) +=
								Eigen::TensorFixedSize<std::complex<double>, Eigen::Sizes<NumElements, 1, 1>>().constant(1);
						}
					}
				}
				return result;
			}();
			// c3(gamma') = (beta' - alpha') / 2 * exp(i(alpha' - beta')omega0 * dt / 2)
			const Eigen::TensorFixedSize<std::complex<double>, Eigen::Sizes<NumElements>> C3 =
				std::complex<double>(-0.5) * IndicesSub.cast<std::complex<double>>() * (dt * omega0 / 2.0 * 1.0i * IndicesSub.cast<std::complex<double>>()).exp();
			// c4(gamma',\gamma) = (beta - alpha) * exp(i(alpha - beta)omega1(0, gamma') * dt / 2)
			const Eigen::TensorFixedSize<std::complex<double>, Eigen::Sizes<NumElements, NumElements>> C4 =
				[&omega1, dt](void) -> Eigen::TensorFixedSize<std::complex<double>, Eigen::Sizes<NumElements, NumElements>>
			{
				const Eigen::TensorFixedSize<std::complex<double>, Eigen::Sizes<NumElements, NumElements>> indices_sub_broadcast =
					IndicesSub.cast<std::complex<double>>()
						.reshape(std::array<std::size_t, 2>{1, NumElements})
						.broadcast(std::array<std::size_t, 2>{NumElements, 1});
				const Eigen::TensorFixedSize<std::complex<double>, Eigen::Sizes<NumElements, NumElements>> omega1_broadcast =
					omega1.chip(1, 0)
						.cast<std::complex<double>>()
						.reshape(std::array<std::size_t, 2>{NumElements, 1})
						.broadcast(std::array<std::size_t, 2>{1, NumElements});
				return -indices_sub_broadcast * (dt / 2.0 * 1.0i * indices_sub_broadcast * omega1_broadcast).exp();
			}();
			// finally, a linearized density matrix
			// rho(gamma')=sum_n{C1(gamma', n)sum_gamma{C2(gamma', n, gamma)rho(gamma', n, gamma)}} + C3(gamma')sum_{gamma}{C4(gamma)rho(gamma',0,gamma)}
			const Eigen::TensorFixedSize<std::complex<double>, Eigen::Sizes<NumElements>> rho = (C1.reshape(std::array<std::size_t, 3>{NumElements, NumOffDiagonalBranches, 1}).broadcast(std::array<std::size_t, 3>{1, 1, NumElements}) * C2 * rho_predict).sum(std::array<std::size_t, 2>{1, 2})
				+ (C3.reshape(std::array<std::size_t, 2>{NumElements, 1}).broadcast(std::array<std::size_t, 2>{1, NumElements}) * C4 * rho_predict.chip(1, 1)).sum(std::array<std::size_t, 1>{1});
			// rho is row-major, but the result should be column major, so a transpose is needed
			result.transpose() = QuantumMatrix<std::complex<double>>::Map(rho.data());
		}
		else
		{
			assert(!"NO INSTANTATION OF MORE THAN TWO LEVEL SYSTEM NOW\n");
		}
	}
	else
	{
		// adiabatic dynamics
		// 7 steps
		// index: {classical dimensions}
		const TensorVector x1 = position_evolve(x0, p0, mass, dt / 2.0, drc);
		// index: {classical dimensions, density matrix element it comes from}
		const DiagonalBranched p1 = momentum_diagonal_evolve(x1, p0, dt, drc);
		const DiagonalBranched x2 = position_evolve(x1, p1, mass, dt / 2.0, drc);
		// construct training set
		const PhasePoints PhaseCoordinatesUsedForPrediction = [&x2, &p1](void) -> PhasePoints
		{
			PhasePoints result(PhaseDim, NumElements);
			Eigen::TensorMap<Eigen::Tensor<double, 2>> result_tensor_map(result.data(), PhaseDim, NumElements);
			result_tensor_map.slice(std::array<std::size_t, 2>{0, 0}, x2.dimensions()) = x2;
			result_tensor_map.slice(std::array<std::size_t, 2>{Dim, 0}, p1.dimensions()) = p1;
			return result;
		}();
		for (std::size_t iPES = 0; iPES < NumPES; iPES++)
		{
			for (std::size_t jPES = 0; jPES <= iPES; jPES++)
			{
				// loop over the lower triangular only
				const std::size_t ElementIndex = iPES * NumPES + jPES;
				if (Kernels[ElementIndex].has_value())
				{
					// the real part
					result(iPES, jPES) += predict_elements_with_variance_comparison(Kernels[ElementIndex].value(), PhaseCoordinatesUsedForPrediction.col(ElementIndex)).value();
				}
				if (const std::size_t SymmetricElementIndex = jPES * NumPES + iPES; iPES != jPES && Kernels[SymmetricElementIndex].has_value())
				{
					// the imaginary part for off-diagonal elemenets
					result(iPES, jPES) += 1.0i * predict_elements_with_variance_comparison(Kernels[SymmetricElementIndex].value(), PhaseCoordinatesUsedForPrediction.col(ElementIndex)).value();
				}
				if (iPES != jPES)
				{
					// change phase for off-diagonal part
					const double omega0 = calculate_omega0(x0, x1, x2, drc, jPES, iPES);
					result(iPES, jPES) *= std::exp(1.0i * omega0 * dt);
				}
			}
		}
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
	static constexpr Direction drc = Direction::Forward;
	auto adiabatic_evolve =
		[](
			ClassicalVector<double>& x,
			ClassicalVector<double>& p,
			const ClassicalVector<double>& mass,
			const double dt,
			const Direction drc,
			const std::size_t iPES,
			const std::size_t jPES) -> void
	{
		auto position_evolve = [&x, &p = std::as_const(p), &mass, dt, drc](void) -> void
		{
			x += drc * dt / 2.0 * (p.array() / mass.array()).matrix();
		};
		auto momentum_diagonal_nonbranch_evolve = [&x = std::as_const(x), &p, &mass, dt, drc, iPES, jPES](void) -> void
		{
			const DiagonalForces f = adiabatic_diagonal_forces(x);
			p += drc * dt / 2.0 * (f.col(iPES) + f.col(jPES));
		};
		position_evolve();
		momentum_diagonal_nonbranch_evolve();
		position_evolve();
	};
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
						[&mass, dt, &Kernels, &adiabatic_evolve, iPES, jPES, ElementIndex](PhaseSpacePoint& psp) -> void
						{
							auto& [r, rho] = psp;
							auto [x, p] = split_phase_coordinate(r);
							if (is_coupling(x, p, mass).any())
							{
								// with coupling, non-adiabatic case
								// calculate its adiabatically evolved phase space coordinate with 2 half steps
								adiabatic_evolve(x, p, mass, dt / 2, drc, iPES, jPES);
								adiabatic_evolve(x, p, mass, dt / 2, drc, iPES, jPES);
							}
							else
							{
								// without coupling, adiabatic case
								// calculate its adiabatically evolved phase space coordinate
								adiabatic_evolve(x, p, mass, dt, drc, iPES, jPES);
							}
							// and use back-propagation to calculate the exact density there
							r << x, p;
							rho = non_adiabatic_evolve_predict(r, mass, dt, Kernels);
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
