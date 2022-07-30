/// @file evolve.cpp
/// @brief Implementation of evolve.h

#include "stdafx.h"

#include "evolve.h"

#include "mc.h"
#include "pes.h"
#include "predict.h"

static constexpr std::size_t NumOffDiagonalBranches = 3; ///< The number of branches resulted by one off-diagonal force, including forward, static and backward
static constexpr std::array<std::ptrdiff_t, NumOffDiagonalBranches> OffDiagonalBranches = {-1, 0, 1}; /// The branches

/// The matrix that saves diagonal forces: i-th column is the force on i-th surface
using DiagonalForces = Eigen::Matrix<double, Dim, NumPES>;
/// The tensor form of @p ClassicalVector<double>
using TensorVector = xt::xtensor_fixed<double, xt::xshape<Dim>>;
/// The tensor form for @f$ P_1 @f$ and @f$ R_2 @f$
using DiagonalBranched = xt::xtensor_fixed<double, xt::xshape<NumElements, Dim>>;
/// The tensor form for @f$ P_2 @f$ and @f$ R_3 @f$
using OffDiagonalBranched = xt::xtensor_fixed<double, xt::xshape<NumOffDiagonalBranches, NumElements, Dim>>;
/// The tensor form for @f$ P_3 @f$ and @f$ R_4 @f$
using TotallyBranched = xt::xtensor_fixed<double, xt::xshape<NumElements, NumOffDiagonalBranches, NumElements, Dim>>;

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
/// @return A mapping vector, that is tensor(0, row, col) to tensor( @p Dim - 1, row, col)
static inline auto tensor_slice(const Tensor3d& tensor, const std::size_t row, const std::size_t col)
{
	const auto view = xt::view(tensor, xt::all(), row, col);
	return Eigen::Map<const ClassicalVector<double>, Eigen::Unaligned, Eigen::InnerStride<NumElements>>(view.data() + view.data_offset());
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
		const auto nac_01 = tensor_slice(adiabatic_coupling(x), 0, 1), f_01 = tensor_slice(adiabatic_force(x), 0, 1);
			f_01 = Eigen::Map<const ClassicalVector<double>, Eigen::Unaligned, Eigen::InnerStride<NumElements>>(f_01_view.data() + f_01_view.data_offset());
		return nac_01.array() * p.array() / mass.array() > CouplingCriterion || f_01.array() > CouplingCriterion;
	}
	else
	{
		const Tensor3d& NAC = adiabatic_coupling(x), Force = adiabatic_force(x);
		ClassicalVector<bool> result = ClassicalVector<bool>::Zero();
		for (std::size_t iPES = 1; iPES < NumPES; iPES++)
		{
			for (std::size_t jPES = 0; jPES < iPES; jPES++)
			{
				const auto nac_ij = tensor_slice(NAC, iPES, jPES), f_ij = tensor_slice(Force, iPES, jPES);
				f = Eigen::Map<const ClassicalVector<double>, Eigen::Unaligned, Eigen::InnerStride<NumElements>>(f_view.data() + f_view.data_offset());
				result = result.array() || nac_ij.array() * p.array() / mass.array() > CouplingCriterion || f_ij.array() > CouplingCriterion;
			}
		}
		return result;
	}
#endif
#undef ADIA
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
				});
		if (IsElementCoupling)
		{
			return true;
		}
	}
	return false;
}

/// @brief To evolve all the positions
/// @tparam xIndices Indices of position
/// @tparam pIndices Indices of momentum
/// @param[in] x Position of classical degree of freedom
/// @param[in] p Momentum of classical degree of freedom
/// @param[in] mass Mass of classical degree of freedom
/// @param[in] dt The time interval of evolution
/// @param[in] drc The direction of evolution
/// @return All branches of momentum vectors
/// @details @f$ x_{i,j} = x_i - \frac{p_{i,j}}{m} dt @f$
template <std::size_t ... xIndices, std::size_t ... pIndices>
static inline xt::xtensor_fixed<double, xt::xshape<pIndices ...>> position_evolve(
	const xt::xtensor_fixed<double, xt::xshape<xIndices ...>>& x,
	const xt::xtensor_fixed<double, xt::xshape<pIndices ...>>& p,
	const ClassicalVector<double>& mass,
	const double dt,
	const Direction drc)
{
	return x + drc * dt * p / xt::adapt(mass.data(), TensorVector::shape_type{});
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
		result.col(iPES) = tensor_slice(force, iPES, iPES);
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
			const TensorVector sum_f = xt::adapt(f.col(iPES).data(), TensorVector::shape_type{}) + xt::adapt(f.col(jPES).data(), TensorVector::shape_type{});
			xt::view(result, iPES * NumPES + jPES, xt::all()) = p + drc * dt / 2.0 * sum_f;
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
	for (std::size_t iOffDiagonalBranch = 0; iOffDiagonalBranch < NumOffDiagonalBranches; iOffDiagonalBranch++)
	{
		for (std::size_t iElement = 0; iElement < NumElements; iElement++)
		{
			xt::view(result, xt::all(), iOffDiagonalBranch, iElement, xt::all()) =
				momentum_diagonal_evolve(
					TensorVector(xt::view(x, iOffDiagonalBranch, iElement, xt::all())),
					TensorVector(xt::view(p, iOffDiagonalBranch, iElement, xt::all())),
					dt,
					drc);
		}
	}
	return result;
}

/// @brief To extract all the 01 vector from a tensor
/// @param[in] x Positions of each element of density matrix
/// @param[in] IsCouple Whether there is coupling for each element or not
/// @param[in] f The function, with classical position as input and the full tensor as output
/// @return All the 01 vector from the return of @p f using position of each element from @p x
static DiagonalBranched construct_all_01_elements(
	const DiagonalBranched& x,
	const ClassicalVector<bool>& IsCouple,
	const std::function<Tensor3d(const ClassicalVector<double>&)>& f)
{
	DiagonalBranched result;
	for (std::size_t iElement = 0; iElement < NumElements; iElement++)
	{
		const auto& xview = xt::view(x, iElement, xt::all());
		xt::view(result, iElement, xt::all()) = xt::view(f(ClassicalVector<double>::Map(xview.data() + xview.data_offset())), xt::all(), 0, 1);
	}
	result *= xt::adapt(IsCouple.data(), {Dim});
	return result;
}

/// @brief To evolve the momentum under the off-diagonal forces
/// @param[in] x Position of classical degree of freedom
/// @param[in] p Momentum of classical degree of freedom
/// @param[in] dt The time interval of evolution
/// @param[in] Couple A vector of 0 or 1, 1 for evolving dimensions and 0 for non-evolving
/// @return All branches of momentum vectors
/// @details @f$ p_{i,j} = p_i + n f_{01}(x_i) dt @f$, where @f$ n = -1, 0, 1 @f$ and @f$ f_{01}=(E_0-E_1)d_{01} @f$
static inline OffDiagonalBranched momentum_offdiagonal_evolve(
	const DiagonalBranched& x,
	const DiagonalBranched& p,
	const double dt,
	const ClassicalVector<bool>& IsCouple)
{
	OffDiagonalBranched result;
	const DiagonalBranched f01 = construct_all_01_elements(x, IsCouple, adiabatic_force);
	const auto n_broadcast = xt::view(xt::adapt(OffDiagonalBranches, xt::xshape<NumOffDiagonalBranches>{}), xt::all(), xt::newaxis(), xt::newaxis());
	return p + dt * n_broadcast * f01;
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
	const TensorVector x2_slice = xt::view(x2, LargeIndex * NumPES + SmallIndex, xt::all());
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
	static const xt::xtensor_fixed<double, xt::xshape<NumElements>> IndicesAdd = xt::reshape_view(xt::view(xt::arange<double>(NumPES), xt::all(), xt::newaxis()) + xt::arange<double>(NumPES), {NumElements});
	static const xt::xtensor_fixed<double, xt::xshape<NumElements>> IndicesSub = xt::reshape_view(xt::view(xt::arange<double>(NumPES), xt::all(), xt::newaxis()) - xt::arange<double>(NumPES), {NumElements});
	const auto& [x, p] = split_phase_coordinate(r);
	const ClassicalVector<bool> IsCouple = is_coupling(x, p, mass);
	QuantumMatrix<std::complex<double>> result = QuantumMatrix<std::complex<double>>::Zero();
	// index: {classical dimensions}
	const TensorVector x0 = xt::adapt(x.data(), TensorVector::shape_type{});
	const TensorVector p0 = xt::adapt(p.data(), TensorVector::shape_type{});
	if (IsCouple.any())
	{
		if constexpr (NumPES == 2)
		{
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
			// omega1(n, gamma) = omega0(R2(gamma, 0), R3(n, gamma, .), R4((1,0), n, gamma, .))
			const xt::xtensor_fixed<double, xt::xshape<NumOffDiagonalBranches, NumElements>> omega1 =
				[&x2, &x3, &x4](void) -> xt::xtensor_fixed<double, xt::xshape<NumOffDiagonalBranches, NumElements>>
			{
				xt::xtensor_fixed<double, xt::xshape<NumOffDiagonalBranches, NumElements>> result;
				for (std::size_t iOffDiagonalBranch = 0; iOffDiagonalBranch < NumOffDiagonalBranches; iOffDiagonalBranch++)
				{
					for (std::size_t iElement = 0; iElement < NumElements; iElement++)
					{
						result(iOffDiagonalBranch, iElement) = calculate_omega0(
							xt::view(x2, iElement, xt::all()),
							xt::view(x3, iOffDiagonalBranch, iElement, xt::all()),
							xt::view(x4, xt::all(), iOffDiagonalBranch, iElement, xt::all()),
							drc,
							0,
							1);
					}
				}
				return result;
			}();
			// phi(n, gamma) = p2(n, gamma, .).dot(d01(R2(gamma, .)))
			const xt::xtensor_fixed<double, xt::xshape<NumOffDiagonalBranches, NumElements>> phi =
				[&x2, &p2, &IsCouple](void) -> xt::xtensor_fixed<double, xt::xshape<NumOffDiagonalBranches, NumElements>>
			{
				const DiagonalBranched nac_01 = construct_all_01_elements(x2, IsCouple, adiabatic_coupling);
				return xt::sum(p2 * nac_01, {2});
			}();
			// rho(gamma', n, gamma) = rho^gamma_W(R4(gamma', n, gamma, .), P3(gamma', n, gamma, .), t)
			const xt::xtensor_fixed<std::complex<double>, xt::xshape<NumElements, NumOffDiagonalBranches, NumElements>> rho_predict =
				[&x4, &p3, &Kernels](void) -> xt::xtensor_fixed<std::complex<double>, xt::xshape<NumElements, NumOffDiagonalBranches, NumElements>>
			{
				xt::xtensor_fixed<std::complex<double>, xt::xshape<NumElements, NumOffDiagonalBranches, NumElements>> result;
				for (std::size_t iPES = 0; iPES < NumPES; iPES++)
				{
					for (std::size_t jPES = 0; jPES < NumPES; jPES++)
					{
						const std::size_t ElementIndex = iPES * NumPES + jPES;
						// construct training set
						const PhasePoints r = [&x4, &p3, ElementIndex](void) -> PhasePoints
						{
							static constexpr std::size_t NumElementsForPrediction = NumElements * NumOffDiagonalBranches;
							static constexpr std::size_t GapSize = NumElements * Dim;
							Eigen::Matrix<double, PhaseDim, NumElementsForPrediction> result;
							for (std::size_t iDim = 0; iDim < Dim; iDim++)
							{
								const auto xview = xt::view(x4, xt::all(), xt::all(), ElementIndex, iDim), pview = xt::view(p3, xt::all(), xt::all(), ElementIndex, iDim);
								result.block<1, NumElementsForPrediction>(iDim, 0) = Eigen::Map<const Eigen::Matrix<double, 1, NumElementsForPrediction>, Eigen::Unaligned, Eigen::InnerStride<GapSize>>(xview.data() + xview.data_offset());
								result.block<1, NumElementsForPrediction>(iDim + Dim, 0) = Eigen::Map<const Eigen::Matrix<double, 1, NumElementsForPrediction>, Eigen::Unaligned, Eigen::InnerStride<GapSize>>(pview.data() + pview.data_offset());
							}
							return result;
						}();
						// get prediction
						if (iPES == jPES)
						{
							xt::view(result, xt::all(), xt::all(), ElementIndex) = xt::adapt(Kernel(r, dynamic_cast<const Kernel&>(*Kernels[ElementIndex].get()), false).get_prediction_compared_with_variance().data(), {NumElements, NumOffDiagonalBranches});
						}
						else if (iPES < jPES)
						{
							// using the corresponding lower-diagonal predictor
							xt::view(result, xt::all(), xt::all(), ElementIndex) = xt::conj(xt::adapt(ComplexKernel(r, dynamic_cast<const ComplexKernel&>(*Kernels[jPES * NumPES + iPES].get()), false).get_prediction_compared_with_variance().data(), {NumElements, NumOffDiagonalBranches}));
						}
						else
						{
							// add to imaginary part of off-diagonal elements
							xt::view(result, xt::all(), xt::all(), ElementIndex) = xt::adapt(ComplexKernel(r, dynamic_cast<const ComplexKernel&>(*Kernels[ElementIndex].get()), false).get_prediction_compared_with_variance().data(), {NumElements, NumOffDiagonalBranches});
						}
					}
				}
				return result;
			}();
			// auxiliary terms: c1, c2, c3, c4
			// C1(gamma', n) = ((1-delta(n, 0))delta(alpha', beta') + cos(phi(0, gamma')dt - (alpha' + beta' + n)pi/2)exp(i(alpha' - beta')omega0 * dt / 2)) / (4 - 2delta(n, 0))
			const xt::xtensor_fixed<std::complex<double>, xt::xshape<NumElements, NumOffDiagonalBranches>> C1 =
				[&phi, omega0, dt](void) -> xt::xtensor_fixed<std::complex<double>, xt::xshape<NumElements, NumOffDiagonalBranches>>
			{
				xt::xtensor_fixed<std::complex<double>, xt::xshape<NumElements, NumOffDiagonalBranches>> result = xt::zeros<decltype(result)::value_type>(decltype(result)::shape_type{});
				result += xt::cos(xt::view(phi, 1, xt::all(), xt::newaxis()) * dt - M_PI / 2.0 * (xt::view(IndicesAdd, xt::all(), xt::newaxis()) + xt::adapt(OffDiagonalBranches, {NumOffDiagonalBranches})))
					* xt::exp(1.0i * xt::view(IndicesSub, xt::all(), xt::newaxis()) / 2.0 * omega0 * dt);
				for (std::size_t iOffDiagonalBranch = 0; iOffDiagonalBranch < NumOffDiagonalBranches; iOffDiagonalBranch++)
				{
					if (OffDiagonalBranches[iOffDiagonalBranch] != 0)
					{
						xt::view(result, xt::all(), iOffDiagonalBranch) += xt::reshape_view(xt::eye<std::complex<double>>(NumPES), {NumElements});
					}
				}
				for (std::size_t iOffDiagonalBranch = 0; iOffDiagonalBranch < NumOffDiagonalBranches; iOffDiagonalBranch++)
				{
					xt::view(result, xt::all(), iOffDiagonalBranch) /= 4.0 - 2.0 * static_cast<double>(OffDiagonalBranches[iOffDiagonalBranch] == 0);
				}
				return result;
			}();
			// c2(gamma', n, gamma) = (1-delta(n, 0))delta(alpha, beta) + cos(phi(n, gamma')dt + (alpha + beta + n)pi/2)exp(i(alpha - beta)omega1(n, gamma') * dt / 2)
			const xt::xtensor_fixed<std::complex<double>, xt::xshape<NumElements, NumOffDiagonalBranches, NumElements>> C2 =
				[&phi, &omega1, dt](void) -> xt::xtensor_fixed<std::complex<double>, xt::xshape<NumElements, NumOffDiagonalBranches, NumElements>>
			{
				xt::xtensor_fixed<std::complex<double>, xt::xshape<NumElements, NumOffDiagonalBranches, NumElements>> result = xt::zeros<decltype(result)::value_type>(decltype(result)::shape_type{});
				result += xt::cos(xt::view(xt::transpose(phi), xt::all(), xt::all(), xt::newaxis()) * dt + M_PI / 2.0 * (IndicesAdd + xt::view(xt::adapt(OffDiagonalBranches, {NumOffDiagonalBranches}), xt::all(), xt::newaxis())))
					* xt::exp(1.0i * IndicesSub / 2.0 * xt::view(xt::transpose(omega1), xt::all(), xt::all(), xt::newaxis()) * dt);
				for (std::size_t iOffDiagonalBranch = 0; iOffDiagonalBranch < NumOffDiagonalBranches; iOffDiagonalBranch++)
				{
					if (OffDiagonalBranches[iOffDiagonalBranch] != 0)
					{
						xt::view(result, xt::all(), iOffDiagonalBranch, xt::all()) += xt::reshape_view(xt::eye<std::complex<double>>(NumPES), {NumElements});
					}
				}
				return result;
			}();
			// c3(gamma') = (beta' - alpha') / 2 * exp(i(alpha' - beta')omega0 * dt / 2)
			const xt::xtensor_fixed<std::complex<double>, xt::xshape<NumElements>> C3 = -IndicesSub / 2.0 * xt::exp(1.0i * IndicesSub / 2.0 * omega0 * dt);
			// c4(gamma',\gamma) = (beta - alpha) * exp(i(alpha - beta)omega1(0, gamma') * dt / 2)
			const xt::xtensor_fixed<std::complex<double>, xt::xshape<NumElements, NumElements>> C4 = -IndicesSub * xt::exp(1.0i * IndicesSub / 2.0 * xt::view(omega1, 1, xt::all(), xt::newaxis()) * dt);
			// finally, a linearized density matrix
			// rho(gamma')=sum_n{C1(gamma', n)sum_gamma{C2(gamma', n, gamma)rho(gamma', n, gamma)}} + C3(gamma')sum_{gamma}{C4(gamma)rho(gamma',0,gamma)}
			const xt::xtensor_fixed<std::complex<double>, xt::xshape<NumElements>> rho = xt::sum(xt::view(C1, xt::all(), xt::all(), xt::newaxis()) * C2 * rho_predict, {1, 2})
				+ xt::sum(xt::view(C3, xt::all(), xt::newaxis()) * C4 * xt::view(rho_predict, xt::all(), 1, xt::all()), {1});
			result = QuantumMatrix<std::complex<double>>::Map(rho.data());
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
			Eigen::Matrix<double, PhaseDim, NumElements> result;
			result.block<Dim, NumElements>(0, 0) = Eigen::Matrix<double, Dim, NumElements>::Map(x2.data());
			result.block<Dim, NumElements>(Dim, 0) = Eigen::Matrix<double, Dim, NumElements>::Map(p1.data());
			return result;
		}();
		for (std::size_t iPES = 0; iPES < NumPES; iPES++)
		{
			for (std::size_t jPES = 0; jPES <= iPES; jPES++)
			{
				// loop over the lower triangular only
				const std::size_t ElementIndex = iPES * NumPES + jPES;
				if (iPES == jPES)
				{
					// diagonal elements
					result(iPES, jPES) = Kernel(PhaseCoordinatesUsedForPrediction.col(ElementIndex), dynamic_cast<const Kernel&>(*Kernels[ElementIndex].get()), false).get_prediction_compared_with_variance().value();
				}
				else
				{
					// off-diagonal elements
					result(iPES, jPES) = std::exp(1.0i * calculate_omega0(x0, x1, x2, drc, jPES, iPES) * dt)
						* ComplexKernel(PhaseCoordinatesUsedForPrediction.col(ElementIndex), dynamic_cast<const ComplexKernel&>(*Kernels[ElementIndex].get()), false).get_prediction_compared_with_variance().value();
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
		[](ClassicalVector<double>& x,
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
			else
			{
				// for strict lower triangular elements, copy the upper part
				density[iPES * NumPES + jPES] = density[jPES * NumPES + iPES];
			}
		}
	}
}
