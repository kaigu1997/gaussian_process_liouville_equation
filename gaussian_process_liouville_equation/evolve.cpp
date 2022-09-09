/// @file evolve.cpp
/// @brief Implementation of evolve.h

#include "stdafx.h"

#include "evolve.h"

#include "mc.h"
#include "pes.h"
#include "predict.h"

/// @brief The number of branches resulted by one off-diagonal force, including forward, static and backward
static constexpr std::size_t NumOffDiagonalBranches = 3;
/// @brief The branches
static constexpr std::array<std::ptrdiff_t, NumOffDiagonalBranches> OffDiagonalBranches = {-1, 0, 1};
/// @brief Index where the value of the branch is 0
static constexpr std::size_t OffDiagonalZeroBranch = 1;

/// @brief The matrix that saves diagonal forces: i-th column is the force on i-th surface
using DiagonalForces = Eigen::Matrix<double, Dim, NumPES>;
/// @brief The tensor form of @p ClassicalVector<double>
using TensorVector = xt::xtensor_fixed<double, xt::xshape<Dim>>;
/// @brief The tensor form for @f$ P_2 @f$ and @f$ R_3 @f$
using OffDiagonalBranched = xt::xtensor_fixed<double, xt::xshape<NumOffDiagonalBranches, Dim>>;
/// @brief The tensor form for @f$ P_3 @f$ and @f$ R_4 @f$
using TotallyBranched = xt::xtensor_fixed<double, xt::xshape<NumTriangularElements, NumOffDiagonalBranches, Dim>>;

/// The direction of dynamics evolving
enum Direction
{
	/// @brief Going back, for density prediction
	Backward = -1,
	/// @brief Going forward, for point evolution
	Forward = 1
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
	[[maybe_unused]] const ClassicalVector<double>& mass
)
{
	[[maybe_unused]] static constexpr double CouplingCriterion = 0.01; // Criteria of whether have coupling or not
	static constexpr bool IsAdiabatic = true;
	if constexpr (IsAdiabatic)
	{
		return ClassicalVector<bool>::Zero();
	}
	else
	{
		if constexpr (NumPES == 2)
		{
			const auto nac_01 = tensor_slice(adiabatic_coupling(x), 0, 1), f_01 = tensor_slice(adiabatic_force(x), 0, 1);
			return nac_01.array() * p.array() / mass.array() > CouplingCriterion || f_01.array() > CouplingCriterion;
		}
		else
		{
			const Tensor3d& NAC = adiabatic_coupling(x);
			const Tensor3d& Force = adiabatic_force(x);
			ClassicalVector<bool> result = ClassicalVector<bool>::Zero();
			for (std::size_t iPES = 1; iPES < NumPES; iPES++)
			{
				for (std::size_t jPES = 0; jPES < iPES; jPES++)
				{
					const auto nac_ij = tensor_slice(NAC, iPES, jPES), f_ij = tensor_slice(Force, iPES, jPES);
					result = result.array() || nac_ij.array() * p.array() / mass.array() > CouplingCriterion || f_ij.array() > CouplingCriterion;
				}
			}
			return result;
		}
	}
}

bool is_coupling(const AllPoints& density, const ClassicalVector<double>& mass)
{
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES < NumPES; jPES++)
		{
			if (
				std::any_of(
					std::execution::par_unseq,
					density(iPES, jPES).cbegin(),
					density(iPES, jPES).cend(),
					[&mass](const PhaseSpacePoint& psp) -> bool
					{
						const auto [x, p] = split_phase_coordinate(psp.get<0>());
						return is_coupling(x, p, mass).any();
					}
				)
			)
			{
				return true;
			}
		}
	}
	return false;
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

/// @brief To calculate the new phase space coordinates under adiabatic dynamics
/// @param[in] x Initial position of classical degree of freedom
/// @param[in] p Initial momentum of classical degree of freedom
/// @param[in] mass Mass of classical degree of freedom
/// @param[in] dt Time interval
/// @param[in] drc The direction of evolution
/// @param[in] RowIndex RowIndex Index of row of the element in density matrix. Must be a valid index (< @p NumPES )
/// @param[in] ColIndex RowIndex Index of row of the element in density matrix. Must be a valid index and no more than @p RowIndex
/// @return Position and momentum after evolving @p dt time
static std::tuple<ClassicalVector<double>, ClassicalVector<double>> adiabatic_evolve(
	ClassicalVector<double> x,
	ClassicalVector<double> p,
	const ClassicalVector<double>& mass,
	const double dt,
	const Direction drc,
	const std::size_t RowIndex,
	const std::size_t ColIndex
)
{
	auto position_evolve = [&x, &p = std::as_const(p), &mass, dt, drc](void) -> void
	{
		x += drc * dt / 2.0 * (p.array() / mass.array()).matrix();
	};
	auto momentum_diagonal_nonbranch_evolve = [&x = std::as_const(x), &p, &mass, dt, drc, RowIndex, ColIndex](void) -> void
	{
		const DiagonalForces f = adiabatic_diagonal_forces(x);
		p += drc * dt / 2.0 * (f.col(RowIndex) + f.col(ColIndex));
	};
	position_evolve();
	momentum_diagonal_nonbranch_evolve();
	position_evolve();
	return std::make_tuple(x, p);
}

/// @brief To calculate @f$ -(\Delta V_{ij}(x_0)+\Delta V_{ij}(x_2))/(2\hbar) @f$
/// @param[in] x0 First position
/// @param[in] x2 Last position
/// @param[in] drc The direction of evolution
/// @param[in] RowIndex RowIndex Index of row of the element in density matrix. Must be a valid index (< @p NumPES )
/// @param[in] ColIndex RowIndex Index of row of the element in density matrix. Must be a valid index and no more than @p RowIndex
/// @return @f$ (\Delta V_{ij}(x_0)+\Delta V_{ij}(x_2))/2\hbar @f$
static double calculate_omega0(
	const ClassicalVector<double>& x0,
	const ClassicalVector<double>& x2,
	const Direction drc,
	const std::size_t RowIndex,
	const std::size_t ColIndex
)
{
	if (RowIndex == ColIndex)
	{
		return 0.0;
	}
	const QuantumVector<double> E0 = adiabatic_potential(x0);
	const QuantumVector<double> E2 = adiabatic_potential(x2);
	return drc * (E0[RowIndex] - E0[ColIndex] + E2[RowIndex] - E2[ColIndex]) / 2.0 / hbar;
}

double get_phase_factor(
	const ClassicalPhaseVector& r,
	const ClassicalVector<double>& mass,
	const double dt,
	const std::size_t NumTicks,
	const std::size_t RowIndex,
	const std::size_t ColIndex
)
{
	if (RowIndex == ColIndex)
	{
		return 0;
	}
	static constexpr Direction drc = Direction::Backward;
	auto [x0, p0] = split_phase_coordinate(r);
	ClassicalVector<double> x2;
	double result = 0.0;
	for (std::size_t iTick = NumTicks; iTick > 0; iTick--)
	{
		std::tie(x2, p0) = adiabatic_evolve(x0, p0, mass, dt, drc, RowIndex, ColIndex);
		result -= calculate_omega0(x0, x2, Direction::Forward, RowIndex, ColIndex);
		x0 = x2;
	}
	return result * dt;
}

/// @brief To predict the density matrix of the given point after evolving 1 time step back
/// @param[in] r Phase space coordinates of classical degree of freedom
/// @param[in] density The exact density of the given element at the given point
/// @param[in] mass Mass of classical degree of freedom
/// @param[in] dt Time interval
/// @param[in] NumTicks Number of @p dt since T=0; i.e., now is @p NumTicks * @p dt
/// @param[in] distribution Phase space distribution function
/// @param[in] RowIndex RowIndex Index of row of the element in density matrix. Must be a valid index (< @p NumPES )
/// @param[in] ColIndex RowIndex Index of row of the element in density matrix. Must be a valid index and no more than @p RowIndex
/// @return The density matrix at the given point after evolving
/// @details This function only deals with non-adiabatic case; for adiabatic evolution, @sa evolve()
static std::complex<double> non_adiabatic_evolve_predict(
	const ClassicalPhaseVector& r,
	const std::complex<double> density,
	const ClassicalVector<double>& mass,
	const double dt,
	const std::size_t NumTicks,
	const DistributionFunction& distribution,
	const std::size_t RowIndex,
	const std::size_t ColIndex
)
{
	using namespace std::literals::complex_literals;
	static constexpr Direction drc = Direction::Backward;
	static auto calculate_lower_triangular_index = [](std::size_t Row, std::size_t Col) constexpr->std::size_t
	{
		return Row * (Row + 1) / 2 + Col;
	};
	static const std::array<double, NumTriangularElements> IndicesAdd = [](void) -> std::array<double, NumTriangularElements>
	{
		std::array<double, NumTriangularElements> result;
		for (std::size_t iPES = 0; iPES < NumPES; iPES++)
		{
			for (std::size_t jPES = 0; jPES <= iPES; jPES++)
			{
				result[calculate_lower_triangular_index(iPES, jPES)] = 0.0 + iPES + jPES;
			}
		}
		return result;
	}();
	static const std::array<double, NumTriangularElements> IndicesSub = [](void) -> std::array<double, NumTriangularElements>
	{
		std::array<double, NumTriangularElements> result;
		for (std::size_t iPES = 0; iPES < NumPES; iPES++)
		{
			for (std::size_t jPES = 0; jPES <= iPES; jPES++)
			{
				result[calculate_lower_triangular_index(iPES, jPES)] = 0.0 + iPES - jPES;
			}
		}
		return result;
	}();
	auto position_evolve =
		[&mass]<std::size_t... xIndices, std::size_t... pIndices>(
			const xt::xtensor_fixed<double, xt::xshape<xIndices...>>& x,
			const xt::xtensor_fixed<double, xt::xshape<pIndices...>>& p,
			const double dt
		)
			->xt::xtensor_fixed<double, xt::xshape<pIndices...>>
	{
		return x + drc * dt * p / xt::adapt(mass.data(), TensorVector::shape_type{});
	};
	const auto& [x0, p0] = split_phase_coordinate(r);
	const ClassicalVector<bool> IsCouple = is_coupling(x0, p0, mass);
	assert(IsCouple.any());
	std::complex<double> result = 0;
	if constexpr (NumPES == 2)
	{
		// x_i and p_{i-1} have same number of elements, and p_i is branching
		// 17 steps
		// index: {classical dimensions}
		const auto [x2, p1] = adiabatic_evolve(x0, p0, mass, dt / 2.0, drc, RowIndex, ColIndex);
		// index: {offdiagonal branching, classical dimensions}
		const auto p2 = [&x2, &p1, dt, &IsCouple](void) -> OffDiagonalBranched
		{
			const TensorVector f01 = xt::view(adiabatic_force(x2), xt::all(), 0, 1) * xt::adapt(IsCouple.data(), TensorVector::shape_type{});
			const auto n_broadcast = xt::view(xt::adapt(OffDiagonalBranches, xt::xshape<NumOffDiagonalBranches>{}), xt::all(), xt::newaxis(), xt::newaxis());
			return xt::adapt(p1.data(), TensorVector::shape_type{}) + dt * n_broadcast * f01;
		}();
		const OffDiagonalBranched x3 = position_evolve(TensorVector(xt::adapt(x2.data(), TensorVector::shape_type{})), p2, dt / 4.0);
		// index: {classical dimensions, density matrix element it goes to, offdiagonal branching, density matrix element it comes from}
		const TotallyBranched p3 = [&x3, &p2, dt = dt / 2.0](void)
		{
			TotallyBranched result;
			// recursively reduce rank, from highest rank to lowest
			for (std::size_t iOffDiagonalBranch = 0; iOffDiagonalBranch < NumOffDiagonalBranches; iOffDiagonalBranch++)
			{
				const auto xview = xt::view(x3, iOffDiagonalBranch, xt::all()), pview = xt::view(p2, iOffDiagonalBranch, xt::all());
				const DiagonalForces f = adiabatic_diagonal_forces(ClassicalVector<double>::Map(xview.data() + xview.data_offset()));
				auto f_col_view = [&f](std::size_t i)
				{
					return xt::adapt(f.col(i).data(), TensorVector::shape_type{});
				};
				for (std::size_t iPES = 0; iPES < NumPES; iPES++)
				{
					for (std::size_t jPES = 0; jPES <= iPES; jPES++)
					{
						xt::view(result, calculate_lower_triangular_index(iPES, jPES), iOffDiagonalBranch, xt::all()) = pview + drc * dt / 2.0 * (f_col_view(iPES) + f_col_view(jPES));
					}
				}
			}
			return result;
		}();
		const TotallyBranched x4 = position_evolve(x3, p3, dt / 4.0);
		// auxiliary terms: evolved elements, phi, omega0, omega1
		const double omega0 = RowIndex == ColIndex ? 0 : calculate_omega0(x0, x2, Direction::Forward, 0, 1);
		// omega1(n, gamma) = omega0(R2(gamma, 0), R3(n, gamVma, .), R4((1,0), n, gamma, .))
		const auto omega1 = [&x2, &x4](void)
		{
			xt::xtensor_fixed<double, xt::xshape<NumOffDiagonalBranches>> result;
			for (std::size_t iOffDiagonalBranch = 0; iOffDiagonalBranch < NumOffDiagonalBranches; iOffDiagonalBranch++)
			{
				const auto xview = xt::view(x4, calculate_lower_triangular_index(1, 0), iOffDiagonalBranch, xt::all());
				result(iOffDiagonalBranch) = calculate_omega0(x2, ClassicalVector<double>::Map(xview.data() + xview.data_offset()), Direction::Forward, 0, 1);
			}
			return result;
		}();
		// phi(n, gamma) = p2(n, gamma, .).dot(d01(R2(gamma, .)))
		const auto phi = [&x2, &p2, &IsCouple](void) -> xt::xtensor_fixed<double, xt::xshape<NumOffDiagonalBranches>>
		{
			const ClassicalVector<double> nac_01 = tensor_slice(adiabatic_coupling(x2), 0, 1).array() * IsCouple.array().cast<double>();
			return xt::sum(p2 * xt::adapt(nac_01.data(), TensorVector::shape_type{}), {1});
		}();
		// rho(gamma', n, gamma) = rho^{gamma'}_W(R4(gamma', n, gamma, .), P3(gamma', n, gamma, .), t)
		const auto rho_predict = [&x4, &p3, density, &mass, dt, NumTicks, &distribution, RowIndex, ColIndex](void)
		{
			xt::xtensor_fixed<std::complex<double>, xt::xshape<NumTriangularElements, NumOffDiagonalBranches>> result;
			for (std::size_t iPES = 0; iPES < NumPES; iPES++)
			{
				for (std::size_t jPES = 0; jPES <= iPES; jPES++)
				{
					const std::size_t TrigIndex = calculate_lower_triangular_index(iPES, jPES);
					// construct training set
					const PhasePoints r = [&x4, &p3, TrigIndex](void) -> Eigen::Matrix<double, PhaseDim, NumOffDiagonalBranches>
					{
						const auto xview = xt::view(x4, TrigIndex, xt::all(), xt::all()), pview = xt::view(p3, TrigIndex, xt::all(), xt::all());
						Eigen::Matrix<double, PhaseDim, NumOffDiagonalBranches> result;
						result.block<Dim, NumOffDiagonalBranches>(0, 0) =
							Eigen::Matrix<double, Dim, NumOffDiagonalBranches>::Map(xview.data() + xview.data_offset());
						result.block<Dim, NumOffDiagonalBranches>(Dim, 0) =
							Eigen::Matrix<double, Dim, NumOffDiagonalBranches>::Map(pview.data() + pview.data_offset());
						return result;
					}();
					// get prediction
					for (std::size_t iOffDiagonalBranch = 0; iOffDiagonalBranch < NumOffDiagonalBranches; iOffDiagonalBranch++)
					{
						if (iPES == RowIndex && jPES == ColIndex && OffDiagonalBranches[iOffDiagonalBranch] == 0)
						{
							// the exact element, assign the exact density
							result(TrigIndex, iOffDiagonalBranch) = density;
						}
						else
						{
							result(TrigIndex, iOffDiagonalBranch) = distribution(r.col(iOffDiagonalBranch), iPES, jPES);
							if (iPES != jPES)
							{
								result(TrigIndex, iOffDiagonalBranch) *= std::exp(1.0i * get_phase_factor(r.col(iOffDiagonalBranch), mass, dt, NumTicks, iPES, jPES));
							}
						}
					}
				}
			}
			return result;
		}();
		// auxiliary terms: c1, c2, c3, c4
		// C1(n, gamma) = ((1-delta(n, 0))delta(alpha, beta) + cos(phi(0, gamma)dt - (alpha + beta + n)pi/2)exp(i(alpha - beta)omega0 * dt / 2)) / (4 - 2delta(n, 0))
		const auto C1 = [&phi, omega0, dt, RowIndex, ColIndex](void)
		{
			xt::xtensor_fixed<std::complex<double>, xt::xshape<NumOffDiagonalBranches>> result = xt::zeros<decltype(result)::value_type>(decltype(result)::shape_type{});
			result += xt::cos(phi(1) * dt - M_PI_2 * (RowIndex + ColIndex + xt::adapt(OffDiagonalBranches.data(), {NumOffDiagonalBranches})))
				* std::exp(1.0i * (0.0 + RowIndex - ColIndex) / 2.0 * omega0 * dt);
			for (std::size_t iOffDiagonalBranch = 0; iOffDiagonalBranch < NumOffDiagonalBranches; iOffDiagonalBranch++)
			{
				if (OffDiagonalBranches[iOffDiagonalBranch] != 0)
				{
					result(iOffDiagonalBranch) += static_cast<double>(RowIndex == ColIndex);
				}
				result(iOffDiagonalBranch) /= 4.0 - 2.0 * static_cast<double>(OffDiagonalBranches[iOffDiagonalBranch] == 0);
			}
			for (std::size_t iOffDiagonalBranch = 0; iOffDiagonalBranch < NumOffDiagonalBranches; iOffDiagonalBranch++)
			{
			}
			return result;
		}();
		// c2(gamma', n, gamma) = (1-delta(n, 0))delta(alpha', beta') + cos(phi(n, gamma')dt + (alpha' + beta' + n)pi/2)exp(i(alpha' - beta')omega1(n, gamma') * dt / 2)
		const auto C2 = [&phi, &omega1, dt, RowIndex, ColIndex](void)
		{
			xt::xtensor_fixed<std::complex<double>, xt::xshape<NumTriangularElements, NumOffDiagonalBranches>> result = xt::zeros<decltype(result)::value_type>(decltype(result)::shape_type{});
			result += xt::cos(phi * dt + M_PI_2 * (xt::view(xt::adapt(IndicesAdd.data(), xt::xshape<NumTriangularElements>{}), xt::all(), xt::newaxis()) + xt::adapt(OffDiagonalBranches.data(), xt::xshape<NumOffDiagonalBranches>{})))
				* xt::exp(1.0i * xt::view(xt::adapt(IndicesSub.data(), xt::xshape<NumTriangularElements>{}), xt::all(), xt::newaxis()) / 2.0 * omega1 * dt);
			for (std::size_t iOffDiagonalBranch = 0; iOffDiagonalBranch < NumOffDiagonalBranches; iOffDiagonalBranch++)
			{
				if (OffDiagonalBranches[iOffDiagonalBranch] != 0)
				{
					for (std::size_t iPES = 0; iPES < NumPES; iPES++)
					{
						result(calculate_lower_triangular_index(iPES, iPES), iOffDiagonalBranch) += 1.0;
					}
				}
				else
				{
					xt::view(result, xt::all(), iOffDiagonalBranch) *= 2.0;
				}
			}
			return result;
		}();
		result += xt::sum(C1 * xt::real(C2 * rho_predict))();
		if (RowIndex != ColIndex)
		{
			result += 1.0i * std::exp(1.0i * omega0 * dt / 2.0) * (std::exp(1.0i * omega1(OffDiagonalZeroBranch) * dt / 2.0) * rho_predict(calculate_lower_triangular_index(1, 0), OffDiagonalZeroBranch)).imag() / 2.0;
		}
	}
	else
	{
		assert(!"NO INSTANTATION OF MORE THAN TWO LEVEL SYSTEM NOW\n");
	}
	return result;
}

/// @details If the point is not in the coupling region, evolve it adiabatically;
/// otherwise, branch the point with non-adiabatic dynamics, then
/// calculate exact density at the new point, and select with monte carlo
void evolve(
	AllPoints& density,
	const ClassicalVector<double>& mass,
	const double dt,
	const std::size_t NumTicks,
	const DistributionFunction& distribution
)
{
	using namespace std::literals::complex_literals;
	static constexpr Direction drc = Direction::Forward;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES <= iPES; jPES++)
		{
			// for row-major, visit upper triangular elements earlier
			// so selection for the upper triangular elements
			std::for_each(
				std::execution::par_unseq,
				density(iPES, jPES).begin(),
				density(iPES, jPES).end(),
				[&mass, dt, NumTicks, &distribution, iPES, jPES](PhaseSpacePoint& psp) -> void
				{
					auto& [r, rho, theta] = psp;
					const auto& [x0, p0] = split_phase_coordinate(r);
					if (is_coupling(x0, p0, mass).any())
					{
						// with coupling, non-adiabatic case
						// save the exact element before adiabatic rotation
						const std::complex<double> exact_element = psp.get_exact_element();
						// calculate its adiabatically evolved phase space coordinate with 2 half steps
						const auto& [x2, p1] = adiabatic_evolve(x0, p0, mass, dt / 2, drc, iPES, jPES);
						theta -= calculate_omega0(x0, x2, drc, iPES, jPES) * dt / 2;
						const auto& [x4, p2] = adiabatic_evolve(x2, p1, mass, dt / 2, drc, iPES, jPES);
						theta -= calculate_omega0(x2, x4, drc, iPES, jPES) * dt / 2;
						// and use back-propagation to calculate the exact density there
						r << x4, p2;
						psp.set_density(non_adiabatic_evolve_predict(r, exact_element, mass, dt, NumTicks - 1, distribution, iPES, jPES));
					}
					else
					{
						// without coupling, adiabatic case
						// calculate its adiabatically evolved phase space coordinate
						const auto& [x2, p1] = adiabatic_evolve(x0, p0, mass, dt, drc, iPES, jPES);
						theta -= calculate_omega0(x0, x2, drc, iPES, jPES) * dt;
						// extract phase factor, the rest remains the same
						rho = distribution(r, iPES, jPES);
						r << x2, p1;
					}
				}
			);
		}
	}
}
