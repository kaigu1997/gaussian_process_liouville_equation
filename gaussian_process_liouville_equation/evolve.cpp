/// @file evolve.cpp
/// @brief Implementation of evolve.h

#include "stdafx.h"

#include "evolve.h"

#include "pes.h"
#include "storage.h"

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

/// @brief The direction of dynamics evolving
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
/// @param[in] dt Time interval
/// @return Whether each classical direction has coupling or not
static ClassicalVector<bool> is_coupling(
	[[maybe_unused]] const ClassicalVector<double>& x,
	[[maybe_unused]] const ClassicalVector<double>& p,
	[[maybe_unused]] const ClassicalVector<double>& mass,
	const double dt
)
{
	[[maybe_unused]] static constexpr double CouplingCriterion = 0; // Criteria of whether have coupling or not
	static constexpr bool IsAdiabatic = false;
	if constexpr (IsAdiabatic)
	{
		return ClassicalVector<bool>::Zero();
	}
	else
	{
		const Tensor3d Force = adiabatic_force(x), NAC = adiabatic_coupling(x);
		const ClassicalVector<double> diag_f_average = [&Force](void) -> ClassicalVector<double>
		{
			ClassicalVector<double> sum = ClassicalVector<double>::Zero();
			for (const std::size_t iPES : std::ranges::iota_view{0ul, NumPES})
			{
				sum += tensor_slice(Force, iPES, iPES);
			}
			return sum / NumPES;
		}();
		if constexpr (NumPES == 2)
		{
			const auto nac_01 = tensor_slice(NAC, 0, 1), f_01 = tensor_slice(Force, 0, 1);
			return (nac_01.array() * p.array() / mass.array()).abs() * dt >= CouplingCriterion
				|| (f_01.array() / diag_f_average.array()).abs() >= CouplingCriterion;
		}
		else
		{
			ClassicalVector<bool> result = ClassicalVector<bool>::Zero();
			for (const std::size_t iPES : std::ranges::iota_view{1ul, NumPES})
			{
				for (const std::size_t jPES : std::ranges::iota_view{0ul, iPES})
				{
					const auto nac_ij = tensor_slice(NAC, iPES, jPES), f_ij = tensor_slice(Force, iPES, jPES);
					result = result.array()
						|| (nac_ij.array() * p.array() / mass.array()).abs() * dt >= CouplingCriterion
						|| (f_ij.array() / diag_f_average.array()).abs() >= CouplingCriterion;
				}
			}
			return result;
		}
	}
}

/// @brief To get the diagonal forces under adiabatic basis at given position
/// @param[in] x Position of classical degree of freedom
/// @return A matrix, i-th row is the force of i-th surface
static DiagonalForces adiabatic_diagonal_forces(const ClassicalVector<double>& x)
{
	const Tensor3d& force = adiabatic_force(x);
	DiagonalForces result;
	for (const std::size_t iPES : std::ranges::iota_view{0ul, NumPES})
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
		x += static_cast<int>(drc) * dt / 2.0 * (p.array() / mass.array()).matrix();
	};
	auto momentum_diagonal_nonbranch_evolve = [&x = std::as_const(x), &p, &mass, dt, drc, RowIndex, ColIndex](void) -> void
	{
		const DiagonalForces f = adiabatic_diagonal_forces(x);
		p += static_cast<int>(drc) * dt / 2.0 * (f.col(RowIndex) + f.col(ColIndex));
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
static inline double calculate_omega0(
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
	return static_cast<int>(drc) * (E0[RowIndex] - E0[ColIndex] + E2[RowIndex] - E2[ColIndex]) / 2.0 / hbar;
}

/// @brief To predict the density matrix of the given point after evolving 1 time step back
/// @param[in] r Phase space coordinates of classical degree of freedom
/// @param[in] density The exact density of the given element at the given point
/// @param[in] mass Mass of classical degree of freedom
/// @param[in] dt Time interval
/// @param[in] distribution Phase space distribution function
/// @param[in] RowIndex RowIndex Index of row of the element in density matrix. Must be a valid index (< @p NumPES )
/// @param[in] ColIndex RowIndex Index of row of the element in density matrix. Must be a valid index and no more than @p RowIndex
/// @return The density matrix at the given point after evolving
/// @details This function only deals with non-adiabatic case; for adiabatic evolution, @sa evolve(), new_point_predict()
static std::complex<double> non_adiabatic_evolve_predict(
	const ClassicalPhaseVector& r,
	const std::optional<std::complex<double>> density,
	const ClassicalVector<double>& mass,
	const double dt,
	const DistributionFunction& distribution,
	const std::size_t RowIndex,
	const std::size_t ColIndex
)
{
	using namespace std::literals::complex_literals;
	static constexpr Direction drc = Direction::Backward;
	static auto calculate_lower_triangular_index = [](const std::size_t Row, const std::size_t Col) constexpr -> std::size_t
	{
		return Row * (Row + 1) / 2 + Col;
	};
	auto position_evolve =
		[&mass]<std::size_t... xIndices, std::size_t... pIndices>(
			const xt::xtensor_fixed<double, xt::xshape<xIndices...>>& x,
			const xt::xtensor_fixed<double, xt::xshape<pIndices...>>& p,
			const double dt
		)
			->xt::xtensor_fixed<double, xt::xshape<pIndices...>>
	{
		return x + static_cast<int>(drc) * dt * p / xt::adapt(mass.data(), TensorVector::shape_type{});
	};
	const auto& [x0, p0] = split_phase_coordinate(r);
	const ClassicalVector<bool> IsCouple = is_coupling(x0, p0, mass, dt);
	if constexpr (NumPES == 2)
	{
		auto offdiagonal_rotation =
			[&mass, &IsCouple](auto rho_view, const ClassicalVector<double>& x, const ClassicalVector<double>& p, const double dt) -> void
		{
			// rhoview are expected to have 3 elements
			assert(rho_view.shape()[0] == NumTriangularElements && rho_view.dimension() == 1);
			// phi: v.dot(NAC)
			const double phi =
				(p.array() / mass.array() * tensor_slice(adiabatic_coupling(x), 0, 1).array() * is_coupling(x, p, mass, dt).array().cast<double>()).sum();
			const double cosphi = std::cos(2.0 * phi * dt), sinphi = std::sin(2.0 * phi * dt);
			// save the old data
			const xt::xtensor_fixed<std::complex<double>, xt::xshape<NumTriangularElements>> rho_save = rho_view;
			rho_view(0) = (1.0 + cosphi) / 2.0 * rho_save(0) - sinphi * rho_save(1).real() + (1.0 - cosphi) / 2.0 * rho_save(2);
			rho_view(1) = sinphi / 2.0 * rho_save(0) + cosphi * rho_save(1).real() + 1.0i * rho_save(1).imag() - sinphi / 2.0 * rho_save(2);
			rho_view(2) = (1.0 - cosphi) / 2.0 * rho_save(0) + sinphi * rho_save(1).real() + (1.0 + cosphi) / 2.0 * rho_save(2);
		};
		// x_i and p_{i-1} have same number of elements, and p_i is branching
		// 17 steps
		// index: {classical dimensions}
		const auto [x2, p1] = adiabatic_evolve(x0, p0, mass, dt / 2.0, drc, RowIndex, ColIndex);
		// index: {offdiagonal branching, classical dimensions}
		// (backward) direction is included. So when evolve forward, the branch correspondence remains the same
		const auto p2 = [&x2, &p1, dt, &IsCouple](void) -> OffDiagonalBranched
		{
			const TensorVector f01 = xt::view(adiabatic_force(x2), xt::all(), 0, 1) * xt::adapt(IsCouple.data(), TensorVector::shape_type{});
			const auto n_broadcast = xt::view(xt::adapt(OffDiagonalBranches, xt::xshape<NumOffDiagonalBranches>{}), xt::all(), xt::newaxis());
			return xt::adapt(p1.data(), TensorVector::shape_type{}) + dt * static_cast<double>(drc) * n_broadcast * f01;
		}();
		const OffDiagonalBranched x3 = position_evolve(TensorVector(xt::adapt(x2.data(), TensorVector::shape_type{})), p2, dt / 4.0);
		// index: {classical dimensions, density matrix element it goes to, offdiagonal branching, density matrix element it comes from}
		const TotallyBranched p3 = [&x3, &p2, dt = dt / 2.0](void)
		{
			TotallyBranched result;
			// recursively reduce rank, from highest rank to lowest
			for (const std::size_t iOffDiagonalBranch : std::ranges::iota_view{0ul, NumOffDiagonalBranches})
			{
				const auto xview = xt::view(x3, iOffDiagonalBranch, xt::all()), pview = xt::view(p2, iOffDiagonalBranch, xt::all());
				const DiagonalForces f = adiabatic_diagonal_forces(ClassicalVector<double>::Map(xview.data() + xview.data_offset()));
				auto f_col_view = [&f](const std::size_t i)
				{
					return xt::adapt(f.col(i).data(), TensorVector::shape_type{});
				};
				for (const std::size_t iPES : std::ranges::iota_view{0ul, NumPES})
				{
					for (const std::size_t jPES : std::ranges::iota_view{0ul, iPES + 1})
					{
						xt::view(result, calculate_lower_triangular_index(iPES, jPES), iOffDiagonalBranch, xt::all()) =
							pview + static_cast<int>(drc) * dt / 2.0 * (f_col_view(iPES) + f_col_view(jPES));
					}
				}
			}
			return result;
		}();
		const TotallyBranched x4 = position_evolve(x3, p3, dt / 4.0);
		// rho(gamma', n, gamma) = rho^{gamma'}_W(R4(gamma', n, gamma, .), P3(gamma', n, gamma, .), t)
		// The predicted density at all those points
		auto rho_predict = [&x4, &p3, density, &distribution, RowIndex, ColIndex](void)
		{
			xt::xtensor_fixed<std::complex<double>, xt::xshape<NumTriangularElements, NumOffDiagonalBranches>> result;
			for (const std::size_t iPES : std::ranges::iota_view{0ul, NumPES})
			{
				for (const std::size_t jPES : std::ranges::iota_view{0ul, iPES + 1})
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
					for (const std::size_t iOffDiagonalBranch : std::ranges::iota_view{0ul, NumOffDiagonalBranches})
					{
						if (iPES == RowIndex && jPES == ColIndex && OffDiagonalBranches[iOffDiagonalBranch] == 0 && density.has_value())
						{
							// the exact element, assign the exact density
							result(TrigIndex, iOffDiagonalBranch) = density.value();
						}
						else
						{
							result(TrigIndex, iOffDiagonalBranch) = distribution(r.col(iOffDiagonalBranch), iPES, jPES);
						}
					}
				}
			}
			return result;
		}();
		xt::xtensor_fixed<std::complex<double>, xt::xshape<NumTriangularElements>> rho_combined_offdiag = xt::zeros<decltype(rho_combined_offdiag)::value_type>(decltype(rho_combined_offdiag)::shape_type{});
		for (const std::size_t iOffDiagonalBranch : std::ranges::iota_view{0ul, NumOffDiagonalBranches})
		{
			// first, an adiabatic evolve. x4 and p3 goes to x2 and p2, rho evolves in the following way
			const auto x4view = xt::view(x4, calculate_lower_triangular_index(1, 0), iOffDiagonalBranch, xt::all());
			rho_predict(calculate_lower_triangular_index(1, 0), iOffDiagonalBranch) *=
				std::exp(calculate_omega0(x2, ClassicalVector<double>::Map(x4view.data() + x4view.data_offset()), Direction::Forward, 0, 1) * dt / 2 * 1.0i);
			// now they are at x2 and p2. A off-diagonal rotation is needed.
			const auto p2view = xt::view(p2, iOffDiagonalBranch, xt::all());
			offdiagonal_rotation(
				xt::view(rho_predict, xt::all(), iOffDiagonalBranch),
				x2,
				ClassicalVector<double>::Map(p2view.data() + p2view.data_offset()),
				dt / 2.0
			);
			// then the off-diagonal force evolution combination with a rotation matrix, p2 comes to p1, x2 keeps the same
			switch (OffDiagonalBranches[iOffDiagonalBranch])
			{
			case -1:
				rho_combined_offdiag +=
					(rho_predict(0, iOffDiagonalBranch) + 2.0 * rho_predict(1, iOffDiagonalBranch).real() + rho_predict(2, iOffDiagonalBranch)) / 4.0;
				break;
			case 0:
			{
				const std::complex<double> value = (rho_predict(0, iOffDiagonalBranch) - rho_predict(2, iOffDiagonalBranch)) / 2.0;
				rho_combined_offdiag(0) += value;
				rho_combined_offdiag(1) += 1.0i * rho_predict(1, iOffDiagonalBranch).imag();
				rho_combined_offdiag(2) -= value;
				break;
			}
			case 1:
			{
				const std::complex<double> value =
					(rho_predict(0, iOffDiagonalBranch) - 2.0 * rho_predict(1, iOffDiagonalBranch).real() + rho_predict(2, iOffDiagonalBranch)) / 4.0;
				rho_combined_offdiag(0) += value;
				rho_combined_offdiag(1) -= value;
				rho_combined_offdiag(2) += value;
				break;
			}
			default:
				assert(!"WRONG BRANCH!");
				break;
			}
		}
		// the another off-diagonal rotation at x2 and p1
		offdiagonal_rotation(
			xt::view(rho_combined_offdiag, xt::all()),
			x2,
			p1,
			dt / 2.0
		);
		// now another adiabatic evolve from (x2, p1) to (x0, p0), i.e., r
		const std::complex<double> result = rho_combined_offdiag(calculate_lower_triangular_index(RowIndex, ColIndex));
		if (RowIndex != ColIndex)
		{
			return result * std::exp(calculate_omega0(x0, x2, Direction::Forward, 0, 1) * dt / 2.0 * 1.0i);
		}
		else
		{
			return result;
		}
	}
	else
	{
		assert(!"NO INSTANTATION OF MORE THAN TWO LEVEL SYSTEM NOW\n");
		return 0.0;
	}
}

/// @details If the point is not in the coupling region, evolve it adiabatically;
/// otherwise, branch the point with non-adiabatic dynamics, then
/// calculate exact density at the new point, and select with monte carlo
void evolve(
	AllPoints& density,
	const ClassicalVector<double>& mass,
	const double dt,
	const DistributionFunction& distribution
)
{
	using namespace std::literals::complex_literals;
	static constexpr Direction drc = Direction::Forward;
	for (const std::size_t iPES : std::ranges::iota_view{0ul, NumPES})
	{
		for (const std::size_t jPES : std::ranges::iota_view{0ul, iPES + 1})
		{
			// for row-major, visit upper triangular elements earlier
			// so selection for the upper triangular elements
			std::for_each(
				std::execution::par_unseq,
				density(iPES, jPES).begin(),
				density(iPES, jPES).end(),
				[&mass, dt, &distribution, iPES, jPES](PhaseSpacePoint& psp) -> void
				{
					auto& [r, rho] = psp;
					const auto& [x0, p0] = split_phase_coordinate(r);
					if (is_coupling(x0, p0, mass, dt).any())
					{
						// with coupling, non-adiabatic case
						// calculate its adiabatically evolved phase space coordinate with 2 half steps
						const auto& [x2, p1] = adiabatic_evolve(x0, p0, mass, dt / 2, drc, iPES, jPES);
						const auto& [x4, p2] = adiabatic_evolve(x2, p1, mass, dt / 2, drc, iPES, jPES);
						// and use back-propagation to calculate the exact density there
						r << x4, p2;
						rho = non_adiabatic_evolve_predict(r, rho, mass, dt, distribution, iPES, jPES);
					}
					else
					{
						// without coupling, adiabatic case
						// calculate its adiabatically evolved phase space coordinate
						const auto& [x2, p1] = adiabatic_evolve(x0, p0, mass, dt, drc, iPES, jPES);
						// extract phase factor, the rest remains the same
						rho = distribution(r, iPES, jPES) * std::exp(-calculate_omega0(x0, x2, drc, iPES, jPES) * dt * 1.0i); // -i(Vi-Vj)dt/hbar
						r << x2, p1;
					}
				}
			);
		}
	}
}

std::complex<double> new_point_predict(
	const ClassicalPhaseVector& r,
	const ClassicalVector<double>& mass,
	const double dt,
	const DistributionFunction& distribution,
	const std::size_t RowIndex,
	const std::size_t ColIndex
)
{
	const auto& [x, p] = split_phase_coordinate(r);
	if (is_coupling(x, p, mass, dt).any())
	{
		return non_adiabatic_evolve_predict(r, std::nullopt, mass, dt, distribution, RowIndex, ColIndex);
	}
	else
	{
		return 0.0;
	}
}

QuantumStorage<bool> is_very_small(
	const AllPoints& density,
	const ClassicalVector<double>& mass,
	const double dt,
	const DistributionFunction& distribution
)
{
	static constexpr double epsilon = power<2>(1e-5); // value larger than this are regarded important
	static const std::size_t NumCheckingPoints = density(0).size();
	QuantumStorage<bool> result(false, false);
	for (const std::size_t iPES : std::ranges::iota_view{0ul, NumPES})
	{
		for (const std::size_t jPES : std::ranges::iota_view{0ul, iPES + 1})
		{
			if (density(iPES, jPES).empty())
			{
				// get some test points
				ElementPoints test_points(density(0).cbegin(), density(0).cbegin() + NumCheckingPoints);
				result(iPES, jPES) =
					std::all_of(
						std::execution::par_unseq,
						test_points.cbegin(),
						test_points.cend(),
						[&mass, dt, &distribution, iPES, jPES](const PhaseSpacePoint& psp) -> bool
						{
							return std::norm(new_point_predict(psp.get<0>(), mass, dt, distribution, iPES, jPES)) < epsilon;
						}
					); // only if all the test points are small, this element in density matrix could be regarded small
			}
			// otherwise the element must not be empty, and thus not small
		}
	}
	return result;
}
