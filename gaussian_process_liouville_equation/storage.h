/// @file storage.h
/// @brief The data structure to store data in this program

#ifndef STORAGE_H
#define STORAGE_H

#include "stdafx.h"

/// @brief To calculate the serialized index of off-diagonal kernels
/// @param[in] RowIndex Index of row. Must be a valid index (< @p NumPES )
/// @param[in] ColIndex Index of column. Must be a valid index and smaller than @p RowIndex
/// @return Index in the strict lower-triangular part
inline constexpr std::size_t calculate_off_diagonal_index(std::size_t RowIndex, std::size_t ColIndex)
{
	return RowIndex * (RowIndex - 1) / 2 + ColIndex;
}

/// @brief To construct array with given data
/// @tparam T Data type in the return array
/// @tparam N Size of the array
/// @tparam Args Type of arguments to construct the objects in result array
/// @param[in] array_args All the arguments to construct an object of type @p T
/// @sa fill_array()
template <typename T, std::size_t N, typename... Args>
std::array<std::decay_t<T>, N> construct_array(std::array<Args, N>&&... array_args)
{
	using TDecay = std::decay_t<T>;
	auto impl = [&array_args...]<std::size_t... II>(std::index_sequence<II...>) -> std::array<TDecay, N>
	{
		auto construct_member = [&array_args...](std::size_t III) -> TDecay
		{
			return TDecay(std::forward<std::decay_t<Args>>(array_args[III])...);
		};
		return {construct_member(II)...};
	};
	return impl(std::make_index_sequence<N>{});
}

/// @brief To fill an array with given data, prevent using @p array.fill() for non-trivial objects
/// @tparam T Data type in the return array
/// @tparam N Size of the array
/// @tparam Args Type of arguments to construct the object filled in result
/// @param[in] args The argument to construct an object of type @p T
/// @return An array full of same object of type @p T
/// @sa construct_array()
template <typename T, std::size_t N, typename... Args>
std::array<std::decay_t<T>, N> fill_array(Args... args)
{
	using TDecay = std::decay_t<T>;
	auto impl = [&args...]<std::size_t... II>(std::index_sequence<II...>) -> std::array<TDecay, N>
	{
		if constexpr (std::is_copy_constructible_v<TDecay>)
		{
			const TDecay& temp = TDecay(std::forward<Args>(args)...);
			auto skip_index = [&temp](std::size_t) -> const TDecay&
			{
				return temp;
			};
			return {skip_index(II)...};
		}
		else
		{
			auto skip_index = [&args...](std::size_t) -> TDecay
			{
				return TDecay(args...);
			};
			return {skip_index(II)...};
		}
	};
	return impl(std::make_index_sequence<N>{});
}

/// @brief To judge whether to pass by value or reference
template <typename T>
using ValOrCRef = std::conditional_t<sizeof(T) <= 2 * sizeof(void*), T, const T&>;

/// @brief A more compact and variable way to store data compared to @p NumPES ^2 elements in an array
/// @tparam DiagDT Type of data corresponding to diagonal elements
/// @tparam OffDiagDT Type of data corresponding to off-diagonal elements
template <typename DiagDT, typename OffDiagDT = DiagDT>
class QuantumStorage
{
public:
	/// @brief Data type of all diagonal data
	using DiagonalArrayType = std::array<DiagDT, NumPES>;
	/// @brief Data type of all off-diagonal data
	using OffDiagonalArrayType = std::array<OffDiagDT, NumOffDiagonalElements>;

	/// @brief Default constructor
	QuantumStorage() = default;

	/// @brief Construct using @p construct_array
	/// @tparam DiagArgs Types of arguments for constructing diagonal data
	/// @tparam OffDiagArgs Types of argument for constructing off-diagonal data
	/// @param[in] diag_args Arguments for constructing diagonal data
	/// @param[in] offdiag_args Argument for constructing off-diagonal data
	template <typename... DiagArgs, typename... OffDiagArgs>
	QuantumStorage(std::tuple<std::array<DiagArgs, NumPES>...> diag_args, std::tuple<std::array<OffDiagArgs, NumOffDiagonalElements>...> offdiag_args):
		diagonal_data(std::apply(construct_array<DiagDT, NumPES, DiagArgs...>, std::move(diag_args))),
		offdiagonal_data(std::apply(construct_array<OffDiagDT, NumOffDiagonalElements, OffDiagArgs...>, std::move(offdiag_args)))
	{
	}

	/// @brief Construct with same data
	/// @param[in] diagonal The data for all diagonal elements
	/// @param[in] offdiagonal The data for all off-diagonal elements
	QuantumStorage(ValOrCRef<DiagDT> diagonal, ValOrCRef<OffDiagDT> offdiagonal):
		diagonal_data(fill_array<DiagDT, NumPES>(diagonal)),
		offdiagonal_data(fill_array<OffDiagDT, NumOffDiagonalElements>(offdiagonal))
	{
	}

	/// @brief To get the editable diagonal array
	/// @return The editable diagonal array
	DiagonalArrayType& get_diagonal_data()
	{
		return diagonal_data;
	}
	/// @brief To get the read-only diagonal array
	/// @return The read-only diagonal array
	const DiagonalArrayType& get_diagonal_data() const
	{
		return diagonal_data;
	}
	/// @brief To get the editable off-diagonal array
	/// @return The editable off-diagonal array
	OffDiagonalArrayType& get_offdiagonal_data()
	{
		return offdiagonal_data;
	}
	/// @brief To get the read-only off-diagonal array
	/// @return The read-only off-diagonal array
	const OffDiagonalArrayType& get_offdiagonal_data() const
	{
		return offdiagonal_data;
	}

	/// @brief Editable version of diagonal element access
	/// @param[in] Index Index of row (and thus column) of the element in density matrix. Must be a valid index (< @p NumPES )
	/// @return Data corresponding to the diagonal element
	DiagDT& operator()(std::size_t Index)
	{
		assert(Index < NumPES);
		return diagonal_data[Index];
	}
	/// @brief Read-only version of diagonal element access
	/// @param[in] Index Index of row (and thus column) of the element in density matrix. Must be a valid index (< @p NumPES )
	/// @return Data corresponding to the diagonal element
	ValOrCRef<DiagDT> operator()(std::size_t Index) const
	{
		assert(Index < NumPES);
		return diagonal_data[Index];
	}
	/// @brief Editable version of off-diagonal (strictly lower-triangular) element access
	/// @param[in] RowIndex RowIndex Index of row of the element in density matrix. Must be a valid index (< @p NumPES )
	/// @param[in] ColIndex RowIndex Index of row of the element in density matrix. Must be a valid index and no more than @p RowIndex
	/// @return Data corresponding to the off-diagonal element
	OffDiagDT& operator()(std::size_t RowIndex, std::size_t ColIndex)
	{
		if constexpr (std::is_same_v<DiagDT, OffDiagDT>)
		{
			if (RowIndex == ColIndex)
			{
				assert(RowIndex < NumPES);
				return diagonal_data[RowIndex];
			}
		}
		return offdiagonal_data[calculate_off_diagonal_index(RowIndex, ColIndex)];
	}
	/// @brief Read-only version of off-diagonal (strictly lower-triangular) element access
	/// @param[in] RowIndex RowIndex Index of row of the element in density matrix. Must be a valid index (< @p NumPES )
	/// @param[in] ColIndex RowIndex Index of row of the element in density matrix. Must be a valid index and no more than @p RowIndex
	/// @return Data corresponding to the off-diagonal element
	ValOrCRef<OffDiagDT> operator()(std::size_t RowIndex, std::size_t ColIndex) const
	{
		if constexpr (std::is_convertible_v<DiagDT, OffDiagDT>)
		{
			if (RowIndex == ColIndex)
			{
				assert(RowIndex < NumPES);
				return diagonal_data[RowIndex];
			}
		}
		return offdiagonal_data[calculate_off_diagonal_index(RowIndex, ColIndex)];
	}
	
private:
	/// @brief Data corresponding to diagonal elements
	DiagonalArrayType diagonal_data;
	/// @brief Data corresponding to off-diagonal elements
	OffDiagonalArrayType offdiagonal_data;
};

/// @brief Abstraction of a phase space point
class PhaseSpacePoint
{
public:
	/// @brief Default constructor
	PhaseSpacePoint() = default;
	/// @brief Constructor with given data
	/// @param[in] R Phase space coordinates
	/// @param[in] DenMatElm Exact density matrix element
	/// @param[in] theta Adiabatic phase factor
	PhaseSpacePoint(const ClassicalPhaseVector& R, std::complex<double> DenMatElm, double theta);

	/// @brief "Move" constructor
	/// @param[in] R Phase space coordinates
	/// @param[in] DenMatElm Exact Density matrix element
	/// @param[in] theta Adiabatic phase factor
	PhaseSpacePoint(ClassicalPhaseVector&& R, std::complex<double> DenMatElm, double theta);

	/// @brief To get an element: coordinates, density matrix element, or adiabatic phase factor
	/// @tparam I Index
	/// @return Reference to the value
	template <std::size_t I>
	std::tuple_element<I, PhaseSpacePoint>::type& get()
	{
		static_assert(I < 3);
		if constexpr (I == 0)
		{
			return r;
		}
		else if constexpr (I == 1)
		{
			return rho;
		}
		else
		{
			return adiabatic_theta;
		}
	}
	/// @brief To get the value of an element: coordinates, density matrix, or adiabatic phase factor
	/// @tparam I Index
	/// @return The value
	template <std::size_t I>
	ValOrCRef<typename std::tuple_element<I, PhaseSpacePoint>::type> get() const
	{
		static_assert(I < 3);
		if constexpr (I == 0)
		{
			return r;
		}
		else if constexpr (I == 1)
		{
			return rho;
		}
		else
		{
			return adiabatic_theta;
		}
	}

	/// @brief To calculate the exact density matrix element (with phase-factor included)
	/// @return The exact density matrix element
	std::complex<double> get_exact_element(void) const;

	/// @brief To calculate the corresponding density matrix when given the exact density matrix
	/// @param[in] DenMatElm The exact density matrix
	void set_density(std::complex<double> DenMatElm);

private:
	/// @brief Phase space coordinates
	ClassicalPhaseVector r;
	/// @brief Density matrix element without adiabatic phase factor
	std::complex<double> rho;
	/// @brief Adiabatic phase factor of off-diagonal elements
	double adiabatic_theta;
};

/// @brief Partial specialization for @p PhaseSpacePoint
template <>
struct std::tuple_size<PhaseSpacePoint> : std::integral_constant<std::size_t, 3>
{
};
/// @brief Partial specialization for @p PhaseSpacePoint
/// @tparam I Index of the element in @p PhaseSpacePoint
template <std::size_t I>
struct std::tuple_element<I, PhaseSpacePoint>
{
	static_assert(I < std::tuple_size<PhaseSpacePoint>::value, 'Index out of bounds for PhaseSpacePoint');
};
/// @brief Specialization for first element of @p PhaseSpacePoint
template <>
struct std::tuple_element<0, PhaseSpacePoint>
{
	/// @brief The type of first element of @p PhaseSpacePoint
	using type = ClassicalPhaseVector;
};
/// @brief Specialization for second element of @p PhaseSpacePoint
template <>
struct std::tuple_element<1, PhaseSpacePoint>
{
	/// @brief The type of second element of @p PhaseSpacePoint
	using type = std::complex<double>;
};
/// @brief Specialization for third element of @p PhaseSpacePoint
template <>
struct std::tuple_element<2, PhaseSpacePoint>
{
	/// @brief The type of third element of @p PhaseSpacePoint
	using type = double;
};

/// @brief Sets of selected phase space points of one density matrix element
using ElementPoints = EigenVector<PhaseSpacePoint>;
/// @brief Sets of selected phase space points of all density matrix elements
using AllPoints = QuantumStorage<ElementPoints>;

#endif // !STORAGE_H
