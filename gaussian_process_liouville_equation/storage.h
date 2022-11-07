/// @file storage.h
/// @brief The data structure to store data in this program

#ifndef STORAGE_H
#define STORAGE_H

#include "stdafx.h"

/// @brief To calculate the serialized index of off-diagonal kernels
/// @param[in] RowIndex Index of row. Must be a valid index (< @p NumPES )
/// @param[in] ColIndex Index of column. Must be a valid index and smaller than @p RowIndex
/// @return Index in the strict lower-triangular part
inline constexpr std::size_t calculate_offdiagonal_index(const std::size_t RowIndex, const std::size_t ColIndex)
{
	assert(RowIndex < NumPES && ColIndex < RowIndex);
	return RowIndex * (RowIndex - 1) / 2 + ColIndex;
}

/// @brief To construct a member in the array
/// @tparam T The type of the member
/// @tparam Args Types of arguments used in constructor
/// @tparam N Size of the array
/// @tparam I The index of the constructed member
/// @param array_args The arguments used in constructor
/// @return The constructed member
template <typename T, std::size_t N, std::size_t I, typename... Args>
	requires std::is_same_v<T, std::decay_t<T>> && (I < N)
T construct_member_in_array(std::array<Args, N>&... array_args)
{
	return T(std::forward<Args>(array_args[I])...);
}
/// @brief Implementation of construct an array of objects from array of arguments
/// @tparam T The constructed type
/// @tparam Args Types of arguments
/// @tparam I Indices from 0 to N-1, with array size of N
/// @param array_args Arguments of the array
/// @return An array of objects
template <typename T, std::size_t... I, typename... Args>
	requires std::is_same_v<std::make_index_sequence<sizeof...(I)>, std::index_sequence<I...>> && std::is_same_v<T, std::decay_t<T>>
std::array<T, sizeof...(I)> construct_array_impl(std::index_sequence<I...>, std::array<Args, sizeof...(I)>&&... array_args)
{
	return {construct_member_in_array<T, sizeof...(I), I>(array_args...)...};
}
/// @brief To construct array with given data
/// @tparam T Data type in the return array
/// @tparam N Size of the array
/// @tparam Args Type of arguments to construct the objects in result array
/// @param[in] array_args All the arguments to construct an object of type @p T
/// @sa fill_array()
/// @return An array of objects
template <typename T, std::size_t N, typename... Args>
std::array<std::decay_t<T>, N> construct_array(std::array<Args, N>&&... array_args)
{
	return construct_array_impl<std::decay_t<T>>(std::make_index_sequence<N>{}, std::forward<std::array<Args, N>>(array_args)...);
}

/// @brief Implementation of construct an array of same objects
/// @tparam T The constructed type
/// @tparam Args Types of arguments to construct the object
/// @tparam I Indices from 0 to N-1, with array size of N
/// @param args Arguments to construct the object
/// @return An array of same objects
template <typename T, std::size_t... I, typename... Args>
	requires std::is_same_v<std::make_index_sequence<sizeof...(I)>, std::index_sequence<I...>> && std::is_same_v<T, std::decay_t<T>>
std::array<T, sizeof...(I)> fill_array_impl(std::index_sequence<I...>, Args&... args)
{
	if constexpr (std::is_copy_constructible_v<T>)
	{
		const T& temp = T(std::forward<Args>(args)...);
		auto skip_index = [&temp](std::size_t) -> const T&
		{
			return temp;
		};
		return {skip_index(I)...};
	}
	else
	{
		auto skip_index = [&args...](std::size_t) -> T
		{
			return T(args...);
		};
		return {skip_index(I)...};
	}
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
	return fill_array_impl<std::decay_t<T>>(std::make_index_sequence<N>{}, args...);
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

	/// @brief Construct with given data
	/// @param[in] diagonal The data for diagonal elements
	/// @param[in] offdiagonal The data for off-diagonal elements
	QuantumStorage(DiagonalArrayType diagonal, OffDiagonalArrayType offdiagonal):
		diagonal_data(std::move(diagonal)),
		offdiagonal_data(std::move(offdiagonal))
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
	DiagDT& operator()(const std::size_t Index)
	{
		assert(Index < NumPES);
		return diagonal_data[Index];
	}
	/// @brief Read-only version of diagonal element access
	/// @param[in] Index Index of row (and thus column) of the element in density matrix. Must be a valid index (< @p NumPES )
	/// @return Data corresponding to the diagonal element
	ValOrCRef<DiagDT> operator()(const std::size_t Index) const
	{
		assert(Index < NumPES);
		return diagonal_data[Index];
	}
	/// @brief Editable version of off-diagonal (strictly lower-triangular) element access
	/// @param[in] RowIndex RowIndex Index of row of the element in density matrix. Must be a valid index (< @p NumPES )
	/// @param[in] ColIndex RowIndex Index of row of the element in density matrix. Must be a valid index and no more than @p RowIndex
	/// @return Data corresponding to the off-diagonal element
	OffDiagDT& operator()(const std::size_t RowIndex, const std::size_t ColIndex)
	{
		if constexpr (std::is_same_v<DiagDT, OffDiagDT>)
		{
			if (RowIndex == ColIndex)
			{
				assert(RowIndex < NumPES);
				return diagonal_data[RowIndex];
			}
		}
		return offdiagonal_data[calculate_offdiagonal_index(RowIndex, ColIndex)];
	}
	/// @brief Read-only version of off-diagonal (strictly lower-triangular) element access
	/// @param[in] RowIndex RowIndex Index of row of the element in density matrix. Must be a valid index (< @p NumPES )
	/// @param[in] ColIndex RowIndex Index of row of the element in density matrix. Must be a valid index and no more than @p RowIndex
	/// @return Data corresponding to the off-diagonal element
	ValOrCRef<OffDiagDT> operator()(const std::size_t RowIndex, const std::size_t ColIndex) const
	{
		if constexpr (std::is_convertible_v<DiagDT, OffDiagDT>)
		{
			if (RowIndex == ColIndex)
			{
				assert(RowIndex < NumPES);
				return diagonal_data[RowIndex];
			}
		}
		return offdiagonal_data[calculate_offdiagonal_index(RowIndex, ColIndex)];
	}

	friend bool operator==(const QuantumStorage<DiagDT, OffDiagDT>&, const QuantumStorage<DiagDT, OffDiagDT>&) = default;
	friend bool operator!=(const QuantumStorage<DiagDT, OffDiagDT>&, const QuantumStorage<DiagDT, OffDiagDT>&) = default;

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
	/// @brief The number of members, used for structed binding and @p get()
	static constexpr std::size_t NumMembers = 2;

	/// @brief Default constructor
	PhaseSpacePoint() = default;
	/// @brief Constructor with given data
	/// @param[in] R Phase space coordinates
	/// @param[in] DenMatElm Exact density matrix element
	/// @param[in] theta Adiabatic phase factor
	PhaseSpacePoint(const ClassicalPhaseVector& R, const std::complex<double> DenMatElm):
		r(R), rho(DenMatElm)
	{
	}

	/// @brief "Move" constructor
	/// @param[in] R Phase space coordinates
	/// @param[in] DenMatElm Exact Density matrix element
	/// @param[in] theta Adiabatic phase factor
	PhaseSpacePoint(ClassicalPhaseVector&& R, const std::complex<double> DenMatElm):
		r(std::move(R)), rho(DenMatElm)
	{
	}

	/// @brief To get an element: coordinates, density matrix element, or adiabatic phase factor
	/// @tparam I Index
	/// @return Reference to the value
	template <std::size_t I>
		requires(I < NumMembers)
	auto get() -> std::tuple_element<I, PhaseSpacePoint>::type&
	{
		if constexpr (I == 0)
		{
			return r;
		}
		else
		{
			return rho;
		}
	}

	/// @brief To get the value of an element: coordinates, density matrix, or adiabatic phase factor
	/// @tparam I Index
	/// @return The value
	template <std::size_t I>
		requires(I < NumMembers)
	auto get() const -> ValOrCRef<typename std::tuple_element<I, PhaseSpacePoint>::type>
	{
		if constexpr (I == 0)
		{
			return r;
		}
		else
		{
			return rho;
		}
	}

private:
	/// @brief Phase space coordinates
	ClassicalPhaseVector r;
	/// @brief Density matrix element without adiabatic phase factor
	std::complex<double> rho;
};

/// @brief Partial specialization for @p PhaseSpacePoint
template <>
struct std::tuple_size<PhaseSpacePoint>: std::integral_constant<std::decay_t<decltype(PhaseSpacePoint::NumMembers)>, PhaseSpacePoint::NumMembers>
{
};
/// @brief Partial specialization for @p PhaseSpacePoint
/// @tparam I Index of the element in @p PhaseSpacePoint
template <std::size_t I>
	requires(I < std::tuple_size<PhaseSpacePoint>::value)
struct std::tuple_element<I, PhaseSpacePoint>
{
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

/// @brief Sets of selected phase space points of one density matrix element
using ElementPoints = EigenVector<PhaseSpacePoint>;
/// @brief Sets of selected phase space points of all density matrix elements
using AllPoints = QuantumStorage<ElementPoints>;

#endif // !STORAGE_H
