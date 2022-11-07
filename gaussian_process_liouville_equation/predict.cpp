/// @file predict.cpp
/// @brief Implementation of predict.h

#include "stdafx.h"

#include "predict.h"

#include "complex_kernel.h"
#include "kernel.h"
#include "pes.h"
#include "storage.h"

/// @brief Type that could add the same type and convert to its type
/// @tparam T The type
template <typename T>
concept Addable = requires(T a, T b) { {a + b} -> std::convertible_to<T>; };

/// @brief To add up tuple
/// @tparam T Types that form the tuple
/// @tparam I Indices corresponding to T, starting from 0
/// @param t1 The first tuple
/// @param t2 The second tuple
/// @return The sum of the two tuple
template <typename... T, std::size_t... I>
	requires std::is_same_v<std::make_index_sequence<sizeof...(I)>, std::index_sequence<I...>> && (sizeof...(T) == sizeof...(I)) && (... && Addable<T>)
static inline std::tuple<T...> sum_tuple(
	const std::tuple<T...>& t1,
	const std::tuple<T...>& t2,
	std::index_sequence<I...>
)
{
	return std::make_tuple(std::get<I>(t1) + std::get<I>(t2)...);
}

/// @brief To calculate the sum of given observables from the given points
/// @tparam T Types of the observables
/// @param density density The selected points in phase space
/// @param observable_calculator How the observables are calculated from the point information
/// @param initial_values The initial value of the observables to sum up from, generally 0
/// @return The sum of given observables from the given points
template <typename... T>
	requires(... && Addable<T>)
static inline std::tuple<T...> calculate_observable_sum(
	const ElementPoints& density,
	const std::function<T(const PhaseSpacePoint&)>&... observable_calculator,
	ValOrCRef<T>... initial_values
)
{
	return std::transform_reduce(
		std::execution::par_unseq,
		density.cbegin(),
		density.cend(),
		std::tuple<T...>(std::forward<T>(initial_values)...),
		[](const std::tuple<T...>& t1, const std::tuple<T...>& t2) -> std::tuple<T...>
		{
			return sum_tuple(t1, t2, std::make_index_sequence<sizeof...(T)>{});
		},
		[&observable_calculator...](const PhaseSpacePoint& psp) -> std::tuple<T...>
		{
			return std::make_tuple(observable_calculator(psp)...);
		}
	);
}

QuantumVector<double> calculate_population_each_surface(const AllPoints& density)
{
	QuantumVector<double> result = QuantumVector<double>::Zero();
	for (const std::size_t iPES : std::ranges::iota_view{0ul, NumPES})
	{
		const ElementPoints& diag_ts = density(iPES);
		if (!diag_ts.empty())
		{
			result[iPES] = std::get<0>(
				calculate_observable_sum<double>(
					diag_ts,
					[](const PhaseSpacePoint& psp) -> double
					{
						return psp.get<1>().real();
					},
					0.0
				)
			);
		}
		// else 0
	}
	return result / result.sum(); // normalize
}

ClassicalPhaseVector calculate_1st_order_average_one_surface(const ElementPoints& density)
{
	assert(!density.empty());
	const auto [ppl, r] = calculate_observable_sum<double, ClassicalPhaseVector>(
		density,
		[](const PhaseSpacePoint& psp) -> double
		{
			return psp.get<1>().real();
		},
		[](const PhaseSpacePoint& psp) -> ClassicalPhaseVector
		{
			const auto& [r, rho] = psp;
			return r * rho.real();
		},
		0.0,
		ClassicalPhaseVector::Zero()
	);
	return r / ppl;
}

ClassicalPhaseVector calculate_standard_deviation_one_surface(const ElementPoints& density)
{
	assert(!density.empty());
	const auto [sum_r, sum_square_r] = calculate_observable_sum<ClassicalPhaseVector, ClassicalPhaseVector>(
		density,
		[](const PhaseSpacePoint& psp) -> ClassicalPhaseVector
		{
			return psp.get<0>();
		},
		[](const PhaseSpacePoint& psp) -> ClassicalPhaseVector
		{
			return psp.get<0>().array().square();
		},
		ClassicalPhaseVector::Zero(),
		ClassicalPhaseVector::Zero()
	);
	return (sum_square_r.array() / density.size() - (sum_r / density.size()).array().square()).sqrt();
}

ClassicalPhaseVector calculate_1st_order_average_all_surface(const AllPoints& density)
{
	ClassicalPhaseVector r_mc = ClassicalPhaseVector::Zero();
	double ppl_mc = 0.0;
	for (const ElementPoints& diag_ts : density.get_diagonal_data())
	{
		if (!diag_ts.empty())
		{
			const auto [ppl_elm, r_elm] = calculate_observable_sum<double, ClassicalPhaseVector>(
				diag_ts,
				[](const PhaseSpacePoint& psp) -> double
				{
					return psp.get<1>().real();
				},
				[](const PhaseSpacePoint& psp) -> ClassicalPhaseVector
				{
					const auto& [r, rho] = psp;
					return r * rho.real();
				},
				0.0,
				ClassicalPhaseVector::Zero()
			);
			r_mc += r_elm;
			ppl_mc += ppl_elm;
		}
	}
	return r_mc / ppl_mc;
}

double calculate_total_energy_average_one_surface(
	const ElementPoints& density,
	const ClassicalVector<double>& mass,
	const std::size_t PESIndex
)
{
	assert(!density.empty());
	const auto [ppl, eng] = calculate_observable_sum<double, double>(
		density,
		[](const PhaseSpacePoint& psp) -> double
		{
			return psp.get<1>().real();
		},
		[&mass, PESIndex](const PhaseSpacePoint& psp) -> double
		{
			const auto& [r, rho] = psp;
			const auto [x, p] = split_phase_coordinate(r);
			return ((p.array().square() / mass.array()).sum() / 2.0 + adiabatic_potential(x)[PESIndex]) * rho.real();
		},
		0.0,
		0.0
	);
	return eng / ppl;
}

QuantumVector<double> calculate_total_energy_average_each_surface(const AllPoints& density, const ClassicalVector<double>& mass)
{
	QuantumVector<double> result = QuantumVector<double>::Zero();
	for (const std::size_t iPES : std::ranges::iota_view{0ul, NumPES})
	{
		result[iPES] = density(iPES).empty() ? 0.0 : calculate_total_energy_average_one_surface(density(iPES), mass, iPES);
	}
	return result;
}

double calculate_total_energy_average_all_surface(const AllPoints& density, const ClassicalVector<double>& mass)
{
	double eng_mc = 0.0, ppl_mc = 0.0;
	for (const std::size_t iPES : std::ranges::iota_view{0ul, NumPES})
	{
		const ElementPoints& diag_ts = density(iPES);
		if (!diag_ts.empty())
		{
			const auto [ppl_elm, eng_elm] = calculate_observable_sum<double, double>(
				diag_ts,
				[](const PhaseSpacePoint& psp) -> double
				{
					return psp.get<1>().real();
				},
				[&mass, iPES](const PhaseSpacePoint& psp) -> double
				{
					const auto& [r, rho] = psp;
					const auto [x, p] = split_phase_coordinate(r);
					return ((p.array().square() / mass.array()).sum() / 2.0 + adiabatic_potential(x)[iPES]) * rho.real();
				},
				0.0,
				0.0
			);
			eng_mc += eng_elm;
			ppl_mc += ppl_elm;
		}
	}
	return eng_mc / ppl_mc;
}

/// @details This function calculates the relative purity of each element (not exact purity). @n
/// To calculate exact purity, a factor of current total purity vs initial total purity times exact initial purity is needed.
QuantumMatrix<double> calculate_purity_each_element(const AllPoints& density)
{
	QuantumMatrix<double> result = QuantumMatrix<double>::Zero();
	for (const std::size_t iPES : std::ranges::iota_view{0ul, NumPES})
	{
		for (const std::size_t jPES : std::ranges::iota_view{0ul, iPES + 1})
		{
			result(iPES, jPES) = std::get<0>(
				calculate_observable_sum<double>(
					density(iPES, jPES),
					[](const PhaseSpacePoint& psp) -> double
					{
						return std::norm(psp.get<1>());
					},
					0.0
				)
			);
		}
	}
	return result.selfadjointView<Eigen::Lower>();
}

AllTrainingSets construct_training_sets(const AllPoints& density)
{
	auto construct_element_training_set = [&density](const std::size_t iPES, const std::size_t jPES) -> ElementTrainingSet
	{
		const ElementPoints& ElementDensity = density(iPES, jPES);
		const std::size_t NumPoints = ElementDensity.size();
		// construct training feature (PhaseDim*N) and training labels (N*1)
		PhasePoints feature = PhasePoints::Zero(PhaseDim, NumPoints);
		Eigen::VectorXcd label = Eigen::VectorXd::Zero(NumPoints);
		// first insert original points
		const auto indices = xt::arange(NumPoints);
		std::for_each(
			std::execution::par_unseq,
			indices.cbegin(),
			indices.cend(),
			[&ElementDensity, &feature, &label](const std::size_t iPoint) -> void
			{
				const auto& [r, rho] = ElementDensity[iPoint];
				feature.col(iPoint) = r;
				label[iPoint] = rho;
			}
		);
		// this happens for coherence, where only real/imaginary part is important
		return std::make_tuple(feature, label);
	};
	AllTrainingSets result;
	for (const std::size_t iPES : std::ranges::iota_view{0ul, NumPES})
	{
		for (const std::size_t jPES : std::ranges::iota_view{0ul, iPES + 1})
		{
			result(iPES, jPES) = construct_element_training_set(iPES, jPES);
		}
	}
	return result;
}

/// @brief To construct the kernels for diagonal elements
/// @tparam I Indices for diagonal elements
/// @param DiagonalParams Parameters for diagonal elements used in the kernel
/// @param DiagonalTrainingSets The feature and label of diagonal elements for training
/// @param[in] IsToCalculateError Whether to calculate LOOCV squared error or not
/// @param[in] IsToCalculateAverage Whether to calculate averages (@<1@>, @<r@>, and purity) or not
/// @param[in] IsToCalculateDerivative Whether to calculate derivative of each kernel or not
/// @return Array of diagonal kernels or @p std::nullopt if 0 populated
template <std::size_t... I>
	requires std::is_same_v<std::make_index_sequence<sizeof...(I)>, std::index_sequence<I...>> && (sizeof...(I) == NumPES)
static std::array<std::optional<TrainingKernel>, NumPES> construct_diagonal_kernels(
	const std::array<ParameterVector, NumPES>& DiagonalParams,
	const std::array<ElementTrainingSet, NumPES>& DiagonalTrainingSets,
	const bool IsToCalculateError,
	const bool IsToCalculateAverage,
	const bool IsToCalculateDerivative,
	std::index_sequence<I...>
)
{
	auto make_single_kernel =
		[&DiagonalParams,
		 &DiagonalTrainingSets,
		 IsToCalculateError,
		 IsToCalculateAverage,
		 IsToCalculateDerivative](const std::size_t II) -> std::optional<TrainingKernel>
	{
		if (std::get<0>(DiagonalTrainingSets[II]).size() != 0)
		{
			return TrainingKernel(DiagonalParams[II], DiagonalTrainingSets[II], IsToCalculateError, IsToCalculateAverage, IsToCalculateDerivative);
		}
		else
		{
			return std::nullopt;
		}
	};
	return {make_single_kernel(I)...};
}

/// @brief To construct the complex kernels for off-diagonal elements
/// @tparam I Indices for off-diagonal elements
/// @param OffDiagonalParams Parameters for off-diagonal elements used in the complex kernel
/// @param OffDiagonalTrainingSets The feature and label of off-diagonal elements for training
/// @param[in] IsToCalculateError Whether to calculate LOOCV squared error or not
/// @param[in] IsToCalculateAverage Whether to calculate averages (@<1@>, @<r@>, and purity) or not
/// @param[in] IsToCalculateDerivative Whether to calculate derivative of each kernel or not
/// @return Array of off-diagonal kernels or @p std::nullopt if 0 populated
template <std::size_t... I>
	requires std::is_same_v<std::make_index_sequence<sizeof...(I)>, std::index_sequence<I...>> && (sizeof...(I) == NumOffDiagonalElements)
static std::array<std::optional<TrainingComplexKernel>, NumOffDiagonalElements> construct_offdiagonal_kernels(
	const std::array<ParameterVector, NumOffDiagonalElements>& OffDiagonalParams,
	const std::array<ElementTrainingSet, NumOffDiagonalElements>& OffDiagonalTrainingSets,
	const bool IsToCalculateError,
	const bool IsToCalculateAverage,
	const bool IsToCalculateDerivative,
	std::index_sequence<I...>
)
{
	static auto is_all_zero = [](const ParameterVector& params) -> bool
	{
		return std::all_of(params.cbegin(), params.cend(), std::bind(std::equal_to<double>{}, std::placeholders::_1, 0));
	};
	auto make_single_complex_kernel =
		[&OffDiagonalParams,
		 &OffDiagonalTrainingSets,
		 IsToCalculateError,
		 IsToCalculateAverage,
		 IsToCalculateDerivative](const std::size_t II) -> std::optional<TrainingComplexKernel>
	{
		if (std::get<0>(OffDiagonalTrainingSets[II]).size() != 0 && !is_all_zero(OffDiagonalParams[II]))
		{
			return TrainingComplexKernel(OffDiagonalParams[II], OffDiagonalTrainingSets[II], IsToCalculateError, IsToCalculateAverage, IsToCalculateDerivative);
		}
		else
		{
			return std::nullopt;
		}
	};
	return {make_single_complex_kernel(I)...};
}

TrainingKernels::TrainingKernels(
	const QuantumStorage<ParameterVector>& ParameterVectors,
	const AllTrainingSets& TrainingSets,
	const bool IsToCalculateError,
	const bool IsToCalculateAverage,
	const bool IsToCalculateDerivative
):
	BaseType(
		construct_diagonal_kernels(
			ParameterVectors.get_diagonal_data(),
			TrainingSets.get_diagonal_data(),
			IsToCalculateError,
			IsToCalculateAverage,
			IsToCalculateDerivative,
			std::make_index_sequence<NumPES>{}
		),
		construct_offdiagonal_kernels(
			ParameterVectors.get_offdiagonal_data(),
			TrainingSets.get_offdiagonal_data(),
			IsToCalculateError,
			IsToCalculateAverage,
			IsToCalculateDerivative,
			std::make_index_sequence<NumOffDiagonalElements>{}
		)
	)
{
}

TrainingKernels::TrainingKernels(const QuantumStorage<ParameterVector>& ParameterVectors, const AllPoints& density):
	TrainingKernels(ParameterVectors, construct_training_sets(density), true, true, false)
{
}

double TrainingKernels::calculate_population(void) const
{
	double result = 0.0;
	for (const std::optional<TrainingKernel>& opt_diag_kernel : BaseType::get_diagonal_data())
	{
		if (opt_diag_kernel.has_value())
		{
			result += opt_diag_kernel->get_population();
		}
	}
	return result;
}

ClassicalPhaseVector TrainingKernels::calculate_1st_order_average(void) const
{
	ClassicalPhaseVector result = ClassicalPhaseVector::Zero();
	for (const std::optional<TrainingKernel>& opt_diag_kernel : BaseType::get_diagonal_data())
	{
		if (opt_diag_kernel.has_value())
		{
			result += opt_diag_kernel->get_1st_order_average();
		}
	}
	return result;
}

/// @details This function uses the population on each surface by analytical integral
/// with energy calculated by averaging the energy of density
double TrainingKernels::calculate_total_energy_average(const QuantumVector<double>& Energies) const
{
	double result = 0.0;
	for (const std::size_t iPES : std::ranges::iota_view{0ul, NumPES})
	{
		const std::optional<TrainingKernel>& opt_diag_kernel = (*this)(iPES);
		if (opt_diag_kernel.has_value())
		{
			// with population by parameters
			result += opt_diag_kernel->get_population() * Energies[iPES];
		}
	}
	return result;
}

/// @details The global purity is the weighted sum of purity of all elements, with weight 1 for diagonal and 2 for off-diagonal
double TrainingKernels::calculate_purity() const
{
	double result = 0.0;
	for (const std::size_t iPES : std::ranges::iota_view{0ul, NumPES})
	{
		for (const std::size_t jPES : std::ranges::iota_view{0ul, iPES + 1})
		{
			if (iPES == jPES)
			{
				if ((*this)(iPES).has_value())
				{
					result += (*this)(iPES)->get_purity();
				}
			}
			else
			{
				if ((*this)(iPES, jPES).has_value())
				{
					result += 2.0 * (*this)(iPES, jPES)->get_purity();
				}
			}
		}
	}
	return result;
}

ParameterVector TrainingKernels::population_derivative(void) const
{
	ParameterVector result(NumPES * KernelBase::NumTotalParameters, 0.0);
	for (const std::size_t iPES : std::ranges::iota_view{0ul, NumPES})
	{
		const std::optional<TrainingKernel> opt_diag_kernel = (*this)(iPES);
		if (opt_diag_kernel.has_value())
		{
			const KernelBase::ParameterArray<double>& ppl_grad = opt_diag_kernel->get_population_derivative();
			std::copy(
				std::execution::par_unseq,
				ppl_grad.cbegin(),
				ppl_grad.cend(),
				result.begin() + iPES * KernelBase::NumTotalParameters
			);
		}
		// else 0
	}
	return result;
}

ParameterVector TrainingKernels::total_energy_derivative(const QuantumVector<double>& Energies) const
{
	ParameterVector result(NumPES * KernelBase::NumTotalParameters, 0.0);
	for (const std::size_t iPES : std::ranges::iota_view{0ul, NumPES})
	{
		const std::optional<TrainingKernel>& opt_diag_kernel = (*this)(iPES);
		if (opt_diag_kernel.has_value())
		{
			// with population derivative by parameters
			const KernelBase::ParameterArray<double> ppl_grad = opt_diag_kernel->get_population_derivative();
			std::transform(
				std::execution::par_unseq,
				ppl_grad.cbegin(),
				ppl_grad.cend(),
				result.begin() + iPES * KernelBase::NumTotalParameters,
				[SurfaceEnergy = Energies[iPES]](const double d) -> double
				{
					return d * SurfaceEnergy;
				}
			);
		}
		// else 0
	}
	return result;
}

ParameterVector TrainingKernels::purity_derivative() const
{
	ParameterVector result(NumTotalParameters, 0.0);
	std::size_t iParam = 0;
	for (const std::size_t iPES : std::ranges::iota_view{0ul, NumPES})
	{
		for (const std::size_t jPES : std::ranges::iota_view{0ul, iPES + 1})
		{
			if (iPES == jPES)
			{
				const std::optional<TrainingKernel> opt_diag_kernel = (*this)(iPES);
				if (opt_diag_kernel.has_value())
				{
					const KernelBase::ParameterArray<double>& purity_deriv = opt_diag_kernel->get_purity_derivative();
					std::copy(
						std::execution::par_unseq,
						purity_deriv.cbegin(),
						purity_deriv.cend(),
						result.begin() + iParam
					);
				}
				// else 0
				iParam += KernelBase::NumTotalParameters;
			}
			else
			{
				const std::optional<TrainingComplexKernel> opt_offdiag_kernel = (*this)(iPES, jPES);
				if (opt_offdiag_kernel.has_value())
				{
					const ComplexKernelBase::ParameterArray<double>& purity_deriv = opt_offdiag_kernel->get_purity_derivative();
					std::transform(
						std::execution::par_unseq,
						purity_deriv.cbegin(),
						purity_deriv.cend(),
						result.begin() + iParam,
						[](const double d) -> double
						{
							return d * 2;
						}
					);
				}
				// else 0
				iParam += ComplexKernelBase::NumTotalParameters;
			}
		}
	}
	return result;
}
