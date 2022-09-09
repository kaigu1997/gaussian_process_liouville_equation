/// @file predict.cpp
/// @brief Implementation of predict.h

#include "stdafx.h"

#include "predict.h"

#include "complex_kernel.h"
#include "kernel.h"
#include "pes.h"
#include "storage.h"

/// @details The reduction is not combined in return because of STUPID clang-format.
ClassicalPhaseVector calculate_1st_order_average_one_surface(const ElementPoints& density)
{
	if (density.empty())
	{
		return ClassicalPhaseVector::Zero();
	}
	const ClassicalPhaseVector result =
		std::transform_reduce(
			std::execution::par_unseq,
			density.cbegin(),
			density.cend(),
			ClassicalPhaseVector(ClassicalPhaseVector::Zero()),
			std::plus<ClassicalPhaseVector>(),
			[](const PhaseSpacePoint& psp) -> ClassicalPhaseVector
			{
				return psp.get<0>();
			}
		);
	return result / density.size();
}

ClassicalPhaseVector calculate_standard_deviation_one_surface(const ElementPoints& density)
{
	if (density.empty())
	{
		return ClassicalPhaseVector::Zero();
	}
	const ClassicalPhaseVector sum_square_r =
		std::transform_reduce(
			std::execution::par_unseq,
			density.cbegin(),
			density.cend(),
			ClassicalPhaseVector(ClassicalPhaseVector::Zero()),
			std::plus<ClassicalPhaseVector>(),
			[](const PhaseSpacePoint& psp) -> ClassicalPhaseVector
			{
				return psp.get<0>().array().square();
			}
		);
	return (sum_square_r.array() / density.size() - calculate_1st_order_average_one_surface(density).array().square()).sqrt();
}

double calculate_total_energy_average_one_surface(
	const ElementPoints& density,
	const ClassicalVector<double>& mass,
	const std::size_t PESIndex
)
{
	if (density.empty())
	{
		return 0.0;
	}
	const double result =
		std::transform_reduce(
			std::execution::par_unseq,
			density.cbegin(),
			density.cend(),
			0.0,
			std::plus<double>(),
			[&mass, PESIndex](const PhaseSpacePoint& psp) -> double
			{
				const auto [x, p] = split_phase_coordinate(psp.get<0>());
				return p.dot((p.array() / mass.array()).matrix()) / 2.0 + adiabatic_potential(x)[PESIndex];
			}
		);
	return result / density.size();
}

QuantumVector<double> calculate_total_energy_average_each_surface(const AllPoints& density, const ClassicalVector<double>& mass)
{
	QuantumVector<double> result = QuantumVector<double>::Zero();
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		result[iPES] = calculate_total_energy_average_one_surface(density(iPES), mass, iPES);
	}
	return result;
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
			[&ElementDensity, &feature, &label](std::size_t iPoint) -> void
			{
				[[maybe_unused]] const auto& [r, rho, phi] = ElementDensity[iPoint];
				feature.col(iPoint) = r;
				label[iPoint] = rho;
			}
		);
		// this happens for coherence, where only real/imaginary part is important
		return std::make_tuple(feature, label);
	};
	AllTrainingSets result;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES <= iPES; jPES++)
		{
			result(iPES, jPES) = construct_element_training_set(iPES, jPES);
		}
	}
	return result;
}

Kernels::Kernels(
	const QuantumStorage<ParameterVector>& ParameterVectors,
	const AllTrainingSets& TrainingSets,
	const bool IsToCalculateError,
	const bool IsToCalculateAverage,
	const bool IsToCalculateDerivative
):
	BaseType(
		[=, &ParameterVectors, &TrainingSets](void)
		{
			std::array<ParameterVector, NumPES> diag_param = ParameterVectors.get_diagonal_data();
			std::array<ElementTrainingSet, NumPES> diag_ts = TrainingSets.get_diagonal_data();
			return std::make_tuple(
				std::move(diag_param),
				std::move(diag_ts),
				fill_array<bool, NumPES>(IsToCalculateError),
				fill_array<bool, NumPES>(IsToCalculateAverage),
				fill_array<bool, NumPES>(IsToCalculateDerivative)
			);
		}(),
		[=, &ParameterVectors, &TrainingSets]()
		{
			static const ParameterVector AllZeroCaseAlternative(ComplexKernel::NumTotalParameters, 1.0);
			static auto is_all_zero = [](const ParameterVector& params) -> bool
			{
				return std::all_of(params.cbegin(), params.cend(), std::bind(std::equal_to<double>{}, std::placeholders::_1, 0));
			};
			std::array<ParameterVector, NumOffDiagonalElements> offdiag_param = ParameterVectors.get_offdiagonal_data();
			std::array<ElementTrainingSet, NumOffDiagonalElements> offdiag_ts = TrainingSets.get_offdiagonal_data();
			for (std::size_t iElement = 0; iElement < NumOffDiagonalElements; iElement++)
			{
				if (is_all_zero(offdiag_param[iElement]))
				{
					offdiag_param[iElement] = AllZeroCaseAlternative;
					std::get<1>(offdiag_ts[iElement]).setZero();
				}
			}
			return std::make_tuple(
				std::move(offdiag_param),
				std::move(offdiag_ts),
				fill_array<bool, NumOffDiagonalElements>(IsToCalculateError),
				fill_array<bool, NumOffDiagonalElements>(IsToCalculateAverage),
				fill_array<bool, NumOffDiagonalElements>(IsToCalculateDerivative)
			);
		}()
	)
{
}

Kernels::Kernels(const QuantumStorage<ParameterVector>& ParameterVectors, const AllPoints& density):
	Kernels(ParameterVectors, construct_training_sets(density), true, true, false)
{
}

double Kernels::calculate_population(void) const
{
	double result = 0.0;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		result += (*this)(iPES).get_population();
	}
	return result;
}

ClassicalPhaseVector Kernels::calculate_1st_order_average(void) const
{
	ClassicalPhaseVector result = ClassicalPhaseVector::Zero();
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		result += (*this)(iPES).get_1st_order_average();
	}
	return result;
}

/// @details This function uses the population on each surface by analytical integral
/// with energy calculated by averaging the energy of density
double Kernels::calculate_total_energy_average(const QuantumVector<double>& Energies) const
{
	double result = 0.0;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		result += (*this)(iPES).get_population() * Energies[iPES];
	}
	return result;
}

/// @details The global purity is the weighted sum of purity of all elements, with weight 1 for diagonal and 2 for off-diagonal
double Kernels::calculate_purity() const
{
	double result = 0.0;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES <= iPES; jPES++)
		{
			if (iPES == jPES)
			{
				result += (*this)(iPES).get_purity();
			}
			else
			{
				result += 2.0 * (*this)(iPES, jPES).get_purity();
			}
		}
	}
	return result;
}

ParameterVector Kernels::population_derivative(void) const
{
	ParameterVector result;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		const Kernel::ParameterArray<double>& grad = (*this)(iPES).get_population_derivative();
		result.insert(result.cend(), grad.cbegin(), grad.cend());
	}
	return result;
}

ParameterVector Kernels::total_energy_derivative(const QuantumVector<double>& Energies) const
{
	ParameterVector result;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		Kernel::ParameterArray<double> grad = (*this)(iPES).get_population_derivative();
		for (double& d : grad)
		{
			d *= Energies[iPES];
		}
		result.insert(result.cend(), grad.cbegin(), grad.cend());
	}
	return result;
}

ParameterVector Kernels::purity_derivative() const
{
	ParameterVector result;
	for (std::size_t iPES = 0; iPES < NumPES; iPES++)
	{
		for (std::size_t jPES = 0; jPES <= iPES; jPES++)
		{
			if (iPES == jPES)
			{
				const Kernel::ParameterArray<double>& grad = (*this)(iPES).get_purity_derivative();
				result.insert(result.cend(), grad.cbegin(), grad.cend());
			}
			else
			{
				ComplexKernel::ParameterArray<double> grad = (*this)(iPES, jPES).get_purity_derivative();
				for (double& d : grad)
				{
					d *= 2;
				}
				result.insert(result.cend(), grad.cbegin(), grad.cend());
			}
		}
	}
	return result;
}
