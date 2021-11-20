/// @file opt.h
/// @brief Interface to functions of parameter optimization

#ifndef OPT_H
#define OPT_H

#include "kernel.h"
#include "mc.h"

/// @brief To store parameters, kernels and optimization algorithms to use.
/// And, to optimize parameters, and then predict density matrix and given point.
class Optimization final
{
public:
	/// @brief Constructor. Initial parameters, kernels and optimization algorithms needed
	/// @param[in] Params Parameters object containing position, momentum, etc
	/// @param[in] KernelTypes The vector containing all the kernel type used in optimization
	/// @param[in] LocalAlgorithm The optimization algorithm for local optimization
	/// @param[in] GlobalAlgorithm The optimization algorithm for global optimization
	Optimization(
		const Parameters& Params,
		const nlopt::algorithm LocalAlgorithm = nlopt::algorithm::LD_LBFGS,
		const nlopt::algorithm GlobalAlgorithm = nlopt::algorithm::G_MLSL_LDS);

	/// @brief To optimize parameters based on the given density
	/// @param[in] density The vector containing all known density matrix
	/// @param[in] mass Mass of classical degree of freedom
	/// @param[in] TotalEnergy The total energy of the system used for energy conservation
	/// @param[in] Purity The purity of the partial Wigner transformed density matrix
	/// @param[in] IsSmall The matrix that saves whether each element is small or not
	/// @return The total error (MSE, log likelihood, etc) of all elements of density matrix
	std::tuple<double, std::vector<int>> optimize(
		const EigenVector<PhaseSpacePoint>& density,
		const ClassicalVector<double>& mass,
		const double TotalEnergy,
		const double Purity,
		const QuantumMatrix<bool>& IsSmall);

	/// @brief To normalize the training set
	/// @param[inout] density The vector containing all known density matrix that needs normalization
	/// @param[in] IsSmall The matrix that saves whether each element is small or not
	/// @return The total population before normalization
	double normalize(EigenVector<PhaseSpacePoint>& density, const QuantumMatrix<bool>& IsSmall) const;

	/// @brief To update the TrainingFeatures and KInvLbls up-to-date
	/// @param[in] density The vector containing all known density matrix
	/// @param[in] IsSmall The matrix that saves whether each element is small or not
	void update_training_set(const EigenVector<PhaseSpacePoint>& density, const QuantumMatrix<bool>& IsSmall);

	/// @brief To predict one of the elements of the density matrix at the given phase space point
	/// @param[in] IsSmall The matrix that saves whether each element is small or not
	/// @param[in] x Position of classical degree of freedom
	/// @param[in] p Momentum of classical degree of freedom
	/// @param[in] ElementIndex The index of the density matrix element
	/// @return The element of the density matrix at the given point
	double predict_element(
		const QuantumMatrix<bool>& IsSmall,
		const ClassicalVector<double>& x,
		const ClassicalVector<double>& p,
		const int ElementIndex) const;

	/// @brief To predict the density matrix at the given phase space point
	/// @param[in] IsSmall The matrix that saves whether each element is small or not
	/// @param[in] x Position of classical degree of freedom
	/// @param[in] p Momentum of classical degree of freedom
	/// @return The density matrix at the given point
	QuantumMatrix<std::complex<double>> predict_matrix(
		const QuantumMatrix<bool>& IsSmall,
		const ClassicalVector<double>& x,
		const ClassicalVector<double>& p) const;

	/// @brief To predict one of the elements of the density matrix at the given phase space points
	/// @param[in] IsSmall The matrix that saves whether each element is small or not
	/// @param[in] x Positions of classical degree of freedom
	/// @param[in] p Momenta of classical degree of freedom
	/// @param[in] ElementIndex The index of the density matrix element
	/// @return The element of the density matrix at the given points
	Eigen::VectorXd predict_elements(
		const QuantumMatrix<bool>& IsSmall,
		const EigenVector<ClassicalVector<double>>& x,
		const EigenVector<ClassicalVector<double>>& p,
		const int ElementIndex) const;

	/// @brief To print the grids of a certain element of density matrix
	/// @param[in] PhaseGrids All the grids required to calculate in phase space
	/// @param[in] ElementIndex The index of the density matrix element
	/// @return A 1-by-N matrix, N is the number of required grids
	Eigen::VectorXd print_element(const QuantumMatrix<bool>& IsSmall, const Eigen::MatrixXd& PhaseGrids, const int ElementIndex) const;

	/// @brief To get the parameter of the corresponding element
	/// @param[in] ElementIndex The index of the density matrix element
	/// @return The parameters of the corresponding element of density matrix
	ParameterVector get_parameter(const int ElementIndex) const
	{
		return ParameterVectors[ElementIndex];
	}

	/// @brief To calculate the population of the given surface
	/// @param[in] IsSmall The matrix that saves whether each element is small or not
	/// @param[in] PESIndex The index of the potential energy surface, corresponding to (PESIndex, PESIndex) in density matrix
	/// @return The population of the given surface
	double calculate_population(const QuantumMatrix<bool>& IsSmall, const int PESIndex) const
	{
		if (IsSmall(PESIndex, PESIndex) == false)
		{
			return Kernels[PESIndex * (NumPES + 1)]->get_population();
		}
		else
		{
			return 0.0;
		}
	}

	/// @brief To calculate the average position and momentum of the given surface
	/// @param[in] IsSmall The matrix that saves whether each element is small or not
	/// @param[in] PESIndex The index of the potential energy surface, corresponding to (PESIndex, PESIndex) in density matrix
	/// @return <x> and <p> of the given surface
	ClassicalPhaseVector calculate_1st_order_average(const QuantumMatrix<bool>& IsSmall, const int PESIndex) const
	{
		if (IsSmall(PESIndex, PESIndex) == false)
		{
			return Kernels[PESIndex * (NumPES + 1)]->get_first_order_average() / Kernels[PESIndex * (NumPES + 1)]->get_population();
		}
		else
		{
			return ClassicalPhaseVector::Zero();
		}
	}

	/// @brief To calculate the purity of the density matrix by parameters
	/// @param[in] IsSmall The matrix that saves whether each element is small or not
	/// @return The purity of the overall partial Wigner transformed density matrix
	double calculate_purity(const QuantumMatrix<bool>& IsSmall) const;

private:
	// static variables
	static const double Tolerance; ///< Relative tolerance
	// local variables
	const int NumPoints;											///< The number of points selected for parameter optimization
	const ParameterVector InitialParameter;							///< The initial parameter for each of the element and for reopt
	std::vector<nlopt::opt> NLOptLocalMinimizers;					///< The vector containing all local NLOPT optimizers, one non-grad for each element
	std::vector<nlopt::opt> NLOptGlobalMinimizers;					///< The vector containing all global NLOPT optimizers
	std::array<ParameterVector, NumElements> ParameterVectors;		///< The parameters for all elements of density matrix
	std::array<Eigen::MatrixXd, NumElements> TrainingFeatures;		///< The training features (phase space coordinates) of each density matrix element
	std::array<std::shared_ptr<const Kernel>, NumElements> Kernels; ///< The kernels

	/// @brief To optimize parameters of each density matrix element based on the given density
	/// @param[inout] density The vector containing all known density matrix
	/// @param[in] IsSmall The matrix that saves whether each element is small or not
	/// @param[inout] minimizers The minimizers to use
	/// @return A vector of int, containing the optimization steps of each element
	std::vector<int> optimize_elementwise(
		const EigenVector<PhaseSpacePoint>& density,
		const QuantumMatrix<bool>& IsSmall,
		std::vector<nlopt::opt>& minimizers);

	/// @brief To optimize parameters of diagonal elements based on the given density and constraint regularizations
	/// @param[inout] density The vector containing all known density matrix
	/// @param[in] mass Mass of classical degree of freedom
	/// @param[in] TotalEnergy The total energy of the system used for energy conservation
	/// @param[in] Purity The purity of the partial Wigner transformed density matrix
	/// @param[in] IsSmall The matrix that saves whether each element is small or not
	/// @param[inout] minimizers The minimizer to use
	/// @return The total error (MSE, log likelihood, etc, including regularizations) of all diagonal elements of density matrix
	std::tuple<double, int> optimize_diagonal(
		const EigenVector<PhaseSpacePoint>& density,
		const ClassicalVector<double>& mass,
		const double TotalEnergy,
		const double Purity,
		const QuantumMatrix<bool>& IsSmall,
		nlopt::opt& minimizer);

	/// @brief To optimize parameters of off-diagonal element based on the given density and purity
	/// @param[inout] density The vector containing all known density matrix
	/// @param[in] Purity The purity of the partial Wigner transformed density matrix, which should conserve
	/// @param[in] IsSmall The matrix that saves whether each element is small or not
	/// @param[inout] minimizers The minimizers to use
	/// @return The total error (MSE, log likelihood, etc, including regularizations) of all off-diagonal elements of density matrix
	std::tuple<double, int> optimize_offdiagonal(
		const EigenVector<PhaseSpacePoint>& density,
		const double Purity,
		const QuantumMatrix<bool>& IsSmall,
		nlopt::opt& minimizer);

	/// @brief To judge whether reoptimization is needed
	/// @param[in] density The vector containing all known density matrix
	/// @param[in] IsSmall The matrix that saves whether each element is small or not
	/// @return True for reopt, false for not
	bool is_reoptimize(const EigenVector<PhaseSpacePoint>& density, const QuantumMatrix<bool>& IsSmall);
};

/// @brief To calculate the average position and momentum of one diagonal element by monte carlo integration
/// @param[in] density The selected density matrices
/// @param[in] PESIndex The index of the potential energy surface, corresponding to (PESIndex, PESIndex) in density matrix
/// @return Average position and momentum
ClassicalPhaseVector calculate_1st_order_average(const EigenVector<PhaseSpacePoint>& density, const int PESIndex);

/// @brief To calculate the average kinetic energy of one diagonal element by monte carlo integration
/// @param[in] density The selected density matrices
/// @param[in] PESIndex The index of the potential energy surface, corresponding to (PESIndex, PESIndex) in density matrix
/// @param[in] mass Mass of classical degree of freedom
/// @return Average kinetic energy
double calculate_kinetic_average(const EigenVector<PhaseSpacePoint>& density, const int PESIndex, const ClassicalVector<double>& mass);

/// @brief To calculate the average potential energy of one diagonal element by monte carlo integration
/// @param[in] density The selected density matrices
/// @param[in] PESIndex The index of the potential energy surface, corresponding to (PESIndex, PESIndex) in density matrix
/// @return Average potential energy
double calculate_potential_average(const EigenVector<PhaseSpacePoint>& density, const int PESIndex);

#endif // !OPT_H
