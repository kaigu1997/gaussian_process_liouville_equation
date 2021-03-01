/// @file evolve.cpp
/// @brief Implementation of evolve.h

#include "stdafx.h"

#include "evolve.h"

#include "pes.h"

/// @brief Calculate the \f$\hat{V}'\f$ operator/matrix
/// @param[in] x Position of classical degree of freedom
/// @param[in] p Momentum of classical degree of freedom
/// @param[in] mass Mass of classical degree of freedom
/// @return The \f$\hat{V}'\f$ operator/matrix
/// @details This function calculated
///
/// \f[
/// \hat{V}'(\vec{R},\vec{P})=\hat{V}_W(\vec{R})-\mathrm{i}\hbar\vec{v}\cdot\vec{D}(\vec{R})
/// \f]
///
/// under the adiabatic basis. Here, \f$ v_i=\frac{P_i}{m_i} \f$,
/// \f$\hat{V}_W\f$ is the PWT quantum subsystem potential operator/matrix,
/// \f$\vec{D}\f$ is the non-adiabatic coupling under adiabatic basis.
static QuantumComplexMatrix get_potential_prime(
	const ClassicalDoubleVector& x,
	const ClassicalDoubleVector& p,
	const ClassicalDoubleVector& mass)
{
	using namespace std::literals::complex_literals; // for using 1.0i
	const Tensor3d& NAC = adiabatic_coupling(x);
	QuantumComplexMatrix result = adiabatic_potential(x).asDiagonal();
	for (int i = 0; i < Dim; i++)
	{
		result -= hbar * p[i] / mass[i] * NAC[i] * 1.0i;
	}
	// keep it self-adjoint
	make_self_adjoint(result);
	return result;
}

/// @details For quantum Liouville operator,
///
/// \f[
/// -\mathrm{i}\hat{\mathcal{L}}^Q\hat{\rho}_W
/// =-\frac{i}{\hbar}\left[\hat{V}_W(\vec{R})-\mathrm{i}\hbar\vec{v}\cdot\vec{D}(\vec{R}),\hat{\rho}_W\right]
/// \f]
///
/// If denote
///
/// \f[
/// \hat{V}'(\vec{R},\vec{P})=\hat{V}_W(\vec{R})-\mathrm{i}\hbar\vec{v}\cdot\vec{D}(\vec{R})
/// \f]
///
/// then
///
/// \f[
/// \mathrm{e}^{-\mathrm{i}\hat{\mathcal{L}}^Q\mathrm{d}t}\hat{\rho}_W(\vec{R},\vec{P})
/// =\mathrm{exp}\left(-\frac{\mathrm{i}\hat{V}'\mathrm{d}t}{\hbar}\right)
/// \hat{\rho}_W(\vec{R},\vec{P})\mathrm{exp}\left(\frac{\mathrm{i}\hat{V}'\mathrm{d}t}{\hbar}\right)
/// \f]
///
/// If the \f$ C \f$ matrix is used for diagonalize \f$ \hat{V}' \f$,
/// or \f$ C^{\dagger}\hat{V}'C=\hat{V}'_{\mathrm{diag}} \f$, then
///
///
/// \f[
/// \mathrm{e}^{-\mathrm{i}\hat{\mathcal{L}}^Q\mathrm{d}t}\hat{\rho}_W(\vec{R},\vec{P})
/// =C\mathrm{exp}\left(-\frac{\mathrm{i}\hat{V}'_{\mathrm{diag}}\mathrm{d}t}{\hbar}\right)C^{\dagger}
/// \hat{\rho}_W(\vec{R},\vec{P})
/// C\mathrm{exp}\left(\frac{\mathrm{i}\hat{V}'_{\mathrm{diag}}\mathrm{d}t}{\hbar}\right)C^{\dagger}
/// \f]
///
/// this gives the evolution of PWTDM under quantum Liouville superoperator.
///
/// This function will not change the number of selected phase points and their PWTDMs.
///
/// Besides, under Trotter expansion, if this superoperator will be done n times,
/// the dt in the parameter list will be global dt over n.
void quantum_liouville(EvolvingDensity& density, const ClassicalDoubleVector& mass, const double dt)
{
	using namespace std::literals::complex_literals; // for using 1.0i
	for (int i = 0; i < density.size(); i++)
	{
		// alias names
		const ClassicalDoubleVector& x = std::get<0>(density[i]);
		const ClassicalDoubleVector& p = std::get<1>(density[i]);
		QuantumComplexMatrix& rho = std::get<2>(density[i]);
		// get V'
		const QuantumComplexMatrix& v_prime = get_potential_prime(x, p, mass);
		// diagonalize this self-adjoint matrix
		Eigen::SelfAdjointEigenSolver<QuantumComplexMatrix> solver(v_prime);
		const QuantumComplexMatrix& TransformMatrix = solver.eigenvectors();
		const QuantumDoubleVector& EigenValues = solver.eigenvalues();
		// rho(t+dt)=C*exp(-iV't/hbar)*C^T*rho*C*exp(iV't/hbar)*C^T
		// rho is self-adjoint
		rho = TransformMatrix * (EigenValues * dt / hbar * -1.0i).array().exp().matrix().asDiagonal() * TransformMatrix.adjoint()
			* rho * TransformMatrix * (EigenValues * dt / hbar * 1.0i).array().exp().matrix().asDiagonal() * TransformMatrix.adjoint();
		make_self_adjoint(rho);
	}
}

/// @details For classical position Liouville superoperator,
///
/// \f{eqnarray*}{
/// &&-\mathrm{i}\hat{\mathcal{L}}^R\hat{\rho}_W=-\vec{v}\frac{\partial\hat{\rho}_W}{\partial\vec{R}} \\
/// \Rightarrow&&\mathrm{e}^{-\mathrm{i}\hat{\mathcal{L}}^R\mathrm{d}t}\hat{\rho}_W(\vec{R},\vec{P})
/// =\hat{\rho}_W(\vec{R}+\vec{v}\mathrm{d}t,\vec{P})
/// \f}
///
/// therefore, simple evolve the position following the Newton rule.
/// In other words, for each direction i, xi_new = xi_old + pi/mi*dt.
///
/// This function will not change the number of selected phase points and their PWTDMs.
///
/// Besides, under Trotter expansion, if this superoperator will be done n times,
/// the dt in the parameter list will be global dt over n.
void classical_position_liouville(EvolvingDensity& density, const ClassicalDoubleVector& mass, const double dt)
{
	for (int i = 0; i < density.size(); i++)
	{
		// alias names; do not need PWTDM
		ClassicalDoubleVector& x = std::get<0>(density[i]);
		const ClassicalDoubleVector& p = std::get<1>(density[i]);
		x = x.array() + dt * p.array() / mass.array();
	}
}

/// @brief Doing the basis transformation for the whole selected PWTDMs
/// @param[inout] density The selected phase points with their PWTDMs
/// @param[in] idx_from The index indicating the representation rho in
/// @param[in] idx_to The index indicating the representation the return value in
/// @details The indices (to and from) are in the range [0, Dim]. [0, Dim) indicates
/// it is one of the force basis, and idx == Dim indicates adiabatic basis.
static void whole_density_basis_transformation(
	EvolvingDensity& density,
	const int idx_from,
	const int idx_to)
{
	for (int i = 0; i < density.size(); i++)
	{
		const ClassicalDoubleVector& x = std::get<0>(density[i]);
		QuantumComplexMatrix& rho = std::get<2>(density[i]);
		rho = basis_transform(rho, x, idx_from, idx_to);
	}
}

/// @brief To do the classical momentum Liouville evolution on one direction on one PWTDM
/// @param[in] psp A phase space point with its PWTDM
/// @param[in] idx_force The index of the force direction
/// @param[in] dt The time-step each PWTDM will evolve
/// @return A vector of phase space points with their PWTDMs which is evolved from the input phase space point
/// @details This function will separate the PWTDM into at most \f$ \frac{N(N+1)}{2} \f$ ones, where \f$ N \f$
/// is the number of PES. For each element, the momentum will change \f$ \frac{f_a+f_b}{2}\mathrm{d}t \f$,
/// where the indices is the index of the element in PWTDM. If some of the elements have the same force,
/// they will be combined to evolve.
EvolvingDensity evolve_one_force_on_one_density(
	const PhaseSpacePoint& psp,
	const int idx_force,
	const double dt)
{
	EvolvingDensity result;
	// alias names
	const auto& [x, p, rho] = psp;
	// get eigen forces
	const QuantumDoubleVector& force = force_basis_force(x, idx_force);
	// prepare for evolve
	ClassicalDoubleVector p_new;
	QuantumComplexMatrix rho_new(NumPES, NumPES);
	std::map<double, int> used_forces; // for saving forces that have been used with the index when it was used
	// loop over all lower-triangular elements, in this order: (0,0)->(1,0)->(2,0)->(1,1)->(3,0)->(2,1)->...
	// first index begin from min(sum, size-1), i.e., on left edge or bottom edge
	for (int sum = 0; sum <= 2 * (NumPES - 1); sum++)
	{
		for (int i = std::min(sum, NumPES - 1), j = sum - i; i >= j; i--, j++)
		{
			const double f = (force[i] + force[j]) / 2.0;
			[[maybe_unused]] const auto& [iterator, insert_success] = used_forces.insert(std::make_pair(f, result.size()));
			if (insert_success == true)
			{
				// successfully insert, meaning there is no such force
				// so the momentum needs evolved, PWTDM not
				p_new = p;
				p_new[idx_force] += f * dt;
				rho_new = QuantumComplexMatrix::Zero(NumPES, NumPES);
				rho_new(i, j) = rho(i, j);
				result.push_back(std::make_tuple(x, p_new, rho_new.selfadjointView<Eigen::Lower>()));
			}
			else
			{
				// unsuccessfully insert, meaning such force exists
				// so used the already calculated momentum, and update PWTDM
				const int idx = iterator->second;
				rho_new = std::get<2>(result[idx]);
				rho_new(i, j) = rho(i, j);
				std::get<2>(result[idx]) = rho_new.selfadjointView<Eigen::Lower>();
			}
		}
	}
	return result;
}

/// @brief To recursively calculate the momentum splitting of each component, and evolve on one direction
/// @param[inout] density The selected phase points with their PWTDMs, generally each point will generate more tha one new PWTDMs
/// @param[in] direction The direction of momentum
/// @param[in] dt The time-step this superoperator will evolve
/// @details This function do trotter expansion on all components of momentum.
/// For example, if there are 3 directions, this functions leads to the order of 3(1/4) 2(1/2) 3(1/4) 1(1) 3(1/4) 2(1/2) 3(1/4)
/// The value in parentheses is the time interval for that calling.
void trotter_expansion_for_classical_momentum_liouville_splitting(EvolvingDensity& density, const int direction, const Optimization& optimizer, const double dt)
{
	if (direction != Dim - 1)
	{
		// boundary condition for recursion
		trotter_expansion_for_classical_momentum_liouville_splitting(density, direction + 1, optimizer, dt / 2.0);
	}
	// regarding that basis transformation needs adiabatic basis as intermediate, such transformation does not increase much computation
	whole_density_basis_transformation(density, Representation::Adiabatic, direction);
	const int InitialSize = density.size(); // there will be extra PWTDMs, so need save original size
	for (int i = 0; i < InitialSize; i++)
	{
		const EvolvingDensity& EvolvedVector = evolve_one_force_on_one_density(density[i], direction, dt);
		// the return vector should have at least one element (from the original PWTDM)
		// so update original place with the first element, and insert other at the end
		density[i] = EvolvedVector[0];
		density.insert(density.cend(), EvolvedVector.cbegin() + 1, EvolvedVector.cend());
	}
	whole_density_basis_transformation(density, direction, Representation::Adiabatic);
	if (direction != Dim - 1)
	{
		// boundary condition for recursion
		trotter_expansion_for_classical_momentum_liouville_splitting(density, direction + 1, optimizer, dt / 2.0);
	}
}

/// @details For classical momentum Liouville superoperator,
///
/// \f[
/// -\mathrm{i}\hat{\mathcal{L}}^P\hat{\rho}_W
/// =-\frac{1}{2}\left(\hat{\vec{F}}_W\cdot\frac{\partial\hat{\rho}_W}{\partial\vec{P}}
/// +\frac{\partial\hat{\rho}_W}{\partial\vec{P}}\cdot\hat{\vec{F}}_W\right)
///	\f]
///
/// If the PWTDM is in force basis (where one of the component of \f$ \hat{\vec{F}}_W \f$
/// is diagonal, or \f$\hat{F}_W^i\f$ is diagonal), then the superoperator evolution would be
///
/// \f[
/// \mathrm{e}^{-\mathrm{i}\hat{\mathcal{L}}^P_i\mathrm{d}t}\hat{\rho}_W(\vec{R},\vec{P})
/// =\sum_{a,b}{\hat{\rho}_W^{ab}(\vec{R},
/// P_0,\dots,P_i+\frac{\mathrm{d}t}{2}(f_a+f_b),\dots,P_{N-1}}
/// \f]
///
/// where \f$ f_a=(\hat{F}_W^i)_{aa} \f$ is the a-th eigenvalue of the i-th direction force.
///
/// If the force on each direction are non-commutative (except some special model this is
/// generally true), the Trotter expansion is used in this function as well for each force.
///
/// As the summation indicates, this function will increase the number of PWTDMs. However,
/// if the sum (or average) of 2 eigenvalues equals to another 2, or \f$ f_a+f_b=f_c+f_d \f$,
/// where \f$ \{a,b\}\neq\{c,d\} \f$, the PWTDM will be combined.
///
/// Besides, under the global Trotter expansion, if this superoperator will be done n times,
/// the dt in the parameter list will be global dt over n.
void classical_momentum_liouville(EvolvingDensity& density, const Optimization& optimizer, const double dt)
{
	trotter_expansion_for_classical_momentum_liouville_splitting(density, 0, optimizer, dt);
}
