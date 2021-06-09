/// @file evolve.cpp
/// @brief Implementation of evolve.h

#include "stdafx.h"

#include "evolve.h"

#include "mc.h"
#include "pes.h"

static const double CouplingCriterion = 0.01; ///< Criteria of whether have coupling or not

/// @brief To judge if current point have large coupling in case of 2-level system
/// @param[in] x Position of classical degree of freedom
/// @param[in] p Momentum of classical degree of freedom
/// @param[in] mass Mass of classical degree of freedom
/// @return Whether have coupling in any of the directions
static bool is_coupling(
	const ClassicalDoubleVector& x,
	const ClassicalDoubleVector& p,
	const ClassicalDoubleVector& mass)
{
	const ClassicalDoubleVector nac_01 = tensor_slice(adiabatic_coupling(x), 0, 1), f_01 = tensor_slice(adiabatic_force(x), 0, 1);
	return nac_01.dot((p.array() / mass.array()).matrix()) > CouplingCriterion || f_01.norm() > CouplingCriterion;
}

/// @brief To judge if current point have large coupling in case of >2 level system
/// @param[in] x Position of classical degree of freedom
/// @return The coupling of each direction
static ClassicalBoolVector is_coupling(const ClassicalDoubleVector& x)
{
	ClassicalBoolVector result = ClassicalBoolVector::Zero();
	const Tensor3d& NAC = adiabatic_coupling(x);
	for (int iDim = 0; iDim < Dim; iDim++)
	{
		result[iDim] = (NAC[iDim].array().abs().sum() > CouplingCriterion);
	}
	return result;
}

/// @brief To evolve the position
/// @param[in] x Position of classical degree of freedom
/// @param[in] p Momentum of classical degree of freedom
/// @param[in] mass Mass of classical degree of freedom
/// @param[in] dt The time interval of evolution
/// @return All branches of momentum vectors
/// @details \f$ x_{i,j} = x_i - \frac{p_{i,j}}{m} dt \f$
static ClassicalVectors position_evolve(
	const ClassicalVectors& x,
	const ClassicalVectors& p,
	const ClassicalDoubleVector& mass,
	const double dt)
{
	assert(p.size() % x.size() == 0);
	const int NumBranch = p.size() / x.size();
	ClassicalVectors result;
	for (int i = 0; i < x.size(); i++)
	{
		for (int j = 0; j < NumBranch; j++)
		{
			result.push_back(x[i].array() - p[i * NumBranch + j].array() / mass.array() * dt);
		}
	}
	return result;
}

/// @brief To get the diagonal forces under adiabatic basis at given position
/// @param[in] x Position of classical degree of freedom
/// @return Bunches of force vectors corresponding to diagonal elements of force matrices of the whole force tensor
static ClassicalVectors adiabatic_diagonal_forces(const ClassicalDoubleVector& x)
{
	const Tensor3d& force = adiabatic_force(x);
	ClassicalVectors result;
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		result.push_back(tensor_slice(force, iPES, iPES));
	}
	return result;
}

/// @brief To evolve the momentum under the adiabatic diagonal forces
/// @param[in] x Position of classical degree of freedom
/// @param[in] p Momentum of classical degree of freedom
/// @param[in] dt The time interval of evolution
/// @return All branches of momentum vectors
/// @details \f$ p_{i,j} = p_i - f_j(x_i) dt \f$, where \f$ f_j = \frac{f_{aa}+f_{bb}}{2} \f$
static ClassicalVectors momentum_diagonal_evolve(const ClassicalVectors& x, const ClassicalVectors& p, const double dt)
{
	assert(x.size() == p.size());
	ClassicalVectors result;
	for (int i = 0; i < x.size(); i++)
	{
		const ClassicalVectors f = adiabatic_diagonal_forces(x[i]);
		for (int iPES = 0; iPES < NumPES; iPES++)
		{
			for (int jPES = 0; jPES <= iPES; jPES++)
			{
				result.push_back(p[i] - (f[iPES] + f[jPES]) / 2.0 * dt);
			}
		}
	}
	return result;
}

/// @brief To calculate sin and cos of \f$ dt(\omega(x_0)+2\omega(x_1)+\omega(x_2))/4 \f$, where \f$ \omega(x) = (E_0(x)-E_1(x))/\hbar \f$
/// @param[in] x0 First position
/// @param[in] x1 Second position
/// @param[in] x2 Third position
/// @param[in] dt Time interval
/// @return sin and cos \f$ dt(\omega(x_0)+2\omega(x_1)+\omega(x_2))/4 \f$
static inline Eigen::Vector2d omega0_of_2_level(
	const ClassicalDoubleVector& x0,
	const ClassicalDoubleVector& x1,
	const ClassicalDoubleVector& x2,
	const double dt)
{
	const QuantumDoubleVector E0 = adiabatic_potential(x0), E1 = adiabatic_potential(x1), E2 = adiabatic_potential(x2);
	const double omega = dt * (E0[0] - E0[1] + 2.0 * (E1[0] - E1[1]) + E2[0] - E2[1]) / 4.0 / hbar;
	Eigen::Vector2d result;
	result << std::sin(omega), std::cos(omega);
	return result;
}

/// @brief To calculate sin and cos of \f$ \frac{\vec{P}\cdot\vec{d}_{01}}{M}dt \f$
/// @param[in] x Position of classical degree of freedom
/// @param[in] p Momentum of classical degree of freedom
/// @param[in] mass Mass of classical degree of freedom
/// @param[in] dt Time interval
/// @return sin and cos \f$ \frac{\vec{P}\cdot\vec{d}_{01}}{M}dt \f$ for each input (x,p) pairs
static inline Eigen::MatrixXd phi_of_2_level(
	const ClassicalVectors& x,
	const ClassicalVectors& p,
	const ClassicalDoubleVector& mass,
	const double dt)
{
	assert(p.size() % x.size() == 0);
	const int NumBranch = p.size() / x.size();
	Eigen::MatrixXd result(p.size(), 2);
	for (int i = 0; i < x.size(); i++)
	{
		const ClassicalDoubleVector& d01 = tensor_slice(adiabatic_coupling(x[i]), 0, 1);
		for (int j = 0; j < NumBranch; j++)
		{
			const int Index = i * NumBranch + j;
			const double phi = dt * (p[Index].array() / mass.array()).matrix().dot(d01);
			result.row(Index) << std::sin(phi), std::cos(phi);
		}
	}
	return result;
}

/// @brief To calculate the element of 2-level system
/// @param[in] phi sin and cos of \f$ dt\phi_{\pm} \f$
/// @param[in] omega1 sin and cos of \f$ dt\omega_{1,\pm}/2 \f$
/// @param[in] rho The predicted density matrix element
/// @param[in] Coe_1 \f$ c_{-1} \f$
/// @param[in] Coe0 \f$ c_0 \f$
/// @param[in] Coe1 \f$ c_1 \f$
/// @return The element
/// @details The element is given by
///
/// \f{eqnarray*}{
/// &&c_{-1}((1+\sin\phi_-)\rho_{00}+2\cos\phi_-\cos\omega_{1,-}\rho_{01}-2\cos\phi_-\sin\omega_{1,-}\rho_{10}+(1-\sin\phi_-)\rho_{11}) \\
/// +&&c_0(\cos\phi\rho_{00}-2\sin\phi\cos\omega_1\rho_{01}+2\sin\phi\sin\omega_1\rho_{10}-\cos\phi\rho_{11}) \\
/// +&&c_1((\sin\phi_+-1)\rho_{00}+2\cos\phi_+\cos\omega_{1,+}\rho_{01}-2\cos\phi_+\sin\omega_{1,+}\rho_{10}-(\sin\phi_++1)\rho_{11})
/// \f}
///
/// where \f$ \rho_{01}=\Re\rho_{10} \f$ and \f$ \rho_{10}=\Im\rho_{10} \f$.
static inline double calculate_2_level_element(
	const Eigen::MatrixXd& phi,
	const Eigen::MatrixXd& omega1,
	const Eigen::MatrixXd& rho,
	const double Coe_1,
	const double Coe0,
	const double Coe1)
{
	return Coe_1 * ((1.0 + phi(0, 0)) * rho(0, 0) + 2.0 * phi(0, 1) * omega1(0, 1) * rho(0, 1) - 2.0 * phi(0, 1) * omega1(0, 0) * rho(0, 2) + (1.0 - phi(0, 0)) * rho(0, 3))
		+ Coe0 * (phi(1, 1) * rho(1, 0) - 2.0 * phi(1, 0) * omega1(1, 1) * rho(1, 1) + 2.0 * phi(1, 0) * omega1(1, 0) * rho(1, 2) - phi(1, 1) * rho(1, 3))
		+ Coe1 * ((1.0 - phi(2, 0)) * rho(2, 0) - 2.0 * phi(2, 1) * omega1(2, 1) * rho(2, 1) + 2.0 * phi(2, 1) * omega1(2, 0) * rho(2, 2) + (1.0 + phi(2, 0)) * rho(2, 3));
}

QuantumComplexMatrix evolve_predict(
	const ClassicalDoubleVector& x,
	const ClassicalDoubleVector& p,
	const ClassicalDoubleVector& mass,
	const double dt,
	const QuantumBoolMatrix IsSmall,
	const Optimization& optimizer)
{
	using namespace std::literals::complex_literals;
	const double dt_2 = dt / 2.0, dt_4 = dt / 4.0;
	QuantumComplexMatrix result;
	if (NumPES == 2)
	{
		// 2-level system
		// x_i and p_{i-1} have same number of elements, and p_i is branching
		if (false)//is_coupling(x, p, mass) == true)
		{
			// 17 steps
			const ClassicalDoubleVector x1 = x.array() - p.array() / mass.array() * dt_4;
			// index: {00, 10, 11}
			const ClassicalVectors p1 = momentum_diagonal_evolve({ x1 }, { p }, dt_2);
			const ClassicalVectors x2 = position_evolve({ x1 }, p1, mass, dt_4);
			// index: {00, 10, 11}, {-1, 0, 1}
			const ClassicalVectors p2 = [](const ClassicalVectors& x, const ClassicalVectors& p, const double dt) -> ClassicalVectors
			{
				ClassicalVectors result;
				for (int i = 0; i < x.size(); i++)
				{
					const ClassicalDoubleVector f_01 = tensor_slice(adiabatic_force(x[i]), 0, 1);
					result.push_back(p[i] - f_01 * dt);
					result.push_back(p[i]);
					result.push_back(p[i] + f_01 * dt);
				}
				return result;
			}(x2, p1, dt);
			const ClassicalVectors x3 = position_evolve(x2, p2, mass, dt_4);
			// index: {00, 10, 11}, {-1, 0, 1}, {00, 10, 11}
			const ClassicalVectors p3 = momentum_diagonal_evolve(x3, p2, dt_2);
			const ClassicalVectors x4 = position_evolve(x3, p3, mass, dt_4);
			// prepare: evolved elements, phi, omega0, omega1
			const Eigen::MatrixXd rho_predict = [](
				const ClassicalVectors& x4,
				const ClassicalVectors& p3,
				const QuantumBoolMatrix& IsSmall,
				const Optimization& optimizer) -> Eigen::MatrixXd
			{
				const int NBranch = x4.size() / 3;
				Eigen::MatrixXd result(NBranch, NumElements);
				for (int iPES = 0; iPES < NumPES; iPES++)
				{
					for (int jPES = 0; jPES <= iPES; jPES++)
					{
						const int ElmIdx = iPES * NumPES + jPES;
						result.col(ElmIdx) = optimizer.predict_elements(
							IsSmall,
							ClassicalVectors(x4.begin() + ElmIdx * NBranch, x4.begin() + (ElmIdx + 1) * NBranch),
							ClassicalVectors(p3.begin() + ElmIdx * NBranch, p3.begin() + (ElmIdx + 1) * NBranch),
							ElmIdx);
						if (iPES != jPES)
						{
							const int SymElmIdx = jPES * NumPES + iPES;
							result.col(SymElmIdx) = optimizer.predict_elements(
								IsSmall,
								ClassicalVectors(x4.begin() + ElmIdx * NBranch, x4.begin() + (ElmIdx + 1) * NBranch),
								ClassicalVectors(p3.begin() + ElmIdx * NBranch, p3.begin() + (ElmIdx + 1) * NBranch),
								SymElmIdx);
						}
					}
				}
				// now the arrangement is: gamma*(n, gamma'); need to rearrange to gamma*(gamma', n) for prediction
				Eigen::MatrixXd rearrange_result(NBranch, NumElements);
				for (int iN = 0; iN < 3; iN++)
				{
					for (int iGammaPrime = 0; iGammaPrime < 3; iGammaPrime++)
					{
						const int OldRow = iN * 3 + iGammaPrime, NewRow = iGammaPrime * 3 + iN;
						rearrange_result.row(NewRow) = result.row(OldRow);
					}
				}
				return rearrange_result;
			}(x4, p3, IsSmall, optimizer);
			const Eigen::MatrixXd phi = phi_of_2_level(x2, p2, mass, dt);
			const Eigen::Vector2d omega0 = omega0_of_2_level(x, x1, x2[1], dt_2);
			const Eigen::MatrixXd omega1 = [](
				const ClassicalVectors& x2,
				const ClassicalVectors& x3,
				const ClassicalVectors& x4,
				const double dt) -> Eigen::MatrixXd
			{
				const int NumBranch1 = x3.size() / x2.size(), NumBranch2 = x4.size() / x3.size();
				Eigen::MatrixXd result(x3.size(), 2);
				for (int i = 0; i < x2.size(); i++)
				{
					for (int j = 0; j < NumBranch1; j++)
					{
						const int Index = i * NumBranch1 + j;
						result.row(Index) = omega0_of_2_level(x2[i], x3[Index], x4[Index * NumBranch2 + 1], dt);
					}
				}
				return result;
			}(x2, x3, x4, dt_2);
			// calculate
			const int NBranch = 3; // the availability of n (off-diagonal momentum evolution branches)
			result(0, 0) = calculate_2_level_element(
				phi.block(0, 0, NBranch, phi.cols()),
				omega1.block(0, 0, NBranch, omega1.cols()),
				rho_predict.block(0, 0, NBranch, rho_predict.cols()),
				(1.0 - phi(1, 0)) / 4.0,
				phi(1, 1) / 2.0,
				(1.0 + phi(1, 0)) / 4.0);
			result(1, 0) = calculate_2_level_element(
				phi.block(NBranch, 0, NBranch, phi.cols()),
				omega1.block(NBranch, 0, NBranch, omega1.cols()),
				rho_predict.block(NBranch, 0, NBranch, rho_predict.cols()),
				phi(1 + NBranch, 1) / 4.0,
				phi(1 + NBranch, 0) / 2.0,
				-phi(1 + NBranch, 1) / 4.0) * (omega0[1] + 1.0i * omega0[0])
				+ (omega1(1 + NBranch, 0) * rho_predict(1 + NBranch, 1) + omega1(1 + NBranch, 1) * rho_predict(1 + NBranch, 2)) * (-omega0[0] + 1.0i * omega0[1]);
			result(1, 1) = calculate_2_level_element(
				phi.block(2.0 * NBranch, 0, NBranch, phi.cols()),
				omega1.block(2.0 * NBranch, 0, NBranch, omega1.cols()),
				rho_predict.block(2.0 * NBranch, 0, NBranch, rho_predict.cols()),
				(1.0 + phi(1 + 2.0 * NBranch, 0)) / 4.0,
				-phi(1 + 2.0 * NBranch, 1) / 2.0,
				(1.0 - phi(1 + 2.0 * NBranch, 0)) / 4.0);
		}
		else
		{
			// 7 steps
			const ClassicalDoubleVector x1 = x.array() - p.array() / mass.array() * dt_2;
			const ClassicalVectors p1 = momentum_diagonal_evolve({ x1 }, { p }, dt);
			const ClassicalVectors x2 = position_evolve({ x1 }, p1, mass, dt_2);
			const Eigen::Vector2d omega = omega0_of_2_level(x, x1, x2[1], dt);
			const double re = optimizer.predict_element(IsSmall, x2[1], p1[1], 1), im = optimizer.predict_element(IsSmall, x2[1], p1[1], 2);
			result(0, 0) = optimizer.predict_element(IsSmall, x2[0], p1[0], 0);
			result(1, 0) = re * (omega[1] + 1.0i * omega[0]) + im * (-omega[0] + 1.0i * omega[1]);
			result(1, 1) = optimizer.predict_element(IsSmall, x2[2], p1[2], 3);
		}
	}
	else
	{
		const ClassicalBoolVector IsCoupling = is_coupling(x);
	}
	return result.selfadjointView<Eigen::Lower>();
}