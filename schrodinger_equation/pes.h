#ifndef PES_H
#define PES_H

// this header file contains the
// parameter used in DVR calculation
// and gives the Potential Energy Surface
// (PES) of different representations:
// diabatic, adiabatic and force-basis.
// Transformation is done by unitary
// matrices calculate by LAPACK in MKL

#include <algorithm>
#include <cmath>
#include <complex>
#include <mkl.h>
#include <utility>
#include "general.h"

#include <mkl.h>
#ifndef EIGEN_USE_MKL_ALL
#define EIGEN_USE_MKL_ALL
#endif // ! EIGEN_USE_MKL_ALL
#include <Eigen/Eigen>

using namespace std;
using namespace Eigen;

/// @enum different models
enum Model
{
	SAC = 1, ///< Simple Avoided Crossing, tully's first model
	DAC, ///< Dual Avoided Crossing, tully's second model
	ECR ///< Extended Coupling with Reflection, tully's third model
};

const int NumPES = 2; ///< the number of potential energy surfaces
/// the Potential matrix and its derivatives
using PESMatrix = Matrix<double, NumPES, NumPES>;

#ifndef TestModel
const Model TestModel = DAC; ///< the model to use
#endif // !TestModel

/// @brief diabatic PES matrix: the analytical form
/// @param x the position
/// @return the potential (subsystem Hamiltonian) at the given point
PESMatrix diabatic_potential(const double x);

/// @brief the absorbing potential: V->V-iE. Here is E only. E is diagonal (=E*I)
/// @param mass the mass of the bath/nucleus
/// @param xmin the original left boundary, extended by an absorbing region
/// @param xmax the original right boundary, extended by an absorbing region
/// @param AbsorbingRegionLength the length of the extended absorbing region
/// @param x the position where to calculate the strength of absorbing potential
/// @return the imaginary part of the absorbing potential at the given point
double absorbing_potential
(
	const double mass,
	const double xmin,
	const double xmax,
	const double AbsorbingRegionLength,
	const double x
);

/// @brief calculate the transformation matrix from diabatic state to adiabatic state
/// @param XCoordinate the coordinate of each grid, i.e., x_i
/// @return the transformation matrix
MatrixXcd diabatic_to_adiabatic(const VectorXd& XCoordinate);

#endif // !PES_H
