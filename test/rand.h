/// @file rand.h
/// This header deals with functions related to randomness: choose a time, choose some points
#ifndef RAND_H
#define RAND_H

#include <chrono>
#include <random>
#include <set>
#include <utility>

#include <mkl.h>
#ifndef EIGEN_USE_MKL_ALL
	#define EIGEN_USE_MKL_ALL
#endif
#include <Eigen/Eigen>

using namespace std;
using namespace Eigen;

const int NPoint = 10; ///< the size of training set

/// @brief choose a time to regress
/// @param[in] total_time_ticks the total time ticks available
/// @return a random int number in range [0, total_time_ticks)
int choose_time(const int total_time_ticks);

/// @brief choose npoint points based on their weight
/// @param[in] data the gridded phase space distribution
/// @param[in] nx the number of x grids
/// @param[in] np the number of p grids
/// @param[in] x the coordinates of x grids
/// @param[in] p the coordinates of p grids
/// @return a pair, first being the pointer to the coordinate table (first[i][0]=xi, first[i][1]=pi), second being the vector containing the phase space distribution at the point (second[i]=P(xi,pi))
set<pair<int, int>> choose_point(const MatrixXd& data, const VectorXd& x, const VectorXd& p);

/// @brief generate a vector of gaussian-type random variables
/// @param[in] length the number of random variables need generating
/// @return a vector containing those random variables
VectorXd gaussian_random_generate(const int length);

#endif // !RAND_H
