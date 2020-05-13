/// @file rand.cpp
/// This file is the implementation of the corresponding header.
#include <chrono>
#include <numeric>
#include <random>
#include <set>
#include <utility>

#include <mkl.h>
#define EIGEN_USE_MKL_ALL
#include <Eigen/Eigen>

#include "rand.h"

using namespace std;
using namespace Eigen;

/// using the uniform int distribution in standard library
/// to generate a random number in range [0, total_time_ticks-1]
int choose_time(const int total_time_ticks)
{
	static uniform_int_distribution<int> uid(0, total_time_ticks);
	return uid(generator);
}

/// using MC to choose a point. If the point has already chosen, redo
set<pair<int, int>> choose_point(const MatrixXd& data, const VectorXd& x, const VectorXd& p)
{
	const int nx = data.rows(), np = data.cols();
	const double weight = accumulate(data.data(), data.data() + nx * np, 0.0);
	uniform_real_distribution<double> urd(0.0, weight);
	set<pair<int, int>> chosen_point;
	int i = 0;
	while (i < NPoint)
	{
		double acc_weight = urd(generator);
		for (int j = 0; j < nx; j++)
		{
			for (int k = 0; k < np; k++)
			{
				acc_weight -= data[j * np + k];
				if (acc_weight < 0)
				{
					auto iter = chosen_point.insert(make_pair(j, k));
					if (iter.second == true)
					{
						i++;
					}
					goto next;
				}
			}
		}
	next:
	}
	return chosen_point;
}

/// using the gassuain distribution in standard library
/// to generate the all the random variables
VectorXd gaussian_random_generate(const int length)
{
	static normal_distribution<double> nd;
	VectorXd result(length);
	for (int i = 0; i < length; i++)
	{
		result(i) = nd(generator);
	}
	return result;
}
