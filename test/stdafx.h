/// @file stdafx.h
/// @brief the header file containing all headers and constants.
///
/// This file will be the precompiled header(PCH), named
/// after STanDard Application Framework eXtensions.

#pragma once
#ifndef STDAFX_H
#define STDAFX_H

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gsl/gsl_multimin.h>
#ifndef EIGEN_USE_MKL_ALL
#define EIGEN_USE_MKL_ALL
#endif
#ifdef __GSL_BLAS_TYPES_H__
#define __MKL_CBLAS_H__
#endif
#include <Eigen/Eigen>
#include <nlopt.hpp>

//#pragma warning(push, 0)
//#pragma warning(disable : 3346 654)
//#pragma warning(pop)

using namespace std;
using namespace Eigen;

const int NPoint = 500; ///< the size of training set

#endif // !STDAFX_H
