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
#include <gsl/gsl_multimin.h>
#include <iostream>
#include <memory>
#include <mkl.h>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#pragma warning(push, 0)
#pragma warning(disable: 3346 654)
#include <shogun/base/init.h>
#include <shogun/base/some.h>
#include <shogun/classifier/LDA.h>
#include <shogun/evaluation/GradientCriterion.h>
#include <shogun/evaluation/GradientEvaluation.h>
#include <shogun/evaluation/MeanSquaredError.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/CombinedKernel.h>
#include <shogun/kernel/DiagKernel.h>
#include <shogun/kernel/GaussianARDKernel.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/machine/gp/GaussianLikelihood.h>
#include <shogun/machine/gp/ExactInferenceMethod.h>
#include <shogun/machine/gp/ZeroMean.h>
#include <shogun/modelselection/GradientModelSelection.h>
#include <shogun/regression/GaussianProcessRegression.h>
#include <shogun/statistical_testing/internals/Kernel.h>
#include <Eigen/Eigen>
#pragma warning(pop)

using namespace std;
using namespace Eigen;
using namespace shogun;

const int NPoint = 100; ///< the size of training set

#endif // !STDAFX_H
