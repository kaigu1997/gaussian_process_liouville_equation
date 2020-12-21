#include <algorithm>
#include <array>
#include <cassert>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <functional>
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

#include <boost/numeric/odeint.hpp>
#ifndef EIGEN_USE_MKL_ALL
#define EIGEN_USE_MKL_ALL
#endif // !EIGEN_USE_MKL_ALL
#include <Eigen/Eigen>
#undef EIGEN_USE_MKL_ALL
#include <nlopt.hpp>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/ConstKernel.h>
#include <shogun/kernel/DiagKernel.h>
#include <shogun/kernel/GaussianARDKernel.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>

const int NPoint = 200;
const int NStep = 500;
const int PhaseDim = 2;
const double pi = 3.14159265358979324;
const double x0 = -10.0, p0 = 14.112, sigma_x = 0.7086, sigma_p = 0.7056;
const double dxmax = 0.125, dpmax = 0.20945;
const double xmin = -15.0, xmax = 15.0, pmin = -11.0208, pmax = 39.2447;
const int NGrid = 241;

using KernelTypeList = std::vector<shogun::EKernelType>;
using KernelList = std::vector<std::pair<double, std::shared_ptr<shogun::Kernel>>>;
using ParameterVector = std::vector<double>;
using WholeTrainingSet = std::tuple<Eigen::MatrixXd, Eigen::VectorXd, KernelTypeList>;
using MatrixVector = std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>>;

static int number_of_overall_hyperparameters(const KernelTypeList& TypesOfKernels)
{
	const int size = TypesOfKernels.size();
	int sum = 0;
	for (int i = 0; i < size; i++)
	{
		sum++; // weight
		switch (TypesOfKernels[i])
		{
		case shogun::EKernelType::K_DIAG:
			break; // no extra hyperparameter for diagonal kernel
		case shogun::EKernelType::K_GAUSSIANARD:
			sum += PhaseDim;
			break;
		default:
			std::cerr << "UNKNOWN KERNEL!\n";
			break;
		}
	}
	return sum;
}

inline double func(double x, double p)
{
	return std::exp(-(std::pow((x - x0) / sigma_x, 2) + std::pow(p - p0 / sigma_p, 2)) / 2.0);
}

WholeTrainingSet generate_training_set(const KernelTypeList& TypesOfKernels)
{
	static std::mt19937_64 generator(std::chrono::system_clock::now().time_since_epoch().count()); // random number generator, 64 bits Mersenne twister engine
	Eigen::MatrixXd training_feature(PhaseDim, NPoint);
	Eigen::VectorXd training_label(NPoint);
	std::uniform_real_distribution<> mc_selection, init_x(xmin, xmax), init_p(pmin, pmax), init_dx(-dxmax, dxmax), init_dp(-dpmax, dpmax);
	for (int i = 0; i < NPoint; i++)
	{
		double x = init_x(generator), p = init_p(generator);
		double weight = func(x, p);
		for (int j = 0; j < NStep; j++)
		{
			const double dx = init_dx(generator), dp = init_dp(generator);
			const double x_new = x + dx, p_new = p + dp;
			if (x_new < xmin || x_new > xmax || p_new < pmin || p_new > pmax)
			{
				continue;
			}
			const double weight_new = func(x_new, p_new);
			if (weight_new < weight && weight_new / weight < mc_selection(generator))
			{
				continue;
			}
			x = x_new;
			p = p_new;
			weight = weight_new;
		}
		training_feature(0, i) = x;
		training_feature(1, i) = p;
		training_label[i] = weight;
	}
	return std::make_tuple(training_feature, training_label, TypesOfKernels);
}

std::vector<ParameterVector> set_initial_value(const KernelTypeList& TypesOfKernels)
{
	static const double DiagMagMin = 1e-8;	// Minimal value of the magnitude of the diagonal kernel
	static const double DiagMagMax = 1e-5;	// Maximal value of the magnitude of the diagonal kernel
	static const double GaussMagMin = 1e-4; // Minimal value of the magintude of the gaussian kernel
	static const double GaussMagMax = 1.0;	// Maximal value of the magnitude of the gaussian kernel
	const int NoVar = number_of_overall_hyperparameters(TypesOfKernels);
	static std::vector<double> lower_bound(NoVar, std::numeric_limits<double>::lowest());
	static std::vector<double> upper_bound(NoVar, std::numeric_limits<double>::max());
	static std::vector<double> hyperparameters(NoVar, 0);
	for (int i = 0, iparam = 0; i < TypesOfKernels.size(); i++)
	{
		switch (TypesOfKernels[i])
		{
		case shogun::EKernelType::K_DIAG:
			lower_bound[iparam] = DiagMagMin;
			upper_bound[iparam] = DiagMagMax;
			hyperparameters[iparam] = DiagMagMin;
			iparam++;
			break;
		case shogun::EKernelType::K_GAUSSIANARD:
			lower_bound[iparam] = GaussMagMin;
			upper_bound[iparam] = GaussMagMax;
			hyperparameters[iparam] = GaussMagMax;
			iparam++;
			for (int j = 0; j < PhaseDim; j++)
			{
				if (j % 2 == 0)
				{
					lower_bound[iparam] = 1.0 / (xmax - xmin);
					hyperparameters[iparam] = 1.0 / sigma_x;
				}
				else
				{
					lower_bound[iparam] = 1.0 / (pmax - pmin);
					hyperparameters[iparam] = 1.0 / sigma_p;
				}
				iparam++;
			}
			break;
		default:
			std::cerr << "UNKNOWN KERNEL!\n";
			break;
		}
	}
	std::vector<ParameterVector> result;
	result.push_back(lower_bound);
	result.push_back(upper_bound);
	result.push_back(hyperparameters);
	return result;
}

std::string indent(const int IndentLevel)
{
	return std::string(IndentLevel, '\t');
}

void print_kernel(
	std::ostream& out,
	const KernelTypeList& TypesOfKernels,
	const std::vector<double>& Hyperparameters,
	const int IndentLevel)
{
	const int NKernel = TypesOfKernels.size();
	for (int i = 0, iparam = 0; i < NKernel; i++)
	{
		out << indent(IndentLevel);
		const double weight = Hyperparameters[iparam++];
		switch (TypesOfKernels[i])
		{
		case shogun::EKernelType::K_DIAG:
			out << "Diagonal Kernel: " << weight << '\n';
			break;
		case shogun::EKernelType::K_GAUSSIANARD:
			out << "Gaussian Kernel with Relevance: " << weight << " * [";
			for (int j = 0; j < PhaseDim; j++)
			{
				if (j != 0)
				{
					out << ", ";
				}
				out << Hyperparameters[iparam++];
			}
			out << "]\n";
			break;
		default:
			std::clog << "UNKNOWN KERNEL!\n";
			break;
		}
	}
}

static KernelList generate_kernels(const KernelTypeList& TypesOfKernels, const ParameterVector& Hyperparameters)
{
	const int NoVar = number_of_overall_hyperparameters(TypesOfKernels);
	assert(Hyperparameters.size() == NoVar || Hyperparameters.size() == NoVar + 1);
	const int NKernel = TypesOfKernels.size();
	KernelList result;
	for (int iKernel = 0, iParam = 0; iKernel < NKernel; iKernel++)
	{
		const double weight = Hyperparameters[iParam++];
		switch (TypesOfKernels[iKernel])
		{
		case shogun::EKernelType::K_DIAG:
			result.push_back(std::make_pair(weight, std::make_shared<shogun::DiagKernel>()));
			break;
		case shogun::EKernelType::K_GAUSSIANARD:
		{
			std::shared_ptr<shogun::GaussianARDKernel> gauss_ard_kernel_ptr = std::make_shared<shogun::GaussianARDKernel>();
			Eigen::VectorXd characteristic(PhaseDim);
			for (int j = 0; j < PhaseDim; j++)
			{
				characteristic[j] = Hyperparameters[iParam++];
			}
			gauss_ard_kernel_ptr->set_vector_weights(shogun::SGVector<double>(characteristic));
			result.push_back(std::make_pair(weight, gauss_ard_kernel_ptr));
			break;
		}
		default:
			std::cerr << "UNKNOWN KERNEL!\n";
			break;
		}
	}
	return result;
}

static inline shogun::SGMatrix<double> generate_shogun_matrix(const Eigen::MatrixXd& mat)
{
	shogun::SGMatrix<double> result(mat.rows(), mat.cols());
	std::copy(mat.data(), mat.data() + mat.size(), result.data());
	return result;
}

static Eigen::MatrixXd get_kernel_matrix(
	const Eigen::MatrixXd& LeftFeature,
	const Eigen::MatrixXd& RightFeature,
	KernelList& Kernels,
	const bool IsTraining = false)
{
	assert(LeftFeature.rows() == PhaseDim && RightFeature.rows() == PhaseDim);
	// construct the feature
	std::shared_ptr<shogun::DenseFeatures<double>> left_feature = std::make_shared<shogun::DenseFeatures<double>>(generate_shogun_matrix(LeftFeature));
	std::shared_ptr<shogun::DenseFeatures<double>> right_feature = std::make_shared<shogun::DenseFeatures<double>>(generate_shogun_matrix(RightFeature));
	const int NKernel = Kernels.size();
	Eigen::MatrixXd result = Eigen::MatrixXd::Zero(LeftFeature.cols(), RightFeature.cols());
	for (int i = 0; i < NKernel; i++)
	{
		const double weight = Kernels[i].first;
		std::shared_ptr<shogun::Kernel>& kernel_ptr = Kernels[i].second;
		if (IsTraining == false && kernel_ptr->get_kernel_type() == shogun::EKernelType::K_DIAG)
		{
			// in case when it is not training feature, the diagonal kernel (working as noise) is not needed
			continue;
		}
		else
		{
			kernel_ptr->init(left_feature, right_feature);
			result += weight * weight * Eigen::Map<Eigen::MatrixXd>(kernel_ptr->get_kernel_matrix());
		}
	}
	return result;
}

static MatrixVector kernel_derivative_over_hyperparameters(const Eigen::MatrixXd& Feature, KernelList& Kernels)
{
	assert(Feature.rows() == PhaseDim);
	// construct the feature
	std::shared_ptr<shogun::DenseFeatures<double>> feature = std::make_shared<shogun::DenseFeatures<double>>(generate_shogun_matrix(Feature));
	const int NKernel = Kernels.size();
	MatrixVector result;
	for (int i = 0; i < NKernel; i++)
	{
		const double weight = Kernels[i].first;
		std::shared_ptr<shogun::Kernel>& kernel_ptr = Kernels[i].second;
		switch (kernel_ptr->get_kernel_type())
		{
		case shogun::EKernelType::K_DIAG:
		{
			kernel_ptr->init(feature, feature);
			// calculate derivative over weight
			result.push_back(weight * Eigen::Map<Eigen::MatrixXd>(kernel_ptr->get_kernel_matrix()));
			break;
		}
		case shogun::EKernelType::K_GAUSSIANARD:
		{
			kernel_ptr->init(feature, feature);
			// calculate derivative over weight
			result.push_back(weight * Eigen::Map<Eigen::MatrixXd>(kernel_ptr->get_kernel_matrix()));
			// calculate derivative over the characteristic matrix elements
			const Eigen::Map<Eigen::MatrixXd>& Characteristic = std::dynamic_pointer_cast<shogun::GaussianARDKernel>(kernel_ptr)->get_weights();
			for (int j = 0; j < PhaseDim; j++)
			{
				const Eigen::Map<Eigen::MatrixXd>& Deriv = kernel_ptr->get_parameter_gradient(std::make_pair("log_weights", std::make_shared<shogun::AnyParameter>()), j);
				result.push_back(weight * weight / Characteristic(j, 0) * Deriv);
			}
			break;
		}
		default:
			std::cerr << "UNKNOWN KERNEL!\n";
			break;
		}
	}
	return result;
}

static double negative_log_marginal_likelihood(const ParameterVector& x, ParameterVector& grad, void* params)
{
	// receive the parameter
	const WholeTrainingSet& Training = *static_cast<WholeTrainingSet*>(params);
	// get the parameters
	const Eigen::MatrixXd& TrainingFeature = std::get<0>(Training);
	const Eigen::VectorXd& TrainingLabel = std::get<1>(Training);
	const KernelTypeList& TypesOfKernels = std::get<2>(Training);
	KernelList Kernels = generate_kernels(TypesOfKernels, x);
	// get kernel and the derivatives of kernel over hyperparameters
	const Eigen::MatrixXd& KernelMatrix = get_kernel_matrix(TrainingFeature, TrainingFeature, Kernels, true);
	// calculate
	Eigen::LLT<Eigen::MatrixXd> DecompOfKernel(KernelMatrix);
	const Eigen::MatrixXd& L = DecompOfKernel.matrixL();
	const Eigen::MatrixXd& KInv = DecompOfKernel.solve(Eigen::MatrixXd::Identity(TrainingFeature.cols(), TrainingFeature.cols())); // inverse of kernel
	const Eigen::VectorXd& KInvLbl = DecompOfKernel.solve(TrainingLabel);														   // K^{-1}*y
	const double result = (TrainingLabel.adjoint() * KInvLbl).value() / 2.0 + L.diagonal().array().abs().log().sum();
	// print current result and combination
	std::clog << indent(2) << result << '\n';
	print_kernel(std::clog, TypesOfKernels, x, 3);
	if (grad.empty() == false)
	{
		// need gradient information, calculated here
		const MatrixVector& KernelDerivative = kernel_derivative_over_hyperparameters(TrainingFeature, Kernels);
		for (int i = 0; i < x.size(); i++)
		{
			grad[i] = ((KInv - KInvLbl * KInvLbl.adjoint()) * KernelDerivative[i]).trace() / 2.0;
		}
		// print current gradient if uses gradient
		print_kernel(std::clog, TypesOfKernels, grad, 4);
	}
	std::clog << std::endl;
	return result;
}

ParameterVector optimize(
	const WholeTrainingSet& TrainingSet,
	const nlopt::algorithm& ALGO,
	const ParameterVector& LowerBound,
	const ParameterVector& UpperBound,
	const ParameterVector& InitialHyperparameters)
{
	const KernelTypeList& TypesOfKernels = std::get<2>(TrainingSet);
	const int NKernel = TypesOfKernels.size();
	// constructor the NLOPT minimizer using Nelder-Mead simplex algorithm
	static const int NoVar = number_of_overall_hyperparameters(TypesOfKernels); // the number of variables to optimize
	assert(InitialHyperparameters.size() == NoVar || InitialHyperparameters.size() == NoVar + 1);
	ParameterVector result;
	// the hyperparameters used in this element
	ParameterVector hyperparameters = InitialHyperparameters;
	if (hyperparameters.size() == NoVar + 1)
	{
		hyperparameters.pop_back();
	}
	// the marginal likelihood of this element
	double marg_ll = 0.0;

	// need optimization
	nlopt::opt minimizer(ALGO, NoVar);
	// set minimizer function
	const nlopt::vfunc minimizing_function = negative_log_marginal_likelihood;
	minimizer.set_min_objective(minimizing_function, const_cast<WholeTrainingSet*>(&TrainingSet));
	// set bounds for noise and characteristic length, as well as for initial values
	minimizer.set_lower_bounds(LowerBound);
	minimizer.set_upper_bounds(UpperBound);
	// set stop criteria
	static const double AbsTol = 1e-10; // tolerance in minimization
	minimizer.set_xtol_abs(AbsTol);
	// set initial step size
	static const double InitStepSize = 0.5; // initial step size
	minimizer.set_initial_step(InitStepSize);

	// optimize
	std::clog << indent(1) << "Begin Optimization" << std::endl;
	try
	{
		nlopt::result result = minimizer.optimize(hyperparameters, marg_ll);
		std::clog << indent(1);
		switch (result)
		{
		case nlopt::result::SUCCESS:
			std::clog << "Successfully stopped";
			break;
		case nlopt::result::STOPVAL_REACHED:
			std::clog << "Stopping value reached";
			break;
		case nlopt::result::FTOL_REACHED:
			std::clog << "Function value tolerance reached";
			break;
		case nlopt::result::XTOL_REACHED:
			std::clog << "Step size tolerance reached";
			break;
		case nlopt::result::MAXEVAL_REACHED:
			std::clog << "Maximum evaluation time reached";
			break;
		case nlopt::result::MAXTIME_REACHED:
			std::clog << "Maximum cpu time reached";
			break;
		}
		std::clog << '\n';
	}
	catch (const std::exception& e)
	{
		std::cerr << indent(1) << "NLOPT optimization failed for " << e.what() << '\n';
	}
	std::clog << indent(1) << "Best Combination is\n";
	ParameterVector grad;
	marg_ll = minimizing_function(hyperparameters, grad, const_cast<WholeTrainingSet*>(&TrainingSet));
	hyperparameters.push_back(marg_ll);
	return hyperparameters;
}

Eigen::MatrixXd predict(
	const WholeTrainingSet& TrainingSet,
	const Eigen::VectorXd& x,
	const Eigen::VectorXd& p,
	const ParameterVector& Hyperparameters)
{
	const Eigen::MatrixXd& TrainingFeature = std::get<0>(TrainingSet);
	const Eigen::VectorXd& TrainingLabel = std::get<1>(TrainingSet);
	const KernelTypeList& TypesOfKernels = std::get<2>(TrainingSet);
	const int nx = x.size(), np = p.size();
	const int NoVar = number_of_overall_hyperparameters(TypesOfKernels);
	Eigen::MatrixXd result(nx, np);
	KernelList Kernels = generate_kernels(TypesOfKernels, Hyperparameters);
	const Eigen::MatrixXd& KernelMatrix = get_kernel_matrix(TrainingFeature, TrainingFeature, Kernels, true);
	const Eigen::VectorXd& KInvLbl = KernelMatrix.llt().solve(TrainingLabel);
	Eigen::VectorXd coord(PhaseDim);
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < np; j++)
		{
			coord[0] = x[i];
			coord[1] = p[j];
			result(i, j) = (get_kernel_matrix(coord, TrainingFeature, Kernels) * KInvLbl).value();
		}
	}
	return result;
}

Eigen::MatrixXd exact(const Eigen::VectorXd& x, const Eigen::VectorXd& p)
{
	const int nx = x.size(), np = p.size();
	Eigen::MatrixXd result(nx, np);
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < np; j++)
		{
			result(i, j) = func(x[i], p[j]);
		}
	}
	return result;
}

int main()
{
	std::clog << std::setprecision(std::numeric_limits<double>::digits10 + 1);
	std::clog.sync_with_stdio(true);
	// the gradient and non-gradient algorithm to use
	const nlopt::algorithm NonGradient = nlopt::algorithm::LN_NELDERMEAD;
	const nlopt::algorithm Gradient = nlopt::algorithm::LD_TNEWTON_PRECOND_RESTART;
	// the kernel to use
	const KernelTypeList TypesOfKernels = { shogun::EKernelType::K_DIAG, shogun::EKernelType::K_GAUSSIANARD };
	// the bounds and initial values for hyperparameters, which is fixed
	const std::vector<ParameterVector>& Initials = set_initial_value(TypesOfKernels);
	const ParameterVector& LowerBound = Initials[0];
	const ParameterVector& UpperBound = Initials[1];
	const ParameterVector& InitialHyperparameters = Initials[2];

	// generate training set
	const WholeTrainingSet& TrainingSet = generate_training_set(TypesOfKernels);
	// train
	const ParameterVector& NonGradientOptimized
		= optimize(
			TrainingSet,
			NonGradient,
			LowerBound,
			UpperBound,
			InitialHyperparameters);
	const ParameterVector& FinalHyperparameters
		= optimize(
			TrainingSet,
			Gradient,
			LowerBound,
			UpperBound,
			NonGradientOptimized);
	
	// predict and output
	const Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(NGrid, xmin, xmax), p = Eigen::VectorXd::LinSpaced(NGrid, pmin, pmax);
	std::ofstream sim("sim.txt"), real("real.txt"), point("point.txt");
	sim << predict(TrainingSet, x, p, FinalHyperparameters) << '\n';
	real << exact(x, p) << '\n';
	const Eigen::MatrixXd& TrainingFeature = std::get<0>(TrainingSet);
	for (int i = 0; i < NPoint; i++)
	{
		point << ' ' << TrainingFeature(0, i) << ' ' << TrainingFeature(1, i);
	}
	point << '\n';
	sim.close();
	real.close();
	point.close();
	return 0;
}