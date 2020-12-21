/// @file io.cpp
/// @brief Definition of input-output functions

#include "io.h"

std::string indent(const int IndentLevel)
{
	return std::string(IndentLevel, '\t');
}

Eigen::VectorXd read_coord(const std::string& filename)
{
	std::vector<double> v;
	double tmp;
	std::ifstream f(filename);
	while (f >> tmp)
	{
		v.push_back(tmp);
	}
	Eigen::VectorXd coord(v.size());
	std::copy(v.cbegin(), v.cend(), coord.data());
	return coord;
}

void read_density(std::istream& in, SuperMatrix& ExactDistribution)
{
	const int nx = ExactDistribution[0][0].rows(), np = ExactDistribution[0][0].cols();
	double tmp;
	// read input
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		for (int jPES = 0; jPES < NumPES; jPES++)
		{
			if (iPES == jPES)
			{
				// diagonal elements
				for (int iGrid = 0; iGrid < nx; iGrid++)
				{
					for (int jGrid = 0; jGrid < np; jGrid++)
					{
						in >> ExactDistribution[iPES][jPES](iGrid, jGrid) >> tmp;
					}
				}
			}
			else if (iPES < jPES)
			{
				// read the real and imaginary part in; real the upper, imag the lower
				for (int iGrid = 0; iGrid < nx; iGrid++)
				{
					for (int jGrid = 0; jGrid < np; jGrid++)
					{
						in >> ExactDistribution[iPES][jPES](iGrid, jGrid) >> ExactDistribution[jPES][iPES](iGrid, jGrid);
					}
				}
			}
			else
			{
				for (int iGrid = 0; iGrid < nx; iGrid++)
				{
					for (int jGrid = 0; jGrid < np; jGrid++)
					{
						// its transpose has been read, so do average
						in >> tmp;
						ExactDistribution[iPES][jPES](iGrid, jGrid) = (ExactDistribution[iPES][jPES](iGrid, jGrid) + tmp) / 2.0;
						in >> tmp;
						ExactDistribution[jPES][iPES](iGrid, jGrid) = (ExactDistribution[jPES][iPES](iGrid, jGrid) + tmp) / 2.0;
					}
				}
			}
		}
	}
}

void print_point(std::ostream& out, const SuperMatrix& TrainingFeatures)
{
	for (int iPES = 0; iPES < NumPES; iPES++)
	{
		for (int jPES = 0; jPES < NumPES; jPES++)
		{
			const Eigen::MatrixXd& TrainingFeatureOfThisElement = TrainingFeatures[iPES][jPES];
			for (int iPoint = 0; iPoint < NPoint; iPoint++)
			{
				for (int iDim = 0; iDim < PhaseDim; iDim++)
				{
					out << ' ' << TrainingFeatureOfThisElement(iDim, iPoint);
				}
			}
			out << '\n';
		}
	}
	out << '\n';
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
#ifndef NOCROSS
				out << '[';
				for (int k = 0; k < PhaseDim; k++)
				{
					if (k <= j)
					{
						if (k != 0)
						{
							out << ", ";
						}
						out << Hyperparameters[iparam++];
					}
					else
					{
						out << ", 0";
					}
				}
				out << ']';
#else
				out << Hyperparameters[iparam++];
#endif
			}
			out << "]\n";
			break;
		default:
			std::clog << "UNKNOWN KERNEL!\n";
			break;
		}
	}
}
