/// @file io.cpp
/// @brief Definition of input-output functions

#include "io.h"

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

void print_point(
	std::ostream& out,
	const std::set<std::pair<int, int>>& point,
	const Eigen::VectorXd& x,
	const Eigen::VectorXd& p)
{
	std::set<std::pair<int, int>>::const_iterator iter = point.cbegin();
	for (int i = 0; i < NPoint; i++, ++iter)
	{
		out << ' ' << x(iter->first) << ' ' << p(iter->second);
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
		for (int j = 0; j < IndentLevel; j++)
		{
			out << '\t';
		}
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
			}
			out << "]\n";
			break;
		default:
			std::clog << "UNKNOWN KERNEL!\n";
			break;
		}
	}
}
