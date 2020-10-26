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

SuperMatrix read_density(std::istream& in, const int NRows, const int NCols)
{
	SuperMatrix result(NRows, NCols);
	double tmp;
	/*
		double tmp;
		Eigen::MatrixXd rho0(nx, np), rho1(nx, np), rho_re(nx, np), rho_im(nx, np);
		// rho00
		for (int i = 0; i < nx; i++)
		{
			for (int j = 0; j < np; j++)
			{
				phase >> rho0(i, j) >> tmp;
			}
		}
		// rho01 and rho10
		for (int i = 0; i < nx; i++)
		{
			for (int j = 0; j < np; j++)
			{
				phase >> rho_re(i, j) >> rho_im(i, j);
			}
		}
		for (int i = 0; i < nx; i++)
		{
			for (int j = 0; j < np; j++)
			{
				phase >> tmp;
				rho_re(i, j) = (rho_re(i, j) + tmp) / 2.0;
				phase >> tmp;
				rho_im(i, j) = (rho_im(i, j) - tmp) / 2.0;
			}
		}
		// rho11
		for (int i = 0; i < nx; i++)
		{
			for (int j = 0; j < np; j++)
			{
				phase >> rho1(i, j) >> tmp;
			}
		}
	*/
	// read input
	for (int i = 0; i < NumPES; i++)
	{
		for (int j = 0; j < NumPES; j++)
		{
			for (int k = 0; k < NRows; k++)
			{
				for (int l = 0; l < NCols; l++)
				{
					if (i == j)
					{
						// diagonal elements
						in >> result(k, l)(i, j) >> tmp;
					}
					else if (i < j)
					{
						// read the real and imaginary part in; real the upper, imag the lower
						in >> result(k, l)(i, j) >> result(k, l)(j, i);
					}
					else
					{
						// its transpose has been read, so do average
						in >> tmp;
						result(k, l)(i, j) = (result(k, l)(i, j) + tmp) / 2.0;
						in >> tmp;
						result(k, l)(j, i) = (result(k, l)(j, i) + tmp) / 2.0;
					}
				}
			}
		}
	}
	return result;
	
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
