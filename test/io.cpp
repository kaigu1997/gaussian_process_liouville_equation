/// @file io.cpp
/// @brief definition of input-output functions

#include "io.h"

VectorXd read_coord(const string& filename)
{
	vector<double> v;
	double tmp;
	ifstream f(filename);
	while (f >> tmp)
	{
		v.push_back(tmp);
	}
	VectorXd coord(v.size());
	copy(v.cbegin(), v.cend(), coord.data());
	return coord;
}

void output_point
(
	const set<pair<int, int>>& point,
	const VectorXd& x,
	const VectorXd& p,
	ostream& out
)
{
	set<pair<int, int>>::const_iterator iter = point.cbegin();
	for (int i = 0; i < NPoint; i++, ++iter)
	{
		out << ' ' << x(iter->first) << ' ' << p(iter->second);
	}
	out << '\n';
}

void print_kernel(CKernel* kernel_ptr, int layer)
{
	for (int i = 0; i < layer; i++)
	{
		cout << '\t';
	}
	cout << kernel_ptr << " - " << kernel_ptr->get_name() << " weight = " << kernel_ptr->get_combined_kernel_weight() << '\n';
	switch (kernel_ptr->get_kernel_type())
	{
	case K_COMBINED:
	{
		CCombinedKernel* combined_kernel_ptr = static_cast<CCombinedKernel*>(kernel_ptr);
		for (int i = 0; i < combined_kernel_ptr->get_num_kernels(); i++)
		{
			print_kernel(combined_kernel_ptr->get_kernel(i), layer + 1);
		}
	}
	break;
	case K_DIAG:
	{
		for (int i = 0; i <= layer; i++)
		{
			cout << '\t';
		}
		cout << "diag = " << static_cast<CDiagKernel*>(kernel_ptr)->kernel(0, 0) << '\n';
	}
	break;
	case K_GAUSSIANARD:
	{
		SGMatrix<double> weight = static_cast<CGaussianARDKernel*>(kernel_ptr)->get_weights();
		for (int i = 0; i <= layer; i++)
		{
			cout << '\t';
		}
		cout << "weight = [";
		for (int i = 0; i < weight.num_rows; i++)
		{
			if (i != 0)
			{
				cout << ", ";
			}
			cout << '[';
			for (int j = 0; j < weight.num_cols; j++)
			{
				if (j != 0)
				{
					cout << ", ";
				}
				cout << weight(i, j);
			}
			cout << ']';
		}
		cout << "]\n";
	}
	break;
	default:
	{
		for (int i = 0; i <= layer; i++)
		{
			cout << '\t';
		}
		cout << "UNIDENTIFIED KERNEL\n";
	}
	}
}
