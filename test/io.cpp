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

void print_kernel(ostream& os, const gsl_vector* x, const int indent)
{
	for (int i = 0; i < indent; i++)
	{
		os << '\t';
	}
	os << abs(gsl_vector_get(x, 3)) << " * [[" << abs(gsl_vector_get(x, 0)) << ", 0], [" << gsl_vector_get(x, 1) << ", " << abs(gsl_vector_get(x, 2)) << "]] + " << abs(gsl_vector_get(x, 4)) << " * I";
}

void print_point(
	ostream& out,
	const set<pair<int, int>>& point,
	const VectorXd& x,
	const VectorXd& p)
{
	set<pair<int, int>>::const_iterator iter = point.cbegin();
	for (int i = 0; i < NPoint; i++, ++iter)
	{
		out << ' ' << x(iter->first) << ' ' << p(iter->second);
	}
	out << '\n';
}
