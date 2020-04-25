/// @file matrix.cpp
/// @brief Implemetation of matrix.h

#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <utility>
#include <mkl.h>
#include "matrix.h"
using namespace std;

/// Transfrom a double array to a complex array by using lambda expression
void real_to_complex(const double* da, Complex* ca, const int length)
{
    generate(ca, ca + length, [i = 0, da](void)mutable->Complex{ return da[i++]; });
}



// RealMatrix functions

RealMatrix::RealMatrix(const int size)
	: length(size), nelements(length * length), content(new double*[length])
{
	content[0] = new double[nelements];
	for (int i = 1; i < length; i++)
	{
		content[i] = content[0] + i * length;
	}
	memset(content[0], 0, nelements * sizeof(double));
}

RealMatrix::RealMatrix(const RealMatrix& matrix)
	: length(matrix.length), nelements(matrix.nelements), content(new double*[length])
{
	content[0] = new double[nelements];
	for (int i = 1; i < length; i++)
	{
		content[i] = content[0] + i * length;
	} 
	copy(matrix.content[0], matrix.content[0] + nelements, content[0]);
}

RealMatrix::RealMatrix(const int size, const double* array)
	: length(size), nelements(length * length), content(new double*[length])
{
	content[0] = new double[nelements];
	for (int i = 1; i < length; i++)
	{
		content[i] = content[0] + i * length;
	}
	copy(array, array + nelements, content[0]);
}

RealMatrix::RealMatrix(const int size, const Index& idx, const double& val)
	:length(size), nelements(length * length), content(new double*[length])
{
	content[0] = new double[nelements];
	for (int i = 1; i < length; i++)
	{
		content[i] = content[0] + i * length;
	}
	memset(content[0], 0, nelements * sizeof(double));
	content[idx.first][idx.second] = val;
}

RealMatrix::~RealMatrix(void)
{
	delete[] content[0];
	delete[] content;
}


int RealMatrix::length_of_matrix(void) const
{
	return length;
}

double* RealMatrix::data(void)
{
	return content[0];
}

const double* RealMatrix::data(void) const
{
	return content[0];
}

void RealMatrix::transform_to_1d(double* array) const
{
	copy(content[0], content[0] + nelements, array);
}

/// Make the real matrix object symmetric. A[i][j]=A[j][i]=(A[i][j]+A[j][i])/2.0
///
/// This is recommended to use ONLY for matrices close to symmetric
/// (e.g. a symmetric matrix after basis transformation).
void RealMatrix::symmetrize(void)
{
	for (int i = 0; i < length; i++)
	{
		for (int j = i + 1; j < length; j++)
		{
			content[i][j] = content[j][i] = (content[i][j] + content[j][i]) / 2.0;
		}
	}
}

double* RealMatrix::operator[](const int idx)
{
	return content[idx];
}

const double* RealMatrix::operator[](const int idx) const
{
	return content[idx];
}

RealMatrix& RealMatrix::operator=(const RealMatrix& rhs)
{
	if (length != rhs.length)
	{
		delete[] content[0];
		delete[] content;
		length = rhs.length;
		nelements = rhs.nelements;
		content = new double* [length];
		content[0] = new double[nelements];
		for (int i = 1; i < length; i++)
		{
			content[i] = content[0] + i * length;
		}
	}
	copy(rhs.content[0], rhs.content[0] + nelements, content[0]);
	return *this;
}

RealMatrix& RealMatrix::operator=(const double* array)
{
	copy(array, array + nelements, content[0]);
	return *this;
}


// overload numerical calculation by VMF

RealMatrix operator+(const RealMatrix& lhs, const double& rhs)
{
	RealMatrix result = lhs;
	for (int i = 0; i < lhs.length; i++)
	{
		result.content[i][i] += rhs;
	}
	return result;
}

RealMatrix operator+(const double& lhs, const RealMatrix& rhs)
{
	RealMatrix result = rhs;
	for (int i = 0; i < rhs.length; i++)
	{
		result.content[i][i] += lhs;
	}
	return result;
}

RealMatrix operator+(const RealMatrix& lhs, const RealMatrix& rhs)
{
	if (lhs.length != rhs.length)
	{
		cerr << "ADD DIFFERENT SIZE REAL MATRIX" << endl;
		exit(100);
	}
	RealMatrix result(lhs.length);
	vmdAdd(result.nelements, lhs.content[0], rhs.content[0], result.content[0], mode);
	return result;
}

RealMatrix& RealMatrix::operator+=(const double& rhs)
{
	for (int i = 0; i < length; i++)
	{
		content[i][i] += rhs;
	}
	return *this;
}

RealMatrix& RealMatrix::operator+=(const RealMatrix& rhs)
{
	if (length != rhs.length)
	{
		cerr << "ADD-ASSIGN DIFFERENT SIZE REAL MATRIX" << endl;
		exit(101);
	}
	double* result = new double[nelements];
	vmdAdd(nelements, content[0], rhs.content[0], result, mode);
	swap(content[0], result);
	for (int i = 1; i < length; i++)
	{
		content[i] = content[0] + i * length;
	}
	delete[] result;
	return *this;
}

RealMatrix operator-(const RealMatrix& lhs, const double& rhs)
{
	RealMatrix result = lhs;
	for (int i = 0; i < lhs.length; i++)
	{
		result.content[i][i] -= rhs;
	}
	return result;
}

RealMatrix operator-(const double& lhs, const RealMatrix& rhs)
{
	RealMatrix result = -1.0 * rhs;
	for (int i = 0; i < rhs.length; i++)
	{
		result.content[i][i] += lhs;
	}
	return result;
}

RealMatrix operator-(const RealMatrix& lhs, const RealMatrix& rhs)
{
	if (lhs.length != rhs.length)
	{
		cerr << "SUBTRACT DIFFERENT SIZE REAL MATRIX" << endl;
		exit(102);
	}
	RealMatrix result(lhs.length);
	vmdSub(result.nelements, lhs.content[0], rhs.content[0], result.content[0], mode);
	return result;
}

RealMatrix& RealMatrix::operator-=(const double& rhs)
{
	for (int i = 0; i < length; i++)
	{
		content[i][i] -= rhs;
	}
	return *this;
}

RealMatrix& RealMatrix::operator-=(const RealMatrix& rhs)
{
	if (length != rhs.length)
	{
		cerr << "SUBSTRACT-ASSIGN DIFFERENT SIZE REAL MATRIX" << endl;
		exit(103);
	}
	double* result = new double[nelements];
	vmdSub(nelements, content[0], rhs.content[0], result, mode);
	swap(content[0], result);
	for (int i = 1; i < length; i++)
	{
		content[i] = content[0] + i * length;
	}
	delete[] result;
	return *this;
}

RealMatrix operator*(const RealMatrix& lhs, const double& rhs)
{
	RealMatrix result(lhs.length);
	double* num = new double[lhs.nelements];
	fill(num, num + lhs.nelements, rhs);
	vmdMul(lhs.nelements, lhs.content[0], num, result.content[0], mode);
	return result;
}

RealMatrix operator*(const double& lhs, const RealMatrix& rhs)
{
	RealMatrix result(rhs.length);
	double* num = new double[rhs.nelements];
	fill(num, num + rhs.nelements, lhs);
	vmdMul(rhs.nelements, num, rhs.content[0], result.content[0], mode);
	return result;
}

RealMatrix& RealMatrix::operator*=(const double& rhs)
{
	double* num = new double[nelements];
	fill(num, num + nelements, rhs);
	double* result = new double[nelements];
	vmdMul(nelements, num, content[0], result, mode);
	swap(content[0], result);
	for (int i = 1; i < length; i++)
	{
		content[i] = content[0] + i * length;
	}
	delete[] result;
	return *this;
}

RealMatrix operator/(const RealMatrix& lhs, const double& rhs)
{
	return lhs * (1.0 / rhs);
}

RealMatrix& RealMatrix::operator/=(const double& rhs)
{
	*this *= 1.0 / rhs;
	return *this;
}



// ComplexMatrix functions

ComplexMatrix::ComplexMatrix(const int size)
	: length(size), nelements(length * length), content(new Complex*[length])
{
	content[0] = new Complex[nelements];
	for (int i = 1; i < length; i++)
	{
		content[i] = content[0] + i * length;
	}
	memset(content[0], 0, nelements * sizeof(Complex));
}

ComplexMatrix::ComplexMatrix(const ComplexMatrix& matrix)
	: length(matrix.length), nelements(matrix.nelements), content(new Complex*[length])
{
	content[0] = new Complex[nelements];
	for (int i = 1; i < length; i++)
	{
		content[i] = content[0] + i * length;
	}
	copy(matrix.content[0], matrix.content[0] + matrix.nelements, content[0]);
}

ComplexMatrix::ComplexMatrix(const int size, const Complex* array)
	: length(size), nelements(length * length), content(new Complex*[length])
{
	content[0] = new Complex[nelements];
	for (int i = 1; i < length; i++)
	{
		content[i] = content[0] + i * length;
	}
	copy(array, array + nelements, content[0]);
}

ComplexMatrix::ComplexMatrix(const RealMatrix& matrix)
	: length(matrix.length), nelements(matrix.nelements), content(new Complex*[length])
{
	content[0] = new Complex[nelements];
	for (int i = 1; i < length; i++)
	{
		content[i] = content[0] + i * length;
	}
	real_to_complex(matrix.content[0], content[0], matrix.nelements);
}

ComplexMatrix::ComplexMatrix(const int size, const double* array)
	: length(size), nelements(length * length), content(new Complex*[length])
{
	content[0] = new Complex[nelements];
	for (int i = 1; i < length; i++)
	{
		content[i] = content[0] + i * length;
	}
	real_to_complex(array, content[0], nelements);
}

ComplexMatrix::ComplexMatrix(const int size, const Index& idx, const Complex& val)
	: length(size), nelements(length * length), content(new Complex*[length])
{
	content[0] = new Complex[nelements];
	for (int i = 1; i < length; i++)
	{
		content[i] = content[0] + i * length;
	}
	memset(content[0], 0, nelements * sizeof(Complex));
	content[idx.first][idx.second] = val;
}

ComplexMatrix::~ComplexMatrix(void)
{
	delete[] content[0];
	delete[] content;
}


int ComplexMatrix::length_of_matrix(void) const
{
	return length;
}

Complex* ComplexMatrix::data(void)
{
	return content[0];
}

const Complex* ComplexMatrix::data(void) const
{
	return content[0];
}

void ComplexMatrix::transform_to_1d(Complex* array) const
{
	copy(content[0], content[0] + nelements, array);
}

/// Make the complex matrix object hermitian: A[i][i].imag=0,
/// A[i][j].real=(A[i][j].real+A[j][i].real)/2,
/// A[i][j].imag=(A[i][j].imag-A[j][i].imag)/2
///
/// This is recommended to use ONLY for matrices close to hermitian
/// (e.g. an hermitian matrix after basis transformation).
void ComplexMatrix::hermitize(void)
{
    // for diagonal elements, set imag to 0
    // for off-diagonal elements, makes them conjugate
    for (int i = 0; i < length; i++)
    {
        for (int j = i; j < length; j++)
        {
            const double real = (content[i][j].real() + content[j][i].real()) / 2.0;
            const double imag = (content[i][j].imag() - content[j][i].imag()) / 2.0;
            content[i][j] = Complex(real, imag);
            content[j][i] = Complex(real, -imag);
        }
    }
}

Complex* ComplexMatrix::operator[](const int idx)
{
	return content[idx];
}

const Complex* ComplexMatrix::operator[](const int idx) const
{
	return content[idx];
}

ComplexMatrix& ComplexMatrix::operator=(const ComplexMatrix& rhs)
{
	if (length != rhs.length)
	{
		delete[] content[0];
		delete[] content;
		length = rhs.length;
		nelements = rhs.nelements;
		content = new Complex * [length];
		content[0] = new Complex[nelements];
		for (int i = 1; i < length; i++)
		{
			content[i] = content[0] + i * length;
		}
	}
	copy(rhs.content[0], rhs.content[0] + nelements, content[0]);
	return *this;
}

ComplexMatrix& ComplexMatrix::operator=(const Complex* array)
{
	copy(array, array + nelements, content[0]);
	return *this;
}


// overload numerical calculation by VMF

ComplexMatrix operator+(const ComplexMatrix& lhs, const Complex& rhs)
{
	ComplexMatrix result = lhs;
	for (int i = 0; i < lhs.length; i++)
	{
		result.content[i][i] += rhs;
	}
	return result;
}

ComplexMatrix operator+(const Complex& lhs, const ComplexMatrix& rhs)
{
	ComplexMatrix result = rhs;
	for (int i = 0; i < rhs.length; i++)
	{
		result.content[i][i] += lhs;
	}
	return result;
}

ComplexMatrix operator+(const ComplexMatrix& lhs, const ComplexMatrix& rhs)
{
	if (lhs.length != rhs.length)
	{
		cerr << "ADD DIFFERENT SIZE RCOMPLEX MATRIX" << endl;
		exit(104);
	}
	ComplexMatrix result(lhs.length);
	vmzAdd(result.nelements, reinterpret_cast<const MKL_Complex16*>(lhs.content[0]), reinterpret_cast<const MKL_Complex16*>(rhs.content[0]), reinterpret_cast<MKL_Complex16*>(result.content[0]), mode);
	return result;
}

ComplexMatrix& ComplexMatrix::operator+=(const Complex& rhs)
{
	for (int i = 0; i < length; i++)
	{
		content[i][i] += rhs;
	}
	return *this;
}

ComplexMatrix& ComplexMatrix::operator+=(const ComplexMatrix& rhs)
{
	if (length != rhs.length)
	{
		cerr << "ADD-ASSIGN DIFFERENT SIZE COMPLEX MATRIX" << endl;
		exit(105);
	}
	Complex* result = new Complex[nelements];
	vmzAdd(nelements, reinterpret_cast<const MKL_Complex16*>(content[0]), reinterpret_cast<const MKL_Complex16*>(rhs.content[0]), reinterpret_cast<MKL_Complex16*>(result), mode);
	swap(content[0], result);
	for (int i = 1; i < length; i++)
	{
		content[i] = content[0] + i * length;
	}
	delete[] result;
	return *this;
}

ComplexMatrix operator-(const ComplexMatrix& lhs, const Complex& rhs)
{
	ComplexMatrix result = lhs;
	for (int i = 0; i < lhs.length; i++)
	{
		result.content[i][i] -= rhs;
	}
	return result;
}

ComplexMatrix operator-(const Complex& lhs, const ComplexMatrix& rhs)
{
	ComplexMatrix result = -1.0 * rhs;
	for (int i = 0; i < rhs.length; i++)
	{
		result.content[i][i] -= lhs;
	}
	return result;
}

ComplexMatrix operator-(const ComplexMatrix& lhs, const ComplexMatrix& rhs)
{
	if (lhs.length != rhs.length)
	{
		cerr << "SUBTRACT DIFFERENT SIZE COMPLEX MATRIX" << endl;
		exit(106);
	}
	ComplexMatrix result(lhs.length);
	vmzSub(result.nelements, reinterpret_cast<const MKL_Complex16*>(lhs.content[0]), reinterpret_cast<const MKL_Complex16*>(rhs.content[0]), reinterpret_cast<MKL_Complex16*>(result.content[0]), mode);
	return result;
}

ComplexMatrix& ComplexMatrix::operator-=(const Complex& rhs)
{
	for (int i = 0; i < length; i++)
	{
		content[i][i] -= rhs;
	}
	return *this;
}

ComplexMatrix& ComplexMatrix::operator-=(const ComplexMatrix& rhs)
{
	if (length != rhs.length)
	{
		cerr << "SUBTRACT-ASSIGN DIFFERENT SIZE COMPLEX MATRIX" << endl;
		exit(107);
	}
	Complex* result = new Complex[nelements];
	vmzSub(nelements, reinterpret_cast<const MKL_Complex16*>(content[0]), reinterpret_cast<const MKL_Complex16*>(rhs.content[0]), reinterpret_cast<MKL_Complex16*>(result), mode);
	swap(content[0], result);
	for (int i = 1; i < length; i++)
	{
		content[i] = content[0] + i * length;
	}
	delete[] result;
	return *this;
}

ComplexMatrix operator*(const ComplexMatrix& lhs, const Complex& rhs)
{
	ComplexMatrix result(lhs.length);
	Complex* num = new Complex[lhs.nelements];
	fill(num, num + lhs.nelements, rhs);
	vmzMul(lhs.nelements, reinterpret_cast<const MKL_Complex16*>(lhs.content[0]), reinterpret_cast<const MKL_Complex16*>(num), reinterpret_cast<MKL_Complex16*>(result.content[0]), mode);
	return result;
}

ComplexMatrix operator*(const Complex& lhs, const ComplexMatrix& rhs)
{
	ComplexMatrix result(rhs.length);
	Complex* num = new Complex[rhs.nelements];
	fill(num, num + rhs.nelements, lhs);
	vmzMul(rhs.nelements, reinterpret_cast<const MKL_Complex16*>(num), reinterpret_cast<const MKL_Complex16*>(rhs.content[0]), reinterpret_cast<MKL_Complex16*>(result.content[0]), mode);
	return result;
}

ComplexMatrix& ComplexMatrix::operator*=(const Complex& rhs)
{
	Complex* num = new Complex[nelements];
	fill(num, num + nelements, rhs);
	Complex* result = new Complex[nelements];
	vmzMul(nelements, reinterpret_cast<const MKL_Complex16*>(num), reinterpret_cast<const MKL_Complex16*>(content[0]), reinterpret_cast<MKL_Complex16*>(result), mode);
	swap(content[0], result);
	for (int i = 1; i < length; i++)
	{
		content[i] = content[0] + i * length;
	}
	delete[] result;
	return *this;
}

ComplexMatrix operator/(const ComplexMatrix& lhs, const Complex& rhs)
{
	return lhs * (1.0 / rhs);
}

ComplexMatrix& ComplexMatrix::operator/=(const Complex& rhs)
{
	*this *= 1.0 / rhs;
	return *this;
}



// ComplexMatrixMatrix functions

ComplexMatrixMatrix::ComplexMatrixMatrix(const int OuterLength, const int InnerLength)
	: length(OuterLength), nelements(length * length), content(new ComplexMatrix*[length])
{
	content[0] = MatrixAllocator.allocate(nelements);
	uninitialized_fill(content[0], content[0] + nelements, ComplexMatrix(InnerLength));
	for (int i = 1; i < length; i++)
	{
		content[i] = content[0] + i * length;
	}
}

ComplexMatrixMatrix::~ComplexMatrixMatrix(void)
{
	destroy(content[0], content[0] + nelements);
	MatrixAllocator.deallocate(content[0], nelements);
	delete[] content;
}

ComplexMatrix* ComplexMatrixMatrix::operator[](const int idx)
{
	return content[idx];
}

const ComplexMatrix* ComplexMatrixMatrix::operator[](const int idx) const
{
	return content[idx];
}

/// Output one element of ComplexMatrix in a line, change column first, then row
///
/// That is to say,
///
/// (line 1) rho[0][0][0][0] rho[0][1][0][0] rho[0][2][0][0] ... rho[0][n][0][0] ... rho[n][n][0][0]
///
/// (line 2) rho[0][0][0][1] rho[0][1][0][1] rho[0][2][0][1] ... rho[0][n][0][1] ... rho[n][n][0][1]
///
/// and so on.
ostream& operator<<(ostream& os, const ComplexMatrixMatrix& rho)
{
	for (int a = 0; a < rho.content[0][0].length_of_matrix(); a++)
	{
		for (int b = 0; b < rho.content[0][0].length_of_matrix(); b++)
		{
			for (int i = 0; i < rho.length; i++)
			{
				for (int j = 0; j < rho.length; j++)
				{
					os << ' ' << rho[i][j][a][b].real() << ' ' << rho[i][j][a][b].imag();
				}
			}
			os << '\n';
		}
	}
	return os;
}
