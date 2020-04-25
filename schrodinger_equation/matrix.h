/// @file matrix.h
/// @brief The declaration of matrices: double and complex-double
///
/// The header file to deal with rowwise 
/// real (double) or complex (double) matrices
/// stored in 1D array, calculated by
/// Vector Mathematical Functions in Intel(R) MKL.
/// Multiplication with vector/matrix could be done via BLAS.
/// Eigenvalue/eigenvector could be gotten from LAPACK.

#ifndef MATRIX_H
#define MATRIX_H

#include <complex>
#include <memory>
#include <mkl.h>
#include <utility>
using namespace std;

/// short version of complex<double>, equivalent to MKL_Complex16
typedef complex<double> Complex;
/// used as indices of matrix to access to matrix elements
typedef pair<int, int> Index;

const MKL_INT64 mode = VML_HA; ///< the mode constant for VMF functions in MKL

// declaration of all kinds of matrix
class RealMatrix;
class ComplexMatrix;

/// @brief transfrom a double array to a complex array
/// @param da the input double array
/// @param ca the output complex<double> array
/// @param length the number of elements in the two arrays
void real_to_complex(const double* da, Complex* ca, const int length);


/// the functions of real matrix
/// @see ComplexMatrix
class RealMatrix
{
private:
    int length; ///< the number of elements in a row/column
    int nelements; ///< the total number of elements, i.e. length^2
    double** content; ///< the memory to save the matrix content
public:
    /// @brief default constructor with all zero
    /// @param size the length of the matrix
    /// @see RealMatrix(), ~RealMatrix()
    RealMatrix(const int size);
    /// @brief copy constructor
    /// @param matrix the matrix to copy from
    /// @see RealMatrix(), ~RealMatrix()
    RealMatrix(const RealMatrix& matrix);
    /// @brief quasi copy constructor
    /// @param size the length of the matrix
    /// @param array the array containing the data for copy, whose length is size^2
    /// @see RealMatrix(), ~RealMatrix()
    RealMatrix(const int size, const double* array);
    /// @brief one element is give number and the other are all zero
    /// @param size the length of the matrix
    /// @param idx the index of a matrix element which is not zero
    /// @param val the value of the non-zero element
    /// @see RealMatrix(), ~RealMatrix()
    RealMatrix(const int size, const Index& idx, const double& val);
    /// @brief destructor
    /// @see RealMatrix()
    ~RealMatrix(void);

    /// @brief the size of the matrix
    /// @return the length of the matrix
    int length_of_matrix(void) const;
    /// @brief direct access to and modify internal data
    /// @return the pointer to the first element of the matrix
    double* data(void);
    /// @brief direct access to internal data only
    /// @return the pointer to the first element of the matrix
    const double* data(void) const;
    /// @brief copy to an array
    /// @param array the array to save the copied elements, at least size^2 length
    void transform_to_1d(double* array) const;
    /// @brief make the RealMatrix object symmetric
    void symmetrize(void);
    /// @brief overload operator[]
    /// @param idx the row number
    /// @return a pointer to the first element of the row
    double* operator[](const int idx);
    /// @brief overload operator[] for const objects
    /// @param idx the row number
    /// @return a pointer to the first element of the row
    const double* operator[](const int idx) const;
    /// @brief overload assignment operator
    /// @param matrix the RealMatrix to deeply copy the data from
    /// @return reference to the object, a RealMatrix
    RealMatrix& operator=(const RealMatrix& matrix);
    /// @brief overload assignment operator
    /// @param array the real array to deeply copy the data from that has size^2 elements
    /// @return reference to the object, a RealMatrix
    RealMatrix& operator=(const double* array);

    // overload numerical calculation by VMF
    /// @brief a RealMatrix adds a real number (times identity)
    /// @param lhs the left operant, a RealMatrix
    /// @param rhs the right operant, a real number, adds to the diagonal elements of the matrix
    /// @return a RealMatrix, the RealMatrix plus the real number times identity
    friend RealMatrix operator+(const RealMatrix& lhs, const double& rhs);
    /// @brief a RealMatrix adds a real number (times identity)
    /// @param lhs the left operant, a real number, adds to the diagonal elements of the matrix
    /// @param rhs the right operant, a RealMatrix
    /// @return a RealMatrix, the real number times identity plus the RealMatrix and 
    friend RealMatrix operator+(const double& lhs, const RealMatrix& rhs);
    /// @brief a RealMatrix adds another RealMatrix
    /// @param lhs the left operant, a RealMatrix
    /// @param rhs the right operant, a RealMatrix
    /// @return a RealMatrix, the sum of the two RealMatrix objects
    friend RealMatrix operator+(const RealMatrix& lhs, const RealMatrix& rhs);
    /// @brief a real number times identity is added to a RealMatrix and this RealMatrix object is changed
    /// @param rhs the right operant, a real number, adds to the diagnoal elements of the matrix
    /// @return reference to the object, a RealMatrix, the sum
    RealMatrix& operator+=(const double& rhs);
    /// @brief a RealMatrix is added to a RealMatrix and this RealMatrix object is changed
    /// @param rhs the right operant, a RealMatrix
    /// @return reference to the object, a RealMatrix, the sum
    RealMatrix& operator+=(const RealMatrix& rhs);
    /// @brief a RealMatrix subtracts a real number (times identity)
    /// @param lhs the left operant, a RealMatrix
    /// @param rhs the right operant, a real number, subtracted from the diagonal elements of the matrix
    /// @return a RealMatrix, the RealMatrix minus the real number times identity
    friend RealMatrix operator-(const RealMatrix& lhs, const double& rhs);
    /// @brief a real number (times identity) subtracts a RealMatrix
    /// @param lhs the left operant, a real number (times identity)
    /// @param rhs the right operant, a RealMatrix
    /// @return a RealMatrix, the RealMatrix minus the real number times identity
    friend RealMatrix operator-(const double& lhs, const RealMatrix& rhs);
    /// @brief a RealMatrix subtracts a RealMatrix
    /// @param lhs the left operant, a RealMatrix
    /// @param rhs the right operant, a RealMatrix
    /// @return a RealMatrix, the left RealMatrix minus the right RealMatrix
    friend RealMatrix operator-(const RealMatrix& lhs, const RealMatrix& rhs);
    /// @brief a real number times identity is subtracted from a RealMatrix and this RealMatrix object is changed
    /// @param rhs the right operant, a real number, subtracted from the diagnoal elements of the matrix
    /// @return reference to the object, a RealMatrix, the difference
    RealMatrix& operator-=(const double& rhs);
    /// @brief a RealMatrix is subtracted from a RealMatrix and this RealMatrix object is changed
    /// @param rhs the right operant, a RealMatrix to
    /// @return reference to the object, a RealMatrix, the difference
    RealMatrix& operator-=(const RealMatrix& rhs);
    /// @brief a real number times a RealMatrix, i.e. all elements are multiplied
    /// @param lhs the left operant, a RealMatrix
    /// @param rhs the right operant, a real number
    /// @return a RealMatrix, the product of the RealMatrix times the real number
    friend RealMatrix operator*(const RealMatrix& lhs, const double& rhs);
    /// @brief a real number times a RealMatrix, i.e. all elements are multiplied
    /// @param lhs the left operant, a real number
    /// @param rhs the right operant, a RealMatrix
    /// @return a RealMatrix, the product of the RealMatrix times the real number
    friend RealMatrix operator*(const double& lhs, const RealMatrix& rhs);
    /// @brief the RealMatrix object is multiplied by the number
    /// @param rhs the right operant, a real number
    /// @return reference to the object, a RealMatrix, the product
    RealMatrix& operator*=(const double& rhs);
    /// @brief a real number divides a RealMatrix, i.e. all elements divide the number
    /// @param lhs the left operant, a RealMatrix
    /// @param rhs the right operant, a real number
    /// @return a RealMatrix, the quotient of the RealMatrix dividing the real number
    friend RealMatrix operator/(const RealMatrix& lhs, const double& rhs);
    /// @brief the RealMatrix object divides the number
    /// @param rhs the right operant, a real number
    /// @return reference to the object, a RealMatrix, the quotient
    RealMatrix& operator/=(const double& rhs);

    // ComplexMatrix could access to RealMatrix for some constructors
    friend class ComplexMatrix;
};

/// the functions of complex matrix, similar to RealMatrix
/// @see RealMatrix, ComplexMatrixMatrix
class ComplexMatrix
{
private:
    int length; ///< the number of elements in a row/column
    int nelements; ///< the total number of elements, i.e. length^2
    Complex** content; ///< the memory to save the matrix content
public:
    /// @brief default constructor with all zero
    /// @param size the length of the matrix
    /// @see ComplexMatrix(), ~ComplexMatrix()
    ComplexMatrix(const int size);
    /// @brief copy constructor
    /// @param matrix the matrix to copy from
    /// @see ComplexMatrix(), ~ComplexMatrix()
    ComplexMatrix(const ComplexMatrix& matrix);
    /// @brief quasi copy constructor
    /// @param size the length of the matrix
    /// @param array the array containing the data for copy, whose length is size^2
    /// @see ComplexMatrix(), ~ComplexMatrix()
    ComplexMatrix(const int size, const Complex* array);
    /// @brief copy constructor from real matrix
    /// @param matrix the matrix to copy from
    /// @see ComplexMatrix(), ~ComplexMatrix()
    ComplexMatrix(const RealMatrix& matrix);
    /// @brief quasi copy constructor from real array
    /// @param size the length of the matrix
    /// @param array the array containing the data for copy, whose length is size^2
    /// @see ComplexMatrix(), ~ComplexMatrix()
    ComplexMatrix(const int size, const double* array);
    /// @brief one element is give number and the other are all zero
    /// @param size the length of the matrix
    /// @param idx the index of a matrix element which is not zero
    /// @param val the value of the non-zero element
    /// @see ComplexMatrix(), ~ComplexMatrix()
    ComplexMatrix(const int size, const Index& idx, const Complex& val);
    /// @brief destructor
    ~ComplexMatrix(void);

    /// @brief the size of the matrix
    /// @return the length of the matrix
    int length_of_matrix(void) const;
    /// @brief direct access to and modify internal data
    /// @return the pointer to the first element of the matrix
    Complex* data(void);
    /// @brief direct access to internal data only
    /// @return the pointer to the first element of the matrix
    const Complex* data(void) const;
    /// @brief copy to an array
    /// @param array the array to save the copied elements, at least size^2 length
    void transform_to_1d(Complex* array) const;
    /// @brief make the ComplexMatrix object hermitian
    void hermitize(void);
    /// @brief overload operator[]
    /// @param idx the row number
    /// @return a pointer to the first element of the row
    Complex* operator[](const int idx);
    /// @brief overload operator[] for const objects
    /// @param idx the row number
    /// @return a pointer to the first element of the row
    const Complex* operator[](const int idx) const;
    /// @brief overload assignment operator
    /// @param matrix the ComplexMatrix to deeply copy the data from
    /// @return reference to the object, a ComplexMatrix
    ComplexMatrix& operator=(const ComplexMatrix& rhs);
    /// @brief overload assignment operator
    /// @param array the complex array to deeply copy the data from that has size^2 elements
    /// @return reference to the object, a ComplexMatrix
    ComplexMatrix& operator=(const Complex* array);

    // overload numerical calculation by VMF
    /// @brief a ComplexMatrix adds a complex number (times identity)
    /// @param lhs the left operant, a ComplexMatrix
    /// @param rhs the right operant, a complex number, adds to the diagonal elements of the matrix
    /// @return a ComplexMatrix, the ComplexMatrix plus the complex number times identity
    friend ComplexMatrix operator+(const ComplexMatrix& lhs, const Complex& rhs);
    /// @brief a ComplexMatrix adds a complex number (times identity)
    /// @param lhs the left operant, a complex number, adds to the diagonal elements of the matrix
    /// @param rhs the right operant, a ComplexMatrix
    /// @return a ComplexMatrix, the complex number times identity plus the ComplexMatrix and 
    friend ComplexMatrix operator+(const Complex& lhs, const ComplexMatrix& rhs);
    /// @brief a ComplexMatrix adds another ComplexMatrix
    /// @param lhs the left operant, a ComplexMatrix
    /// @param rhs the right operant, a ComplexMatrix
    /// @return a ComplexMatrix, the sum of the two ComplexMatrix objects
    friend ComplexMatrix operator+(const ComplexMatrix& lhs, const ComplexMatrix& rhs);
    /// @brief a complex number times identity is added to a ComplexMatrix and this ComplexMatrix object is changed
    /// @param rhs the right operant, a complex number, adds to the diagnoal elements of the matrix
    /// @return reference to the object, a ComplexMatrix, the sum
    ComplexMatrix& operator+=(const Complex& rhs);
    /// @brief a ComplexMatrix is added to a ComplexMatrix and this ComplexMatrix object is changed
    /// @param rhs the right operant, a ComplexMatrix
    /// @return reference to the object, a ComplexMatrix, the sum
    ComplexMatrix& operator+=(const ComplexMatrix& rhs);
    /// @brief a ComplexMatrix subtracts a complex number (times identity)
    /// @param lhs the left operant, a ComplexMatrix
    /// @param rhs the right operant, a complex number, subtracted from the diagonal elements of the matrix
    /// @return a ComplexMatrix, the ComplexMatrix minus the complex number times identity
    friend ComplexMatrix operator-(const ComplexMatrix& lhs, const Complex& rhs);
    /// @brief a complex number (times identity) subtracts a ComplexMatrix
    /// @param lhs the left operant, a complex number (times identity)
    /// @param rhs the right operant, a ComplexMatrix
    /// @return a ComplexMatrix, the ComplexMatrix minus the complex number times identity
    friend ComplexMatrix operator-(const Complex& lhs, const ComplexMatrix& rhs);
    /// @brief a ComplexMatrix subtracts a ComplexMatrix
    /// @param lhs the left operant, a ComplexMatrix
    /// @param rhs the right operant, a ComplexMatrix
    /// @return a ComplexMatrix, the left ComplexMatrix minus the right ComplexMatrix
    friend ComplexMatrix operator-(const ComplexMatrix& lhs, const ComplexMatrix& rhs);
    /// @brief a complex number times identity is subtracted from a ComplexMatrix and this ComplexMatrix object is changed
    /// @param rhs the right operant, a complex number, subtracted from the diagnoal elements of the matrix
    /// @return reference to the object, a ComplexMatrix, the difference
    ComplexMatrix& operator-=(const Complex& rhs);
    /// a ComplexMatrix is subtracted from a ComplexMatrix and this ComplexMatrix object is changed
    /// @param rhs the right operant, a ComplexMatrix to
    /// @return reference to the object, a ComplexMatrix, the difference
    ComplexMatrix& operator-=(const ComplexMatrix& rhs);
    /// @brief a complex number times a ComplexMatrix, i.e. all elements are multiplied
    /// @param lhs the left operant, a ComplexMatrix
    /// @param rhs the right operant, a complex number
    /// @return a ComplexMatrix, the product of the ComplexMatrix times the complex number
    friend ComplexMatrix operator*(const ComplexMatrix& lhs, const Complex& rhs);
    /// @brief a complex number times a ComplexMatrix, i.e. all elements are multiplied
    /// @param lhs the left operant, a complex number
    /// @param rhs the right operant, a ComplexMatrix
    /// @return a ComplexMatrix, the product of the ComplexMatrix times the complex number
    friend ComplexMatrix operator*(const Complex& lhs, const ComplexMatrix& rhs);
    /// @brief the ComplexMatrix object is multiplied by the number
    /// @param rhs the right operant, a complex number
    /// @return reference to the object, a ComplexMatrix, the product
    ComplexMatrix& operator*=(const Complex& rhs);
    /// @brief a complex number divides a ComplexMatrix, i.e. all elements divide the number
    /// @param lhs the left operant, a ComplexMatrix
    /// @param rhs the right operant, a complex number
    /// @return a ComplexMatrix, the quotient of the ComplexMatrix dividing the complex number
    friend ComplexMatrix operator/(const ComplexMatrix& lhs, const Complex& rhs);
    /// @brief the ComplexMatrix object divides the number
    /// @param rhs the right operant, a complex number
    /// @return reference to the object, a ComplexMatrix, the quotient
    ComplexMatrix& operator/=(const Complex& rhs);
};

#endif // !MATRIX_H
