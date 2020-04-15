/******************************************************************************
** Template implementation for "matrix.h"
** by Robert Frysch | May 04, 2018
** Otto von Guericke University Magdeburg
** Institute for Medical Engineering - IMT (Head: Georg Rose)
** Email: robert.frysch@ovgu.de
******************************************************************************/

#include "matrix.h" // optional, only for the IDE
#include <stdexcept>

namespace CTL {

// ### MatrixBase ###
// constructors
template<uint Rows, uint Cols>
MatrixBase<Rows,Cols>::MatrixBase(double fillValue)
{
    std::fill(this->begin(), this->end(), fillValue);
}

template<uint Rows, uint Cols>
MatrixBase<Rows,Cols>::MatrixBase(const double (& initArray)[Rows*Cols])
{
    std::copy_n(initArray, Rows*Cols, _m);
}

// individual element access with 2 indizes
// -> run time boundary check (throws out_of_range)
template<uint Rows, uint Cols>
double & MatrixBase<Rows,Cols>::at(uint row, uint column) noexcept(false)
{
    if(row >= Rows)    throw std::out_of_range("row index exceeds matrix dimentions");
    if(column >= Cols) throw std::out_of_range("column index exceeds matrix dimentions");
    return (*this)[row][column];
}
template<uint Rows, uint Cols>
double MatrixBase<Rows,Cols>::at(uint row, uint column) const noexcept(false)
{
    if(row >= Rows)    throw std::out_of_range("row index exceeds matrix dimentions");
    if(column >= Cols) throw std::out_of_range("column index exceeds matrix dimentions");
    return (*this)[row][column];
}
// -> compile time boundary check (never fails)
template<uint Rows, uint Cols>
template<uint row, uint column>
double & MatrixBase<Rows,Cols>::get() noexcept
{
    static_assert(row < Rows,    "row index must not exceed matrix dimension");
    static_assert(column < Cols, "column index must not exceed matrix dimension");
    return (*this)[row][column];
}
template<uint Rows, uint Cols>
template<uint row, uint column>
double MatrixBase<Rows,Cols>::get() const noexcept
{
    static_assert(row < Rows,    "row index must not exceed matrix dimension");
    static_assert(column < Cols, "column index must not exceed matrix dimension");
    return (*this)[row][column];
}

// individual element access with 1 index
// -> run time boundary check (throws out_of_range)
template<uint Rows, uint Cols>
double & MatrixBase<Rows,Cols>::at(uint n) noexcept(false)
{
    if(n >= Rows*Cols) throw std::out_of_range("index exceeds matrix dimentions");
    return _m[n];
}
template<uint Rows, uint Cols>
double MatrixBase<Rows,Cols>::at(uint n) const noexcept(false)
{
    if(n >= Rows*Cols) throw std::out_of_range("index exceeds matrix dimentions");
    return _m[n];
}
// -> compile time boundary check (never fails)
template<uint Rows, uint Cols>
template<uint n>
double & MatrixBase<Rows,Cols>::get() noexcept
{
    static_assert(n < Rows*Cols, "index must not exceed matrix dimensions");
    return _m[n];
}
template<uint Rows, uint Cols>
template<uint n>
double MatrixBase<Rows,Cols>::get() const noexcept
{
    static_assert(n < Rows*Cols, "index must not exceed matrix dimensions");
    return _m[n];
}

// formatting matrix entries to a string
template<uint Rows, uint Cols>
std::string MatrixBase<Rows,Cols>::info() const
{
    static const uint numSpacesInBetween = 3;
    std::string entries[Rows*Cols];
    std::string * entryPtr = entries;
    size_t maxLength = 0;
    for(auto val : _m)
    {
        *entryPtr = std::to_string(val);
        maxLength = std::max(maxLength,entryPtr->size());
        ++entryPtr;
    }
    maxLength += numSpacesInBetween;
    std::string ret(Rows*Cols*maxLength, SEPARATOR_CHARACTER_FOR_INFO_STRING);
    entryPtr = entries;
    for(uint i = 0; i < Rows; ++i)
    {
        size_t rowOffSet = i*Cols*maxLength;
        for(uint j = 0; j < Cols; ++j)
        {
            size_t colOffSet = j*maxLength+1;
            size_t indentation = maxLength - entryPtr->size() - numSpacesInBetween;
            ret.replace(rowOffSet+colOffSet+indentation, entryPtr->size(), *entryPtr);
            ++entryPtr;
        }
        ret.replace(rowOffSet + Cols*maxLength - 2,2,"|\n");
        ret.replace(rowOffSet,1,1,'|');
    }
    return ret;
}

template<uint Rows, uint Cols>
double MatrixBase<Rows,Cols>::norm() const
{
#ifndef ENABLE_FROBENIUS_NORM
    static_assert(Rows==1 || Cols==1, "norm is currently only defined for vectors/scalars. "
                  "For using 'norm()' as the Frobenius norm of a matrix, "
                  "please define 'ENABLE_FROBENIUS_NORM' before including 'matrix.h'.");
#endif
    double ret = 0.0;
    for(auto val : _m)
        ret += val*val;
    return sqrt(ret);
}


// ### Matrix ###
// factory
template<uint Rows, uint Cols>
template<class Container>
Matrix<Rows,Cols> Matrix<Rows,Cols>::fromContainer(const Container &vector, size_t NthMat, bool * ok)
{
    auto offSet = NthMat*Rows*Cols;
    if( offSet + Rows*Cols > static_cast<size_t>(vector.size()) )
    {
        if(ok) *ok = false;
        return Matrix<Rows,Cols>(0.0);
    }
    Matrix<Rows,Cols> ret;
    auto vecIt = vector.begin() + offSet;
    for(auto & val : ret)
        val = static_cast<double>(*vecIt++);
    if(ok) *ok = true;
    return ret;
}

// single vector extraction
template<uint Rows, uint Cols>
template<uint i>
Matrix<1,Cols> Matrix<Rows,Cols>::row() const
{
    static_assert(i<Rows,"row index must not exceed matrix dimensions");
    Matrix<1,Cols> ret;
    std::copy_n((*this)[i], Cols, ret.begin());
    return ret;
}

template<uint Rows, uint Cols>
template<uint j>
Matrix<Rows,1> Matrix<Rows,Cols>::column() const
{
    static_assert(j<Cols,"column index must not exceed matrix dimensions");
    Matrix<Rows,1> ret;
    auto scrPtr = this->constBegin() + j;
    for(auto & val : ret)
    {
        val = *scrPtr;
        scrPtr += Cols;
    }
    return ret;
}

// unary operators
template<uint Rows, uint Cols>
Matrix<Cols,Rows> Matrix<Rows,Cols>::transposed() const
{
    Matrix<Cols,Rows> ret;
    for(uint column = 0; column < Cols; ++column)
        for(uint row = 0; row < Rows; ++row)
            ret(column,row) = (*this)(row,column);
    return ret;
}

template<uint Rows, uint Cols>
Matrix<Rows,Cols> Matrix<Rows,Cols>::operator- () const
{
    Matrix<Rows,Cols> ret(*this);
    for(auto & val : ret)
        val = -val;
    return ret;
}

// compound assignment
template<uint Rows, uint Cols>
Matrix<Rows,Cols> & Matrix<Rows,Cols>::operator*= (double scalar)
{
    for(auto & val : *this)
        val *= scalar;
    return *this;
}

template<uint Rows, uint Cols>
Matrix<Rows,Cols> & Matrix<Rows,Cols>::operator/= (double scalar)
{
    for(auto & val : *this)
        val /= scalar;
    return *this;
}

template<uint Rows, uint Cols>
Matrix<Rows,Cols> & Matrix<Rows,Cols>::operator+= (const Matrix<Rows,Cols> & other)
{
    auto otherIt = other.begin();
    for(auto & val : *this)
        val += *otherIt++;
    return *this;
}

template<uint Rows, uint Cols>
Matrix<Rows,Cols> & Matrix<Rows,Cols>::operator-= (const Matrix<Rows,Cols> & other)
{
    auto otherIt = other.begin();
    for(auto & val : *this)
        val -= *otherIt++;
    return *this;
}

// binary operators
template<uint Rows, uint Cols>
Matrix<Rows,Cols> Matrix<Rows,Cols>::operator* (double scalar) const
{
    Matrix<Rows, Cols> ret(*this);
    ret *= scalar;
    return ret;
}

template<uint Rows, uint Cols>
Matrix<Rows,Cols> Matrix<Rows,Cols>::operator/ (double scalar) const
{
    Matrix<Rows, Cols> ret(*this);
    ret /= scalar;
    return ret;
}

template<uint Rows, uint Cols>
Matrix<Rows,Cols> Matrix<Rows,Cols>::operator+ (const Matrix<Rows,Cols> & rhs) const
{
    Matrix<Rows,Cols> ret;
    auto m1It = this->begin(), m2It = rhs.begin();
    for(auto & val : ret)
        val = *m1It++ + *m2It++;
    return ret;
}

template<uint Rows, uint Cols>
Matrix<Rows,Cols> Matrix<Rows,Cols>::operator- (const Matrix<Rows,Cols> & rhs) const
{
    Matrix<Rows,Cols> ret;
    auto m1It = this->begin(), m2It = rhs.begin();
    for(auto & val : ret)
        val = *m1It++ - *m2It++;
    return ret;
}

template<uint Rows1, uint Cols1_Rows2>
template<uint Cols2> // standard matrix multiplication
Matrix<Rows1,Cols2> Matrix<Rows1,Cols1_Rows2>::operator* (const Matrix<Cols1_Rows2,Cols2> & rhs) const
{
    Matrix<Rows1,Cols2> ret(0.0);
    const Matrix<Rows1,Cols1_Rows2> & lhs = *this;

    for(uint row = 0; row < Rows1; ++row)
        for(uint col = 0; col < Cols2; ++col)
            for(uint i = 0; i < Cols1_Rows2; ++i)
                ret(row,col) += lhs(row,i) * rhs(i,col);
    return ret;
}


// ### Global functions ###
// concatenation
template<uint Rows, uint Cols1, uint Cols2>
Matrix<Rows,Cols1+Cols2>
horzcat(const Matrix<Rows,Cols1> & m1, const Matrix<Rows,Cols2> & m2)
{
    Matrix<Rows,Cols1+Cols2> ret;
    auto dstPtr = ret.begin();
    for(uint row = 0; row < Rows; ++row)
    {
        std::copy_n(m1[row],Cols1,dstPtr);
        dstPtr += Cols1;
        std::copy_n(m2[row],Cols2,dstPtr);
        dstPtr += Cols2;
    }
    return ret;
}

template<uint Rows1, uint Rows2, uint Cols>
Matrix<Rows1+Rows2,Cols>
vertcat(const Matrix<Rows1,Cols> & m1, const Matrix<Rows2,Cols> & m2)
{
    Matrix<Rows1+Rows2,Cols> ret;
    std::copy( m1.begin(), m1.end(), ret[0] );
    std::copy( m2.begin(), m2.end(), ret[Rows1] );
    return ret;
}

// identity matrix
template<uint N>
Matrix<N,N> eye()
{
    Matrix<N,N> ret(0.0);
    for(uint i = 0; i<N; ++i)
        ret(i,i) = 1.0;
    return ret;
}


} // namespace CTL
