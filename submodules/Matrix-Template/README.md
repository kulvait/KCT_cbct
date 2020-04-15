# Example Code for Matrix Template Class

```cpp
#define _USE_MATH_DEFINES
#include "matrix.h"
#include <vector>
#include <iostream>

int main()
{
    using namespace CT;
    using std::cout;
    using std::endl;

    Matrix<2,2> R({ cos(M_PI_4), -sin(M_PI_4),
                    sin(M_PI_4),  cos(M_PI_4) });
    Matrix<2,1> x({ 1.0, 1.0 });
    auto y = R*x;
    cout << "__Matrix-Vector Multiplication__\n"
         << "R:\n"       << R.info() << endl
         << "x:\n"       << x.info() << endl
         << "x.norm(): " << x.norm() << endl
         << "\ny=R*x:\n" << y.info() << endl
         << "y^T:\n"     << y.transposed().info() << endl
         << endl;

    auto I = eye<3>(); // creates a 3x3 identity matrix
    I *= 42;
    Matrix<3,3> O(1.0); // fill a 3x3 matrix with '1.0'
    cout << "__Matrix-Matrix Multiplication__\n"
         << "I:\n"    << I.info()     << endl
         << "O:\n"    << O.info()     << endl
         << "I*O:\n"  << (I*O).info() << endl;

    std::vector<double> matStack(9*50, 0.1337);
    auto selectedMat = Matrix<3,3>::fromContainer(matStack,42); // select 42th matrix
    Matrix<3,1> t({1.0,3.0,7.0});
    auto catMat = horzcat(selectedMat,-t);
    auto lastRow = catMat.row<2>();
    auto lastCol = catMat.column<3>();
    cout << "__Matrix from Container and Concatenation__\n"
         << "selectedMat:\n" << selectedMat.info()        << endl
         << "t:\n"           << t.info()                  << endl
         << "horzcat(selectedMat,-t):\n" << catMat.info() << endl
         << "lastRow:\n"     << lastRow.info()            << endl
         << "lastColumn:\n"  << lastCol.info()            << endl;

    Matrix<1,4> vector({1.0,2.0,3.0,4.0});
    auto scalar = vector*vector.transposed();
    auto scalarInfo = scalar.info(); scalarInfo.pop_back(); // remove new line from end
    auto func   = [](double val){return val;};
    cout << "__Element Access and Scalars__\n"
         << "vector:\n"         << vector.info()
         << "vector[0]:\t"      << vector[0]      << " (double *)\n"
         << "vector[0][1]:\t"   << vector[0][1]   << " (double)\n"
         << "vector(0,2):\t"    << vector(0,2)    << " (double)\n"
         << "vector(3):\t"      << vector(3)      << " (double)\n"
         << endl;
    cout << "scalar = vector*vector.transposed():\n"
         << "scalar.info():\t"  << scalarInfo     << " (string)\n"
         << "scalar.value():\t" << scalar.value() << " (double)\n"
         << "scalar.ref():\t"   << scalar.ref()   << " (double &)\n"
         << "func(double):\t"   << func(scalar)   << " (implicit cast to 'double')\n"
         << endl;

    #if __cplusplus >= 201402L
    Matrix<1,4> v_t({0.5,1.0,1.5,2.0});
    auto multiCat = vertcat(v_t, 2*v_t, 3*v_t, 4*eye<4>());
    cout << "__C++14 features__\n"
         << "multiCat = vertcat(v_t, 2*v_t, 3*v_t, 4*eye<4>()):\n"
         << multiCat.info() << endl;
    #endif

    return 0;
}
```

possible output:
```
__Matrix-Vector Multiplication__
R:
|_0.707107___-0.707107|
|_0.707107____0.707107|

x:
|1.000000|
|1.000000|

x.norm(): 1.41421

y=R*x:
|0.000000|
|1.414214|

y^T:
|0.000000___1.414214|


__Matrix-Matrix Multiplication__
I:
|42.000000____0.000000____0.000000|
|_0.000000___42.000000____0.000000|
|_0.000000____0.000000___42.000000|

O:
|1.000000___1.000000___1.000000|
|1.000000___1.000000___1.000000|
|1.000000___1.000000___1.000000|

I*O:
|42.000000___42.000000___42.000000|
|42.000000___42.000000___42.000000|
|42.000000___42.000000___42.000000|

__Matrix from Container and Concatenation__
selectedMat:
|0.133700___0.133700___0.133700|
|0.133700___0.133700___0.133700|
|0.133700___0.133700___0.133700|

t:
|1.000000|
|3.000000|
|7.000000|

horzcat(selectedMat,-t):
|_0.133700____0.133700____0.133700___-1.000000|
|_0.133700____0.133700____0.133700___-3.000000|
|_0.133700____0.133700____0.133700___-7.000000|

lastRow:
|_0.133700____0.133700____0.133700___-7.000000|

lastColumn:
|-1.000000|
|-3.000000|
|-7.000000|

__Element Access and Scalars__
vector:
|1.000000___2.000000___3.000000___4.000000|
vector[0]:	0x7fff6123c400 (double *)
vector[0][1]:	2 (double)
vector(0,2):	3 (double)
vector(3):	4 (double)

scalar = vector*vector.transposed():
scalar.info():	|30.000000| (string)
scalar.value():	30 (double)
scalar.ref():	30 (double &)
func(double):	30 (implicit cast to 'double')

__C++14 features__
multiCat = vertcat(v_t, 2*v_t, 3*v_t, 4*eye<4>()):
|0.500000___1.000000___1.500000___2.000000|
|1.000000___2.000000___3.000000___4.000000|
|1.500000___3.000000___4.500000___6.000000|
|4.000000___0.000000___0.000000___0.000000|
|0.000000___4.000000___0.000000___0.000000|
|0.000000___0.000000___4.000000___0.000000|
|0.000000___0.000000___0.000000___4.000000|
```
