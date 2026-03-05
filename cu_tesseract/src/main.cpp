#include "matrix.hpp"

#include <iostream>

using std::cout;
using std::endl;

signed main() {
    Matrix<fp32> m = Matrix<fp32>(3, 3);
    m.rng_fill();
    for (auto i = 0; i < 3; i++) {
        for (auto j = 0; j < 3; j++) {
            cout << m.get(i, j) << " ";
        }
        cout << endl;
    }
}
