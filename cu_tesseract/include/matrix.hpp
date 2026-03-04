#ifndef MATRIX_HPP
#define MATRIX_HPP

#include "types.hpp"

class Matrix {
    private:
        unsigned rows;
        unsigned cols;
        fp32* data;    

    public:
        Matrix(unsigned rows, unsigned cols);
        ~Matrix();

        void set(unsigned row, unsigned col, fp32 value);
        fp32 get(unsigned row, unsigned col) const;
        void rng_fill();
};

#endif