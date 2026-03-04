#ifndef MATRIX_HPP
#define MATRIX_HPP

#include "types.hpp"

class Matrix {
    private:
        std::size_t rows;
        std::size_t cols;
        fp32* data;    

    public:
        Matrix(std::size_t rows, std::size_t cols);
        ~Matrix();

        void set(std::size_t row, std::size_t col, fp32 value);
        fp32 get(std::size_t row, std::size_t col) const;
        void rng_fill();
};

#endif