#ifndef MATRIX_HPP
#define MATRIX_HPP

#include "types.hpp"

#include <cstddef>
#include <random>
#include <stdexcept>

template <typename T>
class Matrix {
    private:
        size_t rows;
        size_t cols;
        T* data;

        static T sample() {;
            std::random_device rd;
            std::mt19937 generator(rd());
            std::uniform_real_distribution<double> dis(0.0, 1.0);
            return static_cast<T>(dis(generator));
        }

    public:
        Matrix(size_t rows, size_t cols): rows(rows), cols(cols), data(new T[rows * cols]) {}

        ~Matrix() {
            delete[] data;
        }

        void set(size_t row, size_t col, T value) {
            if (row >= rows || col >= cols) {
                throw std::out_of_range("Index out of bounds");
            }
            data[row * cols + col] = value;
        }

        T get(size_t row, size_t col) const {
            if (row >= rows || col >= cols) {
                throw std::out_of_range("Index out of bounds");
            }
            return data[row * cols + col];
        }

        void rng_fill() {
            for (size_t i = 0; i < rows; i++) {
                for (size_t j = 0; j < cols; j++) {
                    set(i, j, sample());
                }
            }
        }
};

#endif