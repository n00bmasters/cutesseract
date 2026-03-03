#include "include/matrix.hpp"

#include <stdexcept>
Matrix::Matrix(unsigned rows, unsigned cols) {
    this->rows = rows;
    this->cols = cols;
    this->data = new float[rows * cols];
}

Matrix::~Matrix() {
    delete[] this->data;
}

void Matrix::set(unsigned row, unsigned col, float value) {
    if (row >= this->rows || col >= this->cols) {
        throw std::out_of_range("Index out of bounds");
    }
    this->data[row * this->cols + col] = value;
}

float* Matrix::operator[](unsigned row, unsigned col) {
    if (row >= this->rows) {
        throw std::out_of_range("Index out of bounds");
    }
    return &this->data[row * this->cols + col];
}
