#include "include/matrix.hpp"

#include <stdexcept>
#include "include/types.hpp"
#include <random>

static fp32 sample() { // мб можно границу добавить 
  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_int_distribution<int> dis; 
  return static_cast<fp32>(dis(generator));
}


Matrix::Matrix(std::size_t rows, std::size_t cols) {
    this->rows = rows;
    this->cols = cols;
    this->data = new fp32[rows * cols];
}

Matrix::~Matrix() {
    delete[] this->data;
}

void Matrix::set(std::size_t row, std::size_t col, float value) {
    if (row >= this->rows || col >= this->cols) {
        throw std::out_of_range("Index out of bounds");
    }
    this->data[row * this->cols + col] = value;
}

float Matrix::get(std::size_t row, std::size_t col) const {
    if (row >= this->rows || col >= this->cols) {
        throw std::out_of_range("Index out of bounds");
    }
    return this->data[row * this->cols + col];
}

void Matrix::rng_fill() {
    for (std::size_t i = 0; i < rows; i++){
        for (std::size_t j = 0; j < cols; j++) {
            set(i, j, sample());
        }
    }
}
