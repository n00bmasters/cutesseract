#include "include/matrix.hpp"

#include <stdexcept>
#include <random>

static float sample() { // мб можно границу добавить 
  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_int_distribution<int> dis; 
  return static_cast<float>(dis(generator));
}


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

float Matrix::get(unsigned row, unsigned col) const {
    if (row >= this->rows || col >= this->cols) {
        throw std::out_of_range("Index out of bounds");
    }
    return this->data[row * this->cols + col];
}

void Matrix::rng_fill() {
    for (unsigned i = 0; i < rows; i++){
        for (unsigned j = 0; j < cols; j++) {
            set(i, j, sample());
        }
    }
}
