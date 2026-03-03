#ifndef MATRIX_HPP
#define MATRIX_HPP

class Matrix {
    private:
        unsigned rows;
        unsigned cols;
        float* data;    

    public:
        Matrix(unsigned rows, unsigned cols);
        ~Matrix();
//Treap &Treap::operator=(const Treap &other) {
        float operator[](unsigned row, unsigned col);
        const float operator[](unsigned row, unsigned col) const;

        void set(unsigned row, unsigned col, float value);
};

#endif