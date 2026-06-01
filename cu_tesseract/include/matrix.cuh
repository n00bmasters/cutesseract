#ifndef MATRIX_CUH
#define MATRIX_CUH

#include <random>

#include "dtypes.cuh"
#include "utils.cuh"
#include "MatrixView.cuh"
#include "tensor.cuh"

template <typename T> class Matrix : public Tensor<T> {
  DataLayout layout;

public:
  __host__ Matrix(size_t rows, size_t cols, DataLayout layout,
                  DataDevice target_device)
      : Tensor<T>({rows, cols}), layout(layout) {
    if (layout == DataLayout::COL_WISE) {
        // We need to recompute strides for COL_WISE
        this->set_col_wise_strides();
    }
    if (target_device == DataDevice::CUDA) {
      this->cuda();
    }
  }

  __host__ Matrix(const Matrix &other) : Tensor<T>(other), layout(other.layout) {}

  __host__ Matrix(T* ptr, size_t rows, size_t cols, DataLayout layout, DataDevice device)
      : Tensor<T>({rows, cols}), layout(layout) {
    throw std::runtime_error("wont work with current implementation");
  }

  __host__ Matrix &operator=(const Matrix &other) {
    if (this == &other)
      return *this;
    Tensor<T>::operator=(other);
    layout = other.layout;
    return *this;
  }

  __host__ Matrix(Matrix &&other) : Tensor<T>(std::move(other)), layout(other.layout) {}

  __host__ Matrix &operator=(Matrix &&other) {
    if (this == &other)
      return *this;
    Tensor<T>::operator=(std::move(other));
    layout = other.layout;
    return *this;
  }

  __host__ Matrix operator+(const Matrix &other) const {
    size_t rows = this->get_shape(0);
    size_t cols = this->get_shape(1);
    assert(rows == other.get_shape(0) && cols == other.get_shape(1));
    assert(layout == other.layout);
    assert(this->device == other.device);

    Matrix result(rows, cols, layout, this->device);

    if (this->device == DataDevice::CPU) {
      T* res_ptr = result.item();
      T* this_ptr = this->item();
      T* other_ptr = other.item();
      for (size_t i = 0; i < rows * cols; ++i) {
        res_ptr[i] = this_ptr[i] + other_ptr[i];
      }
    } else {
      dim3 block(256);
      dim3 grid((rows * cols + block.x - 1) / block.x);
      matrix_add_kernel<T><<<grid, block>>>(this->item(), other.item(),
                                            result.item(),
                                            rows * cols);
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    return result;
  }

  __host__ Matrix operator-(const Matrix &other) const {
    size_t rows = this->get_shape(0);
    size_t cols = this->get_shape(1);
    assert(rows == other.get_shape(0) && cols == other.get_shape(1));
    assert(layout == other.layout);
    assert(this->device == other.device);

    Matrix result(rows, cols, layout, this->device);

    if (this->device == DataDevice::CPU) {
      T* res_ptr = result.item();
      T* this_ptr = this->item();
      T* other_ptr = other.item();
      for (size_t i = 0; i < rows * cols; ++i) {
        res_ptr[i] = this_ptr[i] - other_ptr[i];
      }
    } else {
      dim3 block(256);
      dim3 grid((rows * cols + block.x - 1) / block.x);
      matrix_sub_kernel<T><<<grid, block>>>(this->item(), other.item(),
                                            result.item(),
                                            rows * cols);
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    return result;
  }

  __host__ Matrix &operator+=(const Matrix &other) {
    size_t rows = this->get_shape(0);
    size_t cols = this->get_shape(1);
    assert(rows == other.get_shape(0) && cols == other.get_shape(1));
    assert(layout == other.layout);
    assert(this->device == other.device);

    if (this->device == DataDevice::CPU) {
      T* this_ptr = this->item();
      T* other_ptr = other.item();
      for (size_t i = 0; i < rows * cols; ++i) {
        this_ptr[i] += other_ptr[i];
      }
    } else {
      dim3 block(256);
      dim3 grid((rows * cols + block.x - 1) / block.x);
      matrix_add_kernel<T><<<grid, block>>>(
          this->item(), other.item(), this->item(), rows * cols);
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    return *this;
  }

  __host__ Matrix &operator-=(const Matrix &other) {
    size_t rows = this->get_shape(0);
    size_t cols = this->get_shape(1);
    assert(rows == other.get_shape(0) && cols == other.get_shape(1));
    assert(layout == other.layout);
    assert(this->device == other.device);

    if (this->device == DataDevice::CPU) {
      T* this_ptr = this->item();
      T* other_ptr = other.item();
      for (size_t i = 0; i < rows * cols; ++i) {
        this_ptr[i] -= other_ptr[i];
      }
    } else {
      dim3 block(256);
      dim3 grid((rows * cols + block.x - 1) / block.x);
      matrix_sub_kernel<T><<<grid, block>>>(
          this->item(), other.item(), this->item(), rows * cols);
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    return *this;
  }

  __host__ void swap(Matrix &other) {
    Tensor<T>::swap(other);
    std::swap(layout, other.layout);
  }

  __host__ void fill_random(unsigned long long seed = 812ULL) {
    Tensor<T>::fill_random(seed);
  }

  __host__ void to_layout(DataLayout new_layout) {
    if (layout == new_layout)
      return;

    if (this->device == DataDevice::CUDA) {
      throw std::runtime_error(
          ".to_layout not implemented for DataDevice::CUDA. consider using .cpu()");
    }

    size_t rows = this->get_shape(0);
    size_t cols = this->get_shape(1);
    T* ptr = this->item();
    T *new_buffer = new T[rows * cols];
    for (size_t i = 0; i < rows; i++) {
      for (size_t j = 0; j < cols; j++) {
        if (new_layout == DataLayout::ROW_WISE) {
          // Current is COL_WISE: i + j * rows
          new_buffer[i * cols + j] = ptr[i + j * rows];
        } else {
          // Current is DataLayout::ROW_WISE: i * cols + j
          new_buffer[i + j * rows] = ptr[i * cols + j];
        }
      }
    }

   // If we update strides, we don't need to move data. 
    // BUT the original implementation physicaly moved data.
    
    memcpy(ptr, new_buffer, rows * cols * sizeof(T));
    delete[] new_buffer;
    layout = new_layout;
    if (layout == DataLayout::ROW_WISE) {
        this->compute_contiguous_strides();
    } else {
        this->set_col_wise_strides();
    }
  }

  __host__ friend std::ostream &operator<<(std::ostream &os,
                                           const Matrix &matrix) {
    if (matrix.device == DataDevice::CUDA) {
      throw std::runtime_error(
          "data must be on cpu for printing. consider calling .cpu()");
    }
    size_t rows = matrix.get_shape(0);
    size_t cols = matrix.get_shape(1);

    for (size_t i = 0; i < rows; i++) {
      os << "[";
      for (size_t j = 0; j < cols; j++) {
        os << matrix.get(i, j);
        if (j != cols - 1)
          os << ", ";
      }
      os << "]\n";
    }

    return os;
  }

  __host__ std::pair<size_t, size_t> shape() const { 
    return {this->get_shape(0), this->get_shape(1)}; 
  }

  __host__ DataLayout get_layout() const { return layout; }

  __host__ T get(size_t i, size_t j) const {
    size_t rows = this->get_shape(0);
    size_t cols = this->get_shape(1);
    if (i >= rows || j >= cols) {
      throw std::out_of_range("Index out of bounds");
    }
    if (this->device == DataDevice::CUDA) {
      throw std::runtime_error(
          "data must be on cpu to get value. consider calling .cpu()");
    }

    return this->item()[i * this->get_stride(0) + j * this->get_stride(1)];
  }

  __host__ void set(size_t i, size_t j, T val) {
    size_t rows = this->get_shape(0);
    size_t cols = this->get_shape(1);
    if (i >= rows || j >= cols) {
      throw std::out_of_range("Index out of bounds");
    }
    if (this->device == DataDevice::CUDA) {
      throw std::runtime_error(
          "data must be on cpu to set value. consider calling .cpu()");
    }

    this->item()[i * this->get_stride(0) + j * this->get_stride(1)] = val;
  }

private:
    __host__ void set_col_wise_strides() {
        // stride[0] = 1, stride[1] = rows.
        this->set_stride(0, 1);
        this->set_stride(1, this->get_shape(0));
    }
};

#endif

