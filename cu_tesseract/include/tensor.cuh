#pragma once

#include <cassert>
#include <cstddef>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>
#include <random>

#include "dtypes.cuh"
#include "utils.cuh"

template <typename T>
class Tensor {
  T *cpu_ptr;
  T *device_ptr;

  size_t ndim;
  size_t shape_[8];
  size_t strides_[8];
  size_t capacity_;
  bool foreign_pointer = false;
public:
  DataDevice device;

  __host__ Tensor(std::initializer_list<size_t> dims) : ndim(dims.size()), capacity_(1) {
    size_t i = 0;
    for (auto d : dims) {
      shape_[i] = d;
      capacity_ *= d;
      ++i;
    }
    compute_contiguous_strides();
    cpu_ptr = new T[capacity_];
    device_ptr = nullptr;
    device = DataDevice::CPU;
  }

  __host__ Tensor(const std::vector<size_t>& dims) : ndim(dims.size()), capacity_(1) {
    for (size_t i = 0; i < ndim; ++i) {
      shape_[i] = dims[i];
      capacity_ *= dims[i];
    }
    compute_contiguous_strides();
    cpu_ptr = new T[capacity_];
    device_ptr = nullptr;
    device = DataDevice::CPU;
  }

  __host__ ~Tensor() {
    if (foreign_pointer) return;
    if (device == DataDevice::CUDA) {
      CUDA_CHECK(cudaFree(device_ptr));
    } else {
      delete[] cpu_ptr;
    }
  }

  __host__ Tensor(const Tensor &other)
      : cpu_ptr(nullptr), device_ptr(nullptr), ndim(other.ndim),
        capacity_(other.capacity_), device(other.device) {
    for (size_t i = 0; i < ndim; ++i) {
      shape_[i] = other.shape_[i];
      strides_[i] = other.strides_[i];
    }
    if (device == DataDevice::CPU) {
      cpu_ptr = new T[capacity_];
      memcpy(cpu_ptr, other.cpu_ptr, capacity_ * sizeof(T));
    } else {
      CUDA_CHECK(cudaMalloc(&device_ptr, capacity_ * sizeof(T)));
      CUDA_CHECK(cudaMemcpy(device_ptr, other.device_ptr,
                            capacity_ * sizeof(T), cudaMemcpyDeviceToDevice));
    }
  }

  __host__ Tensor &operator=(const Tensor &other) {
    if (this == &other) return *this;

    if (device == DataDevice::CPU) {
      delete[] cpu_ptr;
    } else {
      CUDA_CHECK(cudaFree(device_ptr));
    }

    ndim = other.ndim;
    capacity_ = other.capacity_;
    device = other.device;
    for (size_t i = 0; i < ndim; ++i) {
      shape_[i] = other.shape_[i];
      strides_[i] = other.strides_[i];
    }

    if (device == DataDevice::CPU) {
      cpu_ptr = new T[capacity_];
      memcpy(cpu_ptr, other.cpu_ptr, capacity_ * sizeof(T));
      device_ptr = nullptr;
    } else {
      CUDA_CHECK(cudaMalloc(&device_ptr, capacity_ * sizeof(T)));
      CUDA_CHECK(cudaMemcpy(device_ptr, other.device_ptr,
                            capacity_ * sizeof(T), cudaMemcpyDeviceToDevice));
      cpu_ptr = nullptr;
    }

    return *this;
  }

  __host__ Tensor(Tensor &&other)
      : cpu_ptr(nullptr), device_ptr(nullptr), ndim(0), capacity_(0),
        device(DataDevice::CPU) {
    this->swap(other);
  }

  __host__ Tensor &operator=(Tensor &&other) {
    if (this == &other) return *this;
    this->swap(other);
    return *this;
  }

  __host__ static Tensor view(T* ptr, const std::vector<size_t>& dims,
                              const std::vector<size_t>& strides,
                              DataDevice device) {
    Tensor t;
    t.ndim = dims.size();
    t.capacity_ = 1;
    for (size_t i = 0; i < t.ndim; ++i) {
      t.shape_[i] = dims[i];
      t.strides_[i] = strides[i];
      t.capacity_ *= dims[i];
    }
    t.device = device;
    t.foreign_pointer = true;
    if (device == DataDevice::CUDA) {
      t.device_ptr = ptr;
      t.cpu_ptr = nullptr;
    } else {
      t.device_ptr = nullptr;
      t.cpu_ptr = ptr;
    }
    return t;
  }

  __host__ static Tensor view(T* ptr, const std::vector<size_t>& dims,
                              DataDevice device) {
    Tensor t;
    t.ndim = dims.size();
    t.capacity_ = 1;
    for (size_t i = 0; i < t.ndim; ++i) {
      t.shape_[i] = dims[i];
      t.capacity_ *= dims[i];
    }
    t.compute_contiguous_strides();
    t.device = device;
    t.foreign_pointer = true;
    if (device == DataDevice::CUDA) {
      t.device_ptr = ptr;
      t.cpu_ptr = nullptr;
    } else {
      t.device_ptr = nullptr;
      t.cpu_ptr = ptr;
    }
    return t;
  }

  __host__ __device__ size_t num_elements() const {
    size_t n = 1;
    for (size_t i = 0; i < ndim; ++i) n *= shape_[i];
    return n;
  }

  __host__ __device__ size_t get_ndim() const { return ndim; }
  __host__ __device__ size_t get_shape(size_t d) const { return shape_[d]; }
  __host__ __device__ size_t get_stride(size_t d) const { return strides_[d]; }
  __host__ void set_stride(size_t d, size_t s) { strides_[d] = s; }

  __host__ void compute_contiguous_strides() {
    size_t s = 1;
    for (int i = static_cast<int>(ndim) - 1; i >= 0; --i) {
      strides_[i] = s;
      s *= shape_[i];
    }
  }

  __host__ const size_t* shape_data() const { return shape_; }
  __host__ const size_t* stride_data() const { return strides_; }

  __host__ T *item() const {
    if (device == DataDevice::CPU) {
      return cpu_ptr;
    } else {
      return device_ptr;
    }
  }

  __host__ void cpu() {
    if (device == DataDevice::CPU) return;

    T* new_ptr = new T[capacity_];
    CUDA_CHECK(cudaMemcpy(new_ptr, device_ptr, capacity_ * sizeof(T),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(device_ptr));
    device_ptr = nullptr;
    cpu_ptr = new_ptr;
    device = DataDevice::CPU;
  }

  __host__ void cuda() {
    if (device == DataDevice::CUDA) return;

    T* new_ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&new_ptr, capacity_ * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(new_ptr, cpu_ptr, capacity_ * sizeof(T),
                          cudaMemcpyHostToDevice));
    delete[] cpu_ptr;
    cpu_ptr = nullptr;
    device_ptr = new_ptr;
    device = DataDevice::CUDA;
  }

  __host__ void fill_random(unsigned long long seed = 812ULL) {
    if (device == DataDevice::CUDA) {
      curandGenerator_t gen;
      CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
      CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, seed));
      assert(sizeof(T) == sizeof(fp32));
      CURAND_CHECK(curandGenerateUniform(gen, (float*)device_ptr, capacity_));
      CURAND_CHECK(curandDestroyGenerator(gen));
    } else {
      std::mt19937 gen(seed);
      std::uniform_real_distribution<T> dis(0.0, 1.0);
      for (size_t i = 0; i < capacity_; i++)
        cpu_ptr[i] = dis(gen);
    }
  }

  __host__ void fill_const(T val) {
    if (device == DataDevice::CUDA) {
      T *h_ptr = new T[capacity_];
      for (size_t i = 0; i < capacity_; i++) h_ptr[i] = val;
      CUDA_CHECK(cudaMemcpy(device_ptr, h_ptr, capacity_ * sizeof(T), cudaMemcpyHostToDevice));
      delete[] h_ptr;
    } else {
      for (size_t i = 0; i < capacity_; i++) cpu_ptr[i] = val;
    }
  }

  __host__ void zeros() {
    if (device == DataDevice::CUDA) {
      CUDA_CHECK(cudaMemset(device_ptr, 0, capacity_ * sizeof(T)));
    } else {
      memset(cpu_ptr, 0, capacity_ * sizeof(T));
    }
  }

  __host__ void ones() { fill_const((T)1.0); }

  __host__ Tensor subview(const std::vector<size_t>& offsets,
                          const std::vector<size_t>& sizes) const {
    assert(offsets.size() == ndim && sizes.size() == ndim);

    Tensor out;
    out.ndim = ndim;
    out.device = device;
    out.foreign_pointer = true;

    size_t lin_offset = 0;
    out.capacity_ = 1;
    for (size_t i = 0; i < ndim; ++i) {
      assert(offsets[i] + sizes[i] <= shape_[i]);
      out.shape_[i] = sizes[i];
      out.strides_[i] = strides_[i];
      lin_offset += offsets[i] * strides_[i];
      out.capacity_ *= sizes[i];
    }

    T* base = (device == DataDevice::CPU) ? cpu_ptr : device_ptr;
    out.cpu_ptr = nullptr;
    out.device_ptr = nullptr;
    if (device == DataDevice::CPU) {
      out.cpu_ptr = base + lin_offset;
    } else {
      out.device_ptr = base + lin_offset;
    }
    return out;
  }

  __host__ void swap(Tensor &other) {
    std::swap(cpu_ptr, other.cpu_ptr);
    std::swap(device_ptr, other.device_ptr);
    std::swap(device, other.device);
    std::swap(ndim, other.ndim);
    std::swap(capacity_, other.capacity_);
    std::swap(foreign_pointer, other.foreign_pointer);
    for (size_t i = 0; i < 8; ++i) {
      std::swap(shape_[i], other.shape_[i]);
      std::swap(strides_[i], other.strides_[i]);
    }
  }

  __host__ friend std::ostream &operator<<(std::ostream &os,
                                           const Tensor &t) {
    if (t.device == DataDevice::CUDA) {
      throw std::runtime_error(
          "Tensor must be on CPU to print. Call .cpu() first.");
    }

    if (t.ndim == 1) {
      os << "[";
      for (size_t i = 0; i < t.shape_[0]; ++i) {
        os << t.cpu_ptr[i * t.strides_[0]];
        if (i + 1 < t.shape_[0]) os << ", ";
      }
      os << "]";
    } else if (t.ndim == 2) {
      for (size_t i = 0; i < t.shape_[0]; ++i) {
        os << "[";
        for (size_t j = 0; j < t.shape_[1]; ++j) {
          os << t.cpu_ptr[i * t.strides_[0] + j * t.strides_[1]];
          if (j + 1 < t.shape_[1]) os << ", ";
        }
        os << "]\n";
      }
    } else {
      os << "Tensor(" << t.ndim << "d) [";
      for (size_t i = 0; i < t.ndim; ++i) {
        os << t.shape_[i];
        if (i + 1 < t.ndim) os << "x";
      }
      os << "]";
    }
    return os;
  }

private:
  __host__ Tensor() : cpu_ptr(nullptr), device_ptr(nullptr), ndim(0),
                      capacity_(0), device(DataDevice::CPU) {}
};
