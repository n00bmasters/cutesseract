
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "kernels.cuh"
#include "matrix.cuh"
#include "ndim_kernels.cuh"
#include "tensor.cuh"
#include "strassen_kernel.cuh"
#include "test_class_matrix.cu"

using std::cin;
using std::cout;
using std::endl;
using std::function;
using std::map;
using std::string;
using std::vector;

#ifndef RUNS_NUM
#define RUNS_NUM 4
#endif

enum class FillType { RANDOM, ONES, ZEROS };

typedef function<void(Matrix<fp32> &, Matrix<fp32> &, Matrix<fp32> &)>
    KernelFunc;

template <typename T> T calculate_max_diff(Matrix<T> &A, Matrix<T> &B) {
  assert(A.device == DataDevice::CPU && B.device == DataDevice::CPU);
  size_t rowsA = A.get_shape(0);
  size_t colsA = A.get_shape(1);
  size_t rowsB = B.get_shape(0);
  size_t colsB = B.get_shape(1);
  assert(rowsA == rowsB && colsA == colsB);
  T max_diff = 0.0;
  for (size_t i = 0; i < rowsA; i++) {
    for (size_t j = 0; j < colsA; j++) {
      T diff = std::abs(A.get(i, j) - B.get(i, j));
      if (diff > max_diff) {
        max_diff = diff;
      }
    }
  }
  return max_diff;
}

template <typename T> Matrix<T> mmul_cpu(Matrix<T> &A, Matrix<T> &B) {
  assert(A.device == DataDevice::CPU && B.device == DataDevice::CPU);
  size_t rowsA = A.get_shape(0);
  size_t colsA = A.get_shape(1);
  size_t rowsB = B.get_shape(0);
  size_t colsB = B.get_shape(1);

  assert(colsA == rowsB);
  Matrix<T> C(rowsA, colsB, DataLayout::ROW_WISE, DataDevice::CPU);
  for (size_t i = 0; i < rowsA; i++) {
    for (size_t j = 0; j < colsB; j++) {
      double sum = 0.0;
      for (size_t r = 0; r < colsA; r++) {
        sum += (double)A.get(i, r) * (double)B.get(r, j);
      }
      C.set(i, j, (T)sum);
    }
  }
  return C;
}

template <typename T>
void print_heatmap(Matrix<T> &GPU_C, Matrix<T> &CPU_C, T precision) {
  size_t rowsGPU = GPU_C.get_shape(0);
  size_t colsGPU = GPU_C.get_shape(1);
  size_t rowsCPU = CPU_C.get_shape(0);
  size_t colsCPU = CPU_C.get_shape(1);
  assert(rowsGPU == rowsCPU && colsGPU == colsCPU);
  assert(GPU_C.device == DataDevice::CPU && CPU_C.device == DataDevice::CPU);
  size_t rows = rowsGPU;
  size_t cols = colsGPU;
  size_t grid_r = std::min(rows, (size_t)32);
  size_t grid_c = std::min(cols, (size_t)32);
  size_t step_r = (rows + grid_r - 1) / grid_r;
  size_t step_c = (cols + grid_c - 1) / grid_c;

  cout << "\nError Heatmap (" << grid_r << "x" << grid_c
       << " sampling):" << endl;
  for (size_t i = 0; i < grid_r; i++) {
    for (size_t j = 0; j < grid_c; j++) {
      bool has_error = false;
      for (size_t bi = i * step_r; bi < std::min((i + 1) * step_r, rows); bi++) {
        for (size_t bj = j * step_c; bj < std::min((j + 1) * step_c, cols); bj++) {
          if (std::abs(GPU_C.get(bi, bj) - CPU_C.get(bi, bj)) > precision) {
            has_error = true;
            break;
          }
        }
        if (has_error)
          break;
      }
      cout << (has_error ? "X" : ".");
    }
    cout << endl;
  }
}

void verify_result(Matrix<fp32> &GPU_C, Matrix<fp32> &CPU_C,
                   fp32 precision = 1e-3) {
  assert(GPU_C.device == DataDevice::CPU && CPU_C.device == DataDevice::CPU);
  fp32 max_diff = calculate_max_diff(GPU_C, CPU_C);
  if (max_diff > precision) {
    cout << "[FAILED] Max difference: " << std::scientific << max_diff << endl;
    print_heatmap(GPU_C, CPU_C, precision);
  } else {
    cout << "[PASSED] Max difference: " << std::scientific << max_diff << endl;
  }
}

template <typename T>
void verify_result_tensor(Tensor<T> &GPU_G, Tensor<T> &CPU_G, fp32 precision = 1e-3) {
  assert(GPU_G.device == DataDevice::CPU && CPU_G.device == DataDevice::CPU);
  size_t size = GPU_G.num_elements();
  T* gpu_ptr = GPU_G.item();
  T* cpu_ptr = CPU_G.item();
  T max_diff = 0;
  for (size_t i = 0; i < size; ++i) {
    T diff = std::abs(gpu_ptr[i] - cpu_ptr[i]);
    if (diff > max_diff) max_diff = diff;
  }
  if (max_diff > precision) {
    cout << "[FAILED] Max difference: " << std::scientific << max_diff << endl;
  } else {
    cout << "[PASSED] Max difference: " << std::scientific << max_diff << endl;
  }
}

template <typename T>
void mmul_cpu_nd(Tensor<T> &A, Tensor<T> &B, Tensor<T> &C) {
  size_t ndim = A.get_ndim();
  size_t N = A.get_shape(ndim - 2);
  size_t K = A.get_shape(ndim - 1);
  size_t M = B.get_shape(ndim - 1);
  
  size_t num_batches = 1;
  for (size_t i = 0; i < ndim - 2; ++i) num_batches *= A.get_shape(i);

  T* a_ptr = A.item();
  T* b_ptr = B.item();
  T* c_ptr = C.item();

  size_t batch_size_a = N * K;
  size_t batch_size_b = K * M;
  size_t batch_size_c = N * M;

  for (size_t b = 0; b < num_batches; ++b) {
    T* ba = a_ptr + b * batch_size_a;
    T* bb = b_ptr + b * batch_size_b;
    T* bc = c_ptr + b * batch_size_c;
    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < M; ++j) {
        double sum = 0;
        for (size_t k = 0; k < K; ++k) {
          sum += (double)ba[i * K + k] * (double)bb[k * M + j];
        }
        bc[i * M + j] = (T)sum;
      }
    }
  }
}

void run_test_nd(vector<size_t> shape_A, vector<size_t> shape_B, FillType fill) {
  size_t ndim = shape_A.size();
  vector<size_t> shape_C;
  for (size_t i = 0; i < ndim - 2; ++i) shape_C.push_back(shape_A[i]);
  shape_C.push_back(shape_A[ndim - 2]);
  shape_C.push_back(shape_B[ndim - 1]);

  Tensor<fp32> A(shape_A);
  Tensor<fp32> B(shape_B);
  Tensor<fp32> G(shape_C);
  A.cuda(); B.cuda(); G.cuda();

  if (fill == FillType::RANDOM) {
    A.fill_random(42); B.fill_random(1337);
  } else if (fill == FillType::ONES) {
    A.fill_const(1.0f); B.fill_const(1.0f);
  } else {
    A.zeros(); B.zeros();
  }

  auto start = std::chrono::high_resolution_clock::now();
  gemm_nd<fp32>(A, B, G);
  CUDA_CHECK(cudaDeviceSynchronize());
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration = end - start;
  cout << "Execution time: " << duration.count() << " ms" << endl;

  A.cpu(); B.cpu(); G.cpu();
  Tensor<fp32> C_cpu(shape_C);
  mmul_cpu_nd(A, B, C_cpu);
  verify_result_tensor(G, C_cpu);
}

void run_test(KernelFunc kernel, size_t N, size_t K, size_t M, FillType fill,
              int runs = RUNS_NUM) {
  for (int i = 0; i < runs; i++) {
    Matrix<fp32> A(N, K, DataLayout::ROW_WISE, DataDevice::CUDA);
    Matrix<fp32> B(K, M, DataLayout::ROW_WISE, DataDevice::CUDA);
    Matrix<fp32> G(N, M, DataLayout::ROW_WISE, DataDevice::CUDA);

    if (fill == FillType::RANDOM) {
      A.fill_random((unsigned long long)i);
      B.fill_random((unsigned long long)i + 1337);
    } else if (fill == FillType::ONES) {
      A.ones();
      B.ones();
    } else if (fill == FillType::ZEROS) {
      A.zeros();
      B.zeros();
    }

    auto start = std::chrono::high_resolution_clock::now();
    kernel(A, B, G);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    cout << "Execution time: " << duration.count() << " ms" << endl;

    A.cpu();
    B.cpu();
    G.cpu();
    Matrix<fp32> C = mmul_cpu(A, B);
    verify_result(G, C);
  }
}

void run_benchmark(map<string, KernelFunc> &registry, size_t size = 1024,
                   int trials = 10) {
  cout << "\n--- Benchmarking Kernels (Size: " << size << "x" << size
       << ", Trials: " << trials << ") ---" << endl;

  map<string, double> accumulated_times;

  for (int t = 0; t < trials; t++) {
    Matrix<fp32> A(size, size, DataLayout::ROW_WISE, DataDevice::CUDA);
    Matrix<fp32> B(size, size, DataLayout::ROW_WISE, DataDevice::CUDA);
    A.fill_random((unsigned long long)t);
    B.fill_random((unsigned long long)t + 1337);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (auto const &[name, kernel] : registry) {
      if (!kernel) continue;
      Matrix<fp32> C(size, size, DataLayout::ROW_WISE, DataDevice::CUDA);
      auto start = std::chrono::high_resolution_clock::now();
      kernel(A, B, C);
      CUDA_CHECK(cudaDeviceSynchronize());
      auto end = std::chrono::high_resolution_clock::now();

      std::chrono::duration<double, std::milli> duration = end - start;
      accumulated_times[name] += duration.count();
    }
  }

  for (auto const &[name, kernel] : registry) {
    if (!kernel) continue;
    cout << std::left << std::setw(12) << name << ": " << std::fixed
         << std::setprecision(3) << accumulated_times[name] / trials << " ms"
         << endl;
  }
}

void iterative_stress_test(KernelFunc kernel) {
  for (size_t size = 16; size <= 4096; size *= 2) {
    cout << "\n--- Size: " << size << "x" << size << " ---" << endl;
    try{
      run_test(kernel, size, size, size, FillType::RANDOM, 1);
    }
    catch (const std::exception& e) {
      cout << "Error at size " << size << ": " << e.what() << endl;
      break;
    }
  }
}

void iterative_stress_test_nd() {
  for (int d = 2; d <= 6; ++d) {
    vector<size_t> shape_A(d, 2);
    shape_A[d - 2] = 256;
    shape_A[d - 1] = 256;
    vector<size_t> shape_B(d, 2);
    shape_B[d - 2] = 256;
    shape_B[d - 1] = 256;

    cout << "\n--- ND Dimensions: " << d << " (Batch: ";
    for (int i = 0; i < d - 2; i++) cout << "2 ";
    cout << ") Matrix: 256x256 ---" << endl;
    run_test_nd(shape_A, shape_B, FillType::RANDOM);
  }
}

void menu() {
  map<string, KernelFunc> kernel_registry;
  kernel_registry["Simpe"] = _gemm_nkm_simple_launcher<fp32>;
  kernel_registry["Blocked"] = [](Matrix<fp32> &A, Matrix<fp32> &B,
                                  Matrix<fp32> &C) {
    _gemm_nn_block_launcher<fp32>(A, B, C);
  };
  kernel_registry["Strassen"] = _gemm_strassen_launcher<fp32>;
  kernel_registry["WMMA"] = [](Matrix<fp32> &A, Matrix<fp32> &B, Matrix<fp32> &C) {
    size_t N = A.get_shape(0);
    size_t K = A.get_shape(1);
    size_t M = B.get_shape(1);

    Matrix<fp16> A_fp16(N, K, DataLayout::ROW_WISE, DataDevice::CUDA);
    Matrix<fp16> B_fp16(K, M, DataLayout::ROW_WISE, DataDevice::CUDA);

    size_t threads = 256;
    castFp32ToFp16<<<(N * K + threads - 1) / threads, threads>>>(A.item(), A_fp16.item(), N * K);
    castFp32ToFp16<<<(K * M + threads - 1) / threads, threads>>>(B.item(), B_fp16.item(), K * M);
    CUDA_CHECK(cudaDeviceSynchronize());
    _gemm_nkm_wmma_launcher(A_fp16, B_fp16, C);
  };
  kernel_registry["ND"] = [](Matrix<fp32> &A, Matrix<fp32> &B, Matrix<fp32> &C) {
    gemm_nd<fp32>(A, B, C);
  };
  kernel_registry["Strided"] = [](Matrix<fp32> &A, Matrix<fp32> &B, Matrix<fp32> &C) {
    gemm_nkm_strided<fp32>(A, B, C);
  };
  kernel_registry["ND_3D"] = nullptr; // Marker for special handling

  while (true) {
    cout << "\n=== CuTesseract Test CLI ===" << endl;
    cout << "1. Run Performance Benchmark (1024x1024)" << endl;
    cout << "2. Standard Kernel Verification (512x512)" << endl;
    cout << "3. Iterative Stress Test (16->1024)" << endl;
    cout << "4. ND 3D Verification (Custom Dims)" << endl;
    cout << "5. Exit" << endl;
    cout << "Choice: ";

    int choice;
    if (!(cin >> choice))
      break;

    if (choice == 1) {
      run_benchmark(kernel_registry);
    } else if (choice == 2 || choice == 3 || choice == 4) {
      string kernel_name;
      KernelFunc kernel = nullptr;
      
      if (choice == 4) {
        kernel_name = "ND_3D";
      } else {
        cout << "\nSelect Kernel:" << endl;
        int idx = 1;
        vector<string> names;
        for (auto const &[name, func] : kernel_registry) {
          if (name == "ND_3D") continue;
          cout << idx++ << ". " << name << endl;
          names.push_back(name);
        }
        int k_choice;
        cin >> k_choice;
        if (k_choice < 1 || k_choice > names.size())
          continue;
        kernel_name = names[k_choice - 1];
        kernel = kernel_registry[kernel_name];
      }

      cout << "Select Fill:\n1. Random\n2. Ones\n3. Zeros\nChoice: ";
      int f_choice;
      cin >> f_choice;
      FillType fill = (f_choice == 2 ? FillType::ONES
                                     : (f_choice == 3 ? FillType::ZEROS
                                                      : FillType::RANDOM));

      if (choice == 4) {
        size_t b, n, k, m;
        cout << "Enter Batch, N, K, M: ";
        cin >> b >> n >> k >> m;
        run_test_nd({b, n, k}, {b, k, m}, fill);
      } else if (choice == 2) {
        size_t n, k, m;
        cout << "Enter N, K, M (or 0 0 0 for default 512x512x512): ";
        cin >> n >> k >> m;
        if (n == 0) { n = 512; k = 512; m = 512; }
        run_test(kernel, n, k, m, fill, 1);
      } else {
        if (kernel_name == "ND") {
          iterative_stress_test_nd();
        } else {
          iterative_stress_test(kernel);
        }
      }
    } else if (choice == 5)
      break;
  }
}

int main() {
  menu();
  return 0;
}
