#include "tensor.cuh"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using std::cout;
using std::endl;

// ---------------------------------------------------------------------------
// Construction & Basic Properties
// ---------------------------------------------------------------------------

__host__ void test_initializer_list_construction() {
  Tensor<fp32> t({3, 4, 5});
  assert(t.get_ndim() == 3);
  assert(t.get_shape(0) == 3);
  assert(t.get_shape(1) == 4);
  assert(t.get_shape(2) == 5);
  assert(t.num_elements() == 60);
  assert(t.device == DataDevice::CPU);
  assert(t.item() != nullptr);
  cout << "[PASSED] test_initializer_list_construction" << endl;
}

__host__ void test_vector_construction() {
  std::vector<size_t> dims = {2, 3};
  Tensor<fp64> t(dims);
  assert(t.get_ndim() == 2);
  assert(t.get_shape(0) == 2);
  assert(t.get_shape(1) == 3);
  assert(t.num_elements() == 6);
  cout << "[PASSED] test_vector_construction" << endl;
}

__host__ void test_1d_tensor() {
  Tensor<fp32> t({10});
  assert(t.get_ndim() == 1);
  assert(t.get_shape(0) == 10);
  assert(t.num_elements() == 10);
  assert(t.get_stride(0) == 1);
  cout << "[PASSED] test_1d_tensor" << endl;
}

// ---------------------------------------------------------------------------
// Contiguous Strides
// ---------------------------------------------------------------------------

__host__ void test_contiguous_strides_2d() {
  Tensor<fp32> t({3, 4});
  // Row-major contiguous: stride[0] = cols, stride[1] = 1
  assert(t.get_stride(0) == 4);
  assert(t.get_stride(1) == 1);
  cout << "[PASSED] test_contiguous_strides_2d" << endl;
}

__host__ void test_contiguous_strides_3d() {
  Tensor<fp32> t({2, 3, 4});
  // stride[0] = 3*4 = 12, stride[1] = 4, stride[2] = 1
  assert(t.get_stride(0) == 12);
  assert(t.get_stride(1) == 4);
  assert(t.get_stride(2) == 1);
  cout << "[PASSED] test_contiguous_strides_3d" << endl;
}

// ---------------------------------------------------------------------------
// item() and element access via raw pointer
// ---------------------------------------------------------------------------

__host__ void test_item_pointer_cpu() {
  Tensor<fp32> t({5});
  fp32* ptr = t.item();
  assert(ptr != nullptr);
  // Write values and verify
  for (size_t i = 0; i < 5; ++i) ptr[i] = static_cast<fp32>(i * 1.5f);
  for (size_t i = 0; i < 5; ++i) assert(ptr[i] == i * 1.5f);
  cout << "[PASSED] test_item_pointer_cpu" << endl;
}

// ---------------------------------------------------------------------------
// Copy Construction
// ---------------------------------------------------------------------------

__host__ void test_copy_construction() {
  Tensor<fp32> t1({2, 3});
  for (size_t i = 0; i < 6; ++i) t1.item()[i] = static_cast<fp32>(i);

  Tensor<fp32> t2(t1);
  assert(t2.get_ndim() == t1.get_ndim());
  assert(t2.get_shape(0) == t1.get_shape(0));
  assert(t2.get_shape(1) == t1.get_shape(1));
  assert(t2.num_elements() == t1.num_elements());

  // Data should be independent
  for (size_t i = 0; i < 6; ++i) assert(t2.item()[i] == static_cast<fp32>(i));
  t2.item()[0] = 999.0f;
  assert(t1.item()[0] == 0.0f);  // original unchanged

  cout << "[PASSED] test_copy_construction" << endl;
}

// ---------------------------------------------------------------------------
// Copy Assignment
// ---------------------------------------------------------------------------

__host__ void test_copy_assignment() {
  Tensor<fp32> t1({2, 3});
  for (size_t i = 0; i < 6; ++i) t1.item()[i] = static_cast<fp32>(i);

  Tensor<fp32> t2({4, 5});
  t2 = t1;
  assert(t2.get_ndim() == 2);
  assert(t2.get_shape(0) == 2);
  assert(t2.get_shape(1) == 3);
  assert(t2.num_elements() == 6);
  for (size_t i = 0; i < 6; ++i) assert(t2.item()[i] == static_cast<fp32>(i));

  cout << "[PASSED] test_copy_assignment" << endl;
}

// ---------------------------------------------------------------------------
// Move Construction
// ---------------------------------------------------------------------------

__host__ void test_move_construction() {
  Tensor<fp32> t1({2, 3});
  for (size_t i = 0; i < 6; ++i) t1.item()[i] = static_cast<fp32>(i);
  fp32* old_ptr = t1.item();

  Tensor<fp32> t2(std::move(t1));
  assert(t2.get_ndim() == 2);
  assert(t2.get_shape(0) == 2);
  assert(t2.get_shape(1) == 3);
  assert(t2.item() == old_ptr);  // pointer stolen

  cout << "[PASSED] test_move_construction" << endl;
}

// ---------------------------------------------------------------------------
// Move Assignment
// ---------------------------------------------------------------------------

__host__ void test_move_assignment() {
  Tensor<fp32> t1({2, 3});
  for (size_t i = 0; i < 6; ++i) t1.item()[i] = static_cast<fp32>(i);
  fp32* old_ptr = t1.item();

  Tensor<fp32> t2({4, 5});
  t2 = std::move(t1);
  assert(t2.get_ndim() == 2);
  assert(t2.get_shape(0) == 2);
  assert(t2.item() == old_ptr);

  cout << "[PASSED] test_move_assignment" << endl;
}

// ---------------------------------------------------------------------------
// Self-assignment (copy)
// ---------------------------------------------------------------------------

__host__ void test_self_assignment() {
  Tensor<fp32> t({3, 4});
  for (size_t i = 0; i < 12; ++i) t.item()[i] = static_cast<fp32>(i);
  t = t;  // self-assignment should be a no-op
  assert(t.get_ndim() == 2);
  assert(t.get_shape(0) == 3);
  assert(t.get_shape(1) == 4);
  for (size_t i = 0; i < 12; ++i) assert(t.item()[i] == static_cast<fp32>(i));

  cout << "[PASSED] test_self_assignment" << endl;
}

// ---------------------------------------------------------------------------
// View (foreign pointer, contiguous)
// ---------------------------------------------------------------------------

__host__ void test_view_contiguous() {
  fp32 data[6] = {1, 2, 3, 4, 5, 6};
  Tensor<fp32> t = Tensor<fp32>::view(data, {2, 3}, DataDevice::CPU);

  assert(t.get_ndim() == 2);
  assert(t.get_shape(0) == 2);
  assert(t.get_shape(1) == 3);
  assert(t.num_elements() == 6);
  assert(t.item() == data);        // shares pointer
  assert(t.item()[0] == 1.0f);
  assert(t.item()[5] == 6.0f);

  // Verify strides are contiguous
  assert(t.get_stride(0) == 3);
  assert(t.get_stride(1) == 1);

  cout << "[PASSED] test_view_contiguous" << endl;
}

__host__ void test_view_with_explicit_strides() {
  fp32 data[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  // View as 3x4 with strides (4, 1) — standard contiguous sub-window
  Tensor<fp32> t = Tensor<fp32>::view(data, {3, 4}, {4, 1}, DataDevice::CPU);

  assert(t.get_stride(0) == 4);
  assert(t.get_stride(1) == 1);
  assert(t.item()[3] == 4.0f);  // row 0, col 3
  assert(t.item()[4] == 5.0f);  // row 1, col 0

  cout << "[PASSED] test_view_with_explicit_strides" << endl;
}

// ---------------------------------------------------------------------------
// subview
// ---------------------------------------------------------------------------

__host__ void test_subview_2d() {
  Tensor<fp32> t({4, 4});
  for (size_t i = 0; i < 4; ++i)
    for (size_t j = 0; j < 4; ++j)
      t.item()[i * 4 + j] = static_cast<fp32>(i * 10 + j);

  // Take a 2x2 subview starting at (1, 1)
  Tensor<fp32> sv = t.subview({1, 1}, {2, 2});

  assert(sv.get_ndim() == 2);
  assert(sv.get_shape(0) == 2);
  assert(sv.get_shape(1) == 2);
  assert(sv.num_elements() == 4);

  // Original data at (1,1)=11, (1,2)=12, (2,1)=21, (2,2)=22
  assert(sv.item()[0 * sv.get_stride(0) + 0 * sv.get_stride(1)] == 11.0f);
  assert(sv.item()[0 * sv.get_stride(0) + 1 * sv.get_stride(1)] == 12.0f);
  assert(sv.item()[1 * sv.get_stride(0) + 0 * sv.get_stride(1)] == 21.0f);
  assert(sv.item()[1 * sv.get_stride(0) + 1 * sv.get_stride(1)] == 22.0f);

  // Modifying subview should modify original (shared pointer)
  sv.item()[1 * sv.get_stride(0) + 1 * sv.get_stride(1)] = 99.0f;
  assert(t.item()[(1 + 1) * 4 + (1 + 1)] == 99.0f); // sv(1,1) is t(2,2)

  cout << "[PASSED] test_subview_2d" << endl;
}

__host__ void test_subview_1d() {
  Tensor<fp32> t({10});
  for (size_t i = 0; i < 10; ++i) t.item()[i] = static_cast<fp32>(i);

  Tensor<fp32> sv = t.subview({3}, {4});
  assert(sv.get_ndim() == 1);
  assert(sv.get_shape(0) == 4);
  assert(sv.num_elements() == 4);
  assert(sv.item()[0] == 3.0f);
  assert(sv.item()[3] == 6.0f);

  cout << "[PASSED] test_subview_1d" << endl;
}

// ---------------------------------------------------------------------------
// cpu() / cuda() transfer (compile-time only; run-time requires GPU)
// ---------------------------------------------------------------------------

__host__ void test_device_transfer_noop() {
  // Calling cpu() on CPU tensor should be a no-op
  Tensor<fp32> t({3, 3});
  t.cpu();
  assert(t.device == DataDevice::CPU);
  assert(t.item() != nullptr);

  cout << "[PASSED] test_device_transfer_noop" << endl;
}

// ---------------------------------------------------------------------------
// num_elements
// ---------------------------------------------------------------------------

__host__ void test_num_elements() {
  Tensor<fp32> t0({});
  assert(t0.num_elements() == 1);

  Tensor<fp32> t1({5});
  assert(t1.num_elements() == 5);

  Tensor<fp32> t2({2, 3, 4});
  assert(t2.num_elements() == 24);

  cout << "[PASSED] test_num_elements" << endl;
}

// ---------------------------------------------------------------------------
// Edge cases: zero-dim tensor (scalar-like)
// ---------------------------------------------------------------------------

__host__ void test_scalar_tensor() {
  Tensor<fp32> t({});
  assert(t.get_ndim() == 0);
  assert(t.num_elements() == 1);
  t.item()[0] = 42.0f;
  assert(t.item()[0] == 42.0f);

  cout << "[PASSED] test_scalar_tensor" << endl;
}

// ---------------------------------------------------------------------------
// Destructor doesn't double-free on move
// ---------------------------------------------------------------------------

__host__ void test_move_destructor_safety() {
  Tensor<fp32> t1({100});
  {
    Tensor<fp32> t2(std::move(t1));
    // t2 destructor runs here — should free only once
  }
  // t1 was moved-from; its destructor should handle nullptr safely
  cout << "[PASSED] test_move_destructor_safety" << endl;
}

// ---------------------------------------------------------------------------
// swap
// ---------------------------------------------------------------------------

__host__ void test_swap() {
  Tensor<fp32> a({2, 2});
  Tensor<fp32> b({3, 3});
  a.item()[0] = 1.0f;
  b.item()[0] = 2.0f;

  a.swap(b);
  assert(a.get_shape(0) == 3);
  assert(b.get_shape(0) == 2);
  assert(a.item()[0] == 2.0f);
  assert(b.item()[0] == 1.0f);

  cout << "[PASSED] test_swap" << endl;
}

// ---------------------------------------------------------------------------
// set_stride
// ---------------------------------------------------------------------------

__host__ void test_set_stride() {
  Tensor<fp32> t({4, 5});
  t.set_stride(0, 10);
  t.set_stride(1, 2);
  assert(t.get_stride(0) == 10);
  assert(t.get_stride(1) == 2);

  cout << "[PASSED] test_set_stride" << endl;
}

// ---------------------------------------------------------------------------
// shape_data / stride_data
// ---------------------------------------------------------------------------

__host__ void test_shape_stride_data() {
  Tensor<fp32> t({2, 3, 4});
  const size_t* sp = t.shape_data();
  assert(sp[0] == 2 && sp[1] == 3 && sp[2] == 4);
  const size_t* stp = t.stride_data();
  assert(stp[0] == 12 && stp[1] == 4 && stp[2] == 1);

  cout << "[PASSED] test_shape_stride_data" << endl;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
  cout << "=== Tensor Unit Tests ===" << endl;

  test_initializer_list_construction();
  test_vector_construction();
  test_1d_tensor();
  test_contiguous_strides_2d();
  test_contiguous_strides_3d();
  test_item_pointer_cpu();
  test_copy_construction();
  test_copy_assignment();
  test_move_construction();
  test_move_assignment();
  test_self_assignment();
  test_view_contiguous();
  test_view_with_explicit_strides();
  test_subview_2d();
  test_subview_1d();
  test_device_transfer_noop();
  test_num_elements();
  test_scalar_tensor();
  test_move_destructor_safety();
  test_swap();
  test_set_stride();
  test_shape_stride_data();

  cout << "\n=== All Tensor Tests Passed ===" << endl;
  return 0;
}
