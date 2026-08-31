// RUN: triton-opt --triton-to-linalg --split-input-file %s | FileCheck %s

// Unit tests for ConverterUtils::getBoundarySizes() (PR #1838,
// "[TritonToLinalg](fix) boundary size Error for make_block_ptr").
//
// `tl.make_block_ptr` is lowered to `tt.make_tensor_ptr` before
// TritonToLinalg runs.  getBoundarySizes() reconstructs the per-axis
// in-bounds size by decomposing the flat block offset with the full-shape
// strides; two defects in that decomposition are covered here:
//
//  1. (primary) An axis that is *not* boundary-checked never reduced the
//     flat offset (offset % stride), so a checked trailing axis saw the
//     offset contributed by the leading axis, shrinking its boundary to 0
//     and silently dropping the loaded/stored data.  The fix strips every
//     non-zero-stride axis, checked or not.
//

// Primary case: axis 0 (stride 64) is *not* checked, axis 1 is.  The block
// starts at (16, 36) of a (64, 60) tensor: the flat offset 16*64 + 36 = 1060
// must be reduced by axis 0 first, leaving 36 for axis 1 -> boundary
// 60 - 36 = 24.  Before the fix the un-reduced offset 1060 was used,
// clamping the axis-1 boundary to 0 ([16, 0]) and dropping the whole load.

// CHECK-LABEL: func.func @boundary_size_unchecked_axis
// CHECK: memref.subview {{.*}}[0, 0] [16, 24] [1, 1]
// CHECK: memref.copy
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @boundary_size_unchecked_axis(
      %base_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %out_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}
  ) attributes {noinline = false} {
    %c16_i32 = arith.constant 16 : i32
    %c36_i32 = arith.constant 36 : i32
    %c1_i64 = arith.constant 1 : i64
    %c60_i64 = arith.constant 60 : i64
    %c64_i64 = arith.constant 64 : i64
    %load_ptr = tt.make_tensor_ptr %base_ptr, [%c64_i64, %c60_i64], [%c64_i64, %c1_i64], [%c16_i32, %c36_i32] {order = array<i32: 1, 0>} : <tensor<16x32xf32>>
    %data = tt.load %load_ptr {boundaryCheck = array<i32: 1>} : !tt.ptr<tensor<16x32xf32>>
    %store_ptr = tt.make_tensor_ptr %out_ptr, [%c64_i64, %c60_i64], [%c64_i64, %c1_i64], [%c16_i32, %c36_i32] {order = array<i32: 1, 0>} : <tensor<16x32xf32>>
    tt.store %store_ptr, %data {boundaryCheck = array<i32: 1>} : !tt.ptr<tensor<16x32xf32>>
    tt.return
  }
}