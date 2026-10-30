// RUN: triton-opt --triton-to-structured '--discrete-mask-access-conversion=compile-on-910-95=True compile-mode=simd_simt_template' '--triton-to-unstructure=compile-on-910-95=True compile-mode=simd_simt_template' %s --split-input-file | FileCheck %s

// tt.load -> ascend.indirect_load
tt.func public @triton_indirect_load_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<i64>, %arg2: !tt.ptr<f32>, %arg3: i32) attributes {noinline = false} {
  %cst = arith.constant dense<32> : tensor<8x1xi32>
  %c8_i32 = arith.constant 8 : i32
  %0 = tt.get_program_id x : i32
  %1 = arith.muli %0, %c8_i32 : i32
  %2 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
  %3 = tt.splat %1 : i32 -> tensor<8xi32>
  %4 = arith.addi %3, %2 : tensor<8xi32>
  %5 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %6 = tt.splat %arg1 : !tt.ptr<i64> -> tensor<32x!tt.ptr<i64>>
  %7 = tt.addptr %6, %5 : tensor<32x!tt.ptr<i64>>, tensor<32xi32>
  %8 = tt.load %7 : tensor<32x!tt.ptr<i64>>
  %9 = tt.expand_dims %2 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
  %10 = tt.splat %arg3 : i32 -> tensor<8x1xi32>
  %11 = arith.muli %9, %10 : tensor<8x1xi32>
  %12 = tt.expand_dims %8 {axis = 0 : i32} : tensor<32xi64> -> tensor<1x32xi64>
  %13 = arith.extsi %11 : tensor<8x1xi32> to tensor<8x1xi64>
  %14 = tt.broadcast %13 : tensor<8x1xi64> -> tensor<8x32xi64>
  %15 = tt.broadcast %12 : tensor<1x32xi64> -> tensor<8x32xi64>
  %16 = arith.addi %14, %15 : tensor<8x32xi64>
  %17 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<8x32x!tt.ptr<f32>>
  %18 = tt.addptr %17, %16 : tensor<8x32x!tt.ptr<f32>>, tensor<8x32xi64>
  %19 = tt.load %18 : tensor<8x32x!tt.ptr<f32>>
  %20 = math.exp %19 : tensor<8x32xf32>
  %21 = tt.expand_dims %4 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
  %22 = arith.muli %21, %cst : tensor<8x1xi32>
  %23 = tt.expand_dims %5 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
  %24 = tt.broadcast %22 : tensor<8x1xi32> -> tensor<8x32xi32>
  %25 = tt.broadcast %23 : tensor<1x32xi32> -> tensor<8x32xi32>
  %26 = arith.addi %24, %25 : tensor<8x32xi32>
  %27 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<8x32x!tt.ptr<f32>>
  %28 = tt.addptr %27, %26 : tensor<8x32x!tt.ptr<f32>>, tensor<8x32xi32>
  tt.store %28, %20 : tensor<8x32x!tt.ptr<f32>>
  tt.return
}

// CHECK-LABEL: tt.func public @triton_indirect_load_kernel
// CHECK: ascend.indirect_load{{.*}} : <f32>, {{.*}} : tensor<8x32xi64> {{.*}}-> tensor<8x32xf32>
// CHECK: tt.store

// -----

// Block-pointer load with mayDiscretememaccess must not use SIMT indirect_load.
tt.func public @block_ptr_load_skip_indirect(%arg0: !tt.ptr<f32>) -> tensor<32x32xf32> attributes {noinline = false} {
  %c32_i64 = arith.constant 32 : i64
  %c1_i64 = arith.constant 1 : i64
  %c0_i32 = arith.constant 0 : i32
  %0 = tt.make_tensor_ptr %arg0, [%c32_i64, %c32_i64], [%c32_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x32xf32>>
  %1 = tt.load %0 : !tt.ptr<tensor<32x32xf32>>
  annotation.mark %1 {mayDiscretememaccess} : tensor<32x32xf32>
  tt.return %1 : tensor<32x32xf32>
}

// CHECK-LABEL: tt.func public @block_ptr_load_skip_indirect
// CHECK-NOT: ascend.indirect_load

// -----

// Keep the mask, row address and load in one gather loop during CV splitting.
// CHECK-LABEL: @axis_discrete_row_gather
// CHECK: scf.for
// CHECK: scf.if
// CHECK: tt.load
// CHECK: tensor.insert_slice
// CHECK: } {ExtractedLoadOrStore, hivm.parallel_loop}
tt.func @axis_discrete_row_gather(%data: !tt.ptr<bf16>, %indices: !tt.ptr<i32>) -> tensor<16x16xbf16> {
  %zero = arith.constant dense<0> : tensor<16xi32>
  %stride = arith.constant dense<16> : tensor<16xi32>
  %other = arith.constant dense<0.0> : tensor<16x16xbf16>
  %range = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
  %idx_base = tt.splat %indices : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
  %idx_ptr = tt.addptr %idx_base, %range : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
  %idx = tt.load %idx_ptr : tensor<16x!tt.ptr<i32>>
  %valid = arith.cmpi sge, %idx, %zero : tensor<16xi32>
  %safe_idx = arith.select %valid, %idx, %zero : tensor<16xi1>, tensor<16xi32>
  %rows = arith.muli %safe_idx, %stride : tensor<16xi32>
  %rows_2d = tt.expand_dims %rows {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
  %cols_2d = tt.expand_dims %range {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %row_offsets = tt.broadcast %rows_2d : tensor<16x1xi32> -> tensor<16x16xi32>
  %col_offsets = tt.broadcast %cols_2d : tensor<1x16xi32> -> tensor<16x16xi32>
  %offsets = arith.addi %row_offsets, %col_offsets : tensor<16x16xi32>
  %base = tt.splat %data : !tt.ptr<bf16> -> tensor<16x16x!tt.ptr<bf16>>
  %ptr = tt.addptr %base, %offsets : tensor<16x16x!tt.ptr<bf16>>, tensor<16x16xi32>
  %valid_2d = tt.expand_dims %valid {axis = 1 : i32} : tensor<16xi1> -> tensor<16x1xi1>
  %mask = tt.broadcast %valid_2d : tensor<16x1xi1> -> tensor<16x16xi1>
  %value = tt.load %ptr, %mask, %other : tensor<16x16x!tt.ptr<bf16>>
  tt.return %value : tensor<16x16xbf16>
}
