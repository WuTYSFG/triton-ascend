// RUN: triton-opt --op-classifier %s | FileCheck %s

// ============================================================================
// Part 1 — iter_args init must not leak the outside VECTOR consumer back in
// ============================================================================
// Regression test for `OpClassifierPass`:
//   When a scf.for loop has its iter_args init consumed as the `outs` of a
//   CUBE linalg.matmul inside the body, but the loop's yielded value flows
//   into a VECTOR arith op outside the loop, the init must NOT be wrongly
//   classified as VECTOR_AND_CUBE.
//
//   Before the fix, `propagateVectorUpstream` propagated VECTOR onto the init
//   through `scf.for` operands; the init then split into a CUBE original
//   and a VECTOR clone -- and a faulty `getForInitCoreType` bound the
//   iter_args to the VECTOR clone while the body still needed the CUBE
//   original (which became dead code).
//
//   After the fix:
//     1. `propagateVectorUpstream` skips iter_args init defining ops.
//     2. `getForInitCoreType` classifies the init by its body-side compute
//        consumers (here: a CUBE linalg.matmul), so the init mirrors the
//        body and stays CUBE-only. The outside VECTOR arith consumer does
//        not leak back into the init.
//
//   Mirrors syy.mlir's %3 pattern: a `linalg.fill` init consumed as the
//   outs of a CUBE matmul inside the scf.for body, with the yielded value
//   feeding a VECTOR `arith.mulf` after the loop.

module {
  // CHECK-LABEL: func.func @iter_args_init_stays_cube_with_outer_vector_consumer
  // The init's `tensor.empty` must be classified CUBE-only -- matches the
  // body-side CUBE consumer, NOT split with VECTOR.
  // CHECK: tensor.empty() {ssbuffer.core_type = "CUBE"}
  // The init's `linalg.fill` must be classified CUBE-only.
  // CHECK: linalg.fill {{.*}}{ssbuffer.core_type = "CUBE"}
  // The body-side matmul must stay CUBE-only.
  // CHECK: linalg.matmul {{.*}}{ssbuffer.core_type = "CUBE"}
  // The outside-loop VECTOR consumer must stay VECTOR-only (does NOT pollute
  // the iter_args init).
  // CHECK: arith.mulf {{.*}}{ssbuffer.core_type = "VECTOR"}
  // Negative guard: no clone of the init should have been produced for the
  // VECTOR consumer. Before the fix the init was marked CUBE_AND_VECTOR and
  // got cloned; the VECTOR clone routed the iter_args off the CUBE original.
  // CHECK-NOT: linalg.fill {{.*}}{ssbuffer.core_type = "VECTOR"}
  // CHECK-NOT: linalg.fill {{.*}}{ssbuffer.core_type = "VECTOR_AND_CUBE"}
  // CHECK-NOT: tensor.empty() {{.*}}{ssbuffer.core_type = "VECTOR"}
  // CHECK-NOT: tensor.empty() {{.*}}{ssbuffer.core_type = "VECTOR_AND_CUBE"}
  func.func @iter_args_init_stays_cube_with_outer_vector_consumer(
      %a: tensor<16x64xf16>,
      %b: tensor<64x32xf16>,
      %extra: tensor<16x32xf32>) -> tensor<16x32xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %cst = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<16x32xf32>
    %init = linalg.fill ins(%cst : f32) outs(%empty : tensor<16x32xf32>) -> tensor<16x32xf32>

    %result = scf.for %i = %c0 to %c8 step %c1 iter_args(%acc = %init) -> (tensor<16x32xf32>) {
      // Body-side CUBE consumer: matmul writes `outs(%acc)`.
      %mm = linalg.matmul ins(%a, %b : tensor<16x64xf16>, tensor<64x32xf16>)
          outs(%acc : tensor<16x32xf32>) -> tensor<16x32xf32>
      scf.yield %mm : tensor<16x32xf32>
    }

    // Outside-loop VECTOR consumer: must NOT propagate onto `%init`.
    // (Mirrors syy.mlir's `%86 = arith.mulf %85, %53`.)
    %outside = arith.mulf %result, %extra : tensor<16x32xf32>
    return %outside : tensor<16x32xf32>
  }

  // -----

  // ==========================================================================
  // Mixed body-side consumers: iter_arg is used by BOTH a CUBE op
  // (matmul outs(%acc)) AND a VECTOR op (arith.addf reads %acc). The init
  // must be classified CUBE_AND_VECTOR (handleCubeAndVector will then split
  // it into a CUBE original + a VECTOR clone), NOT short-circuited to
  // CUBE_ONLY as the original `classifyIterArgFromBody` did.
  // ==========================================================================
  // CHECK-LABEL: func.func @iter_args_init_with_mixed_cube_and_vector_consumers
  // The init `linalg.fill` is classified CUBE_AND_VECTOR, so handleCubeAndVector
  // splits it: a CUBE original (kept by the iter_args / matmul) and a VECTOR
  // clone (for the addf side). Without the collect-all logic the init would be
  // CUBE_ONLY and no split would happen.
  // CHECK: linalg.fill {{.*}}{ssbuffer.core_type = "CUBE"}
  // CHECK: linalg.matmul {{.*}}{ssbuffer.core_type = "CUBE"}
  // CHECK: arith.addf {{.*}}{ssbuffer.core_type = "VECTOR"}
  func.func @iter_args_init_with_mixed_cube_and_vector_consumers(
      %a: tensor<16x64xf16>,
      %b: tensor<64x32xf16>) -> tensor<16x32xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %cst = arith.constant 0.0 : f32
    %bias = tensor.empty() : tensor<16x32xf32>
    %empty = tensor.empty() : tensor<16x32xf32>
    %init = linalg.fill ins(%cst : f32) outs(%empty : tensor<16x32xf32>) -> tensor<16x32xf32>

    %result = scf.for %i = %c0 to %c8 step %c1 iter_args(%acc = %init) -> (tensor<16x32xf32>) {
      // CUBE consumer: matmul writes `outs(%acc)`.
      %mm = linalg.matmul ins(%a, %b : tensor<16x64xf16>, tensor<64x32xf16>)
          outs(%acc : tensor<16x32xf32>) -> tensor<16x32xf32>
      // VECTOR consumer: arith.addf reads `%acc`. The combined user set is
      // {CUBE, VECTOR}; classifyIterArgFromBody must collect both and
      // resolve to CUBE_AND_VECTOR, not short-circuit to CUBE_ONLY.
      %sum = arith.addf %acc, %bias : tensor<16x32xf32>
      scf.yield %mm : tensor<16x32xf32>
    }

    return %result : tensor<16x32xf32>
  }

  // -----

  // ==========================================================================
  // Part 2 — nested scf.for: the outer loop's per-iter_arg classification must
  // mirror the inner loop's, since each outer iter_arg flows directly into one
  // of the inner iter_args.
  //
  // Pattern:
  //   %outer = scf.for ... iter_args(%a = fill, %b = fill) -> (...) {
  //     %inner = scf.for ... iter_args(%x = %a, %y = %b) -> (...) {
  //       %vx = arith.addf %x, %x          // consumes %x -> VECTOR body user
  //       %mm = linalg.matmul ... %y=%q,%r=%y  // consumes %y -> CUBE body user
  //       scf.yield %vx, %mm
  //     }
  //     scf.yield %inner#0, %inner#1
  //   }
  //
  // Inner-loop classification: %x (VECTOR), %y (CUBE)  -> "VECTOR, CUBE"
  // Outer-loop classification: %a (= %x) VECTOR, %b (= %y) CUBE  -> "VECTOR, CUBE"
  //
  // Before the fix the outer loop was mis-classified as "CUBE, CUBE" because
  // processYieldOperation used the op-level `opCoreTypes[innerFor]` (simplified
  // to OP_CUBE_ONLY since any CUBE -> CUBE_ONLY) instead of the per-operand
  // "VECTOR, CUBE" attribute that Step 8 stamps on the inner for.
  // ==========================================================================
  func.func @nested_for_mixed_iter_arg(
      %a: tensor<64x64xf32>, %b: tensor<64x64xf32>, %q: tensor<64x64xf32>,
      %c0: i32, %c3: i32, %c1: i32) -> (tensor<64x64xf32>, tensor<64x64xf32>) {
    // CHECK-LABEL: func.func @nested_for_mixed_iter_arg
    // CHECK:       scf.for %{{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) -> (tensor<64x64xf32>, tensor<64x64xf32>)
    // CHECK:         %{{.*}} = scf.for %{{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) -> (tensor<64x64xf32>, tensor<64x64xf32>)
    // CHECK:           linalg.matmul {{.*}} outs(%{{.*}} : tensor<64x64xf32>) -> tensor<64x64xf32>
    // CHECK:           arith.addf %{{.*}}, %{{.*}} {{.*}} : tensor<64x64xf32>
    // CHECK:           scf.yield {ssbuffer.core_type = "VECTOR, CUBE"} %{{.*}}, %{{.*}} : tensor<64x64xf32>, tensor<64x64xf32>
    // CHECK:         } {ssbuffer.core_type = "VECTOR, CUBE"}
    // CHECK:         scf.yield {ssbuffer.core_type = "VECTOR, CUBE"} %{{.*}}, %{{.*}} : tensor<64x64xf32>, tensor<64x64xf32>
    // CHECK:       } {ssbuffer.core_type = "VECTOR, CUBE"}

    %outer:2 = scf.for %i = %c0 to %c3 step %c1 iter_args(%arg_x = %a, %arg_y = %b) -> (tensor<64x64xf32>, tensor<64x64xf32>) : i32 {
      %inner:2 = scf.for %j = %c0 to %c3 step %c1 iter_args(%x = %arg_x, %y = %arg_y) -> (tensor<64x64xf32>, tensor<64x64xf32>) : i32 {
        // %y feeds matmul (CUBE).
        %mm = linalg.matmul ins(%q, %q : tensor<64x64xf32>, tensor<64x64xf32>)
                            outs(%y : tensor<64x64xf32>) -> tensor<64x64xf32>
        // %x feeds an arith tensor op (VECTOR).
        %vx = arith.addf %x, %x : tensor<64x64xf32>
        scf.yield %vx, %mm : tensor<64x64xf32>, tensor<64x64xf32>
      }
      scf.yield %inner#0, %inner#1 : tensor<64x64xf32>, tensor<64x64xf32>
    }
    return %outer#0, %outer#1 : tensor<64x64xf32>, tensor<64x64xf32>
  }

  // -----

  // Same nested shape but with the inner results yielded in the opposite
  // order, proving the per-operand lookup is positional rather than
  // "any CUBE -> CUBE" at the op level: the outer loop must come out
  // "CUBE, VECTOR" while the inner one stays "VECTOR, CUBE".
  func.func @nested_for_swapped_iter_arg(
      %a: tensor<64x64xf32>, %b: tensor<64x64xf32>, %q: tensor<64x64xf32>,
      %c0: i32, %c3: i32, %c1: i32) -> (tensor<64x64xf32>, tensor<64x64xf32>) {
    // CHECK-LABEL: func.func @nested_for_swapped_iter_arg
    // CHECK:       scf.for %{{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) -> (tensor<64x64xf32>, tensor<64x64xf32>)
    // CHECK:         %{{.*}} = scf.for %{{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) -> (tensor<64x64xf32>, tensor<64x64xf32>)
    // CHECK:           linalg.matmul {{.*}} outs(%{{.*}} : tensor<64x64xf32>) -> tensor<64x64xf32>
    // CHECK:           arith.addf %{{.*}}, %{{.*}} {{.*}} : tensor<64x64xf32>
    // CHECK:           scf.yield {ssbuffer.core_type = "VECTOR, CUBE"} %{{.*}}, %{{.*}} : tensor<64x64xf32>, tensor<64x64xf32>
    // CHECK:         } {ssbuffer.core_type = "VECTOR, CUBE"}
    // CHECK:         scf.yield {ssbuffer.core_type = "CUBE, VECTOR"} %{{.*}}, %{{.*}} : tensor<64x64xf32>, tensor<64x64xf32>
    // CHECK:       } {ssbuffer.core_type = "CUBE, VECTOR"}

    %outer:2 = scf.for %i = %c0 to %c3 step %c1 iter_args(%arg_x = %a, %arg_y = %b) -> (tensor<64x64xf32>, tensor<64x64xf32>) : i32 {
      %inner:2 = scf.for %j = %c0 to %c3 step %c1 iter_args(%x = %arg_x, %y = %arg_y) -> (tensor<64x64xf32>, tensor<64x64xf32>) : i32 {
        // %y feeds matmul (CUBE).
        %mm = linalg.matmul ins(%q, %q : tensor<64x64xf32>, tensor<64x64xf32>)
                            outs(%y : tensor<64x64xf32>) -> tensor<64x64xf32>
        // %x feeds an arith tensor op (VECTOR).
        %vx = arith.addf %x, %x : tensor<64x64xf32>
        scf.yield %vx, %mm : tensor<64x64xf32>, tensor<64x64xf32>
      }
      scf.yield %inner#1, %inner#0 : tensor<64x64xf32>, tensor<64x64xf32>
    }
    return %outer#0, %outer#1 : tensor<64x64xf32>, tensor<64x64xf32>
  }
}
