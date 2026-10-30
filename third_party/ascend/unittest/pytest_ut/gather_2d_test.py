"""
2D gather test cases reproducing the tmp6 pointer-analysis scenario.

Scenario (as requested):
    offsets_x = tl.load(idx_ptr + col)[None, :]     # data-dependent column offsets
    offsets_y = tl.arange(0, N)[:, None]            # continuous row offsets
    addr      = offsets_x * N + offsets_y            # [N, M]
    val       = tl.load(data_ptr + addr)

Pointer analysis result: 低维（最后一维 / innermost）离散 unstructured，
高维（第一维 / outermost）连续 structured —— 和 tmp6 的 x3 轴场景完全同构。

Two variants:

  Variant A  "low-discrete / high-continuous"（本题要的场景）
      addr[i,j] = offsets_x[j] * N + i     (offsets_x = tl.load(idx_ptr + col), [1, M])
      axis 0 (rows): structured   (arange participates with coefficient 1)
      axis 1 (cols): unstructured (data-dependent offsets_x)
      Without the alignment guard this would already split correctly:
      only axis 1 becomes one scf.for and axis 0 stays a vectorized slice.
      BUT the unstructured axis IS the innermost axis, so the 32-byte guard
      in UnstructureConversionPass.cpp fires (sizeInByte walk breaks at the
      innermost axis -> 2 bytes for fp16 -> 2 % 32 != 0 ->
      setUnstructured(rank) marks BOTH axes unstructured):
          -> 2 nested scf.for loops  (2D analog of tmp6's 4 loops)
      This variant is therefore the minimal repro of the guard's effect:
      the "continuous high axis" gets scalarized purely because the discrete
      axis sits at the innermost position.

  Variant B  "high-discrete / low-continuous"（对照，离散轴在非最内层）
      addr[i,j] = offsets_y[i] * M + j     (offsets_y = tl.load(idx_ptr + row), [N, 1])
      axis 0 (rows): unstructured (data-dependent offsets_y)
      axis 1 (cols): structured   (arange)
      The innermost structured run is BLOCK_M elements = 2*BLOCK_M bytes.
      With BLOCK_M >= 16 (fp16) the 32-byte guard passes, so only the discrete
      axis is split into ONE scf.for; axis 1 stays a vectorized slice.
      With BLOCK_M < 16 the guard fires again and both axes scalarize.

Usage:
  python3 gather_2d_test.py --check-inputs            # CPU-only: eager contract, no kernel
  python3 gather_2d_test.py --variant A               # NPU:0, compile_mode=simt_only
  python3 gather_2d_test.py --variant B --compile-mode simd_simt_template
  python3 gather_2d_test.py --variant A --debug       # dump kernel.ttir / .ttadapter.mlir

With --debug the ascend backend prints a triton-opt command line; append
"--mlir-print-ir-after=triton-to-structured" / "--mlir-print-ir-after=triton-to-unstructure"
to inspect ImplicitPermute (随路, now in Structured) and the scf.for splitting.

Note: Variant A still needs PtrAnalysis support for load-sourced discrete
offsets before 随路 can rewrite it; today visitOperand fails on that form so
Unstructure still emits two nested loops. Regular strided 随路 (test_implicit_permute)
is covered by the Structured migration.
"""

import argparse
import os
import sys

import torch

try:
    import torch_npu  # noqa: F401
except Exception:  # pragma: no cover - sandbox without CANN runtime
    torch_npu = None

import triton
import triton.language as tl

# ============================================================================
# Kernels (plain Triton, no inductor wrapper)
# ============================================================================


@triton.jit
def gather_2d_low_discrete(in_ptr0, in_ptr1, out_ptr0, N, M, n_rows, BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr):
    """Variant A: low/innermost axis discrete, high axis continuous.

    addr[i, j] = offsets_x[j] * N + i,  offsets_x = load(in_ptr0 + col)[1, M]
    """
    pid = tl.program_id(0)
    row = pid * BLOCK_N + tl.arange(0, BLOCK_N)[:, None]  # [BLOCK_N, 1]
    col = tl.arange(0, BLOCK_M)[None, :]  # [1, BLOCK_M]
    row_mask = row < N
    col_mask = col < M
    # offsets_x: data-dependent column offsets (tmp4-analog, load[None, :])
    offsets_x = tl.load(in_ptr0 + col, mask=col_mask, other=0)  # [1, BLOCK_M]
    # offsets_y = arange(0, N)[:, None]  (row above, plus program offset)
    addr = offsets_x * N + row  # [BLOCK_N, BLOCK_M], stride = [N, 1]
    mask = row_mask & col_mask
    val = tl.load(in_ptr1 + addr, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr0 + (row * M + col), val, mask=mask)


@triton.jit
def gather_2d_high_discrete(in_ptr0, in_ptr1, out_ptr0, N, M, n_rows, BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr):
    """Variant B: high/outermost axis discrete, low axis continuous.

    addr[i, j] = offsets_y[i] * M + j,  offsets_y = load(in_ptr0 + row)[N, 1]
    """
    pid = tl.program_id(0)
    row = pid * BLOCK_N + tl.arange(0, BLOCK_N)[:, None]  # [BLOCK_N, 1]
    col = tl.arange(0, BLOCK_M)[None, :]  # [1, BLOCK_M]
    row_mask = row < N
    col_mask = col < M
    # offsets_y: data-dependent row offsets (load[:, None])
    offsets_y = tl.load(in_ptr0 + row, mask=row_mask, other=0)  # [BLOCK_N, 1]
    # offsets_x = arange(0, M)[None, :]  (col above)
    addr = offsets_y * M + col  # [BLOCK_N, BLOCK_M]
    mask = row_mask & col_mask
    val = tl.load(in_ptr1 + addr, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr0 + (row * M + col), val, mask=mask)


_KERNELS = {
    "A": gather_2d_low_discrete,
    "B": gather_2d_high_discrete,
}

# ============================================================================
# Eager references
# ============================================================================


def _eager_reference(variant, idx, data, N, M):
    """out[i, j] = data[idx[j] * N + i]  (A) / data[idx[i] * M + j]  (B)."""
    idx64 = idx.to(torch.int64)
    if variant == "A":
        base = idx64 * N  # [M] row-base per column
        row = torch.arange(N, dtype=torch.int64)[:, None]  # [N, 1]
        addr = base[None, :] + row  # [N, M]
    else:
        base = idx64 * M  # [N] row-base per row
        col = torch.arange(M, dtype=torch.int64)[None, :]  # [1, M]
        addr = base[:, None] + col  # [N, M]
    return data.flatten()[addr].to(torch.float32)


def _naive_reference(variant, idx, data, N, M):
    """Explicit Python-loop reference; cross-checks _eager_reference."""
    idx_list = idx.tolist()
    data_flat = data.to(torch.float32).flatten().tolist()
    out = [[0.0] * M for _ in range(N)]
    if variant == "A":
        for i in range(N):
            for j in range(M):
                out[i][j] = data_flat[idx_list[j] * N + i]
    else:
        for i in range(N):
            for j in range(M):
                out[i][j] = data_flat[idx_list[i] * M + j]
    return torch.tensor(out, dtype=torch.float32)


def _prepare_inputs(device, seed, variant, N, M, n_rows):
    """Mirror the runner's data generation style (seeded, normal_(0, 0.2))."""
    torch.manual_seed(seed)
    idx_len = M if variant == "A" else N
    idx = torch.randint(0, n_rows, (idx_len, ), dtype=torch.int32, device=device)
    data_len = n_rows * (N if variant == "A" else M)
    data = torch.empty(data_len, dtype=torch.float16, device=device).normal_(0.0, 0.2)
    out = torch.empty((N, M), dtype=torch.float32, device=device)
    return idx, data, out


def _compare(name, eager, out):
    if eager.shape != out.shape:
        raise AssertionError(f"{name}: shape mismatch {tuple(eager.shape)} vs {tuple(out.shape)}")
    diff = (eager - out.to(eager.dtype)).abs().max().item()
    ok = diff <= 1e-3  # fp16 data -> fp32 compute, tolerance for accumulated rounding
    print(f"[{name}] max_abs_diff={diff:.3e} {'OK' if ok else 'FAIL'}")
    if not ok:
        raise AssertionError(f"{name}: max abs diff {diff:.3e} exceeds tolerance")


# ============================================================================
# CPU-only contract check (no NPU required)
# ============================================================================


def _check_inputs(seeds, N, M, n_rows):
    for variant in ("A", "B"):
        for seed in seeds:
            idx, data, _ = _prepare_inputs("cpu", seed, variant, N, M, n_rows)
            eager = _eager_reference(variant, idx, data, N, M)
            naive = _naive_reference(variant, idx, data, N, M)
            _compare(f"eager-vs-naive v{variant} seed{seed}", naive, eager)
    print("--check-inputs OK")


# ============================================================================
# NPU run
# ============================================================================


def _run_kernel(variant, device, seed, iters, N, M, n_rows, BLOCK_N, BLOCK_M, num_warps, num_stages):
    if torch_npu is None:
        raise RuntimeError("torch_npu is not importable in this environment")
    torch.npu.set_device(device)

    kernel = _KERNELS[variant]
    idx, data, out = _prepare_inputs(device, seed, variant, N, M, n_rows)
    grid = (triton.cdiv(N, BLOCK_N), )

    launch_kwargs = dict(num_warps=num_warps, num_stages=num_stages, compile_mode="simd")

    kernel[grid](idx, data, out, N, M, n_rows, BLOCK_N=BLOCK_N, BLOCK_M=BLOCK_M, **launch_kwargs)

    eager = _eager_reference(variant, idx.to("cpu"), data.to("cpu"), N, M)
    _compare(f"kernel-vs-eager v{variant}", eager, out.to("cpu"))

    if iters > 1:
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            kernel[grid](idx, data, out, N, M, n_rows, BLOCK_N=BLOCK_N, BLOCK_M=BLOCK_M, **launch_kwargs)
        end.record()
        end.synchronize()
        print(f"[timing] avg {start.elapsed_time(end) / iters:.3f} ms over {iters} iters")

    print(f"shape out={tuple(out.shape)} data_len={data.numel()} grid={grid} "
          f"BLOCK_N={BLOCK_N} BLOCK_M={BLOCK_M} num_warps={num_warps} "
          f"num_stages={num_stages}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=["A", "B"], default="A",
                    help="A: low-discrete/high-continuous (requested); B: contrast")
    ap.add_argument("--check-inputs", action="store_true",
                    help="CPU-only eager-vs-naive contract check (no NPU launch)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 7, 42])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--device", type=str, default="npu:1")
    ap.add_argument("--N", type=int, default=128, help="rows (continuous dim, high axis)")
    ap.add_argument("--M", type=int, default=64, help="cols (gather dim for variant A)")
    ap.add_argument("--n-rows", type=int, default=23, help="rows of the data matrix")
    ap.add_argument("--BLOCK_N", type=int, default=128)
    ap.add_argument("--BLOCK_M", type=int, default=64)
    ap.add_argument("--num-warps", type=int, default=4)
    ap.add_argument("--num-stages", type=int, default=2)
    ap.add_argument("--debug", action="store_true", help="pass debug=True to launch; backend dumps TTIR/adapter IR")

    args = ap.parse_args()

    if args.check_inputs:
        _check_inputs(args.seeds, args.N, args.M, args.n_rows)
        return

    _run_kernel(args.variant, args.device, args.seed, args.iters, args.N, args.M, args.n_rows, args.BLOCK_N,
                args.BLOCK_M, args.num_warps, args.num_stages)


if __name__ == "__main__":
    main()
