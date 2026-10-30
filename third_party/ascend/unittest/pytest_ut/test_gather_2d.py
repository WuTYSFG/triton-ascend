# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""2D gather 的精度与性能 pytest 用例（由 gather_2d_test_origin.py 改写）。

复用了同级 test_common.py 的接口：
  * test_common.generate_tensor   —— 生成 CPU 数据张量；
  * test_common.validate_cmp      —— 按 dtype 容差校验内核输出 vs 参考。

变体语义（pointer-analysis / unstructure 场景）：
  A  "low-discrete / high-continuous"   addr[i,j] = offsets_x[j] * N + i
      离散轴在最内层 -> 32 字节对齐保护触发，两轴均标量化（2 层 scf.for）；
  C  A 的显式 trans 重写（"normal read + tl.trans"），语义与 A 一致，
      内轴连续 -> 只标量化外层离散轴（1 层 scf.for）；
  B  "high-discrete / low-continuous"   addr[i,j] = offsets_y[i] * M + j
      连续轴在最内层（BLOCK_M >= 16 时保持向量化）。

性能用例只做正确性与有限正时延的合理性断言，具体数值请用 pytest -s
查看打印的报告，不要在 CI 中依赖绝对阈值。
"""

import math

import pytest
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

import test_common


@triton.jit
def gather_2d_low_discrete(in_ptr0, in_ptr1, out_ptr0, N, M, n_rows,
                           BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr):
    """Variant A: low/innermost axis discrete, high axis continuous."""
    pid = tl.program_id(0)
    row = pid * BLOCK_N + tl.arange(0, BLOCK_N)[:, None]       # [BLOCK_N, 1]
    col = tl.arange(0, BLOCK_M)[None, :]                        # [1, BLOCK_M]
    row_mask = row < N
    col_mask = col < M
    offsets_x = tl.load(in_ptr0 + col, mask=col_mask, other=0)  # [1, BLOCK_M]
    addr = offsets_x * N + row                                   # [BLOCK_N, BLOCK_M]
    mask = row_mask & col_mask
    val = tl.load(in_ptr1 + addr, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr0 + (row * M + col), val, mask=mask)


@triton.jit
def gather_2d_low_discrete_trans(in_ptr0, in_ptr1, out_ptr0, N, M, n_rows,
                                 BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr):
    """Variant C: explicit-transpose rewrite of A ("normal read + tl.trans")."""
    pid = tl.program_id(0)
    base_i = pid * BLOCK_N + tl.arange(0, BLOCK_N)      # [BLOCK_N]  continuous axis
    col = tl.arange(0, BLOCK_M)                          # [BLOCK_M]  gather axis
    i_mask = base_i < N
    col_mask = col < M
    offsets_x = tl.load(in_ptr0 + col, mask=col_mask, other=0)   # [BLOCK_M]
    base = offsets_x[:, None] * N                        # [BLOCK_M, 1]
    addr = base + base_i[None, :]                        # [BLOCK_M, BLOCK_N]
    mask = col_mask[:, None] & i_mask[None, :]
    val = tl.load(in_ptr1 + addr, mask=mask, other=0.0).to(tl.float32)
    valt = tl.trans(val)                                 # [N, M] store-natural
    off = base_i[:, None] * M + col[None, :]             # [BLOCK_N, BLOCK_M]
    tl.store(out_ptr0 + off, valt, i_mask[:, None] & col_mask[None, :])


@triton.jit
def gather_2d_high_discrete(in_ptr0, in_ptr1, out_ptr0, N, M, n_rows,
                            BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr):
    """Variant B: high/outermost axis discrete, low axis continuous."""
    pid = tl.program_id(0)
    row = pid * BLOCK_N + tl.arange(0, BLOCK_N)[:, None]       # [BLOCK_N, 1]
    col = tl.arange(0, BLOCK_M)[None, :]                        # [1, BLOCK_M]
    row_mask = row < N
    col_mask = col < M
    offsets_y = tl.load(in_ptr0 + row, mask=row_mask, other=0)  # [BLOCK_N, 1]
    addr = offsets_y * M + col                                   # [BLOCK_N, BLOCK_M]
    mask = row_mask & col_mask
    val = tl.load(in_ptr1 + addr, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr0 + (row * M + col), val, mask=mask)


KERNELS = {
    "A": gather_2d_low_discrete,
    "C": gather_2d_low_discrete_trans,
    "B": gather_2d_high_discrete,
}

# Variant C 是 A 的实现重写，语义上与 A 共享同一 eager 参考。
VARIANT_REF = {"A": "A", "C": "A", "B": "B"}


def _eager_reference(variant, idx, data, N, M):
    """out[i, j] = data[idx[j] * N + i]  (A) / data[idx[i] * M + j]  (B)."""
    idx64 = idx.to(torch.int64)
    if variant == "A":
        base = idx64 * N                                # [M] row-base per column
        row = torch.arange(N, dtype=torch.int64)[:, None]  # [N, 1]
        addr = base[None, :] + row                       # [N, M]
    else:
        base = idx64 * M                                 # [N] row-base per row
        col = torch.arange(M, dtype=torch.int64)[None, :]  # [1, M]
        addr = base[:, None] + col                       # [N, M]
    return data.flatten()[addr].to(torch.float32)


def _naive_reference(variant, idx, data, N, M):
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
    torch.manual_seed(seed)
    idx_len = M if variant == "A" else N
    idx = torch.randint(0, n_rows, (idx_len,), dtype=torch.int32, device=device)
    data_len = n_rows * (N if variant == "A" else M)
    data = test_common.generate_tensor((data_len,), "float16").to(device)
    out = torch.empty((N, M), dtype=torch.float32, device=device)
    return idx, data, out


def _launch(variant, idx, data, out, N, M, n_rows,
            BLOCK_N, BLOCK_M, num_warps, num_stages):
    grid = (triton.cdiv(N, BLOCK_N),)
    KERNELS[variant][grid](idx, data, out, N, M, n_rows,
                           BLOCK_N=BLOCK_N, BLOCK_M=BLOCK_M,
                           num_warps=num_warps, num_stages=num_stages,
                           compile_mode="simd")
    return grid


def _assert_close(variant, idx, data, out, N, M):
    """内核输出 vs CPU eager 参考，按 float32 容差校验（test_common.validate_cmp）。"""
    eager = _eager_reference(VARIANT_REF[variant], idx.to("cpu"), data.to("cpu"), N, M)
    assert eager.shape == out.shape, \
        f"variant {variant}: shape mismatch {tuple(eager.shape)} vs {tuple(out.shape)}"
    test_common.validate_cmp("float32", out, eager)


def _run_kernel(variant, N, M, n_rows, BLOCK_N, BLOCK_M,
                num_warps=4, num_stages=2, seed=0):
    idx, data, out = _prepare_inputs("npu", seed, variant, N, M, n_rows)
    grid = _launch(variant, idx, data, out, N, M, n_rows, BLOCK_N, BLOCK_M, num_warps, num_stages)
    torch.npu.synchronize()
    _assert_close(variant, idx, data, out, N, M)
    return idx, data, out, grid


def _benchmark(variant, N, M, n_rows, BLOCK_N, BLOCK_M,
               num_warps=4, num_stages=2, warmup=10, iters=100, seed=0):
    """warmup 后用 NPU Event 计时 `iters` 次，返回 (avg_ms, idx, data, out, grid)。"""
    idx, data, out = _prepare_inputs("npu", seed, variant, N, M, n_rows)
    grid = (triton.cdiv(N, BLOCK_N),)

    def once():
        KERNELS[variant][grid](idx, data, out, N, M, n_rows,
                               BLOCK_N=BLOCK_N, BLOCK_M=BLOCK_M,
                               num_warps=num_warps, num_stages=num_stages,
                               compile_mode="simd")

    for _ in range(warmup):
        once()
    torch.npu.synchronize()

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        once()
    end.record()
    end.synchronize()

    return start.elapsed_time(end) / iters, idx, data, out, grid


# ============================================================================
# 精度测试
# ============================================================================

SEEDS = [0, 1, 7, 42]
N, M, N_ROWS = 128, 64, 23
BLOCK_N, BLOCK_M = 128, 64


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("variant", ["A", "C", "B"])
def test_gather_2d_kernel_vs_eager(variant, seed):
    """NPU：内核输出必须与 eager torch 参考一致（fp16 数据 -> fp32 计算）。"""
    _run_kernel(variant, N, M, N_ROWS, BLOCK_N, BLOCK_M, seed=seed)


@pytest.mark.parametrize("N,M,n_rows,BLOCK_N,BLOCK_M", [
    (100, 50, 23, 128, 64),  # N/M 均非整除 block -> 行/列边界掩码生效
    (64, 32, 23, 128, 64),   # grid 只有 1 个 program，大半 tile 被掩码
    (128, 8, 23, 128, 8),    # BLOCK_M=8 < 16 (fp16) -> 32 字节对齐保护触发
])
@pytest.mark.parametrize("variant", ["A", "C", "B"])
def test_gather_2d_boundary_shapes(variant, N, M, n_rows, BLOCK_N, BLOCK_M):
    """NPU：非整除与保护阈值附近的 shape 仍需输出正确。"""
    _run_kernel(variant, N, M, n_rows, BLOCK_N, BLOCK_M)


@pytest.mark.parametrize("seed", SEEDS)
def test_gather_2d_trans_matches_low_discrete(seed):
    """变体 C（显式 trans 重写）与变体 A 输出逐位一致。"""
    _, _, out_a, _ = _run_kernel("A", N, M, N_ROWS, BLOCK_N, BLOCK_M, seed=seed)
    _, _, out_c, _ = _run_kernel("C", N, M, N_ROWS, BLOCK_N, BLOCK_M, seed=seed)
    assert torch.equal(out_a.cpu(), out_c.cpu())


# ============================================================================
# 性能测试
# ============================================================================

VARIANTS = ["A", "C", "B"]
WARMUP, ITERS = 10, 100


@pytest.mark.parametrize("variant", VARIANTS)
def test_gather_2d_perf_latency(variant):
    """默认 shape 下测量平均内核延迟（ms），并校验结果仍正确。"""
    avg_ms, idx, data, out, grid = _benchmark(
        variant, N, M, N_ROWS, BLOCK_N, BLOCK_M, warmup=WARMUP, iters=ITERS)
    torch.npu.synchronize()
    _assert_close(variant, idx, data, out, N, M)
    assert math.isfinite(avg_ms) and avg_ms > 0
    print(f"[perf] v{variant} N={N} M={M} grid={grid} "
          f"BLOCK_N={BLOCK_N} BLOCK_M={BLOCK_M} avg={avg_ms:.3f} ms "
          f"over {ITERS} iters")


@pytest.mark.parametrize("N,M,n_rows,BLOCK_N,BLOCK_M", [
    (128, 64, 23, 128, 64),
    (512, 256, 23, 128, 64),
    (2048, 1024, 23, 128, 64),
])
@pytest.mark.parametrize("variant", ["A", "C"])
def test_gather_2d_perf_scaling(variant, N, M, n_rows, BLOCK_N, BLOCK_M):
    """A/C 在不同数据规模下的延迟与读写吞吐量报告。"""
    avg_ms, idx, data, out, grid = _benchmark(
        variant, N, M, n_rows, BLOCK_N, BLOCK_M, warmup=WARMUP, iters=ITERS)
    torch.npu.synchronize()
    _assert_close(variant, idx, data, out, N, M)
    assert math.isfinite(avg_ms) and avg_ms > 0
    bytes_moved = (idx.numel() + data.numel()) * data.element_size() * 2  # 读 + 写
    gbps = bytes_moved / avg_ms / 1e6 if avg_ms > 0 else 0.0
    print(f"[perf] v{variant} N={N} M={M} grid={grid} avg={avg_ms:.3f} ms "
          f"~{gbps:.1f} GB/s")


def test_gather_2d_perf_variant_report():
    """同 shape 下对比 A / C / B 的延迟，观察标量化差异的影响。"""
    results = {}
    for variant in VARIANTS:
        avg_ms, idx, data, out, grid = _benchmark(
            variant, N, M, N_ROWS, BLOCK_N, BLOCK_M, warmup=WARMUP, iters=ITERS)
        torch.npu.synchronize()
        _assert_close(variant, idx, data, out, N, M)
        results[variant] = avg_ms
    for variant in VARIANTS:
        rel = results[variant] / results["A"] if results["A"] > 0 else float("nan")
        print(f"[perf] v{variant}: {results[variant]:.3f} ms (x{rel:.2f} vs A)")
