# 第五章 多维张量切分

> 导航：[← 单核数据运算](./04-single-core.md) · [目录](./README.md) · [下一章：调优方法论与进阶技巧 →](./06-tuning.md)

Triton 算子处理多维张量时，核心思想是将高维数据映射到硬件的 Block、Core、硬件单元中。本章提供二维与三维张量的典型处理示例。

---

## 5.1 二维张量切分：以矩阵乘法（GEMM）为例

对于二维矩阵乘法，通常需要在高度（M）和宽度（N）上进行二维切分，并在深度（K）上进行循环迭代。

```python
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # 1. 任务划分：计算当前 Block 在 M 和 N 维度上的坐标
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 2. 定义块指针（Block Pointers），处理多维步长（Strides）
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    # 3. 循环迭代 K 维度进行累加计算
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float16)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (offs_k[None, :] < K))
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_bn[None, :] < N))
        accumulator += tl.dot(a, b)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    tl.store(c_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn, accumulator)
```

**要点**：
- `pid_m` / `pid_n` 分别对应 M / N 维度上的 block 编号
- `stride_*` 显式处理多维步长，避免假设连续内存
- K 维度通过循环分块累加

---

## 5.2 三维及以上张量切分：以 Batched GEMM 为例

处理三维张量（如 `[Batch, M, N]`）时，可以将 `Batch` 维度（B）直接映射到 Triton 的 `Grid` 维度上，或者将其与 `M/N` 维度展平后重新映射。

### 启动 `Grid` 时增加 `Batch` 维度

```python
grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']), B)
```

### 核函数实现

```python
@triton.jit
def batched_matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, B, ...):
    # 获取当前 Batch 的索引
    pid_b = tl.program_id(2)

    # 根据 Batch 索引计算全局内存的基地址偏移
    a_batch_ptr = a_ptr + pid_b * M * K
    b_batch_ptr = b_ptr + pid_b * K * N
    c_batch_ptr = c_ptr + pid_b * M * N

    # 后续 M、N、K 维度的切分与二维 GEMM 完全一致，只需替换基地址指针即可
    # ...
```

**要点**：
- `tl.program_id(2)` 取得 Batch 维度的索引
- 每个 Batch 独立计算自己的 `a_batch_ptr` / `b_batch_ptr` / `c_batch_ptr`
- 后续 M / N / K 维度的切分逻辑与二维 GEMM 一致

---

## 小结

| 维度 | 映射方式 | 关键点 |
| --- | --- | --- |
| 1D | `pid` + 内部分块 | 见 [第四章](./04-single-core.md) |
| 2D (M, N) | `pid_m` / `pid_n` 分别对应 M / N | 显式处理 `stride_*` |
| 3D (B, M, N) | `pid_b` 对应 Batch，M/N 走 2D 逻辑 | 用 `pid_b * M * K` 计算基地址偏移 |
| 更高维 | 展平 + 反向映射 | 见 [2.1 设置最大硬件核数](./02-multi-core.md#21-设置最大硬件核数) 的展平示例 |

> 💡 **核心思想**：把高维张量的"批/通道"维度映射到 Grid，把"空间/计算"维度映射到 Block 内的 `tl.arange`。
