# 第四章 单核数据运算

> 导航：[← 单核数据搬运](./03-data-movement.md) · [目录](./README.md) · [下一章：多维张量切分 →](./05-multi-dim.md)

## 4.1 开发目标

在昇腾 NPU 单核上实现基础数据运算算子（如加减乘除、激活函数、简单矩阵元素运算）。保证算子在单核内高效执行，为后续多核并行和分布式扩展打下基础。

---

## 4.2 开发步骤

### 1. 确定算子功能

- 明确输入/输出张量的形状、数据类型（`float16` / `float32` / `int32` 等）
- 确认是否需要广播、边界处理

### 2. 编写核函数（kernel）

单核运算通常对应块级的数据处理。下面以**向量加法**为例：

```python
@triton.jit
def add_kernel(x_ptr,    # Pointer to first input vector
               y_ptr,    # Pointer to second input vector
               output_ptr,  # output 向量的指针
               n_elements,  # 向量的大小
               BLOCK_SIZE: tl.constexpr,  # 每个进程需要处理的元素个数
               # 注意：constexpr 属性表示它可以被用作 shape 值
               ):
    pid = tl.program_id(axis=0)  # 使用 1D launch grid，所以 axis 为 0
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
```

#### 调用示例

```python
def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output
```

#### 正确性测试

使用上述函数计算两个 `torch.Tensor` 对象的 element-wise sum，并测试其正确性：

```python
torch.manual_seed(0)
size = 98432
x = torch.rand(size, device='npu')
y = torch.rand(size, device='npu')
output_torch = x + y
output_triton = add(x, y)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')
# Out:
# tensor([1.3713, 1.3076, 0.4940, ..., 0.6724, 1.2141, 0.9733], device='npu')
# tensor([1.3713, 1.3076, 0.4940, ..., 0.6724, 1.2141, 0.9733], device='npu')
# The maximum difference between torch and triton is 0.0
```

---

## 4.3 单核运算的关键点

- **块级数据处理**：每个计算块负责一小段数据，保证并行性
- **边界检查**：使用 `mask` 或 `if (tid < N)` 避免越界
- **块大小选择**：合理设置 `block` 和 `grid`

---

## 4.4 性能要点

### 1. 访存优化

- 保证连续访问
- 使用对齐的 stride，避免跨行/跨列跳跃式访问
- 尽量让数据块大小对齐到 32 字节边界
- 输入输出 buffer 在分配时保证对齐，避免访存性能下降

**示例**：

```python
BLOCK_SIZE = 256  # 256 * 4 bytes = 1024 bytes，对齐良好

@triton.jit
def vec_add_kernel(X, Y, Z, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    # 计算当前 block 负责的 index 范围
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # mask 防止越界
    mask = offsets < N

    # 连续访存：offsets 是连续的
    x = tl.load(X + offsets, mask=mask)
    y = tl.load(Y + offsets, mask=mask)

    z = x + y

    # 连续写回
    tl.store(Z + offsets, z, mask=mask)


def vec_add(x, y):
    assert x.numel() == y.numel()
    N = x.numel()

    # 分配对齐内存（PyTorch 默认已经对齐到 64 字节）
    z = torch.empty_like(x)

    # grid：每个 block 处理 BLOCK_SIZE 个元素
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)

    vec_add_kernel[grid](x, y, z, N, BLOCK_SIZE=BLOCK_SIZE)

    return z
```

### 2. 子块划分

- 将大矩阵分解为小 block，每个 block 在 UB 内完成计算
- 子块划分要兼顾访存连续性和计算单元利用率

**示例**：

```python
BLOCK_M = 64   # 每个 block 处理 64 行
BLOCK_N = 64   # 每个 block 处理 64 列
BLOCK_K = 32   # 内部累加维度

@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)  # block 在 M 方向的 id
    pid_n = tl.program_id(1)  # block 在 N 方向的 id

    # 当前 block 对应的起始坐标
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # 初始化累加器
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # 循环分块计算
    for k in range(0, K, BLOCK_K):
        a = tl.load(
            A + (offs_m[:, None] * stride_am + (offs_k[None, :] + k) * stride_ak),
            mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K),
            other=0.0
        )
        b = tl.load(
            B + ((offs_k[:, None] + k) * stride_bk + offs_n[None, :] * stride_bn),
            mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N),
            other=0.0
        )
        acc += tl.dot(a, b)

    # 写回结果
    c = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

---

## 小结

| 阶段 | 关注点 |
| --- | --- |
| 功能确定 | 输入输出 shape、dtype、是否广播 |
| 核函数编写 | 块级处理、mask 边界检查、constexpr 参数 |
| 性能优化 | 访存连续 + 32B 对齐 + 合理子块划分 + Autotune 调参 |
