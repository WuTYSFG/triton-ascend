# 第六章 调优方法论与进阶技巧

> 导航：[← 多维张量切分](./05-multi-dim.md) · [目录](./README.md) · [第一章 概述 →](./01-overview.md)

在昇腾（Ascend）架构的高性能计算体系中，`Tiling`（切分）设计的本质是在**高并行度、高计算密度与低资源开销**之间寻找最佳平衡点。作为连接算法逻辑与底层硬件的桥梁，切分参数（`BLOCK_SIZE`）的设定直接决定了核心计算单元（Cube 与 Vector）的利用率，进而影响整体的计算吞吐与访存效率。

---

## 6.1 核心硬件约束

### Cube 单元的矩阵对齐约束

昇腾的矩阵计算单元（`Cube Unit`）在硬件底层天然支持 `16×16` 的基础计算粒度。

因此，`BLOCK_SIZE_M` 和 `BLOCK_SIZE_N` 建议设置为 `16` 的倍数。若切分尺寸未满足对齐要求，硬件在执行时会进行额外的数据填充（Padding）操作，引入额外的指令开销，导致性能下降。

### Vector 单元的访存与向量化约束

`Vector` 单元主要用于处理逐元素（`Element-wise`）操作、规约（`Reduction`）以及复杂的非线性激活函数。由于 `Vector` 单元没有 `Cube` 单元那样严格的矩阵维度对齐要求，其切分策略主要受限于片上内存（UB）容量、内存对齐规则以及向量化指令的宽度。

与 Cube 单元不同，Vector 单元没有严格的矩阵维度对齐限制，其切分策略的核心瓶颈转移到了片上内存（UB）容量、内存对齐规则以及向量化指令的宽度上。合理的 `BLOCK_SIZE_K` 或一维切分大小，需要确保单次加载的数据能够被向量化指令高效处理，同时避免超出 UB 的存储上限。

| 单元 | 主要约束 | 切分要点 |
| --- | --- | --- |
| **Cube** | 16×16 矩阵对齐 | `BLOCK_M` / `BLOCK_N` 必须为 16 的倍数 |
| **Vector** | UB 容量 + 32/512 字节对齐 + 向量化宽度 | `BLOCK_SIZE_K` / 一维 `BLOCK_SIZE` 以向量化指令为优化目标 |

---

## 6.2 通用切分方法论

在实际的工程实践中，确定最佳的 `BLOCK_SIZE` 并非单纯的数学推导，而是需要结合硬件特性与算子特征进行多维度的权衡：

### 1. 资源容量与计算密度的权衡

切分块过大虽然能提高单次计算的密度，减少循环开销，但极易超出片上 SRAM/UB 的容量限制，引发数据溢出或增加 L1/L2 Cache 的访问压力；反之，切分块过小会导致计算单元频繁等待数据搬运，无法掩盖访存延迟。

> 切分大小应尽可能填满 UB 的可用空间，同时为中间变量和循环控制预留必要的缓冲。

### 2. 硬件对齐与边界处理

除了满足 Cube 的 16 对齐外，还需考虑内存总线的对齐要求（如 32B 或 64B 对齐）。在处理非对齐的边界数据（Tail Processing）时，建议采用独立的标量或低效 Vector 处理路径，避免为了强行对齐而引入过多的无效计算（Padding Compute）。

### 3. 动态适配与性能测试

不同算子的访存模式（Memory-bound 或 Compute-bound）对切分大小的敏感度不同。建议建立基于基准测试（Benchmark）的调优闭环：

- 遍历不同的 `BLOCK_SIZE` 组合
- 监控 Cube/Vector 的实际利用率
- 监控 UB 占用率及指令流水线气泡
- 收敛到当前硬件平台下的最优切分配置

---

## 6.3 最佳实践：使用 Autotune 自动调优

由于硬件资源的动态变化，手动调试参数组合效率极低。**强烈建议**使用 `triton.autotune` 遍历预设的参数空间，让编译器自动寻找当前硬件下的最优解：

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def optimized_kernel(...):
    # 算子实现
    ...
```

更多 Autotune 用法参见 [3.6 Triton Autotune 自动调优](./03-data-movement.md#36-triton-autotune-自动调优)。

---

## 6.4 内存对齐与边界补齐规则

在进行 Vector 切分时，必须严格遵守昇腾 NPU 的内存对齐规则，否则会触发自动补齐（Padding），导致严重的性能恶化：

| 算子类型 | 尾轴大小要求 |
| --- | --- |
| **VV 类算子**（Vector-Vector） | 尾轴大小必须能被 **32 Bytes** 整除 |
| **CV 类算子**（Cube-Vector 融合） | 尾轴大小必须能被 **512 Bytes** 整除 |

### 规避策略

如果模型中存在形状不规则的 Tensor（如 `(2048, 3)`），直接操作会导致严重的自动补齐。建议通过 `reshape` 和 `trans` 操作，将长轴（如 2048）裂出一根对齐轴（如 16）借给短轴，从而让两个轴都满足对齐要求。详细示例参见 [3.2 尽量保证 Tensor 的尾轴大小数据对齐](./03-data-movement.md#32-尽量保证-tensor-的尾轴大小数据对齐)。

---

## 6.5 向量化指令与循环展开

Vector 单元的优势在于单条指令可并行处理多个数据。在编写 Triton 算子时：

### 避免标量退化

在 NPU 上，`int64` 或 `int32` 的比较操作（如 `tl.where` 中的掩码判断）无法启用向量化，会退化为标量（Scalar）计算，导致流水线断流。**应尽量将索引转换为 `float32` 类型进行比较**。

### 循环展开（Loop Unrolling）

适度展开循环可以减少标量控制指令（如判断、自增）的开销，增加指令级并行（ILP）。**通常展开 2-4 倍是较稳妥的选择**。

---

## 6.6 Vector 算子切分经验参数

与 Cube 算子不同，Vector 算子的 `BLOCK_SIZE` 主要取决于 UB 空间和向量寄存器的宽度：

| 算子场景 | 推荐 BLOCK_SIZE | 适用说明 |
| --- | --- | --- |
| 纯逐元素操作 (Add / ReLU) | 1024 / 2048 | 计算密度低，需切出较大 Block 以摊薄 DMA 搬运开销 |
| 复杂非线性 (Softmax / GELU) | 256 / 512 | 中间变量较多，受限于 UB 空间，需适当减小 Block |
| 规约操作 (Sum / Mean) | 512 / 1024 | 需考虑规约树的高度，Block 过小会导致规约效率下降 |
| 非对齐短序列 | 视对齐情况而定 | 优先进行 reshape 对齐，避免使用过小的 Block 触发补齐 |

---

## 6.7 纯 Vector 算子的 Grid 映射

对于纯 Vector 算子，**Grid 的分核数量应直接等于硬件的 Vector 核数量**。例如，Atlas A2 系列通常有 48 个 Vector 核，若下发的 Block 数远超 48，多出的任务会排队等待，引入额外的核启动与初始化开销。

> 更多关于核数设置的内容，参见 [2.1 设置最大硬件核数](./02-multi-core.md#21-设置最大硬件核数)。

### 纯 Vector 算子的 Grid 设定示例

将分核数固定为硬件 Vector 核数量，避免冗余调度：

```python
grid = lambda meta: (min(num_elements // meta['BLOCK_SIZE'], VECTOR_CORE_NUM),)

@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # 使用向量化加载与计算
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(c_ptr + offsets, a + b, mask=mask)
```

---

## 6.8 双缓冲（Double Buffering）与流水线

为了隐藏 Vector 计算与 Global Memory 搬运之间的延迟，建议在切分时结合双缓冲技术（`multibuffer`）：

- 在 UB 中开辟两块缓冲区（A 和 B）
- 当 Vector 单元正在计算缓冲区 A 的数据时，DMA 单元异步地将下一块数据搬运到缓冲区 B
- 计算完成后，立即切换到缓冲区 B 进行计算，同时 DMA 开始为 A 填充新数据

> ⚠️ **注意**：开启双缓冲后，UB 的可用空间会减半，因此 `BLOCK_SIZE` 也需要相应调小。

与存算并行的关系参见 [3.4 存算并行](./03-data-movement.md#34-存算并行)。

---

## 小结

| 维度 | 关键建议 |
| --- | --- |
| Cube 切分 | `BLOCK_M` / `BLOCK_N` 取 16 的倍数 |
| Vector 切分 | 关注 UB 容量、32/512B 对齐、向量化宽度 |
| 自动调优 | 优先使用 `triton.autotune`，避免手动调参 |
| 对齐规避 | reshape + trans 把长轴裂出对齐轴借给短轴 |
| 标量退化 | 比较操作尽量转 `float32` |
| 循环展开 | 展开 2-4 倍较稳妥 |
| 双缓冲 | 开启后 `BLOCK_SIZE` 需相应调小 |
