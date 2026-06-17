# 第三章 单核数据搬运

> 导航：[← 多核任务并行](./02-multi-core.md) · [目录](./README.md) · [下一章：单核数据运算 →](./04-single-core.md)

## 3.1 设置合适的循环内数据分块大小（BLOCK SIZE）

以 `add_kernel` 为例，变量和操作共同决定了片上内存空间的占用大小，通过修改 `BLOCK_SIZE` 大小可以调整循环内数据分块和计算中间结果占用的大小。如果超过上限则算子编译时会提示预期占用大小并报错。

要达到最大计算访存比，`BLOCK_SIZE` 需要在不超出片上空间时尽可能大，这可以通过 Triton-Ascend 的 [Autotune](#36-triton-autotune-自动调优) 预先设置不同的 `BLOCK_SIZE`，运行时会自动选取最优设置。

```python
import triton.language as tl

@triton.jit
def add_kernel(x_ptr,
               y_ptr,
               out_ptr,
               n,  # 元素总数量
               BLOCK_SIZE: tl.constexpr,  # 分块元素数量
               ):
    pid = tl.program_id(0)
    NUM_CORE = tl.num_programs(0)
    NUM_BLOCKS = tl.cdiv(n, BLOCK_SIZE)
    for block_idx in range(pid, NUM_BLOCKS, NUM_CORE):
        block_start = block_idx * BLOCK_SIZE
        # 分块大小为 BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        # 加载 x, y 数据到片上内存
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)

        output = x + y

        tl.store(out_ptr + offsets, output, mask=mask)
```

---

## 3.2 尽量保证 Tensor 的尾轴大小数据对齐

### 【描述】

对于 VV 类算子需要调用 Vector 核计算时，昇腾硬件的 UB 要求 Tensor 的尾轴大小能被 32 Bytes 整除，而对于 CV 类算子需要调用 Vector 核和 Cube 核计算时，要求 Tensor 的尾轴大小能被 512 Bytes 整除，若尾轴长度不足则会自动补齐。

在此前提下，对模型中 shape 为 `(2048, 3)` 和 `(2048, 1)` 的 Tensor 的种种操作，都会因为自动补齐导致性能明显恶化，此时可考虑通过转置操作将对齐轴转到低维，直到 `store` 时再转置为原始状态，从而规避自动补齐，优化计算速度。同时由于转置操作本身也受自动补齐规则的影响，因此同样需要特殊技巧来规避补齐。

这里列出一个 **"借轴转置"** 的 tip，适用于 **`tensor.numel() % 256Byte == 0`** 的场景。

> **注**：VV 类算子表示该类算子在运算过程中只使用了 Vector Core；CV 类算子表示该类算子运算过程中既使用了 AI Core 又使用了 Vector Core。

### 【示例】

```python
# conv_state = tensor([2048, 3], bfloat16)
conv_state = tl.load(conv_state_ptr + conv_batch_offs * conv_batch_stride + doffs * 3 + tl.arange(0, 2048 * 3))
# 当成 1D tensor load，此时由于 numel 对齐，不会自动补齐。
conv_state_T = conv_state.reshape(128, 16 * 3).trans().reshape(16, 3 * 128).trans().reshape(3 * 2048,)
# 长轴(2048)裂出一根对齐轴(16)借给短轴(3)，从而让两个轴都对齐
```

---

## 3.3 先将数据搬运到 UB 上，再从 UB 中 select 目标值

### 【描述】

在 NPU 的离散场景下，可以先将数据搬运到 UB，再从 share 中 select 目标值。

### 【示例】

```python
@triton.jit
def pick_kernel(
        x_ptr,
        idx_ptr,
        y_ptr,
        stride_x,
        stride_idx,
        stride_y,
        M: tl.constexpr,
        N: tl.constexpr
):
    pid = tl.program_id(0)
    rn = tl.arange(0, N)

    idx = tl.load(idx_ptr + rn * stride_idx)
    mask = idx < M

    # 原先写法
    # val = tl.load(x_ptr + idx * stride_x, mask=mask)
    # 修改后写法
    rm = tl.arange(0, M)
    x_shared = tl.load(x_ptr + rm * stride_x)  # [M]
    val = tl.gather(x_shared, idx, 0)

    tl.store(y_ptr + rn * stride_y, val, mask=mask)
```

### 【优化前后性能分析和对比】

通过 `msprof` 工具执行用例可得到 `PROF_*` 文件夹，里面包含了 `op_summary_*.csv` 文件，该文件可以帮助分析流水情况（注：`*` 表示时间戳，性能数据采集参考方法见 [性能分析 debug_guide](../debug_guide/profiling.md)）。

|  | Op Name | aiv_mte2_time(us) | aiv_mte2_ratio |
| :--- | :--- | :--- | :--- |
| 未优化 | pick_kernel | 0.686 | 0.008 |
| 优化 | pick_kernel | 1.041 | 0.066 |

通过分析表格中的数据可以发现，优化前后的 `aiv_mte2_time(us)` 和 `aiv_mte2_ratio` 差距较大。优化方案通过先将大部分数据搬运到 UB 上，减少小批量数据通过 L2 搬运到 UB 的次数，减少了 L2 搬运到 UB 上的总时间。

---

## 3.4 存算并行

Triton-Ascend 支持两种数据处理模式：**存算串行** 和 **存算并行**。

| 模式 | 描述 | 特点 |
| --- | --- | --- |
| **存算串行** | 先从全局内存搬运数据到片上内存，完成计算后，再搬运下一批数据 | 存在明显的空闲等待时间，效率较低 |
| **存算并行** | 在搬运第一批数据至片上内存的同时，已开始对其执行计算；随后继续搬运第二批数据，形成"搬运+计算"重叠的流水线式操作 | 显著提升整体吞吐率 |

实现存算并行的关键在于合理设计数据切分（Tiling）策略，使得在当前批次数据计算过程中，能够提前准备下一阶段所需的数据，从而实现数据搬运与计算过程的并行化。

> 目前，编译器默认配置 `multiBuffer=True`，**默认支持存算并行**。

---

## 3.5 Tiling 优化

AI Core 进行计算的时候要先将数据搬运至片上内存，而片上内存的空间通常远小于 AI Core 要处理的总数据量。以 Atlas 800T/I A2 产品为例，片上内存容量为 192 KB，默认开启 double buffer 特性后容量还会减至原来的一半。因此算子计算时需要对数据进行分块操作，每次只加载处理其中的一小部分数据。

### 【示例】

```python
@libentry()
@triton.autotune(configs=runtime.get_tuned_config("masked_fill"), key=["N"])
@triton.jit
def masked_fill_kernel(inp, expand_mask, value, out, N, BLOCK_SIZE: tl.constexpr, BLOCK_SIZE_SUB: tl.constexpr):
    pid = tl.program_id(axis=0)
    base_offset = pid * BLOCK_SIZE

    # 计算需要处理的块的总数
    num_sub_blocks = BLOCK_SIZE // BLOCK_SIZE_SUB

    # 针对每个子块进行循环处理
    for sub_block_idx in range(num_sub_blocks):
        # 计算当前子块的偏移量
        sub_offset = base_offset + sub_block_idx * BLOCK_SIZE_SUB
        offsets = sub_offset + tl.arange(0, BLOCK_SIZE_SUB)
        mask = offsets < N
        # Load input and mask
        input_vals = tl.load(inp + offsets, mask=mask, other=0)
        fill_mask_vals = tl.load(expand_mask + offsets, mask=mask, other=0).to(tl.int1)

        # Write the original input first
        tl.store(out + offsets, input_vals, mask=mask)

        # Overlay and write value at the position that needs to be filled
        value_to_write = tl.full([BLOCK_SIZE_SUB], value, dtype=input_vals.dtype)
        overwrite_vals = tl.where(fill_mask_vals, value_to_write, tl.load(out + offsets, mask=mask, other=0))
        tl.store(out + offsets, overwrite_vals, mask=mask)
```

---

## 3.6 Triton Autotune 自动调优

在 Tiling 分块优化中，`BLOCK_SIZE`、`BLOCK_SIZE_SUB` 等分块参数的取值直接影响算子性能，但手动调试参数组合效率低且难以找到最优值。`triton.autotune` 是 Triton 框架提供的自动调优工具，能遍历预设的参数配置，通过实际运行对比性能，自动选择最优参数组合，是 Tiling 优化的核心配套手段。

### 核心作用

- **自动遍历参数空间**：针对 `BLOCK_SIZE`、`BLOCK_SIZE_SUB` 等 `constexpr` 类型的分块参数，批量测试不同取值的性能。
- **性能基准对比**：以算子的执行耗时为指标，筛选出适配当前硬件的最优参数。
- **缓存调优结果**：调优后的最优配置会被缓存，后续调用算子时直接复用，避免重复调优。

### 简单示例

```python
import triton.language as tl

@triton.autotune(
    configs=[  # 待测试的参数配置列表，参数候选值需要是 2 的幂次
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
    ],
    key=['n_elements'],  # 调优维度：参数取值依赖的输入维度
)
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
```

> **注**：设置以下环境变量，便可打印出最优参数信息：
> ```bash
> export TRITON_PRINT_AUTOTUNING=1
> ```

---

## 3.7 如何在 NPU 上避免 UB OVERFLOW

### 【描述】

在 NPU 上，UB 或者 L1 Size 存在上限，当出现该错误时，需要减少单次搬运的数据量，以 `for` 循环的方式处理长序列场景。

### 【错误信息示例】

```text
E triton.compiler.errors.MLIRCompilationError:
E ///--------------------- [ERROR][Triton][BEG]-------------------------
E [ConvertLinalgRToBinary] encounters error:
E loc("/tmp/tmpsb6qkdih/kernel.ttadapter.mlir":2:1): error: Failed to run BishengHIR pipeline
E
E loc("/tmp/tmpsb6qkdih/kernel.ttadapter.mlir":3:3): error: ub overflow, requires 3072256 bits while 1572864 bits available! (possible reason
large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer. )
```

### 【注意】

- A2 系列产品 UB 大小为 192 KB（1 572 864 bits）

### 【处理方法】

1. 减小 `BLOCK_SIZE` 参数
2. 启用 [Tiling 子块划分](#35-tiling-优化) 把大块切分为多个小块
3. 减少循环内同时存在的中间变量
4. 关闭双缓冲（参考 [6.8](./06-tuning.md#68-双缓冲double-buffering与流水线)），但这会影响存算并行效果
