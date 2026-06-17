# 第二章 多核任务并行

> 导航：[← 概述](./01-overview.md) · [目录](./README.md) · [下一章：单核数据搬运 →](./03-data-movement.md)

## 2.1 设置最大硬件核数

在一个 Triton 算子中，通常使用 `grid` 进行分核操作。对于 GPU 而言，其计算核心 SM 通常是几十到几百量级。但是对于昇腾 NPU 平台而言，其计算核心 AI Core 的数量在几十个的量级。

虽然运行时接口允许下发并发任务数最大为 65535，但超过物理核数的部分是通过新一轮的下发来完成的。如果直接将 GPU 上的 Triton 算子拿到昇腾平台上运行，这些大量的任务会引入可观的核启动和核初始化时的额外开销，影响到算子性能表现。

因此，需要针对昇腾平台特性修改分核逻辑。最推荐的做法是**将分核的数量直接固定为硬件的物理核数**，在核内做更为细致的数据分块：

- 对于纯 Vector 算子，分核数等于 **Vector 核数量**
- 对于 CV 融合算子，分核数等于 **Cube 核数量**（通常为 Vector 核数量的一半），算子执行时会按 1:2 的比例调用 Vector 核

一般而言，在 NPU 卡上，一个计算核心 AI Core 含有一个 cube 核，每个 cube 核配有两个 vector 核，因此可以通过以下接口获取 **Vector 核数（`vectorcore_num`）** 与 **Cube 核数量（`aicore_num`）**：

```python
import torch
import triton.runtime.driver as driver
import torch_npu

device = torch_npu.npu.current_device()
properties = driver.active.utils.get_device_properties(device)
vectorcore_num = properties["num_vectorcore"]
aicore_num = properties["num_aicore"]
```

### 参考示例

先固定核数，再通过内部循环分批处理任务分块：

```python
NUM_CORE = vectorcore_num
grid = (NUM_CORE,)
_attn_fwd[grid](Q, K, V, M, Out, acc, scale......)

@triton.jit
def _attn_fwd(Q, K, V, M, Out, acc, scale,
              ......
              stride_qz, stride_qh,
              Z: tl.constexpr, H: tl.constexpr,
              N_CTX: tl.constexpr,
              HEAD_DIM: tl.constexpr,
              BLOCK_M: tl.constexpr,
              BLOCK_N: tl.constexpr,
              STAGE: tl.constexpr
              ):
    # 计算任务总量，将三维任务(Z, H, M)展平为一维总任务数
    NUM_BLOCKS_M = N_CTX // BLOCK_M
    NUM_BLOCKS = NUM_BLOCKS_M * Z * H

    # 每个核根据自己标识选取要处理的任务
    pid = tl.program_id(0)                  # 当前核的唯一 ID
    NUM_CORE = tl.num_programs(0)           # 获取固定启动的总核数

    # 循环规则：range(pid, NUM_BLOCKS, NUM_CORE) 实现"跨步分配任务"
    #   - 起始值 pid：每个核从自己的 ID 开始取任务，避免任务重叠
    #   - 步长 NUM_CORE：按总核数跨步，确保任务均匀分配到各个核
    for block_idx in range(pid, NUM_BLOCKS, NUM_CORE):
        # 计算每次任务的数据偏移
        # 【核心：一维任务索引反向还原为原始多维索引】
        # block_idx 是展平后的一维任务索引，通过整除/取余拆分回原始维度
        # 1. 拆分 Z+H 合并轴 & M 分块轴：
        #   - 整除 NUM_BLOCKS_M：提取 Z+H 合并轴的索引（task_hz_idx）
        #   - 取余 NUM_BLOCKS_M：提取 M 维度的分块索引（task_m_idx）
        task_hz_idx = block_idx // NUM_BLOCKS_M
        task_m_idx = block_idx % NUM_BLOCKS_M
        # 2. 拆分 Z+H 合并轴为原始 Z 轴和 H 轴：
        #   - 整除 H：还原 Z 轴索引（off_z）
        #   - 取余 H：还原 H 轴索引（off_h）
        off_z = task_hz_idx // H
        off_h = task_hz_idx % H
        # 3. 计算数据偏移量：根据还原的 Z/H 索引，定位 Q/K/V 张量中对应的数据起始位置
        qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
```

## 小结

| 维度 | GPU | NPU |
| --- | --- | --- |
| 计算核心数 | 几十到几百 | 几十 |
| 并发任务上限 | 通常远超物理核数 | 最多 65535，但超额部分需多轮下发 |
| 推荐分核策略 | 任务量级即可 | 固定为物理核数，核内分块 |
| 获取核数接口 | `torch.cuda.get_device_properties` | `driver.active.utils.get_device_properties` |

> 💡 **关键点**：不要照搬 GPU 的"任务数 = 数据量 / BLOCK_SIZE"分核策略。在 NPU 上，**将分核数固定为物理核数**才能避免额外的核启动开销。
