# Triton 算子性能调优指南

## 第一篇：基础篇 — 理解硬件差异，建立心智模型

### 第 1 章 GPU 与 NPU 微架构差异：Triton 算子开发迁移视角

- 1.1 执行模型与 Grid 映射：Warp 调度 → 物理核绑定
  - GPU：SIMT 执行模型，32 线程/Warp 锁步执行
    - Grid 与物理 SM **解耦**：可启动远超 SM 数量的 Block，硬件自动排队调度
    - `num_warps` 控制每个 Block 的并行度，`num_stages` 控制软件流水线深度
    - 开发者只需按逻辑任务量设置 Grid（如 `cdiv(N, BLOCK_SIZE)`），硬件负责调度
  - NPU：SIMD 执行模型（默认），一个 Program 绑定一个 AI Core，无线程概念
    - Grid 与物理核**强绑定**：Grid 规模超出物理核数时，Block需分批（轮转）调度，每次切换都会产生额外的设备侧开销（Block调度与参数装载、核内上下文初始化等）；且跨Block的存算流水线无法并行，前一个Block需排空流水线后，下一个Block才重新预热启动。
    - `num_warps` / `num_stages` **无效**（NPU 无 Warp、无 GPU 式软件流水线）
    - 开发者需主动查询核心数并将 Grid 对齐：`grid = (num_cores,)` + 核内 for 循环
  - **对 Triton 代码的影响**：
    - 迁移第一步：删除 `num_warps`、`num_stages`，加入 `multibuffer` 参数
    - Grid 设计从"逻辑分块"变为"物理核持久化"（Persistent Kernel 模式）
    - 纯 Vector 算子用 `num_vectorcore`（910\_95 为 aicore 的 2 倍），CV 融合算子用 `num_aicore`
- 1.2 片上存储管理：Shared Memory 手动管理 → UB 编译器管理
  - GPU：Shared Memory（48-164 KB/SM）由程序员手动控制
    - 通过 `num_stages` 隐式控制 Shared Memory 多缓冲深度
    - Tensor Core 输出直接写入寄存器，后续运算立即可用
    - 开发者通过 `__shared__` 或 Triton 编译器间接管理 Shared Memory 布局
  - NPU：UB（248KB 可用）由编译器全权管理，开发者**不能手动分配**
    - UB 是 Vector 的唯一工作区：所有 `tl.load` 的数据、中间计算结果、`tl.store` 前的数据都在 UB
    - Cube 输出进入 L0C（256KB），需经 fixpipe 搬到 UB 后 Vector 才能处理
    - 开发者通过 `BLOCK_SIZE` **间接控制** UB 占用（而非直接分配 Shared Memory）
    - `multibuffer=True` 启用双缓冲，但 UB 可用空间减半（248KB → 124KB）
  - **对 Triton 代码的影响**：
    - `BLOCK_SIZE` 不仅是 Tiling 参数，还直接决定 UB 占用——选错即 `ub overflow` 编译错误
    - UB 预算公式：`Σ(每个 tensor 的 BLOCK_SIZE × dtype_size) ≤ 248KB`（开启 multibuffer 则 ≤ 124KB）
    - 大 BLOCK + 核内二级分块（`BLOCK_SIZE_SUB`）是 NPU 常用模式：核间大块控并行度，核内小块控 UB

***

## 第二篇：分析篇 — 性能剖析与瓶颈定位

### 第 2 章 性能分析方法论（调优的前提）

- 2.1 性能数据采集（数据从哪来）
  - torch\_npu.profiler（Python API）
    - `_ExperimentalConfig` 配置（AiCMetrics.PipeUtilization）
    - 采集 schedule 设计（wait / warmup / active / repeat）
  - msprof op 命令行：单算子级性能剖析
  - 仿真流水图（离线采集）
  - 辅助分析环境变量：`TRITON_PRINT_AUTOTUNING` / `TRITON_ALL_BLOCKS_PARALLEL` / `TRITON_DEBUG` / `TRITON_BENCH_METHOD`
  - **仓库文档参考**（部分覆盖，按内容相关度排序）：
    - 上板采集与 PipeUtilization 数据解读（走 msprof 命令行路径，非 torch\_npu.profiler）：[profiling.md](../debug_guide/profiling.md)
    - `torch_npu.profiler` 计时路径的实现与"设置预热和运行次数"（warmup/repeat，do\_bench\_npu 内部机制）：[do\_bench\_npu\_guide.md](../debug_guide/do_bench_npu_guide.md)
    - `TRITON_BENCH_METHOD="npu"` 下用 `torch_npu.profiler.profile` 采集每个 config 的片上时间：[06\_autotune\_example.md](../examples/06_autotune_example.md)
  - **覆盖缺口**：`_ExperimentalConfig` 配置、schedule 的 wait/active/repeat 在当前仓库文档中尚无直接说明，建议结合 CANN《性能调优工具用户指南》与 PyTorch profiler 官方文档补充
- 2.2 关键性能指标解读（数据怎么看）
  - 指标来源：采集阶段的 op\_summary / PipeUtilization.csv（见 2.1）
  - `aiv_vec_ratio`（Vector 利用率，理想 > 80%）
  - `aiv_mte2_ratio`（搬入占比，理想 < 50%）
  - `aiv_scalar_ratio`（标量利用率，警惕 > 30% 的标量退化）
  - `aic_cube_ratio`（Cube 利用率，理想 > 80%）
- 2.3 结果可视化与流水图解读（怎么直观呈现）
  - TensorBoard 可视化与结果解读（当前仓库无直接覆盖，参考 PyTorch profiler 官方文档）
  - Chrome Tracing / MindStudio Insight 流水图分析
  - 指令时间线诊断规则（断流检测、空白段识别）
- 2.4 瓶颈分类与优化方向决策树（看完怎么办）
  - 计算密集型 → 增大分块 / CV 融合
  - 访存密集型 → 存算并行 / care\_padding / 对齐优化
  - 标量退化 → 数据类型优化（避免 int64）
  - 同步开销 → 减少同步 / 调整编译参数

***

## 第三篇：调优篇 — 系统化性能优化方法论

### 第 3 章 Grid 映射与分核策略：让所有核心都饱和

> **核心理念**：NPU 的 Grid 与物理核强绑定，分核策略直接决定硬件利用率。优化的第一步不是改算法，而是确保每个核心都有足够的工作量。
>
> **（1）Grid 与物理核的强绑定关系。** GPU 的 Grid 与物理 SM 解耦——开发者只需按逻辑任务量设置 Grid（如 `cdiv(N, BLOCK_SIZE)`），硬件自动排队调度，可以启动远超 SM 数量的 Block（如 A100 有 108 个 SM，但 Grid 可以是数万个 Block），硬件的 Block Scheduler 负责将 Block 分配到空闲 SM。NPU 则完全不同：一个 Program 绑定一个 AI Core，Grid 与物理核**强绑定**。当 Grid 超出物理核数时，硬件分批调度（如 910\_95 有 \~40 个 AICore，Grid=120 时分 3 批执行），每批切换有额外的设备侧初始化开销（寄存器清零、流水线冲刷），核启动开销急剧增加。因此 NPU 上的最佳实践是 Grid = 物理核数，让每个核通过核内循环持久化处理多个任务块。
>
> **（2）核心数查询 API 与核型选择。** NPU 的 AI Core 包含 1 个 Cube（矩阵乘单元）+ 2 个 Vector（向量计算单元）。因此有两种核心数：`num_aicore`（AI Core 总数，含 Cube+Vector）和 `num_vectorcore`（Vector 核数，910\_95 上为 aicore 的 2 倍）。查询方式：`import triton.runtime.driver as driver; properties = driver.active.utils.get_device_properties(device); aicore_num = properties["num_aicore"]; vector_num = properties["num_vectorcore"]`。选择规则：**纯 Vector 算子**（无 `tl.dot`，如 RMSNorm、RoPE、Softcap）使用 `num_vectorcore` 分核，充分利用 2 倍的 Vector 核数；**CV 融合算子**（含 `tl.dot`，如 Flash Attention、Fused MatMul）使用 `num_aicore` 分核，因为 Cube 单元是计算瓶颈。选错核型会导致 Vector 算子只用一半 Vector 核，或 CV 算子无法启用 Cube。
>
> **（3）coreDim 硬件约束。** NPU 的 Grid 各维度乘积受 `coreDim ≤ 65535 (UINT16_MAX)` 硬件约束——这是硬件调度器用 16 位整数索引 program\_id 的限制。多维 Grid（如 `(blocks_m, q_head, batch)` = `(128, 32, 8)` = 32768）在 head 数和 batch 数较大时容易触及上限，编译报错 `coreDim exceeds UINT16_MAX`。展平为一维 `(NUM_CORES,)` 可彻底规避此约束。

- 3.1 Persistent Kernel：从逻辑 Grid 到物理核的持久化映射
  - **问题**：多维逻辑 Grid（如 Flash Attention 的 `(blocks_m, q_head, batch)`）使每个 program 只处理一个逻辑块，Grid 规模 `blocks_m × q_head × batch` （如 128×32×8=32768）远超物理核数，造成频繁的核间调度与切换开销。且多维 Grid 可能触及 `coreDim ≤ 65535` 的硬件上限
  - **方案**：将 Grid 展平为一维 `(NUM_CORES,)`，核内通过 `for` 循环以步长 `NUM_CORES` 遍历全部任务块（interleaved 分配），并从 `block_idx` 反推原始的多维索引。
  - **协同优化**：
    - 核心数动态获取（`device_properties.get("num_vectorcore")`），不硬编码——910B 与 950 核心数不同
    - 纯 Vector 算子使用 `num_vectorcore`，CV 融合算子用 `num_aicore`
    - 持久化循环天然形成软件流水线——上一轮 Vector 计算与下一轮 MTE2 加载重叠
  ```python
  # ── Before: 多维逻辑 Grid，每 program 处理一个 (M块, head, batch) 组合 ──
  @triton.jit
  def fwd_kernel(q_ptr, k_ptr, v_ptr, ...,
                 BLOCK_M: tl.constexpr, ...):
      start_m = tl.program_id(0)    # M 维度 block 索引
      start_qh = tl.program_id(1)   # head 维度
      start_b = tl.program_id(2)    # batch 维度
      # ... 单块计算逻辑 ...

  # 调用处：Grid 远超物理核数
  grid = lambda META: (triton.cdiv(max_seqlen_q, META["BLOCK_M"]),
                       q_head, batch_size)
  fwd_kernel[grid](q, k, v, ...)

  # ── After: 一维物理 Grid，每 program 循环处理多个块（interleaved 分配）──
  @triton.jit
  def fwd_kernel(q_ptr, k_ptr, v_ptr, ...,
                 AICORE_NUM: tl.constexpr, MAX_Q_LEN: tl.constexpr,
                 BATCH_SIZE: tl.constexpr, q_head: tl.constexpr,
                 kv_head: tl.constexpr, BLOCK_M: tl.constexpr, ...):
      pid = tl.program_id(0)
      NUM_BLOCKS_M = tl.cdiv(MAX_Q_LEN, BLOCK_M)
      NUM_BLOCKS = NUM_BLOCKS_M * BATCH_SIZE * q_head
      # interleaved 遍历：步长 = AICORE_NUM，各核交替领取任务
      for block_idx in range(pid, NUM_BLOCKS, AICORE_NUM):
          # 从一维 block_idx 反推多维索引
          task_hz_idx = block_idx // NUM_BLOCKS_M
          start_m = block_idx % NUM_BLOCKS_M
          start_b = task_hz_idx // q_head
          start_qh = task_hz_idx % q_head
          start_kvh = start_qh * kv_head // q_head  # GQA head 映射
          # ... 单块计算逻辑（与 Before 相同）...

  # 调用处：Grid = 物理核数，核心数动态获取
  import triton.runtime.driver as driver
  properties = driver.active.utils.get_device_properties(device)
  NUM_CORES = properties["num_aicore"]  # CV 融合算子用 aicore
  grid = (NUM_CORES,)
  fwd_kernel[grid](q, k, v, ...,
                   AICORE_NUM=NUM_CORES,
                   MAX_Q_LEN=max_seqlen_q, ...)
  ```
  - **案例**：FA v8 fwd/bwd — 3D Grid `(blocks_m, q_head, batch)` → 1D Grid `(AICORE_NUM,)` + interleaved 循环
  - **案例**：RoPE — 3D Grid `(seq_blocks, bs, head)` → 1D Grid `(bs,)` + 核内遍历 seq\_len 和 head
- 3.2 自适应分核：按任务量缩减核数
  - **问题**：当任务总量小于总核数时（如 batch\_size=4 而 num\_vectorcore=80），部分核心分不到任务，产生空转——`range(pid, total_tasks, num_cores)` 中 `pid >= total_tasks` 的核心循环体为空，属于无效启动、造成资源空转。
  - **方案**：根据实际任务量动态缩减 Grid 核心数，确保每个核心至少分到一个完整任务块，避免无效核心参与调度
  ```python
  # ── Before: 固定核心数，部分核心空转 ──
  NUM_CORES = VECTOR_NUM  # 80 核
  grid = (NUM_CORES,)
  # 当 total_tasks=4 时，76 个核心空转

  # ── After: 根据任务量动态缩减 ──
  # 计算每核至少分到一个 BLOCK_SIZE 大小的任务块
  total_tasks = triton.cdiv(numel, BLOCK_SIZE)
  if total_tasks < VECTOR_NUM:
      NUM_CORES = total_tasks  # 缩减到实际任务数
  else:
      NUM_CORES = VECTOR_NUM
  grid = (NUM_CORES,)
  ```
  - **案例**：RMSNorm — `n_rows < num_cores` 时缩减 Grid 到 `n_rows`
- 3.3 SPLIT\_K 拆分：K 维度并行化
  - **问题**：当 M×N 维度不足以饱和所有核心时（如 decode 场景的小 batch，M=1、N=head\_dim=128，只有 1×8=8 个输出块），大量核心闲置。此时 M×N 的并行度远小于物理核数
  - **方案**：在 K 维度上拆分 matmul，多个 program 并行计算部分和，通过原子操作或锁机制累加到最终输出
  - **关键细节**：SPLIT\_K 个 program 写同一输出地址，必须用 `tl.atomic_add` 累加（无锁）或 `LOCK` 数组 + `tl.store`（有锁）；autotune 时需 `reset_to_zero=["output_ptr"]` 避免多次评估累加
  ```python
  # ── Before: 单 program 处理完整 K 维度，M×N 并行度不足 ──
  @triton.jit
  def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, ...,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                    BLOCK_K: tl.constexpr):
      pid_m = tl.program_id(0)
      pid_n = tl.program_id(1)
      acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
      # 单 program 遍历完整 K 维度
      for k in range(0, K, BLOCK_K):
          a = tl.load(a_ptr + ...)
          b = tl.load(b_ptr + ...)
          acc += tl.dot(a, b)
      tl.store(c_ptr + ..., acc)  # 直接写入，无冲突

  grid = (cdiv(M, BLOCK_M), cdiv(N, BLOCK_N))  # M×N 并行度

  # ── After: K 维度拆分，多 program 并行 + 原子累加 ──
  @triton.jit
  def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, ...,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                    BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr):
      pid_m = tl.program_id(0)
      pid_n = tl.program_id(1)
      pid_k = tl.program_id(2)  # K 维度拆分索引
      acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
      # 每个 program 只处理 K/SPLIT_K 部分
      k_start = pid_k * K // SPLIT_K
      k_end = (pid_k + 1) * K // SPLIT_K
      for k in range(k_start, k_end, BLOCK_K):
          a = tl.load(a_ptr + ...)
          b = tl.load(b_ptr + ...)
          acc += tl.dot(a, b)
      # 原子累加：SPLIT_K 个 program 写同一地址
      tl.atomic_add(c_ptr + ..., acc)

  # Grid 增加 K 维度：M×N×SPLIT_K 并行度
  grid = (cdiv(M, BLOCK_M), cdiv(N, BLOCK_N), SPLIT_K)
  ```
  - **案例**：Fused MatMul bwd\_w — `SPLIT_K` 参数（8/16）拆分 K 维度，配合 `LOCK_W` 数组处理写冲突
  - **案例**：FA bwd\_qkv — `dq` 用 `tl.atomic_add` 累加多核结果，autotune 配 `reset_to_zero=["dq_ptr"]`
- 3.4 Sorted Round-Robin 负载均衡：稀疏注意力的任务不均衡问题
  - **问题**：稀疏注意力（Flex Attention / SWA）与 GQA 场景中，不同任务块的工作量差异极大：causal mask 使上三角全零，前几个 query block 只需处理少量 key block，而最后一个 query block 要处理全部 key block。此时若按stride 方式遍历任务（`range(pid, total, num_cores)`），各核任务大小混杂不均，导致各核总工作量严重失衡：分配到靠后 query block 的核满载，分配到靠前的核大量空转。
  - **方案**：主机端先枚举所有任务 → 计算 workload（如每个 query block 需处理的 key block 数）→ 按 workload 降序排序 → 以轮询方式分配到各核（round-robin）→ kernel 端通过任务张量读取本核的任务列表
  - **原理**：降序排序保证大任务优先被分配，轮询则让每个核在排序序列上等距采样（核 0 取第1、第 C+1 、第 2C+1...大的任务），因此各核拿到的任务集合在大小分布上高度同构，总工作量趋于均衡；同时避免"最后才遇到一个超大任务，已无处摊平"的情况。
  ```python
  # ── Before: kernel 内 stride 遍历，各核工作量严重不均 ──
  @triton.jit
  def attn_kernel(q_ptr, k_ptr, ...):
      pid = tl.program_id(0)
      # interleaved 遍历：靠前的 query block 工作量小，靠后的大
      for block_idx in range(pid, total_tasks, num_cores):
          start_m = block_idx % NUM_BLOCKS_M
          # causal mask：start_m 越大，需处理的 key block 越多
          num_kv_blocks = start_m + 1  # 工作量随 start_m 线性增长
          for kv in range(num_kv_blocks):
              ...  # 靠前的核只执行 1 轮，靠后的执行 NUM_BLOCKS_M 轮

  # ── After: 主机端预排序 + 轮询分配，各核均衡 ──
  # 主机端：计算每个任务的 workload 并降序排序
  def assign_tasks(total_tasks, num_cores):
      workloads = [compute_workload(t) for t in range(total_tasks)]
      tasks_sorted = sorted(range(total_tasks), key=lambda t: workloads[t],
                            reverse=True)  # 降序：重任务在前
      # 轮询分配：核 0 拿 tasks_sorted[0], tasks_sorted[num_cores], ...
      core_tasks = [[] for _ in range(num_cores)]
      for i, task in enumerate(tasks_sorted):
          core_tasks[i % num_cores].append(task)  # round-robin
      # 展平为连续张量供 kernel 读取
      core_tasks_ranges = [...]  # 每核的任务起止偏移
      core_tasks_flat = tensor([task for core in core_tasks for task in core])
      return core_tasks_ranges, core_tasks_flat

  # kernel 端：通过任务张量读取分配到本核的任务
  @triton.jit
  def attn_kernel(q_ptr, k_ptr, core_tasks_ptr, core_tasks_ranges_ptr, ...):
      pid = tl.program_id(0)
      start_offset = tl.load(core_tasks_ranges_ptr + pid * 2)
      end_offset = tl.load(core_tasks_ranges_ptr + pid * 2 + 1)
      # 只遍历分配到本核的任务（已均衡）
      for i in range(start_offset, end_offset):
          block_idx = tl.load(core_tasks_ptr + i)
          start_m = block_idx % NUM_BLOCKS_M
          ...  # 各核总工作量趋于一致
  ```
  - **案例**：FlexAttention — 主机端 sorted round-robin 分配，消除 causal mask 导致的工作量不均
  - **案例**：SWA prefill — 滑动窗口场景的 workload 排序分配

### 第 4 章 Tiling 与分块策略：在 UB 约束下找到最优分块

> **核心理念**：BLOCK\_SIZE 不仅是 Tiling 参数，还直接决定 UB 占用——选错即 `ub overflow`。Tiling 的本质是在 UB 容量、对齐约束和计算效率之间找到最优平衡点。
>
> **（1）UB 容量与预算公式。** NPU 的 UB（Unified Buffer）是 Vector 的唯一工作区——所有 `tl.load` 的数据、中间计算结果、`tl.store` 前的数据都在 UB。910\_95（A5）的 UB 硬件为 256KB，编译器预留 8KB 后实际可用 **248KB**；910B（A2）的 UB 为 **192KB**。与 GPU 的 Shared Memory（开发者手动管理、通过 `__shared__` 或 `num_stages` 间接控制）不同，NPU 的 UB 由编译器全权管理，开发者通过 `BLOCK_SIZE` **间接控制** UB 占用。UB 预算公式：`所需UB = Σ(每个 tensor 的 BLOCK_SIZE × dtype_bytes) × multibuffer副本数 ≤ UB可用`。例如 FP16 的 RMSNorm kernel，输入 `x` (BLOCK\_SIZE×2B) + weight `w` (BLOCK\_SIZE×2B) + 输出 `y` (BLOCK\_SIZE×2B) + 中间 `mean/var` (32B) ≈ BLOCK\_SIZE×6B + 32B。开启 `multibuffer=True` 后 UB 可用空间减半（910\_95 为 124KB，910B 为 96KB），因为双缓冲需要两份 UB 空间。超出 UB 容量时编译报错 `ub overflow`，必须减小 BLOCK\_SIZE 或减少核内同时存活的 tensor 数量。
>
> **（2）对齐约束。** NPU 有两层对齐要求：**Vector 尾轴 32B 连续对齐**是硬约束——FP16（2B）尾轴元素须为 16 的倍数，FP32（4B）须为 8 的倍数，否则 DMA 引擎无法发起连续搬运；**CV 融合算子（含** **`tl.dot`）要求 512B 对齐**——FP16 尾轴元素须为 256 的倍数，因为 Cube 单元的矩阵分块粒度更大。非对齐场景需要额外的 Mask 处理（加载时用 mask 截断多余元素），但会引入性能开销。此外，单值缓冲区（如归约结果 mean/var/max/sum）逻辑上只需 4B，但硬件要求 32B 对齐，实际分配 32B。
>
> **（3）计算效率与分块大小的关系。** 大 BLOCK 提升 Cube 利用率和数据复用（矩阵乘的 BLOCK\_M×BLOCK\_N 越大，每个元素被复用的次数越多），但受 UB 容量限制；小 BLOCK 增加 Grid 分核数（更多并行度），但 Cube 利用率低。NPU 上 Cube 单元的最优分块通常是 128（`BLOCK_M=BLOCK_N=BLOCK_K=128` 是 Fused MatMul 的收敛值），因为 Cube 的 `mmad` 指令以 128×128 矩阵为基本计算单元。对于纯 Vector 算子（无 `tl.dot`），BLOCK\_SIZE 可以更大（如 Softcap 的 BLOCK\_SIZE=2048），因为 Vector 的 UB 压力小于 CV 融合算子。

- 4.1 核内多块循环：提升 MTE2/MTE3/计算流水并行度
  - **问题**：标准 Grid 模式下每个 program 只处理一个数据块，Grid 规模大（如 `cdiv(numel, BLOCK_SIZE)=1000`）但核内无流水线深度——每个 program 只做一次 load→compute→store，无法利用存算并行
  - **方案**：新增 `BLOCK_NUM` 参数，单 program 内循环处理多个数据块，Grid 相应缩小 `BLOCK_NUM` 倍。核内多块循环减少 Grid 规模（降低核切换开销），同时实现存算并行（上一块 compute 与下一块 load 重叠）
  ```python
  # ── Before: 每 program 一个块，Grid 大但无核内流水线 ──
  @triton.jit
  def softcap_fwd_kernel(x_ptr, y_ptr, n_elements, softcap,
                         BLOCK_SIZE: tl.constexpr):
      pid = tl.program_id(0)
      block_start = pid * BLOCK_SIZE
      offsets = block_start + tl.arange(0, BLOCK_SIZE)
      mask = offsets < n_elements
      x = tl.load(x_ptr + offsets, mask=mask)
      y = softcap * libdevice.tanh(x.to(tl.float32) / softcap).to(x.dtype)
      tl.store(y_ptr + offsets, y, mask=mask)

  grid = (triton.cdiv(n_elements, BLOCK_SIZE),)  # Grid 可能很大

  # ── After: 每 program 处理 BLOCK_NUM 个块，Grid 缩小 ──
  @triton.jit
  def softcap_fwd_kernel(x_ptr, y_ptr, n_elements, softcap,
                         BLOCK_SIZE: tl.constexpr,
                         BLOCK_NUM: tl.constexpr):   # 新增多块参数
      pid = tl.program_id(0)
      for i in range(BLOCK_NUM):
          # 每 i 轮处理一个 BLOCK_SIZE 大小的块
          block_start = pid * BLOCK_SIZE * BLOCK_NUM + i * BLOCK_SIZE
          offsets = block_start + tl.arange(0, BLOCK_SIZE)
          mask = offsets < n_elements
          x = tl.load(x_ptr + offsets, mask=mask)
          y = softcap * libdevice.tanh(x.to(tl.float32) / softcap).to(x.dtype)
          tl.store(y_ptr + offsets, y, mask=mask)

  # Grid 缩小 BLOCK_NUM 倍，核内循环实现存算并行
  grid = (triton.cdiv(n_elements, BLOCK_SIZE * BLOCK_NUM),)
  ```
  - **案例**：Softcap — 新增 `BLOCK_NUM` 参数（1/2/4/8/16/32/64），核内循环处理多块，BLOCK\_SIZE×BLOCK\_NUM 决定每 program 总数据量
- 4.2 1D 展平替代 2D：小 shape 场景的索引开销消除
  - **问题**：小 shape 下 2D row-blocked kernel 的 2D 索引开销占比高——`offsets = row[:, None] * stride + col[None, :]` 需要二维广播乘加，当 `n_cols` 较小时（如 head\_dim=128），索引计算的时间接近甚至超过实际计算时间
  - **方案**：改为 1D flatten kernel，将二维 `(rows, cols)` 展平为一维 `rows*cols`，消除 2D 索引，提升内存合并（连续地址访问）
  ```python
  # ── Before: 2D kernel，小 shape 下索引开销高 ──
  @triton.jit
  def gelu_fwd_kernel(x_ptr, y_ptr, n_rows, n_cols, ...,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
      pid_m = tl.program_id(0)
      pid_n = tl.program_id(1)
      row = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
      col = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
      # 2D 索引：行偏移 × stride + 列偏移
      offsets = row[:, None] * n_cols + col[None, :]
      mask = (row[:, None] < n_rows) & (col[None, :] < n_cols)
      x = tl.load(x_ptr + offsets, mask=mask)
      y = x * 0.5 * (1.0 + tl.tanh(0.7978 * (x + 0.0447 * x * x * x)))
      tl.store(y_ptr + offsets, y, mask=mask)

  grid = (triton.cdiv(n_rows, BLOCK_M), triton.cdiv(n_cols, BLOCK_N))

  # ── After: 1D flatten kernel，消除 2D 索引 ──
  @triton.jit
  def gelu_fwd_kernel(x_ptr, y_ptr, n_elements, ...,
                      BLOCK_SIZE: tl.constexpr):
      pid = tl.program_id(0)
      # 1D 连续偏移，无二维广播
      offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
      mask = offsets < n_elements
      x = tl.load(x_ptr + offsets, mask=mask)
      y = x * 0.5 * (1.0 + tl.tanh(0.7978 * (x + 0.0447 * x * x * x)))
      tl.store(y_ptr + offsets, y, mask=mask)

  grid = (triton.cdiv(n_rows * n_cols, BLOCK_SIZE),)
  ```
  - **案例**：GELU / SiLU — `n_cols <= 10240` 或 `n_rows < num_cores` 时切换到 1D flatten kernel
- 4.3 移除 next\_power\_of\_2：避免不必要的 padding
  - **问题**：`next_power_of_2(d)` 将实际维度向上取整到 2 的幂（如 d=300 → 512，d=192 → 256），导致 padding 浪费 UB 空间（多分配 70% 的 UB）和多余循环迭代。更关键的是，当 `BLOCK_DIN = next_power_of_2(Din)` 使 `BLOCK_DIN > Din` 时，归约循环需要执行多次（无法单 Pass 完成），阻碍算子融合
  - **方案**：当实际维度满足 32B 对齐要求时（FP16 的 d 须为 16 的倍数），直接使用实际维度 `D_ROUNDED = d`，消除 padding 浪费并使循环只执行一次
  ```python
  # ── Before: 向上取整到 2 的幂，padding 浪费 + 多次循环 ──
  @triton.jit
  def rmsnorm_fwd_kernel(x_ptr, w_ptr, y_ptr, n_rows, ...,
                         D_ROUNDED: tl.constexpr):
      # D_ROUNDED = triton.next_power_of_2(d)
      # 当 d=300 时 D_ROUNDED=512，多分配 212 个元素的 UB 空间
      offs_d = tl.arange(0, D_ROUNDED)
      mask_d = offs_d < d  # 需要 mask 截断 padding 区域
      x = tl.load(x_ptr + row * d + offs_d, mask=mask_d, other=0.0)
      # 当 D_ROUNDED > d 时，循环可能执行多次
      mean = tl.sum(x * x, axis=0) / d
      ...

  # 调用处
  D_ROUNDED = triton.next_power_of_2(d)  # 300 → 512

  # ── After: 直接使用实际维度，无 padding + 单次循环 ──
  @triton.jit
  def rmsnorm_fwd_kernel(x_ptr, w_ptr, y_ptr, n_rows, ...,
                         D_ROUNDED: tl.constexpr):
      # D_ROUNDED = d（当 d 满足 32B 对齐时）
      offs_d = tl.arange(0, D_ROUNDED)
      # D_ROUNDED == d，无需 mask（已对齐）
      x = tl.load(x_ptr + row * d + offs_d)
      # D_ROUNDED == d，循环只执行一次 → 单 Pass 融合
      mean = tl.sum(x * x, axis=0) / d
      ...

  # 调用处
  D_ROUNDED = d  # 直接用 300（假设 300 是 16 的倍数，或已对齐）
  ```
  - **案例**：RMSNorm — `D_ROUNDED=triton.next_power_of_2(d)` → `D_ROUNDED=d`，使归约循环只执行一次
  - **案例**：Gate RMSNorm — `BLOCK_DIN=min(3264, next_power_of_2(Din))` → `BLOCK_DIN=Din`，使循环只执行一次，实现单 Pass 融合
- 4.4 SWA 循环结构优化：合并 vs 拆分
  - **问题**：滑动窗口注意力（SWA）中，全局窗口块（始终满密度，无需 local mask）和非全局窗口块（需 causal+local mask）的循环结构影响性能。两种循环结构在不同全局窗口占比下性能差异显著
  - **两种策略**（根据全局窗口占比选择）：
    - **合并**（全局窗口占比低时更优）：将两个独立循环合并为一个，用 `tl.where` 统一 mask，减少循环数量和 Grid 开销
    - **拆分**（全局窗口占比高时更优）：将统一循环拆分为两段——全局段无 mask 开销（满密度直接计算），非全局段只执行 causal+local mask
  ```python
  # ── 合并策略（全局窗口占比低时更优）──
  # Before: 两个独立循环，循环开销翻倍
  for kv in range(global_window_blocks):
      k = tl.load(k_block_ptr); qk = tl.dot(q, k)
      mask = global_mask  # 全局窗口 mask
      qk = tl.where(mask, qk, -inf)
  for kv in range(local_window_blocks):
      k = tl.load(k_block_ptr); qk = tl.dot(q, k)
      mask = causal_local_mask  # causal + local mask
      qk = tl.where(mask, qk, -inf)

  # After: 合并为一个循环，用 tl.where 统一 mask
  for kv in range(all_kv_blocks):
      k = tl.load(k_block_ptr); qk = tl.dot(q, k)
      is_global = (kv < global_threshold)
      # 统一 mask：全局段用 global_mask，非全局段用 causal_local_mask
      mask = tl.where(is_global, global_mask, causal_local_mask)
      qk = tl.where(mask, qk, -inf)

  # ── 拆分策略（全局窗口占比高时更优）──
  # Before: 统一循环，全局段也承受 mask 检查开销
  for kv in range(all_kv_blocks):
      k = tl.load(k_block_ptr); qk = tl.dot(q, k)
      # 每轮都检查 global_window_check，即使全局段不需要
      mask = causal_mask & (local_window_check | global_window_check)
      qk = tl.where(mask, qk, -inf)

  # After: 拆分为两段，全局段零 mask 开销
  for kv in range(global_blocks):       # 段 1：全局窗口，无 mask 开销
      k = tl.load(k_block_ptr); qk = tl.dot(q, k)
      # 全局窗口满密度，无需 mask
  for kv in range(local_blocks):        # 段 2：非全局窗口，causal + local mask
      k = tl.load(k_block_ptr); qk = tl.dot(q, k)
      mask = causal_mask & local_window_mask
      qk = tl.where(mask, qk, -inf)
  ```
  - **案例**：SWA prefill — 根据全局窗口占比选择合并或拆分策略

### 第 5 章 Autotune 调优策略：让调优器替你找到最优配置

> **核心理念**：NPU 的最优 BLOCK 尺寸取决于 UB 容量、数据类型、矩阵形状和硬件架构的复杂交互，人工试错效率低下。Autotune 让调优器自动搜索最优配置，是 NPU 算子调优的标准武器。
>
> **（1）Config 内容差异。** GPU 的 `triton.Config` 包含 Tiling 参数 + `num_stages`（软件流水线深度，1\~4）+ `num_warps`（线程束数量，4/8/16/32）。NPU 的 Config **仅含 Tiling 参数**——`num_stages` 和 `num_warps` 在 NPU 上无意义（NPU 无 GPU 式 warp 和软件流水线 stage），必须移除。NPU 的流水线并行通过 `multibuffer`（UB 双缓冲）和核内 `for` 循环实现，而非 `num_stages`。此外，NPU 的 Config 可以将编译参数（如 `multibuffer`、`enable_vf_fusion`）作为搜索维度。
>
> **（2）候选值约束。** NPU 的 BLOCK 候选值受 UB 容量硬约束——大 BLOCK 提升 Cube 利用率和数据复用，但超出 UB 容量则编译失败（`ub overflow`）。经验公式：`最大Tiling元素数 ≈ UB_SIZE / (dtype_bytes × buffer数量)`，如 FP16（2B）3 buffers 时 248KB/(2×3) ≈ 40960 元素。此外，NPU 的 BLOCK 值应满足 32B 对齐（FP16 须为 16 的倍数），使用 2 的幂次值（32/64/128/256/512/1024/2048）天然满足对齐要求。非 2 的幂的值（如 100、86）也可使用，但需确保对齐。
>
> **（3）key 参数化与缓存。** `key` 参数指定调优触发维度——当 key 值变化时重新调优，相同值复用缓存。NPU 上需要特别注意：不同 dtype（fp16 vs bf16）的 UB 占用不同（bf16 和 fp16 都是 2B，但中间计算可能不同），应将 dtype 加入 key；不同算子属性（如 FA 的 QK\_DIM/V\_DIM/MASK\_FN/SPARSE\_OPT）影响最优 Tiling，也应加入 key。Autotune 的首次运行会尝试所有候选 Config，配置数过多会显著增加首次运行时间——建议总配置数控制在 20 个以内，使用 `prune_configs_by` 提前剪枝不可行配置。

- 5.1 BLOCK 尺寸搜索：从小 BLOCK 起步逐步增大
  - **问题**：固定小 BLOCK 导致 Cube 利用率低和数据复用差，但盲目增大可能溢出 UB。开发者难以人工判断特定 shape 下的最优 BLOCK 值
  - **方案**：从小 BLOCK 起步，通过 Autotune 逐步增大搜索最优值。候选值可包含非 2 的幂以精确匹配实际维度（如序列长度 100）
  ```python
  # ── Before: 固定小值，无法适配不同 shape ──
  @triton.jit
  def fwd_kernel(..., BLOCK_M: tl.constexpr):
      # BLOCK_M 固定为 32，大 shape 下 Cube 利用率低
      ...

  # ── After: Autotune 搜索，自动选择最优 BLOCK ──
  @triton.autotune(
      configs=[
          triton.Config({"BLOCK_M": 32}),
          triton.Config({"BLOCK_M": 64}),
          triton.Config({"BLOCK_M": 128}),
          triton.Config({"BLOCK_M": 256}),
          # 场景：非 2 的幂精确匹配实际维度
          triton.Config({"BLOCK_M": 100}),  # 序列长度恰好 100
      ],
      key=["QK_DIM", "V_DIM"],  # 按 head_dim 缓存最优 config
  )
  @triton.jit
  def fwd_kernel(..., BLOCK_M: tl.constexpr):
      ...
  ```
  - **案例**：FA bwd\_q blkM 32→64→128，bwd\_kv blkN 32→64→128，bwd\_preprocess 32→128/256
  - **案例**：Fused MatMul fwd 从搜索空间收敛到 BM=BN=BK=128（Cube 单元最优分块）
  - **案例**：RoPE TOKEN\_BLOCK\_SIZE=3（非 2 的幂精确匹配）
- 5.2 多维组合搜索：从单 config 到全空间
  - **问题**：单维度搜索无法捕捉 BLOCK\_M × BLOCK\_N × BLOCK\_K 的交互效应——如 BLOCK\_M=128 在 BLOCK\_N=32 时最优，但在 BLOCK\_N=128 时可能 BLOCK\_M=64 更优（UB 约束）
  - **方案**：从单 config 扩展到多维组合搜索，使用列表推导生成笛卡尔积
  ```python
  # ── Before: 单 config，无法捕捉维度交互 ──
  @triton.autotune(
      configs=[triton.Config({"BLOCK_M": 128})],  # 只有一个配置
      key=["M", "N"],
  )

  # ── After: 多维组合搜索，覆盖 BM×BN×BK 全空间 ──
  @triton.autotune(
      configs=[
          triton.Config({"BLOCK_M": BM, "BLOCK_N": BN, "BLOCK_K": BK})
          for BM in [128, 256, 512]
          for BN in [64, 128, 256, 512]
          for BK in [32, 64, 128, 512]
      ],  # 3×4×4 = 48 个配置
      key=["M", "N", "K"],
  )
  @triton.jit
  def matmul_kernel(..., BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                    BLOCK_K: tl.constexpr):
      ...
  ```
  - **案例**：SwiGLU — 从单 config 扩展到 BM×BN×BK 三维搜索
- 5.3 Key 参数化：按形状/dtype 缓存最优 config
  - **问题**：不同输入形状/属性的最优 config 不同，无 key 参数化时跨形状重复调优（每次 shape 变化都重新搜索全部 Config）。更隐蔽的问题：不同 dtype（fp16 vs bf16）的 UB 占用和计算路径不同，但若 dtype 不在 key 中，fp16 选出的 config 可能对 bf16 不是最优
  - **方案**：按输入形状/属性缓存最优 config；将 dtype 也加入 key（通过 `X_ptr.dtype` 引用指针的 dtype）
  ```python
  # ── Before: 基础 key，跨 shape/dtype 重复调优 ──
  @triton.autotune(
      configs=[...],
      key=["n_rows", "n_cols"],  # 缺少 dtype，fp16/bf16 共用 config
  )

  # ── After: 扩展 key（含 dtype + 算子属性）──
  @triton.autotune(
      configs=[...],
      # 将 dtype 加入 key：fp16 和 bf16 各自缓存最优 config
      key=["n_rows", "n_cols", "X_ptr.dtype"],
  )

  # 或按算子属性缓存（Flash Attention 场景）
  @triton.autotune(
      configs=get_fwd_configs(),
      # QK_DIM/V_DIM 影响 UB 占用，MASK_FN/SPARSE_OPT 影响计算量
      key=["QK_DIM", "V_DIM", "MASK_FN", "SPARSE_OPT"],
  )
  ```
  - **案例**：FA v5 — 按 QK\_DIM/V\_DIM/MASK\_FN/SPARSE\_OPT 缓存
  - **案例**：RMSNorm — 将 dtype 加入 key（fp16 vs bf16 UB 占用不同）
- 5.4 搜索空间剪枝：提前过滤无效 config
  - **问题**：大搜索空间中包含两类无效 Config：(1) 会溢出 UB 的 Config（BLOCK\_M×BLOCK\_N 过大），首次运行时编译失败浪费时间；(2) 特定架构上永远不会胜出的 Config（如 910B 上大 BLOCK 因 UB 只有 192KB 而总是不如小 BLOCK）。这些无效 Config 浪费首次运行编译时间
  - **方案一（UB 剪枝）**：通过 `prune_configs_by` 的 `early_config_prune` 回调函数，在编译前过滤会溢出 UB 的 Config
  - **方案二（架构剪枝）**：按架构裁剪搜索空间，只保留该架构上可能胜出的 Config
  ```python
  # ── UB 剪枝：过滤会溢出 UB 的 Config ──
  def _prune_oversized_tiles(configs, named_args, **kwargs):
      # 过滤 BLOCK_M × BLOCK_N 超过 UB 容量的 Config
      return [c for c in configs
              if c.kwargs['BLOCK_M'] * c.kwargs['BLOCK_N'] <= 1048576]

  @triton.autotune(
      configs=[...],  # 原始搜索空间
      prune_configs_by={"early_config_prune": _prune_oversized_tiles},
      key=["M", "N"],
  )

  # ── 架构剪枝：按架构裁剪搜索空间 ──
  def get_configs():
      all_configs = [
          triton.Config({"BLOCK_M": BM, "BLOCK_N": BN})
          for BM in [32, 64, 128, 256]
          for BN in [32, 64, 128, 256]
      ]
      if is_910():  # 910B UB 只有 192KB，大 BLOCK 容易溢出
          # 只保留 BLOCK_M ≤ 128 的 Config（910B 胜出 config）
          return [c for c in all_configs if c.kwargs['BLOCK_M'] <= 128]
      return all_configs  # 910_95 UB 248KB，全空间搜索

  @triton.autotune(configs=get_configs(), key=["M", "N"])
  ```
  - **案例**：RMSNorm / Int8GEMM — UB 溢出 config 提前过滤
  - **案例**：SWA a2 — 8 configs → 1 config（910B 架构剪枝）
- 5.5 编译参数加入搜索
  - **问题**：编译参数（如 `enable_vf_fusion`、`multibuffer`）的最优值取决于具体 shape，人工选择容易遗漏。特别是 `multibuffer`：开启后 UB 减半但实现存算并行，某些 shape 下开启更优（计算密集），某些 shape 下关闭更优（UB 受限需要大 BLOCK）
  - **方案**：将编译参数作为 Autotune 搜索维度，让调优器自动选择
  ```python
  # ── 编译参数加入搜索 ──
  @triton.autotune(
      configs=[
          # 同一 BLOCK 下，multibuffer 和 enable_vf_fusion 各试一次
          triton.Config({"BLOCK_M": 8}, multibuffer=True, enable_vf_fusion=True),
          triton.Config({"BLOCK_M": 8}, multibuffer=True, enable_vf_fusion=False),
          triton.Config({"BLOCK_M": 8}, multibuffer=False, enable_vf_fusion=True),
          triton.Config({"BLOCK_M": 8}, multibuffer=False, enable_vf_fusion=False),
      ],
      key=["M", "N"],
  )
  ```
  - **案例**：RMSNorm — `enable_vf_fusion` 作为搜索维度加入 autotune configs
  - **案例**：RoPE — `multibuffer` 作为搜索维度加入 autotune configs
- 5.6 autotune 结果磁盘缓存：避免重复调优
  - **问题**：每次运行都重新 autotune 浪费时间，特别适用于固定 shape 的生产环境（如推理场景的 seq\_len 固定）
  - **方案**：开启 autotune 的**磁盘缓存**——`cache_results=True`，或环境变量 `TRITON_CACHE_AUTOTUNING=1` 全局开启。首次调用正常编译并 benchmark 全部 config，把每个 config 的耗时写入 `TRITON_CACHE_DIR/<hash>/<函数名>.autotune.json`；后续相同 `key` 的调用直接加载缓存、取耗时最短的 config，**跳过整个搜索阶段**。注意这是**按 key 缓存**而非永久固定：`key` 参数取值、dtype、编译选项或 config 列表任一变化都会使缓存失效并重新调优
  ```python
  @triton.autotune(
      configs=[
          triton.Config({"BLOCK_SIZE": 1024}, multibuffer=True),
          triton.Config({"BLOCK_SIZE": 2048}, multibuffer=True),
          triton.Config({"BLOCK_SIZE": 1024}, multibuffer=False),
      ],
      key=["n"],            # n 不变 → 命中磁盘缓存，跳过 benchmark
      cache_results=True,   # 关键：开启磁盘缓存
  )
  @triton.jit
  def kernel(x_ptr, y_ptr, n, BLOCK_SIZE: tl.constexpr):
      offs = tl.arange(0, BLOCK_SIZE)
      v = tl.load(x_ptr + offs)
      tl.store(y_ptr + offs, v)
  ```
  - **缓存文件内容**：`<函数名>.autotune.json` 保存 `key` 与每个 config 的实测耗时（中位数/下界/上界），命中后按耗时取 `min`
  ```json
  {
    "key": [8192],
    "configs_timings": [
      [{"kwargs": {"BLOCK_SIZE": 1024}, "multibuffer": true}, [0.023, 0.021, 0.030]],
      [{"kwargs": {"BLOCK_SIZE": 2048}, "multibuffer": true}, [0.031, 0.029, 0.035]],
      [{"kwargs": {"BLOCK_SIZE": 1024}, "multibuffer": false}, [0.040, 0.038, 0.045]]
    ]
  }
  ```
  - **彻底固定配置**：若 shape 长期不变，可调优一次后把最优 config 直接写死、去掉 `@triton.autotune`，连首次调优开销也省掉
  ```python
  @triton.jit
  def kernel(x_ptr, y_ptr, n, BLOCK_SIZE: tl.constexpr):
      offs = tl.arange(0, BLOCK_SIZE)
      v = tl.load(x_ptr + offs)
      tl.store(y_ptr + offs, v)

  # 直接指定最优 config（BLOCK_SIZE=1024 + multibuffer=True），跳过搜索
  kernel[(grid,)](x, y, n, BLOCK_SIZE=1024, multibuffer=True)
  ```
  - **案例**：RoPE / 推理场景 — seq\_len 固定使 `key` 不变，后续推理直接命中缓存
- 5.7 动态序列长度：避免每个新长度都触发 Autotune
  - **问题**：在自回归解码、变长输入等整网场景中，切分轴 `seq_len` 可能连续取 `1001、1002、1003……`。如果使用 `key=["seq_len"]`，每个未见过的长度都会产生新的调优缓存项。相同的完整 key 可以复用已有结果，但开启 5.6 节的磁盘缓存只能解决相同 key 的跨进程复用，不能自动让新长度沿用附近长度的配置。
  - **选择原则**：将大量精确长度归并为有限的性能区间，并让 kernel 在运行时处理真实长度。
    **方案1：长度分桶：用桶值做调优 key，用真实长度执行。**
  将长度向上归并到 2 的幂，例如 `1001、1002、1024` 的桶值均为 `1024`，`1025` 的桶值为 `2048`。也可以根据性能拐点设置不等宽区间。桶值只控制配置复用，不要求将输入 padding 到桶上界；grid、循环边界和 mask 仍使用真实长度。
  ```python
  def length_bucket(seq_len):
      return triton.next_power_of_2(max(128, seq_len))

  @triton.autotune(
      configs=[
          triton.Config({"BLOCK_SIZE": 256}),
          triton.Config({"BLOCK_SIZE": 512}),
          triton.Config({"BLOCK_SIZE": 1024}),
      ],
      key=["seq_bucket"],  # seq_len 的精确值不进入调优 key
      cache_results=True,
  )
  @triton.jit(do_not_specialize=["seq_len", "seq_bucket"])
  def add_one_bucket_kernel(X, Y, seq_len, seq_bucket,
                            BLOCK_SIZE: tl.constexpr):
      pid = tl.program_id(0)
      for block_id in range(pid, tl.cdiv(seq_len, BLOCK_SIZE),
                            tl.num_programs(0)):
          offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
          mask = offsets < seq_len
          x = tl.load(X + offsets, mask=mask, other=0.0)
          tl.store(Y + offsets, x + 1.0, mask=mask)

  def add_one_bucket(x, num_cores):
      y = torch.empty_like(x)
      seq_len = x.numel()
      if seq_len == 0:
          return y
      seq_bucket = length_bucket(seq_len)
      grid = lambda meta: (
          min(num_cores, triton.cdiv(seq_len, meta["BLOCK_SIZE"])),
      )
      add_one_bucket_kernel[grid](x, y, seq_len, seq_bucket)
      return y
  ```
  同桶的第一次调用会在当时的真实长度上 benchmark，后续直接复用，**不会对桶内每个长度继续搜索**。应选择有代表性的长度预热，并验证桶下界、上界及常见长度的正确性和性能；若桶内性能差异明显，拆分桶，或离线在多个代表点上综合选择配置。所有候选配置都必须对桶内全部长度正确，不能依赖首次调优长度的整除性，也不能省略必要的尾部 mask。

  **方案2：启发式分派：离线调优形成规则，线上直接选择配置。**

  在目标硬件、dtype、布局和典型长度上离线测量，找到不同配置的性能交叉点，再将结果写成区间表或条件分支。线上只选择配置并调用普通 `@triton.jit` kernel，不执行 `@triton.autotune` 的搜索；也可以用 `@triton.heuristics` 把选配置的规则直接挂到 kernel 上，在 launch 时动态算出 `BLOCK_SIZE`，效果与手动分派等价：
  ```python
  def select_block_size(seq_len):
      if seq_len <= 512:
          return 256
      if seq_len <= 4096:
          return 512
      return 1024  # 为更长输入提供覆盖路径

  @triton.heuristics({"BLOCK_SIZE": lambda args: select_block_size(args["seq_len"])})
  @triton.jit(do_not_specialize=["seq_len"])
  def add_one_fixed_kernel(X, Y, seq_len, BLOCK_SIZE: tl.constexpr):
      pid = tl.program_id(0)
      for block_id in range(pid, tl.cdiv(seq_len, BLOCK_SIZE),
                            tl.num_programs(0)):
          offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
          mask = offsets < seq_len
          x = tl.load(X + offsets, mask=mask, other=0.0)
          tl.store(Y + offsets, x + 1.0, mask=mask)

  def add_one_dispatch(x, num_cores):
      y = torch.empty_like(x)
      seq_len = x.numel()
      if seq_len == 0:
          return y
      grid = lambda meta: (
          min(num_cores, triton.cdiv(seq_len, meta["BLOCK_SIZE"])),
      )
      add_one_fixed_kernel[grid](x, y, seq_len)  # BLOCK_SIZE 由 heuristics 计算
      return y
  ```
  **调优缓存与编译缓存需要分别处理**：缩小 autotune key 只减少 benchmark 次数。如果将每步变化的真实长度标为 `tl.constexpr`，即使命中调优缓存，JIT 仍可能为每个长度编译新版本。上述示例将真实长度保留为运行时参数，并用 `do_not_specialize` 避免按该标量的值或对齐特征进一步特化；它不禁止 `seq_bucket` 作为 autotune key。有限的 `BLOCK_SIZE: tl.constexpr` 仍会对应不同编译版本，可在上线前预热各桶或分派分支。`prune_configs_by` 只能减少每次搜索的候选数，无法消除大量精确长度带来的新 key。

### 第 6 章 CV 流水线优化：让 Cube 和 Vector 并行工作

> **核心理念**：NPU 的 Cube 和 Vector 是物理分离的两个执行单元——不优化则串行执行，流水线气泡严重。CV 流水线优化的核心是让 Cube 和 Vector 并行工作，消除气泡。
>
> **（1）Cube/Vector 分离架构与数据通路。** 910\_95 的每个 AI Core 由 1 Cube（矩阵乘单元）+ 2 Vector（向量计算单元）+ Scalar 组成。`tl.dot` 映射到 Cube 的 `mmad` 指令，数据路径是 `GM →(MTE2)→ L1 →(MTE1)→ L0A/L0B → Cube → L0C →(fixpipe)→ UB`。关键点：Cube 输出先写入 **L0C**（256KB，NZ 格式——一种分块压缩格式），需 **fixpipe** 硬件流水线搬运到 UB 并完成格式转换（NZ→ND，即分块压缩→常规行优先）后 Vector 才能处理。fixpipe 还支持**随路操作**（在搬运同时完成）：bias 累加、精度转换（fp32→fp16）、转置——可替代后续 Vector 操作，是融合后处理的关键硬件特性。A5（910\_95）的 fixpipe L0C→UB **直通**（零 GM 中转，单步完成）；A2（910B）的 L0C→**GM**→UB（额外两次 GM 访问，性能损失显著）。
>
> **（2）PIPE 模型与串行执行问题。** NPU 的硬件流水线用 PIPE 枚举标识：`PIPE_M`（Cube 计算流水线）、`PIPE_V`（Vector 计算流水线）、`PIPE_FIX`（fixpipe 流水线）、`PIPE_MTE2`（GM→UB 搬入）、`PIPE_MTE3`（UB→GM 搬出）。不优化时，Cube 计算（PIPE\_M）和 Vector 计算（PIPE\_V）串行执行——Vector 必须等 Cube 的 fixpipe（PIPE\_FIX）完成后才能开始，形成流水线气泡。
>
> **（3）CV 流水线优化的核心思想。** 通过**指令重排序**，将没有数据依赖的 Cube 操作（`tl.dot`）提前到 Vector 操作之前，使 Cube 和 Vector 并行执行：当 Cube 执行矩阵计算时，Vector 同时执行向量计算。优化后的执行时序：Cube 先启动 `tl.dot`，Vector 随后并行执行独立计算，最后 Vector 处理依赖 Cube 结果的操作。

- 6.1 指令重排序 + VF 融合控制：让 Cube 提前启动
  - **问题一**：无数据依赖的 `tl.dot` 排在 Vector 计算之后（原始代码顺序），Cube 等待 Vector 完成后才启动，整个 Vector 执行期间 Cube 空闲
  - **方案一**：将无数据依赖的 `tl.dot` 提前到 Vector 计算之前，使 Cube 与 Vector 并行。注意：依赖 Cube 结果的操作（如 `acc += pv`）必须延后
  - **问题二**：前驱 Vector 计算与 `tl.dot` 的 VF（Vector Function）融合链过长，导致 UB 溢出或阻止 dot 调度到 Cube（编译器将 `ds` 的计算融合到 dot 的 VF 链中，使整个链在 Vector 执行，Cube 无法独立调度）
  - **方案二**：在 `tl.dot` 之前插入 `compile_hint("break_vf")`，打断 VF 融合链，确保 `ds` 独立计算后 `tl.dot` 单独调度到 Cube
  ```python
  # ── 指令重排序：让 Cube 提前启动 ──
  # Before: Vector 先执行，Cube 等待（Flash Attention softmax 循环体）
  m_new = tl.maximum(m, tl.max(s, 1))
  alpha = tl.math.exp2((m - m_new) * qk_scale)  # Vector
  p = tl.math.exp2((s - m_new[:, None]) * qk_scale)
  p_sum = tl.sum(p, 1)       # Vector
  acc *= alpha[:, None]      # Vector
  v = load_if(v_block_ptr, False, True)
  acc += tl.dot(p.to(dtype), v)  # Cube（等以上全部 Vector 完成后启动）

  # After: Cube 提前启动，与 Vector 并行
  m_new = tl.maximum(m, tl.max(s, 1, propagate_nan=True),
                     propagate_nan=tl.PropagateNan.ALL)
  p = tl.math.exp(s - m_new[:, None])
  v = load_if(v_block_ptr, False, True)
  pv = tl.dot(p.to(dtype), v)          # Cube 提前启动（无依赖 acc）
  p_sum = tl.sum(p, 1)                  # Vector（与 Cube 并行）
  alpha = tl.math.exp(m - m_new)        # Vector（与 Cube 并行）
  acc = acc * alpha[:, None] + pv       # 合并：依赖 Cube 结果，放最后
  l = l * alpha + p_sum
  m = m_new

  # ── break_vf：打断 VF 融合链（Flash Attention 反向）──
  import triton.language.extra.cann.extension as extension

  # Before: ds 可能被融合到 dot 的 VF 链，阻止 Cube 调度
  ds = p * (dp - d[:, None])
  ds = tl.where(mask, ds, 0.0)
  ds = ds.to(dtype)
  dq += tl.dot(ds, k)  # ds 的 VF 链可能过长，UB 溢出或 Cube 无法独立调度

  # After: 打断 VF 链，ds 独立计算后 tl.dot 单独调度到 Cube
  ds = p * (dp - d[:, None])
  ds = tl.where(mask, ds, 0.0)
  ds = ds.to(dtype)
  extension.compile_hint(ds, "break_vf")  # 在此处打断 VF 融合
  dq += tl.dot(ds, k)  # tl.dot 独立调度到 Cube
  ```
  - **案例**：FA v4/v8 fwd — `tl.dot(p, v)` 提前到 `p_sum` / `alpha` 之前
  - **案例**：FA bwd\_q/bwd\_qkv — `ds` 计算后插入 `break_vf`，确保 `tl.dot` 调度到 Cube
- 6.2 CV 协同配置：子块绑定与完整七参数
  - **问题**：CV 融合算子需要 Cube 和 Vector 以 1:2 比例协同（匹配 1 Cube + 2 Vector 架构），但默认配置不启用协同——Cube 和 Vector 各自独立调度，无法实现流水线并行。此外，CV 融合算子需要一组完整的编译参数来控制 CV 协同行为
  - **方案一**：CV 融合算子启用 `enable_auto_bind_sub_block=True`（自动绑定子块，实现 CV 1:2 协同），纯 Vector 算子禁用（无 Cube 需要协同）
  - **方案二**：CV 融合算子配置完整的七参数集，精确控制 CV 协同、多缓冲、同步策略等
  ```python
  # ── 子块绑定决策 ──
  # CV 融合算子（含 tl.dot）：启用子块绑定，实现 CV 1:2 协同
  fwd_kernel[grid](..., enable_auto_bind_sub_block=True)
  # 纯 Vector 算子（无 tl.dot）：禁用子块绑定
  rmsnorm_kernel[grid](..., enable_auto_bind_sub_block=False)

  # ── CV 完整七参数配置（CV 融合算子专用）──
  fwd_kernel[grid](
      ...,
      enable_mixed_cv=True,                              # 启用 Cube/Vector 混合调度
      sync_solver=True,                                  # 自动求解 CV 间同步点
      enable_auto_bind_sub_block=True,                   # 自动绑定子块（CV 1:2 协同）
      set_workspace_multibuffer=2,                       # workspace 双缓冲
      limit_auto_multi_buffer_of_local_buffer="no-limit", # 不限制 UB 多缓冲分配
      enable_flatten=False,                              # CV 融合必须 False（不能展平）
      multibuffer=True,                                  # UB 双缓冲，存算并行
  )

  # ── 纯 Vector 算子配置（只需 2 个参数）──
  rmsnorm_kernel[grid](
      ...,
      multibuffer=True,
      limit_auto_multi_buffer_of_local_buffer="no-limit",
  )
  ```
  - **案例**：FA v8 全系列 — 完整七参数配置
  - **案例**：FA bwd\_q — 从 `enable_auto_bind_sub_block=False` 改为启用
- 6.3 Scale 融合：减少 FixPipe 后处理
  - **问题**：`tl.dot` 之后的 scale 乘法、累加等操作在 fixpipe 阶段或 Vector 单元执行，增加 Vector 单元压力。特别是 online softmax 中的 `alpha` 缩放 + PV matmul + 累加三步操作，每步都占用 Vector 周期
  - **方案一（Scale 融合到输入）**：将 dot 之后的 scale 乘法移到 dot 输入中，减少一次 Vector 乘法
  - **方案二（Accumulator-in-dot）**：将 alpha 缩放 + PV matmul + 累加融合为单条 `tl.dot` 三参数调用 `tl.dot(p, v, acc)`，利用 Cube 的硬件累加器
  - **方案三（k\_scale post-dot）**：将 k\_scale 从 pre-dot（dot 前对 k 乘 scale）移到 post-dot（dot 后与结果融合），消除额外 Vector 乘法
  ```python
  # ── Scale 融合到输入（SDPA 反向）──
  # Before: scale 在 dot 之后，额外一次 Vector 乘法
  dq += tl.dot(ds, k) * scale
  # After: scale 融入 ds（dot 输入），消除 dot 后乘法
  ds = ds * scale
  dq += tl.dot(ds, k)

  # ── Accumulator-in-dot（SDPA 前向 online softmax）──
  # Before: 缩放 + matmul + 累加（三步，三次 Vector/Cube 操作）
  block_o = alpha[:, None] * block_o + tl.dot(block_p, block_v)
  # After: 融合为单条 dot 三参数形式（Cube 硬件累加器完成缩放+累加）
  block_o = tl.dot(block_p, block_v, block_o * alpha[:, None])

  # ── k_scale post-dot（LightningIndexer）──
  # Before: pre-dot Vector 乘法（对 k 整个矩阵乘 scale）
  k = k * k_scale[:, None]
  qk = tl.dot(q, tl.trans(k))
  # After: scale 融合到后续 ReLU（只对 qk 结果乘，元素更少）
  qk = tl.dot(q, tl.trans(k))
  relu_qk = tl.maximum(qk * k_scale[None, :], 0.0)
  ```
  - **案例**：SDPA fwd — accumulator-in-dot 融合
  - **案例**：SDPA bwd — scale 融合到 matmul 输入
  - **案例**：LightningIndexer — k\_scale post-dot 融合

### 第 7 章 存算并行与软件流水线：让数据搬运与计算重叠

> **核心理念**：NPU 的数据搬运由专用 DMA 引擎执行，与计算单元物理分离。存算并行的核心是利用这一物理分离，让 MTE2（搬入）与 Vector/Cube 计算**自动重叠**。
>
> **（1）专用 DMA 引擎与物理分离。** GPU 上 `tl.load` / `tl.store` 通过 SM 内部的 LD/ST 队列执行，与计算共享 Warp 调度资源——需要 `cp.async` + `num_stages` 显式管理异步拷贝才能实现存算重叠，否则 load 和 compute 串行执行。NPU 则有**物理独立的专用 DMA 引擎**：MTE2（GM↔L1/UB，数据搬入引擎）和 MTE3（UB→GM/L1，数据搬出引擎），与 Vector/Cube 计算单元是不同的硬件 Pipeline（PIPE\_MTE2/PIPE\_MTE3 vs PIPE\_V/PIPE\_M）。这意味着 DMA 搬运和计算**天然并行**——当 MTE2 搬入下一块数据时，Vector 可以同时处理当前块数据，无需开发者显式管理异步操作。
>
> **（2）multibuffer 双缓冲机制。** `multibuffer=True` 启用 UB 双缓冲（Ping-Pong）：将 UB 分为两份，一份供当前块计算（Vector 读取），另一份供下一块搬入（MTE2 写入），两份交替使用。这使得下一块的 MTE2 搬入与当前块的 Vector 计算**自动重叠**——无需修改代码逻辑，只需开启参数。权衡：开启后 UB 可用空间减半（910\_95 从 248KB → 124KB，910B 从 192KB → 96KB），可能需要减小 BLOCK\_SIZE。910\_95 的 multibuffer **默认关闭**（需显式 `multibuffer=True`），910B 默认开启。
>
> **（3）核内 for 循环——存算并行的标准模式。** 持久化 kernel（Grid = 物理核数）的核内 `for` 循环天然形成软件流水线——上一轮 compute 与下一轮 load 自动重叠。这是因为循环内 `tl.load`（MTE2）和 `tl.dot`/计算（Cube/Vector）分属不同 PIPE，编译器自动调度它们并行执行。循环不变量外提（hoisting）和指针基址+offset 重算（避免 `+=`）直接影响 DMA 并行效率——循环内重复加载不变量或使用 `+=` 引入 RAW 依赖都会阻止 MTE2 并行发射。

- 7.1 核内循环 + 循环不变量外提：存算并行的基础模式
  - **问题**：标准 Grid 模式下每 program 一个块，无核内流水线深度（只做一次 load→compute→store，无法重叠）；循环内重复加载不变量（如 weight、col\_offsets）是纯浪费——每轮重新发起 MTE2 搬运相同数据
  - **方案一**：用核内 for 循环替代大 Grid，使上一轮 compute 与下一轮 load 重叠（持久化 kernel 模式）
  - **方案二**：将循环内不变的计算/加载提到循环外（hoisting），只加载一次，循环内直接复用
  ```python
  # ── 核内循环实现存算并行（RMSNorm 持久化 kernel）──
  # Before: 每 program 一个块，无核内流水线
  @triton.jit
  def rmsnorm_kernel(x_ptr, w_ptr, y_ptr, n_rows, ...):
      pid = tl.program_id(0)
      # 单行处理，无循环
      row = pid
      x = tl.load(x_ptr + row * stride + offs_d)
      ...  # compute
      tl.store(y_ptr + row * stride + offs_d, y)

  grid = (n_rows,)  # Grid 可能远超物理核数

  # After: 持久化 kernel，核内循环处理多行
  @triton.jit
  def rmsnorm_kernel(x_ptr, w_ptr, y_ptr, n_rows, ...,
                     NUM_CORES: tl.constexpr):
      pid = tl.program_id(0)
      # 核内循环：上一轮 compute 与下一轮 load 自动重叠
      for row in range(pid, n_rows, NUM_CORES):
          x = tl.load(x_ptr + row * stride + offs_d)  # 本轮 load (MTE2)
          # 上轮 compute (Vector) 与本轮 load 并行
          mean = tl.sum(x * x, axis=0) / D
          rstd = 1.0 / tl.sqrt(mean + eps)
          y = (x * rstd[:, None]) * w + b
          tl.store(y_ptr + row * stride + offs_d, y)

  grid = (NUM_CORES,)  # Grid = 物理核数

  # ── 循环不变量外提（RMSNorm weight 加载）──
  # Before: 每轮重复加载 weight（循环不变量）
  for row in range(pid, n_rows, NUM_CORES):
      w = tl.load(w_ptr + offs_d)  # weight 每轮重新加载（浪费）
      x = tl.load(x_ptr + row * stride + offs_d)
      y = (x * rstd[:, None]) * w + b
      tl.store(y_ptr + ..., y)

  # After: weight 循环外加载一次，循环内直接复用
  w = tl.load(w_ptr + offs_d)  # 循环外加载一次
  col_offsets = tl.arange(0, D_ROUNDED)  # 循环外计算一次
  for row in range(pid, n_rows, NUM_CORES):
      x = tl.load(x_ptr + row * stride + col_offsets)
      y = (x * rstd[:, None]) * w + b
      tl.store(y_ptr + ..., y)
  ```
  - **案例**：RMSNorm — 持久化 kernel + weight/col\_offsets 循环外加载
- 7.2 循环手动展开：双路并行增加流水线深度
  - **问题**：单路循环的流水线深度有限——每轮只有一个 load 和一个 compute，load/compute 重叠不充分（MTE2 搬入一块后 Vector 计算一块，MTE2 空闲等待 Vector 完成）
  - **方案**：每轮处理 2 个（或更多）数据块，双路 load/compute/store——路 0 的 compute 与路 1 的 load 重叠，MTE2 和 Vector 都满载
  ```python
  # ── Before: 单路循环，流水线深度有限 ──
  @triton.jit
  def rmsnorm_kernel(x_ptr, ..., NUM_CORES: tl.constexpr):
      pid = tl.program_id(0)
      for row in range(pid, n_rows, NUM_CORES):
          x = tl.load(x_ptr + row * stride + offs_d)  # 单路 load
          mean = tl.sum(x * x, axis=0) / D             # 单路 compute
          y = (x * rstd[:, None]) * w
          tl.store(y_ptr + row * stride + offs_d, y)   # 单路 store

  # ── After: 双路展开，路 0 compute 与路 1 load 重叠 ──
  @triton.jit
  def rmsnorm_kernel(x_ptr, ..., NUM_CORES: tl.constexpr):
      pid = tl.program_id(0)
      # 每轮处理 2 行，步长翻倍
      for row in range(pid * 2, n_rows, NUM_CORES * 2):
          # 路 0 和路 1 的 load 可并行发射（无依赖）
          x_0 = tl.load(x_ptr + (row) * stride + offs_d)      # 路 0 load
          x_1 = tl.load(x_ptr + (row + 1) * stride + offs_d)  # 路 1 load
          # 路 0 compute（与路 1 load 重叠）
          mean_0 = tl.sum(x_0 * x_0, axis=0) / D
          rstd_0 = 1.0 / tl.sqrt(mean_0 + eps)
          y_0 = (x_0 * rstd_0[:, None]) * w
          # 路 1 compute
          mean_1 = tl.sum(x_1 * x_1, axis=0) / D
          rstd_1 = 1.0 / tl.sqrt(mean_1 + eps)
          y_1 = (x_1 * rstd_1[:, None]) * w
          # 双路 store
          tl.store(y_ptr + (row) * stride + offs_d, y_0)
          tl.store(y_ptr + (row + 1) * stride + offs_d, y_1)
  ```
  - **案例**：RMSNorm — 每轮处理 2 行（`pid*2` 步长），双路 load/compute/store
- 7.3 multibuffer 启用：Ping-Pong 双缓冲
  - **问题**：当不启用 Ping-Pong 双缓冲时， load 和 compute 串行执行——MTE2 搬入完成后 Vector 才能开始计算，MTE2 在 Vector 计算期间空闲
  - **方案**：当前 multibuffer 已默认开启，或者通过设置 `multibuffer=True` 的方式手动启用，UB 双缓冲使下一 tile 的内存加载与当前 tile 的计算重叠。Ping-Pong 模式：buffer A 供当前 tile 计算，buffer B 供下一 tile 搬入，交替使用
  - **权衡**：开启后 UB 可用空间减半（248KB → 124KB），需评估对 BLOCK\_SIZE 的影响——可能需要减小 BLOCK\_SIZE 或减少核内同时存活的 tensor
  ```python
  # ── Before: multibuffer 默认关闭（910_95），load 和 compute 串行 ──
  rope_kernel[grid](
      in_ptr, pos_ptr, cu_seqlens, out_ptr,
      head, base, DIM, max_seq_len, REVERSE,
      # 无 multibuffer 参数，使用默认值 False
  )

  # ── After: 显式开启 multibuffer ──
  rope_kernel[grid](
      in_ptr, pos_ptr, cu_seqlens, out_ptr,
      head, base, DIM, max_seq_len, REVERSE,
      multibuffer=True,  # 显式开启 Ping-Pong 双缓冲
  )
  ```
  - **案例**：RoPE / Softcap / FA / RMSNorm — 显式开启 multibuffer
- 7.4 指针依赖消除：避免循环内 `+=` 引入 RAW 依赖
  - **问题**：循环内 `ptrs += BLOCK_SIZE` 引入 Read-After-Write（RAW）依赖——当前迭代的指针值依赖前一次迭代的写入结果，导致编译器无法重排指令，MTE2 load 指令必须等上一轮 `+=` 完成后才能发射，阻塞流水线
  - **方案**：改为每轮独立计算 offset（`base + k * BLOCK_SIZE`），消除迭代间 RAW 依赖——每轮的指针地址只依赖循环变量 `k` 和循环外常量 `base`，编译器可自由重排 load 指令
  ```python
  # ── Before: += 引入 RAW 依赖，阻塞 MTE2 并行发射 ──
  @triton.jit
  def matmul_kernel(a_ptr, b_ptr, ...):
      # 循环外计算基址
      a_ptrs = a_ptr + (offset_m[:, None] * K + offset_k[None, :])
      b_ptrs = b_ptr + (offset_k[:, None] * N + offset_n[None, :])
      for k in range(0, tl.cdiv(K, BLOCK_K)):
          a = tl.load(a_ptrs)  # load 必须等上一轮 a_ptrs += 完成后才能发射
          b = tl.load(b_ptrs)
          acc += tl.dot(a, b)
          a_ptrs += BLOCK_K       # RAW：下轮 load 等上轮完成
          b_ptrs += BLOCK_K * N   # RAW：下轮 load 等上轮完成

  # ── After: 每轮独立计算 offset，无 RAW 依赖 ──
  @triton.jit
  def matmul_kernel(a_ptr, b_ptr, ...):
      # 循环外计算基址（不变量）
      a_ptrs_base = a_ptr + (offset_m[:, None] * K + offset_k[None, :])
      b_ptrs_base = b_ptr + (offset_k[:, None] * N + offset_n[None, :])
      for k in range(0, tl.cdiv(K, BLOCK_K)):
          # 每轮独立计算地址：base + k * stride（无迭代间依赖）
          a_ptrs = a_ptrs_base + k * BLOCK_K
          b_ptrs = b_ptrs_base + k * BLOCK_K * N
          a = tl.load(a_ptrs)  # 无 RAW，可提前发射
          b = tl.load(b_ptrs)
          acc += tl.dot(a, b)
  ```
  - **案例**：Fused MatMul fwd — `a_ptrs_base + k * BLOCK_SIZE_K` 替代循环内 `+=`
  - **案例**：RMSNorm — 行方向 offset 每轮独立计算
- 7.5 高级流水线：parallel 循环 + Preload/Cool-down
  - **问题一**：无携带依赖的循环（循环内各轮独立）未被编译器识别为可并行，多个 GM load 串行发射——编译器保守地按顺序发射各轮 load，即使它们之间无数据依赖
  - **方案一**：用 CANN extension 的 `parallel` 标记无携带依赖的循环，提示编译器并行发射各轮 load
  - **问题二**：逐块串行发射 Cube 任务，流水线深度不足——每轮只发射一个 `tl.dot`，Cube 单元在 fixpipe 期间空闲
  - **方案二**：外层循环多发射 `preload` 个 Cube 任务（预加载多块数据），最后 `preload` 次循环处理 cool-down（等待 Cube 结果回写）
  ```python
  # ── parallel 循环：并行化多个无依赖 GM load ──
  import triton.language.extra.cann.extension as extension

  # Before: 普通循环，编译器串行发射各轮 load
  for page_iter in range(PAGE_AGGREGATION_NUM):
      k_slice = tl.load(k_ptr + page_iter * stride)  # 串行 load

  # After: 标记为可并行，编译器并行发射各轮 load
  for page_iter in extension.parallel(0, PAGE_AGGREGATION_NUM):
      k_slice = tl.load(k_ptr + page_iter * stride)  # 并行 load

  # ── Preload + Cool-down：多发射 Cube 任务 ──
  preload = 3  # 预加载 3 个块
  # 外层循环多发射 preload 个 Cube 任务
  end_block = end_block + preload  # 扩展循环范围，多发射 3 个块
  for sq_loop in range(start, end_block, step):
      if sq_loop < last_loop:
          # 正常路径：发射 Cube 任务 + preload 下一个块的数据
          qk = tl.dot(q, k)  # Cube 任务
          k_next = tl.load(...)  # 预加载下一块
      else:
          # cool-down：最后 preload 次循环，等待 Cube 结果回写
          # 不发射新 Cube 任务，只处理已发射的结果
          acc = acc + qk_result
  ```
  - **案例**：FA/SWA page aggregation — parallel 循环并行化多个 GM load
  - **案例**：fa-opt/fast\_fa\_fwd — preload=3 多发射 + cool-down
