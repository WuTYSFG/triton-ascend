### 第 8 章 算子融合：消除不必要的内存往返

> **核心理念**：Memory-Bound 算子的性能瓶颈在于 GM 访问带宽。算子融合通过将多个操作合并到单个 kernel 中，让中间数据保留在寄存器/UB 中，消除 GM 写回与重新加载。
>
> **（1）GM 往返的代价。** 未融合的算子链中，每个操作的输出都要写回 GM（Global Memory，HBM），下一个操作再从 GM 读回——每次 GM 往返消耗数百到数千个时钟周期（GM 带宽约 1-2 TB/s，而 UB 带宽约 30-50 TB/s，差距 20-30 倍）。以 `RMSNorm(x) * gated` 为例，未融合时：kernel 1 计算 `gated = silu(x @ w_gate)` → 写回 GM → kernel 2 读回 `gated` → 计算 `rmsnorm(gated)` → 写回 GM。两次 GM 往返消耗大量带宽。融合后，`gated` 全程保留在寄存器/UB 中，GM 访问次数从 4 次（2 读 + 2 写）降为 2 次（1 读 + 1 写）。
>
> **（2）融合的层次。** NPU 上的算子融合从低到高有四个层次：**逐元素融合**（最简单，如 `x * sigmoid(x)` 融合为 silu，在同一 Vector 指令流中完成）→ **多 kernel 融合**（消除 kernel 启动开销和中间 GM 访问，如反向 QKV 三合一）→ **Pass 融合**（消除 scratch 中间存储，如 Two-Pass → Single-Pass，中间数据全程在寄存器）→ **计算单元升级融合**（Vector → Cube，如逐 Q head 的 Vector 乘改为按 KV head 分组的 Cube 矩阵乘）。
>
> **（3）NPU 特有的融合手段。** 除了通用融合手段，NPU 还有两种独特融合方式：**3D Block Pointer 融合**——用 3D block\_ptr `(BLOCK, Q_HEAD_NUM, V_DIM)` 一次加载所有 head 的数据，利用硬件 DMA 描述符的多维搬运能力，消除逐 head 串行加载；**布局一致性融合**——统一数据布局（如全程 D-T 布局），消除 kernel 内的 `tl.trans` 转置操作（每次转置需要一次 UB 内数据搬运 + 格式转换，开销显著）。

- 8.1 多 Kernel 融合：消除启动开销与中间 GM 访问
  - **问题**：多个独立 kernel 串行执行，每个 kernel 的输出写回 GM，下一个 kernel 再读回；更严重的是，前端计算（如反向传播中的 Q\@K^T 和 softmax 概率矩阵 P）在多个 kernel 中重复执行——bwd\_q kernel 计算 P，bwd\_kv kernel 又计算一遍 P
  - **方案**：将多个独立 kernel 融合为单 kernel，前端计算只执行一遍，中间数据全程在寄存器/UB
  - **三种典型场景**：
    - 反向 QKV 三合一：bwd\_q + bwd\_kv → bwd\_qkv（dq 用 atomic\_add 或 per-core workspace 处理多核写冲突）
    - 逐元素操作融合：GELU(x) + GELU(x)\*y → 单 kernel（消除中间 GELU(x) 的 GM 往返）
    - softmax + 累积采样：softmax + multinomial → 单 kernel（消除概率矩阵的 GM 往返）
  ```python
  # ── 反向 QKV 三合一（Flash Attention 反向）──
  # Before: 两个独立 kernel，前端计算（P 矩阵）执行两遍
  # kernel 1: bwd_q → 计算 P = softmax(Q@K^T) → dq = P @ (do - sum) @ K
  bwd_q_kernel[grid](q, k, v, do, dq, ...)  # P 在此计算
  # kernel 2: bwd_kv → 重新计算 P → dk = P^T @ do @ Q, dv = P^T @ do @ V
  bwd_kv_kernel[grid](q, k, v, do, dk, dv, ...)  # P 又计算一遍

  # After: 单 kernel，前端计算只执行一遍
  # bwd_qkv → dq, dk, dv 同时输出
  # P 只计算一次，dq/dk/dv 共享 P 矩阵
  bwd_qkv_kernel[grid](
      q, k, v, do,
      dq, dk, dv,  # 三合一输出
      ...,
      # dq 用 atomic_add 处理多核写冲突（多个 program 写同一 dq 地址）
  )
  # autotune 需配 reset_to_zero=["dq_ptr"] 避免多次评估累加
  ```
  - **案例**：FA v8 bwd — `fused bwd_qkv_kernel`，dq 用 atomic\_add
  - **案例**：GELU+Mul — 两个逐元素操作融合为单 kernel
  - **案例**：Sample — `_top_k_sample_kernel` 融合 softmax + multinomial
- 8.2 Two-Pass → Single-Pass 融合：消除 scratch 中间存储
  - **问题**：两趟计算中，Pass 1 的中间结果写回 GM（scratch buffer），Pass 2 再读回继续计算——两次 GM 访问（一次写 + 一次读），且 scratch buffer 占用额外显存。以 Gate RMSNorm 为例：Pass 1 计算 `gated = silu(x @ w_gate)` → 写回 GM scratch → Pass 2 读回 scratch → 计算 `rmsnorm(gated)`
  - **方案**：融合为单趟，中间数据（gated 值）全程在寄存器/UB，消除 scratch GM 往返和额外显存占用
  ```python
  # ── Before: 两趟，gated 写回 GM scratch ──
  # Pass 1: compute gated → write to scratch (GM)
  gated = silu(x @ w_gate)  # 前端计算
  torch.empty_like(gated).copy_(gated)  # 写回 GM scratch

  # Pass 2: read scratch → compute RMSNorm
  gated_from_gm = load_from_scratch()  # 从 GM 读回
  mean = tl.sum(gated_from_gm ** 2, axis=0) / D
  y = gated_from_gm * rsqrt(mean + eps) * weight

  # ── After: 单趟，gated 全程在寄存器/UB ──
  # gated 值不写回 GM，直接在寄存器中传给 RMSNorm 计算
  gated = silu(x @ w_gate)  # 前端计算，结果在寄存器
  # 直接对寄存器中的 gated 做 RMSNorm（无需 GM 往返）
  mean = tl.sum(gated ** 2, axis=0) / D  # gated 在寄存器，零 GM 访问
  y = gated * rsqrt(mean + eps) * weight
  ```
  - **案例**：Gate RMSNorm — 两趟融合为单趟，gated 值全程保留在寄存器
  - **案例**：RMSNorm 单 pass — 配合 `BLOCK_DIN=Din`（移除 next\_power\_of\_2）使归约循环只执行一次
- 8.3 维度融合：多 Head 一次加载 + 头循环展平
  - **问题一**：逐 head 加载数据，核内 head 循环串行执行——每个 head 独立发起 GM load，MTE2 搬运串行，且 head 循环的循环开销（索引计算、边界检查）重复执行
  - **方案一（3D Block Pointer）**：用 3D block\_ptr `(BLOCK, Q_HEAD_NUM, V_DIM)` 一次加载所有 head，利用硬件 DMA 描述符的多维搬运能力，单次 MTE2 完成所有 head 数据搬入
  - **问题二**：核内头循环串行处理各 KV head，无法利用核间并行
  - **方案二（头展平到 Grid）**：将核内头循环展平到 Grid 维度，`h = pid % num_kv_heads`，让不同核并行处理不同 head
  ```python
  # ── 3D Block Pointer：多 Head 一次加载（RoPE）──
  # Before: 逐 head 加载，核内 head 循环串行
  for head in range(num_heads):
      x = tl.load(make_block_ptr(
          base=in_ptr + begin * head * DIM + head * DIM,
          shape=(len, DIM),          # 2D：单 head
          strides=(head * DIM, 1),
          block_shape=(BLOCK_M, DIM // 2),
      ))
      # ... 旋转计算 ...

  # After: 3D block_ptr 一次加载所有 head
  x = tl.load(make_block_ptr(
      base=in_ptr + begin * head * DIM,
      shape=(len, head, DIM),        # 3D：所有 head
      strides=(head * DIM, DIM, 1),  # 三维 stride
      block_shape=(BLOCK_M, head, DIM // 2),  # 一次加载所有 head
      order=(2, 1, 0),               # 最内维 DIM 连续
  ))
  # ... 旋转计算（3D 广播）...

  # ── 头展平到 Grid：利用核间并行（KV Cache）──
  # Before: 核内串行处理各 head
  grid = (num_chunks,)
  for h in range(num_kv_heads):  # 核内串行
      process(h)

  # After: Grid 维度展开，各核并行处理不同 head
  grid = (num_chunks * num_kv_heads,)
  h = pid % num_kv_heads  # 从 pid 反推 head 索引
  process(h)
  ```
  - **案例**：FA v8 bwd\_preprocess — 3D block\_ptr 融合所有 head 的归约
  - **案例**：KV Cache — 头维度展平到 Grid
- 8.4 计算单元升级融合：Vector → Cube
  - **问题**：GQA decode 场景下（GROUP\_SIZE 个 Q head 共享一组 KV），逐 Q head 的 Vector 逐元素乘 + 归约效率低——每个 Q head 独立做 `qk = tl.sum(q[None, :] * k, axis=...)`，Vector 单元逐元素乘 + 归约，Cube 单元空闲
  - **方案**：将逐 Q head 的 Vector 乘法改为按 KV head 分组的 Cube 矩阵乘，GROUP\_SIZE 个 Q head 融合为一个矩阵乘的 M 维度——Cube 单元的 `mmad` 指令一次完成 GROUP\_SIZE 个 Q head 与 KV 的点积
  ```python
  # ── Before: Vector 逐元素乘 + 归约（GQA decode）──
  # 逐 Q head 处理，Vector 单元做逐元素乘 + sum 归约
  for q_head in range(GROUP_SIZE):
      q = tl.load(q_ptr + q_head * stride)  # 单 Q head
      # Vector 逐元素乘 + 归约（Cube 空闲）
      qk = tl.sum(q[None, :] * k, axis=head_dim)  # Vector 乘 + sum
      # ... 后续 softmax ...

  # ── After: Cube 矩阵乘，GROUP_SIZE 个 Q head 融合 ──
  # 按 KV head 分组，GROUP_SIZE 个 Q head 融合为矩阵乘的 M 维
  q_group = tl.load(q_ptr + kv_head * GROUP_SIZE * stride)  # GROUP_SIZE 个 Q head
  # Cube 矩阵乘：一次完成 GROUP_SIZE 个 Q head 与 K 的点积
  qk = tl.dot(q_group, tl.trans(k))  # Cube mmad，任务数减少 GROUP_SIZE 倍
  # ... 后续 softmax ...
  ```
  - **案例**：FA/SWA decode — GQA group-fusion + Cube grid
- 8.5 Mask 预计算融合 + Group 融合 + 布局一致性
  - **问题一**：运行时 mask\_fn 逐元素计算 mask 开销大——每个 program 在 kernel 内动态计算 causal mask、local mask、attention arg mask 的组合，占用大量 Vector 周期
  - **方案一**：预计算 mask\_tensor（主机端 CPU 计算），kernel 内直接 load 预计算结果，将 Vector 计算转为 MTE2 搬入
  - **问题二**：逐 group 循环调用 RMSNorm 有 kernel 启动开销——每个 group 独立启动一个 kernel，N 个 group 有 N 次启动开销
  - **方案二**：单 kernel 融合所有 group，通过 `head_to_group` 查找表 + `tl.where` 权重选择，在核内循环中处理不同 group 的不同 weight
  - **问题三**：kernel 内 `tl.trans` 转置操作昂贵——每次转置需要一次 UB 内数据搬运 + 格式转换
  - **方案三**：统一数据布局，全程 D-T 布局一致（输入、权重、输出都用同一布局），消除所有转置
  ```python
  # ── Mask 预计算（Flash Attention）──
  # Before: 运行时逐元素计算 mask（kernel 内 Vector 计算）
  @triton.jit
  def mask_fn(q_attn_arg, k_attn_arg, q_offset, k_offset):
      tril_causal = q_offset[:, None] >= k_offset[None, :]
      attn_args_mask = (q_attn_arg == k_attn_arg) | (k_attn_arg == 0)
      return (tril_causal & attn_args_mask) | (q_offset == k_offset)

  mask = mask_fn(q_attn_arg, k_attn_arg, offset_m, offset_n)  # Vector 计算

  # After: 直接加载预计算 mask（kernel 内 MTE2 搬入）
  # 主机端预计算 mask_tensor [batch, max_q_len, max_k_len]
  mask = tl.load(mask_block_ptr)  # MTE2 搬入，零 Vector 计算

  # ── Group RMSNorm 融合 ──
  # Before: 逐 group 循环，N 个 kernel 启动
  for group in range(num_groups):
      rmsnorm_kernel[grid](x[group], weight[group])  # 每次启动一个 kernel

  # After: 单 kernel 融合所有 group
  @triton.jit
  def group_rmsnorm_kernel(x_ptr, weight_ptr, head_to_group_ptr, ...):
      pid = tl.program_id(0)
      head = pid  # 每个 program 处理一个 head
      group = tl.load(head_to_group_ptr + head)  # 查找 head 属于哪个 group
      # 用 tl.where 选择对应 group 的 weight
      w = tl.load(weight_ptr + group * stride + offs_d)
      ...  # RMSNorm 计算

  # ── Conv1D 布局一致性 ──
  # Before: 加载后转置 × 3 次（输入、权重、输出各转置一次）
  x = tl.trans(tl.load(x_ptr + ...))   # 转置 1：输入
  w = tl.trans(tl.load(w_ptr + ...))   # 转置 2：权重
  result = tl.dot(x, w)
  out = tl.trans(result)                # 转置 3：输出
  tl.store(out_ptr + ..., out)

  # After: 全程 D-T 布局一致，无转置
  x = tl.load(x_ptr + ...)   # D-T 布局加载，不转置
  w = tl.load(w_ptr + ...)   # D-T 布局加载，不转置
  result = tl.dot(x, w)      # D-T 布局矩阵乘
  tl.store(out_ptr + ..., result)  # D-T 布局存储，不转置
  ```
  - **案例**：FA v5/v8 bwd — mask\_tensor 替代 mask\_fn
  - **案例**：GroupRMSNorm — `_group_rmsnorm_interleaved_kernel`
  - **案例**：Conv1D — 4 次 trans → 全程布局一致

### 第 9 章 Mask 计算优化：最小化边界处理开销

> **核心理念**：Mask 操作是纯开销——不产生有用的计算结果，只为保证边界正确性。Mask 优化的目标是系统性地消除不必要的 mask 计算，将必要的 mask 开销降到最低。
>
> **（1）Mask 的 NPU 代价。** 每条 mask 指令占用 Vector 执行周期——mask 的布尔比较（如 `offs < N`）、布尔运算（如 `mask1 & mask2`）、类型转换（如 `mask.to(float)`）都在 Vector 单元执行，与有用计算竞争 Vector 资源。更隐蔽的代价：乘法 mask（`(1 - mask) * -inf`）需要类型转换 + 乘法 + 减法三条指令，而 `tl.where(mask, value, -inf)` 只需一条选择指令。此外，mask 的布尔运算结果在 NPU 上是 int1 类型，参与浮点运算时需要显式类型转换，否则可能触发隐式转换的不确定行为。
>
> **（2）三个优化层次。** Mask 优化有三个层次，效率从低到高：**实现方式优化**——将乘法 mask 改为 `tl.where`，将多 mask 组合改为预计算单 mask，减少指令数量；**预计算与缓存**——将运行时 mask 计算移到主机端 CPU（kernel 外），kernel 内只需一次 MTE2 load 搬入预计算结果，将 Vector 计算转为 DMA 搬入；**冗余消除**——利用对齐信息和 constexpr 编译期常量，完全消除 mask 代码（死代码消除），这是最高效的——完全消除 mask 代码而非优化其执行。
>
> **（3）冗余 mask 的常见场景。** 当 `BLOCK_SIZE = HEAD_DIM` 时，尾维 mask（`offs_k < HEAD_DIM`）恒为 True，是死代码；当 Grid 已保证 pid 不越界时，`% M` 取模运算（防止 pid 越界）是冗余的；当 mask 已覆盖边界或 BLOCK 整除维度时，额外的边界检查 mask 是重复的。这些冗余 mask 可以通过 `tl.constexpr` 编译期常量控制，让编译器直接消除整段代码。

- 9.1 Mask 实现方式优化：tl.where 替代乘法 mask
  - **问题**：`(1 - mask) * -inf` 的乘法 mask 需要三条指令：`mask.to(float)`（bool→float 类型转换）+ `1.0 - mask_float`（减法）+ `result * 1e6`（乘法）。三条 Vector 指令只为实现一个条件选择
  - **方案**：`tl.where(mask, value, -inf)` 只需一条 Vector 选择指令——硬件直接按 mask 条件选择 value 或 -inf，无需类型转换和乘法
  ```python
  # ── Before: 乘法 mask（3 条指令）──
  @triton.jit
  def attn_kernel(q_ptr, k_ptr, ...):
      # ... 计算 qk 分数 ...
      mask = (offs_m[:, None] < Q_LEN) & (offs_n[None, :] < K_LEN)
      # 乘法 mask：类型转换 + 减法 + 乘法（3 条 Vector 指令）
      block_s = qk - (1.0 - mask.to(tl.float32)) * 1e6  # 越界位置减去大数

  # ── After: where mask（1 条指令）──
  @triton.jit
  def attn_kernel(q_ptr, k_ptr, ...):
      # ... 计算 qk 分数 ...
      mask = (offs_m[:, None] < Q_LEN) & (offs_n[None, :] < K_LEN)
      # where mask：硬件条件选择（1 条 Vector 指令）
      block_s = tl.where(mask, qk, -1e6)  # 越界位置直接置为 -inf
  ```
- 9.2 预计算 Mask + 全局缓存：从运行时计算到直接加载
  - **问题**：多个 mask load + 布尔运算在 kernel 内逐元素计算，开销大。以 SWA 为例：causal mask + local window mask + attention arg mask + query mask，4 次 mask load + 3 次布尔运算（`&` 和 `|` 组合），每次都占用 Vector 周期。更严重的是，这些 mask 在每次 forward 时都重新计算，即使 mask 参数（window\_size 等）没有变化
  - **方案一**：预计算单 mask 替代多 mask 组合——主机端将所有 mask 组合为一个预计算 mask\_tensor，kernel 内只需 1 次 MTE2 load
  - **方案二**：mask 参数不变时缓存预计算 mask 张量——按 mask 参数（如 window\_size）作为缓存键，参数不变时直接返回缓存的 mask\_tensor，避免重复预计算
  ```python
  # ── 预计算单 mask（SWA prefill）──
  # Before: 4 次 load + 3 次布尔运算（kernel 内逐元素计算）
  @triton.jit
  def swa_kernel(q_ptr, k_ptr, ...):
      # 4 次 mask load
      tril_mask = gen_mask_tril(offs_m, offs_n)       # causal 下三角
      triu_mask = gen_mask_triu(offs_m, offs_n)       # attention arg 上三角
      bound_mask = gen_mask_bound(offs_n, K_LEN)      # 边界 mask
      q_mask = gen_mask_q(offs_m, Q_LEN)              # query 边界
      # 3 次布尔运算组合
      mask = tril_mask & (triu_mask | bound_mask) & q_mask  # Vector 计算

  # After: 1 次 load（主机端预计算单 mask）
  @triton.jit
  def swa_kernel(q_ptr, k_ptr, mask_ptr, ...):
      # 1 次 MTE2 load，零 Vector 计算
      mask = tl.load(mask_ptr + offsets)  # 预计算 mask 直接加载

  # ── 全局缓存（SWA prefill）──
  # Before: 每次调用重算预计算 mask
  def swa_forward(q, k, window_size, ...):
      mask = precompute_mask(window_size, ...)  # 每次 forward 都重算
      swa_kernel[grid](q, k, mask, ...)

  # After: mask 参数不变时缓存命中
  _GLOBAL_WINDOW_SIZE = None
  _CACHED_MASK = None

  def swa_forward(q, k, window_size, ...):
      global _GLOBAL_WINDOW_SIZE, _CACHED_MASK
      if _GLOBAL_WINDOW_SIZE == window_size:
          mask = _CACHED_MASK  # 缓存命中，零计算
      else:
          mask = precompute_mask(window_size, ...)
          _CACHED_MASK = mask
          _GLOBAL_WINDOW_SIZE = window_size
      swa_kernel[grid](q, k, mask, ...)
  ```
  - **案例**：SWA prefill — `gen_mask_causal_with_window` JIT + `_GLOBAL_WINDOW_SIZE` 缓存
- 9.3 冗余 Mask 消除：利用对齐信息删除死代码
  - **问题**：当 BLOCK\_SIZE = HEAD\_DIM、BLOCK 整除维度、或 mask 已覆盖边界时，额外 mask 检查是死代码——条件恒为 True，但运行时仍会生成 mask 指令并执行
  - **四种消除场景**：
    - 尾维 mask：当 `BLOCK_SIZE = HEAD_DIM` 时，`offs_k < HEAD_DIM` 恒为 True（因为 `offs_k = arange(0, HEAD_DIM)`），移除 `& (offs_k < HEAD_DIM)`
    - masked\_out\_rows 安全保护：非首块路径中 m\_i 已有有效值（上一轮的 m\_new），不存在全 -inf 行，`masked_out_rows` 检查是冗余的
    - KV\_LEN / IS\_FULL\_BLOCKS 条件 mask：当 mask 已覆盖边界或 BLOCK 整除维度时，额外的 KV\_LEN 边界检查重复
    - `% M / % N` 取模运算：Grid 已保证 pid 不越界（`pid < NUM_BLOCKS_M`），取模 `% M` 是冗余的，且 `%` 运算在 NPU 上导致标量退化（性能下降 5-10 倍）
  ```python
  # ── 尾维 mask 消除（FlexAttention）──
  # Before: 双维检查（即使 BLOCK_K == HEAD_DIM，尾维检查仍执行）
  mask = (offs_m[:, None] < Q_LEN) & (offs_k[None, :] < HEAD_DIM)
  # 当 BLOCK_K == HEAD_DIM 时，offs_k = arange(0, HEAD_DIM) < HEAD_DIM 恒为 True

  # After: HEAD_DIM = BLOCK_SIZE，尾维检查冗余，移除
  mask = (offs_m[:, None] < Q_LEN)  # 只保留 M 维边界检查

  # ── masked_out_rows 消除（FlexAttention online softmax）──
  # Before: 安全保护（3 条指令），防止全 -inf 行导致 NaN
  masked_out_rows = (m_ij == float('-inf'))  # 检查是否全 -inf
  m_ij_masked = tl.where(masked_out_rows, 0, m_ij)  # 全 -inf 时置 0
  alpha = tl.exp(m_i - m_ij_masked)  # 安全的 exp

  # After: 非首块路径中 m_i 已有有效值，直接使用（消除 3 条指令）
  alpha = tl.exp(m_i - m_ij)  # 无需安全保护

  # ── 取模运算消除（SwiGLU fwd）──
  # Before: 取模防止越界（且 % 导致标量退化）
  offset = (pid * BLOCK + tl.arange(0, BLOCK)) % M  # % 运算，标量退化

  # After: Grid 已保证 pid 不越界，取模冗余
  offset = pid * BLOCK + tl.arange(0, BLOCK)  # 连续地址，向量化
  ```
  - **案例**：FlexAttention — 移除尾维 mask、masked\_out\_rows、KV\_LEN 边界保护（6 处）
  - **案例**：SwiGLU fwd — offset 移除取模
- 9.4 编译期 Mask 消除：constexpr 控制代码生成
  - **问题**：当列数/token 数对齐时（如 `n_cols % BLOCK_SIZE == 0`），mask 代码是死代码（条件恒为 True），但运行时条件判断仍会生成 mask 指令——编译器无法在运行时判断条件是否恒为 True，保守地生成 mask 代码
  - **方案**：用编译期常量（`tl.constexpr`）控制是否生成 mask 代码——将"是否对齐"的判断从运行时提升到编译时，编译器在编译期直接消除整段死代码
  ```python
  # ── Before: 总是生成 mask（即使列已对齐）──
  @triton.jit
  def gelu_fwd_kernel(x_ptr, y_ptr, n_cols, ...,
                      BLOCK_SIZE: tl.constexpr):
      pid = tl.program_id(0)
      offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
      # 总是生成 mask，即使 n_cols % BLOCK_SIZE == 0（mask 恒为 True）
      mask = (offs < n_cols)
      x = tl.load(x_ptr + offs, mask=mask, other=0.0)
      y = x * 0.5 * (1.0 + tl.tanh(0.7978 * (x + 0.0447 * x * x * x)))
      tl.store(y_ptr + offs, y, mask=mask)

  # ── After: constexpr 控制，编译期消除 mask ──
  @triton.jit
  def gelu_fwd_kernel(x_ptr, y_ptr, n_cols, ...,
                      BLOCK_SIZE: tl.constexpr,
                      COL_ALIGNED: tl.constexpr):  # 编译期常量
      pid = tl.program_id(0)
      offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
      if not COL_ALIGNED:
          # 仅不对齐时生成 mask 代码（编译期决定）
          mask = (offs < n_cols)
          x = tl.load(x_ptr + offs, mask=mask, other=0.0)
          y = x * 0.5 * (1.0 + tl.tanh(0.7978 * (x + 0.0447 * x * x * x)))
          tl.store(y_ptr + offs, y, mask=mask)
      else:
          # 对齐时无 mask（编译器消除 if not 分支）
          x = tl.load(x_ptr + offs)  # 无 mask
          y = x * 0.5 * (1.0 + tl.tanh(0.7978 * (x + 0.0447 * x * x * x)))
          tl.store(y_ptr + offs, y)  # 无 mask

  # 调用处：编译期传入 COL_ALIGNED
  COL_ALIGNED = (n_cols % BLOCK_SIZE == 0)  # 编译期计算
  gelu_fwd_kernel[grid](x, y, n_cols, COL_ALIGNED=COL_ALIGNED, ...)
  ```
  - **案例**：GELU — `COL_ALIGNED` constexpr 消除 mask
  - **案例**：VisionRoPE — `TOKEN_ALIGNED` constexpr 消除 mask

### 第 10 章 内存访问与数据通路：优化数据搬运

> **核心理念**：NPU 的数据搬运由专用 DMA 引擎执行，与计算单元物理分离。内存访问优化的核心是：减少搬运次数、提升搬运效率、利用硬件数据通路特性。
>
> **（1）连续地址流依赖。** NPU 的 DMA 引擎（MTE2/MTE3）依赖**连续地址流**——DMA 描述符以固定 stride 发起连续搬运，离散访存（stride 最低维 ≠ 1，即非连续地址）性能急剧下降。这与 GPU 的 Warp 合并访问（coalesced access）模式不同：GPU 上 32 线程的 Warp 可以合并访问分散地址（只要地址在同一 cache line），NPU 的 DMA 引擎则需要严格的连续地址。离散访存时 DMA 引擎退化为逐元素搬运，带宽利用率可能降低数倍。
>
> **（2）Block Pointer 的硬件优势。** `tl.make_block_ptr` 利用硬件 DMA 描述符自动处理边界——DMA 引擎根据 block\_ptr 的 shape/strides/block\_shape 自动计算搬运计划，越界位置自动填充 0（无需手动 mask）。这比手动指针算术 + mask 更高效：手动方式需要 Vector 单元计算 offset 和 mask（占用 Vector 周期），block\_ptr 方式由 DMA 描述符硬件完成（零 Vector 开销）。但注意：NPU 上 block\_ptr 的 **stride 必须反映真实内存布局**，不能通过交换 stride 实现转置（GPU 上可以），转置语义只能通过 **order 参数**表达。
>
> **（3）`%`** **取模的标量退化。** NPU 上 `%` 取模运算通过标量 ALU 执行，导致向量运算退化为逐元素标量计算——地址不连续，DMA 引擎无法发起连续搬运，性能可能下降 **5-10 倍**。应尽量用 mask 替代取模：mask 是向量化的比较操作（保持连续地址），取模是标量运算（破坏连续地址）。
>
> **（4）CANN extension 切片指令。** CANN extension 的 `extract_slice` / `insert_slice` 直接映射 NPU DMA 切片指令，无通用版本的范围检查开销——通用 Triton 的 slice 操作在运行时检查 offset 和 shape 是否越界（占用 Vector 周期），CANN 扩展版本跳过检查直接发起 DMA 切片。
>
> **（5）care\_padding 依赖跳过。** `care_padding=False` 跳过 padding 区域的依赖检查——默认情况下，编译器检查 load 的 padding 区域（BLOCK\_SIZE > 实际维度的部分）是否与其他操作有依赖，阻止 load 指令并行发射。设置 `care_padding=False` 后，编译器跳过此检查（通过 mask 保证正确性），允许 load 指令提前并行发射。

- 10.1 预计算缓存：从运行时计算到直接加载
  - **问题**：kernel 内的 sin/cos 等超越函数计算、每次 forward 的权重转置都是昂贵的重复计算。以 RoPE 为例：每个 program 在 kernel 内计算 `freqs = pos * inv_freq; sin = tl.sin(freqs); cos = tl.cos(freqs)`，`tl.sin`/`tl.cos` 是昂贵的超越函数（多条 Vector 指令），且每次 forward 都重复计算相同的 sin/cos 值
  - **方案一（Sin/Cos 预计算）**：CPU 预计算 sin/cos 表后传入 kernel 直接 load，将超越函数计算转为 MTE2 搬入
  - **方案二（权重转置缓存）**：按 `data_ptr()` 键缓存转置结果，消除重复 O(N·K) 转置——每次 forward 都做 `weight.t().contiguous()` 是 O(N·K) 的数据搬运，但权重内容不变时转置结果可缓存复用
  ```python
  # ── Sin/Cos 预计算（RoPE）──
  # Before: kernel 内计算超越函数（多条 Vector 指令）
  @triton.jit
  def rope_kernel(in_ptr, pos_ptr, out_ptr, ...):
      pos = tl.load(pos_ptr + offsets)
      freqs = pos[:, None] * inv_freq[None, :]  # 计算频率
      sin = tl.sin(freqs)  # 昂贵的超越函数（多条 Vector 指令）
      cos = tl.cos(freqs)  # 昂贵的超越函数
      x1 = x0 * cos - y0 * sin  # 旋转
      ...

  # After: 直接 load 预计算结果（1 次 MTE2 load）
  @triton.jit
  def rope_kernel(in_ptr, pos_ptr, sin_cache_ptr, cos_cache_ptr, out_ptr, ...):
      # 直接加载预计算 sin/cos（零超越函数计算）
      sin = tl.load(sin_cache_ptr + offsets)  # MTE2 搬入
      cos = tl.load(cos_cache_ptr + offsets)  # MTE2 搬入
      x1 = x0 * cos - y0 * sin  # 旋转
      ...

  # 主机端预计算
  def rope_forward(x, position, ...):
      sin_cache, cos_cache = precompute_sin_cos(position, base, DIM)  # CPU 计算
      rope_kernel[grid](x, position, sin_cache, cos_cache, out, ...)

  # ── 权重转置缓存（GEMM）──
  # Before: 每次 forward 重算转置（O(N·K) 数据搬运）
  def gemm_forward(x, weight, ...):
      weight_t = weight.t().contiguous()  # 每次 forward 都转置
      gemm_kernel[grid](x, weight_t, ...)

  # After: 按 data_ptr() 缓存，权重不变时复用
  _bt_cache = {}  # 全局缓存字典

  def gemm_forward(x, weight, ...):
      key = weight.data_ptr()  # 用张量内存地址作为缓存键
      if key not in _bt_cache:
          _bt_cache[key] = weight.t().contiguous()  # 首次计算并缓存
      weight_t = _bt_cache[key]  # 后续直接复用
      gemm_kernel[grid](x, weight_t, ...)
  ```
  - **案例**：RoPE — sin/cos 预计算缓存
  - **案例**：GEMM — `_bt_cache` 消除重复转置
- 10.2 数据聚合与 CANN 切片：提升 Cube 利用率与切片效率
  - **问题一**：小 page 场景下（PagedAttention 的 page\_size 较小，如 16/32），逐页 matmul 的 Cube 利用率低——每个 page 独立做 `tl.dot(q, k_page)`，Cube 单元的矩阵乘规模太小（M=BLOCK\_M, N=page\_size），Cube 利用率不足
  - **方案一（L1 Cache 聚合）**：多个页用 `insert_slice` 聚合为单个大 tile 后再做 matmul——PAGE\_AGG\_NUM 个 page 聚合为 `(PAGE_AGG_NUM * BLOCK_N, D)` 的大矩阵，Cube 一次处理大矩阵，利用率大幅提升
  - **问题二**：通用 Triton 的 slice 操作有范围检查开销——`tl.extract_slice` 在运行时检查 offset 和 shape 是否越界，占用 Vector 周期
  - **方案二（CANN extension）**：用 CANN 扩展版本替代通用 slice，直接映射 DMA 切片指令，跳过范围检查
  ```python
  # ── L1 Cache 聚合（FA/SWA page aggregation）──
  # Before: 逐页加载 + 小 matmul，Cube 利用率低
  @triton.jit
  def fa_kernel(q_ptr, k_ptr, ...):
      for kv_block in range(num_kv_blocks):
          k = tl.load(k_ptr + kv_block * page_stride)  # 单页加载
          qk = tl.dot(q, tl.trans(k))  # 小 matmul，Cube 利用率低
          # ... softmax ...

  # After: 聚合为大 tile + 大 matmul，Cube 利用率高
  @triton.jit
  def fa_kernel(q_ptr, k_ptr, ..., PAGE_AGG_NUM: tl.constexpr):
      PAGE_AGG = PAGE_AGG_NUM  # 聚合页数
      for kv_block in range(0, num_kv_blocks, PAGE_AGG):
          # 预分配大 tile
          k = tl.zeros((PAGE_AGG * BLOCK_N, D), dtype=tl.float16)
          for i in range(PAGE_AGG):
              # 加载单页并插入大 tile 的对应位置
              k_slice = tl.load(k_ptr + (kv_block + i) * page_stride)
              k = insert_slice(k, k_slice, offset=i * BLOCK_N, axis=0)
          # 大 matmul，Cube 利用率高
          qk = tl.dot(q, tl.trans(k))  # 矩阵规模增大 PAGE_AGG 倍
          # ... softmax ...

  # ── CANN extension slice（RoPE / Conv1D / GroupGEMM / VisionRoPE）──
  # Before: 通用版本（运行时范围检查）
  slice_result = tl.extract_slice(tensor, offsets, shape, strides)
  # 运行时检查 offsets + shape 是否越界（Vector 开销）

  # After: CANN 扩展（直接映射 DMA 指令，跳过检查）
  import triton.language.extra.cann.extension as extension
  slice_result = extension.extract_slice(tensor, offsets, shape, strides)
  # 直接发起 DMA 切片指令，零范围检查开销
  ```
  - **案例**：FA/SWA prefill — 小 page 聚合提升 Cube 利用率
  - **案例**：RoPE / Conv1D / GroupGEMM / VisionRoPE — CANN extension slice
- 10.3 访存并行优化：care\_padding + Load 重排序
  - **问题一**：padding 区域的依赖检查阻止 load 指令并行发射——当 BLOCK\_SIZE > 实际维度时，load 的 padding 部分（越界位置）被编译器检查是否与其他操作有依赖，此检查阻止 load 提前发射
  - **方案一**：`care_padding=False` 跳过 padding 区域依赖检查（通过 mask 保证正确性），允许 load 指令提前并行发射
  - **问题二**：无依赖 load 排在依赖 load 之后，串行执行——编译器不重排 load 顺序，如果排在前的 load 因等待上一轮 store 而阻塞，排在后的无依赖 load 也无法提前发射
  - **方案二**：将无依赖 load 提前到有依赖 load 之前，使无依赖 load 与上一轮 store 并行执行
  ```python
  # ── care_padding（KV Cache）──
  # Before: 默认检查 padding，load 被阻塞
  @triton.jit
  def kv_cache_kernel(k_ptr, ...):
      # padding 区域依赖检查阻止 load 并行发射
      k = tl.load(k_ptr + offsets, mask=mask)  # 默认 care_padding=True

  # After: 跳过 padding 检查，load 提前并行发射
  @triton.jit
  def kv_cache_kernel(k_ptr, ...):
      # 跳过 padding 依赖检查（mask 保证正确性）
      k = tl.load(k_ptr + offsets, mask=mask, care_padding=False)  # 允许并行发射

  # ── Load 重排序（Quant/RoPE）──
  # Before: 依赖 load 先执行，无依赖 load 后执行（串行）
  @triton.jit
  def kernel(x_ptr, scale_ptr, ...):
      for i in range(num_iters):
          # x 的 load 依赖上一轮 store（同一地址），阻塞
          x = tl.load(x_ptr + offsets)
          # scale 的 load 无依赖，但必须等 x 的 load 完成后才能发射
          scale = tl.load(scale_ptr + offsets)
          # ... compute ...
          tl.store(x_ptr + offsets, result)  # 下一轮 x load 依赖此 store

  # After: 无依赖 load 先发射，两个 load 并行
  @triton.jit
  def kernel(x_ptr, scale_ptr, ...):
      for i in range(num_iters):
          # scale 的 load 无依赖，先发射（与上一轮 store 并行）
          scale = tl.load(scale_ptr + offsets)
          # x 的 load 依赖上一轮 store，后发射
          x = tl.load(x_ptr + offsets)
          # ... compute ...
          tl.store(x_ptr + offsets, result)
  ```
  - **案例**：KV Cache — `care_padding=False` 提升数据加载并行度
  - **案例**：Quant/RoPE — Load 重排序隐藏内存延迟
- 10.4 Block Pointer + 内存分配优化
  - **问题一**：手动指针算术 + mask 比 Block Pointer 效率低——手动方式需要 Vector 单元计算 `offsets = row[:, None] * stride + col[None, :]` 和 `mask = (row < M) & (col < N)`，占用 Vector 周期；Block Pointer 由 DMA 描述符硬件自动处理边界，零 Vector 开销
  - **方案一**：用 `tl.make_block_ptr` 替代手动指针算术，利用硬件 DMA 描述符自动处理边界
  - **问题二**：`torch.clone()` 拷贝整个张量数据，配合原地 kernel 引入 RAW 冒险——`clone` 分配新内存并拷贝数据（O(N) 数据搬运），原地 kernel 读写同一地址引入 Read-After-Write 冒险
  - **方案二**：用 `torch.empty_like()`（仅分配内存，无拷贝）替代 `clone`（分配 + 拷贝），配合非原地 kernel（读写不同地址）
  ```python
  # ── Block Pointer 替代手动指针算术 ──
  # Before: 手动指针算术 + mask（Vector 计算 offset 和 mask）
  @triton.jit
  def kernel(ptr, M, N, stride, ...,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
      pid_m = tl.program_id(0)
      pid_n = tl.program_id(1)
      row = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
      col = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
      # 手动计算 offset（Vector 计算）
      offsets = row[:, None] * stride + col[None, :]
      # 手动计算 mask（Vector 计算）
      mask = (row[:, None] < M) & (col[None, :] < N)
      x = tl.load(ptr + offsets, mask=mask)  # 手动 load

  # After: block_ptr 自动处理边界（DMA 描述符硬件完成）
  @triton.jit
  def kernel(ptr, M, N, stride, ...,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
      pid_m = tl.program_id(0)
      pid_n = tl.program_id(1)
      # block_ptr：DMA 描述符自动处理边界，零 Vector 开销
      block_ptr = tl.make_block_ptr(
          base=ptr,
          shape=(M, N),
          strides=(stride, 1),
          offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
          block_shape=(BLOCK_M, BLOCK_N),
          order=(1, 0),
      )
      x = tl.load(block_ptr)  # 自动 load，越界位置填充 0

  # ── empty_like 替代 clone（RoPE）──
  # Before: 分配 + 拷贝（O(N) 数据搬运 + 原地 RAW 冒险）
  def rope_forward(q, ...):
      q_out = torch.clone(q)  # 分配新内存 + 拷贝 q 的所有数据
      rope_kernel[grid](q_out, ...)  # 原地修改 q_out（读写同一地址）

  # After: 仅分配（无拷贝，非原地无 RAW）
  def rope_forward(q, ...):
      q_out = torch.empty_like(q)  # 仅分配内存，不拷贝数据
      rope_kernel[grid](q, q_out, ...)  # 非原地：读 q，写 q_out（不同地址）
  ```
  - **案例**：VisionRoPE — 3D block\_ptr for cos/sin/q/k
  - **案例**：RoPE — `empty_like` 节省一个 q 张量大小的显存峰值
- 10.5 循环不变量外提：Bias 提前加载 + Cast 位置提前
  - **问题一**：bias 在 K 循环内每轮重复加载——bias 是循环不变量（不随 K 变化），但在循环内每轮 `tl.load(bias_ptr + offsets)`，重复发起 MTE2 搬运相同数据
  - **方案一**：bias 在循环外提前加载一次 + `broadcast_to` 显式广播，循环内直接复用
  - **问题二**：`.to(dtype)` 在 `tl.trans` 之后，cast 结果无法被多处复用——`p.to(dtype)` 在 `tl.trans(p)` 之后，每次需要转置后的 p 时都要重新 cast
  - **方案二**：将 cast 提前到 transpose 之前，cast 结果可被 dv 和 dk 两处复用（`tl.trans` 只是布局变换，不改变数据）
  ```python
  # ── Bias 提前加载（Fused MatMul fwd）──
  # Before: 每轮重复加载 bias（循环不变量在循环内）
  @triton.jit
  def matmul_kernel(a_ptr, b_ptr, bias_ptr, c_ptr, ...):
      bias = tl.load(bias_ptr + offsets)  # 循环外？不，在循环内
      for k in range(0, K, BLOCK_K):
          bias = tl.load(bias_ptr + offsets)  # 每轮重复加载（浪费）
          a = tl.load(a_ptr + ...)
          b = tl.load(b_ptr + ...)
          acc += tl.dot(a, b) + bias[None, :]  # bias 每轮加一次

  # After: 循环外加载一次 + 显式广播
  @triton.jit
  def matmul_kernel(a_ptr, b_ptr, bias_ptr, c_ptr, ...):
      # 循环外加载一次
      bias = tl.load(bias_ptr + offsets)
      bias = tl.broadcast_to(bias[None, :])  # 显式广播到矩阵维度
      for k in range(0, K, BLOCK_K):
          a = tl.load(a_ptr + ...)
          b = tl.load(b_ptr + ...)
          acc += tl.dot(a, b)  # 循环内不加 bias
      acc += bias  # 循环外加一次

  # ── Cast Hoisting（FA v5 bwd_kv / FA v8 bwd_q）──
  # Before: trans 后 cast，重复转换
  # p 需要转置后用于 dv 和 dk 两处，每处都重新 cast
  dv = tl.dot(tl.trans(p).to(dtype), do)       # trans 后 cast（第 1 次）
  dk = tl.dot(tl.trans(p).to(dtype), do_other)  # trans 后 cast（第 2 次，重复）

  # After: trans 前 cast 一次，复用结果
  p = p.to(dtype)                       # trans 前 cast（只 1 次）
  dv = tl.dot(tl.trans(p), do)          # 复用 cast 后的 p
  dk = tl.dot(tl.trans(p), do_other)    # 复用 cast 后的 p
  ```
  - **案例**：Fused MatMul fwd — bias 循环外提前加载
  - **案例**：FA v5 bwd\_kv / FA v8 bwd\_q — p 的 cast 提前复用
- 10.6 指令简化：Mask 类型转换 + tl.full 初始化
  - **问题一**：布尔 mask 的隐式类型推断导致不确定行为——`mask = (q_offset <= k_offset)` 产生 bool 类型，参与浮点运算时（如 `block_s = tl.where(mask, block_s, -inf)`）可能触发隐式转换的不确定行为
  - **方案一**：对布尔 mask 进行显式类型转换 `.to(tl.int32)`，确保类型安全
  - **问题二**：`tl.zeros([M]) - float("inf")` 需要先创建零张量再减（2 条指令：`tl.zeros` 创建零张量 + 减法），用于初始化 online softmax 的最大值数组 `m_i`
  - **方案二**：`tl.full([M], float("-inf"))` 直接填充目标值（1 条指令），消除减法操作
  ```python
  # ── Mask 类型转换（FA v4 fwd）──
  # Before: 隐式 bool（可能触发不确定行为）
  @triton.jit
  def fa_kernel(...):
      q_offset = tl.arange(0, BLOCK_M) * stride
      k_offset = tl.arange(0, BLOCK_N) * stride
      # 隐式 bool 类型，参与 where 时可能不确定
      mask = (q_offset <= k_offset)
      block_s = tl.where(mask, block_s, -1e6)

  # After: 显式 int32（类型安全）
  @triton.jit
  def fa_kernel(...):
      q_offset = tl.arange(0, BLOCK_M) * stride
      k_offset = tl.arange(0, BLOCK_N) * stride
      # 显式转换为 int32，确保类型安全
      mask = (q_offset <= k_offset).to(tl.int32)
      block_s = tl.where(mask, block_s, -1e6)

  # ── tl.full 替代 zeros - inf（FlexAttention）──
  # Before: 2 条指令（创建零张量 + 减法）
  @triton.jit
  def fa_kernel(...):
      # 先创建零张量，再减去 inf（2 条 Vector 指令）
      m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

  # After: 1 条指令（直接填充 -inf）
  @triton.jit
  def fa_kernel(...):
      # 直接填充目标值（1 条 Vector 指令）
      m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
  ```
  - **案例**：FA v4 fwd — mask\_fn 显式类型转换
  - **案例**：FlexAttention — `m_i` 初始化优化

***

### 第 11 章 编译器图优化：固定模式匹配的自动优化

> **核心理念**：Triton-Ascend 的图优化 pass 在 IR 图上做固定模式匹配（pattern match），命中特定子图后自动重写为更优的等价形式。开发者无需（也无法）手写这些变换，但理解匹配的模式与触发条件，可以预判编译器行为、定位"为什么没被优化"，并知道何时需要显式干预。

> **背景知识**：图优化 pass 针对 IR 图中的固定子图形态做自动重写，主要覆盖五类模式——**融合类**（逐元素链、dot 后处理、向量函数打包）、**循环结构类**（多缓冲标记）、**布局类**（NZ↔ND 转换）、**访存地址与布局类**（取模线性化、读写置换、store 合并、跨步改写）、**程序映射与 Grid 变换类**（独立轴张量化、持久化任务挖掘、行合并）。每个 pass 只认自己关心的子图结构，未命中则原样保留。

- 11.1 匹配机制概述：IR 图 → pass 扫描固定子图 → 重写
  - **匹配对象**：pass 扫描特定子图结构，常见有三类触发条件——**结构模式**（op 类型与连接关系，如 `mulf + addf`、循环内 `load/store` 对）、**布局/属性约束**（如 `ascend.dot` 的 `fractal_a/b/c`、block 尺寸 16×16）、**循环结构约束**（候选必须位于 `scf.for`/`scf.while` 内且支持每迭代轮转槽位）
  - **重写结果**：命中后替换为更优等价形式（融合、布局转换、多缓冲标记等），未命中则原样保留
  - **与手写代码的关系**：绝大多数自动生效；少数场景需要提供 hint 或开关配合（见 11.8）
- 11.2 自动融合类 pattern：逐元素链合并、dot 后处理与 fixpipe 随路、VF 打包
  - **逐元素链合并**：`x * sigmoid(x)` 等连续逐元素链被合并为单指令（silu），`mul + add` 折叠为 fma，连续 cast 折叠
  - **dot 后处理随路**：`a*b + c` 或带 bias/精度转换的结果不再单独做一次 Vector 运算——编译器把 bias 累加、fp32→fp16、转置融进 **fixpipe** 从 L0C 搬出时的随路操作（按下游消费方式选择 NZ2ND / NZ2DN / NZ2NZ 模式），这是 Cube 结果喂给 Vector 前最常见的自动融合
  - **VF 打包**：连续向量运算链被 outline 成独立向量函数（`--vf-fusion-mode`，默认 max-parallel），减少指令间同步；代价是链过长时 UB 溢出或吞掉 dot 的独立调度（回指第 6 章）
- 11.3 循环结构 pattern：多缓冲标记
  - **多缓冲标记**：循环内 `load/store` 对命中多缓冲模式，在分配点打 `annotation.mark {hivm.multi_buffer = N}` 标记，后续经内存规划分配 N 份地址、按迭代计数取模选择槽位（回指第 7 章）。本地 Buffer 的 N 固定为 2，GM Workspace 的 N 由 `--set-workspace-multibuffer` 指定
  - **硬约束**：候选 Buffer 从分配点到最外层祖先循环必须都是 `scf.for`/`scf.while`，位于 `scf.parallel`/`scf.forall` 内的 Buffer 不会被标记
- 11.4 布局 pattern：NZ↔ND 转换与消除
  - **分形布局推断**：`ascend.dot` 的 `fractal_a/b/c` 属性触发布局推断（如 DOTA\_ND\_TO\_zN），编译器在 GM 的 ND 与 Cube 的 NZ（`[N1, M1, 16, 16]` 分形块）之间自动插入 `convert_layout`
  - **fixpipe 收尾转换**：Cube 结果在 L0C 为 NZ，需 fixpipe 搬运到 UB 并转回 ND，Vector 才能逐元素处理（回指第 10 章）
  - **转换消除**：连续的 `convert_layout` 被合并/消除；布局一致的数据流可省掉整段转换
- 11.5 访存地址与布局优化 pattern：把离散、低效的内存访问改写为整块连续访问
  - **共同目标**：针对**地址表达式本身**（取模、跨步、维度顺序）做改写，不动数据布局（与 11.4 的 convert\_layout 类 pattern 互补），让后端把离散访问识别为整块连续内存、一次 DMA 搬完，同时消除标量地址计算。下文每个 pattern 均给出触发模式、改写、收益与约束
- 11.5.1 ConvertModuloToMask：取模线性化
  - **触发模式**：kernel 写 `offs % n` 让越界下标绕回数组开头以避免非法访问，但写回处已有 `offs < n` 掩码丢弃越界数据——读进来的绕回数据本就无用
  - **DSL 写法（触发示例）**：load 用取模绕回地址，store 带 `offs < n` 掩码
  ```python
  @triton.jit
  def kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
      pid = tl.program_id(0)
      offs = pid * BLOCK + tl.arange(0, BLOCK)
      wrap = offs % n                  # 越界下标绕回数组开头
      mask = offs < n                  # store 的守卫掩码
      v = tl.load(x_ptr + wrap)        # load 无掩码（读绕回数据）
      tl.store(y_ptr + offs, v, mask=mask)
  ```
  - **改写**：去掉取模让地址恢复连续，同时给 load 补上同样的掩码（越界位置读 0）
  ```ttir
  # Before：offs % n 把连续地址打乱成跳跃的，load 无掩码
  %wrap = arith.remsi %offs, %n       # offs % n
  %addr = arith.addi %kb, %wrap
  %v    = tt.load %wa                 # load 无掩码
  tt.store %oa, %v, %gb               # store 已带 offs < n 掩码

  # After：地址恢复线性，load 补掩码 + 零填充
  %cmp  = arith.cmpi slt, %offs, %n   # offs < n
  %addr = arith.addi %kb, %offs       # 地址线性化
  %v    = tt.load %wa, %cmp, %cst     # 补掩码，越界读 0
  tt.store %oa, %v, %cmp
  ```
  - **收益**：地址线性化后后端可识别为整块连续内存，用一次 DMA 搬完，而非因取模被迫逐元素离散访问
  - **约束**：除数必须是运行时变量（编译期常量交由 TritonToStructured 处理更优，它保留绕回语义并改写为跨步访问，完全等价）；下标必须是一维 i32、无 layout，长度等于 make\_range 长度；必须存在 store 上的 offs < n 守卫作为依据，因此归约轴上的取模无法处理（归约轴无输出坐标，不会有这种掩码）；取模结果只能流向 load 地址，不能流向 other、tt.trans、tt.reshape 或其他计算；expand\_dims 的轴必须唯一；不支持跨嵌套 scf.for 的多条循环携带地址
- 11.5.2 LoadStoreTranspose：读写维度置换
  - **触发模式**：分析 load/store 的地址表达式，判断张量各维的实际内存步长。若最内维（代码中变化最快的那维）在内存里是跳着走的，则触发
  - **DSL 写法（触发示例）**：tile 最内维（cols）在内存中跳着走——列主序访问
  ```python
  @triton.jit
  def kernel(x_ptr, y_ptr, M: tl.constexpr, N: tl.constexpr):
      rows = tl.arange(0, M)           # 32 行
      cols = tl.arange(0, N)           # 16 列
      # 元素 (r, c) 的地址 = r + c * M：内维步长 M，读一个元素跳 M 个元素
      offs = rows[:, None] + cols[None, :] * M
      v = tl.load(x_ptr + offs)        # [32, 16]
      tl.store(y_ptr + offs, v + 1.0)
  ```
  - **改写**：把整个"读→算→写"闭包的维度顺序调换，使最内维变成内存中真正连续的那维；同步改写掩码、assert、循环携带指针，以及循环之后 reduce 的轴号
  ```ttir
  # Before：最内维 16 列每列相隔 32 个元素，读一个元素跳 128 字节
  %off = arith.addi %mb, %nb          # 32×16 偏移
  %sa  = tt.addptr %sp, %off
  %x   = tt.load %sa : tensor<32x16x!tt.ptr>
  %y   = arith.addf %x, %one
  tt.store %da, %y

  # After：调换维度顺序，最内维 32 个元素紧邻，一次连续搬 128 字节
  %off = arith.addi %mb, %nb          # 16×32 偏移
  %sa  = tt.addptr %sp, %off
  %x   = tt.load %sa : tensor<16x32x!tt.ptr>
  %y   = arith.addf %x, %cst
  tt.store %da, %y
  ```
  - **收益**：改写前每读一个元素跳 128 字节、16 个元素是 16 次离散访问；改写后最内维 32 个元素紧邻，一次连续搬 128 字节
  - **约束**：指针必须是 addptr(splat(标量指针), 偏移) 形式，维度 ≥2、静态 shape、无 layout，不支持 block pointer；每一维须恰好由一个 make\_range 驱动，且各维步长大小关系能在结构上证明（运行时步长需掩码提供取值范围）；必须同时有 load 和 store 且存在 load→store 数据通路，所有访存点置换一致；不支持 volatile、padding、boundary\_check；load 的 other 必须是均匀常量；闭包内只允许逐元素算子、splat、broadcast、expand\_dims、addptr、trans、常量，不支持矩阵乘和 reshape；循环结果只能被单输入 tt.reduce 消费；不跨 scf.if/scf.while 区域边界；遇到用户自己写的张量 assert 直接放弃
- 11.5.3 StoreCoalescing：相邻 store 合并
  - **触发模式**：函数内有几次 store 写入同一块内存的相邻区段（前一次结尾正好接下一次开头）
  - **DSL 写法（触发示例）**：对同一基址的相邻区段分多次 store，首尾严格相接
  ```python
  @triton.jit
  def kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):
      offs0 = tl.arange(0, BLOCK)
      offs1 = tl.arange(0, BLOCK) + BLOCK
      v0 = tl.load(x_ptr + offs0)
      v1 = tl.load(x_ptr + offs1)
      tl.store(y_ptr + offs0, v0)      # [0, BLOCK)
      tl.store(y_ptr + offs1, v1)      # [BLOCK, 2*BLOCK)，紧邻上一段
  ```
  - **改写**：先在寄存器中把数据拼成一个大张量，最后一次性写出
  ```ttir
  # Before：两段相邻 store 分两次写
  tt.store %base, %v0                 # [0, 64)
  tt.store %base + 64, %v1            # [64, 128)

  # After：寄存器拼成 128 元素大张量，一次写出
  %t = tensor.insert_slice %v0 into %empty[0] [64] [1]
  %t = tensor.insert_slice %v1 into %t[64] [64] [1]
  tt.store %base, %t                  # 一次 128 元素连续写
  ```
  - **收益**：两次 256 字节的写变成一次 512 字节的写，搬运次数减半、单次更长，GM 写带宽利用率更高
  - **约束**：只处理函数最外层 store（循环或分支内的不处理）；store 不能带掩码或 boundary\_check，且须能静态证明按行主序连续排列；基地址必须是同一 SSA 值（纯语法比较，不做别名推断）；元素类型、cache/evict 修饰及其他属性须完全一致；至少两段，段间不重叠且首尾严格相接；合并后总字节数不超过 UB 预算；被推迟的 store 与中间夹着的其他访存必须能证明不冲突；不支持 i1 和不足一字节的类型；偏移量须在 i32 范围内；simt\_only 下关闭
- 11.5.4 MergeConcatLoadBuffer：concat 缓冲合并
  - **触发模式**：在降级到 memref 之后生效。带掩码的 load 降级后会变成"分配整块缓冲 → 用填充值填满 → 把有效部分拷入"，若紧接着做 tl.cat 拼接，就会出现两块缓冲各写一半
  - **DSL 写法（触发示例）**：两个带掩码 load 后紧跟 tl.cat 拼接
  ```python
  @triton.jit
  def kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
      offs = tl.arange(0, BLOCK)
      mask = offs < n
      a = tl.load(x_ptr + offs, mask=mask, other=0.0)     # 带掩码 load
      b = tl.load(x_ptr + BLOCK + offs, mask=mask, other=0.0)
      c = tl.cat(a, b)                  # 拼接 → 降级后两块缓冲各写一半
      tl.store(y_ptr + tl.arange(0, 2 * BLOCK), c)
  ```
  - **改写**：把两块缓冲合并为一块；若能证明两次拷贝覆盖了后续全部读取范围，两个填充操作都删除
  ```ttir
  # Before：两块 32×512 缓冲各写一半（各含一次 fill + copy + to_tensor）
  %a0 = memref.alloc() : memref<32x512xf32>
  linalg.fill ins(%cst) outs(%a0)
  memref.copy %ss0, %sd0              # 写 [0, 256)
  %t0 = bufferization.to_tensor %a0
  %a1 = memref.alloc() : memref<32x512xf32>
  linalg.fill ins(%cst) outs(%a1)
  memref.copy %ss1, %sd1              # 写 [256, 512)
  %t1 = bufferization.to_tensor %a1
  %ex  = tensor.extract_slice %t0[0, 0] [32, 256] [1, 1]
  %ins = tensor.insert_slice %ex into %t1[0, 0] [32, 256] [1, 1]

  # After：合并为一块缓冲，拷贝写不同 subview，两次填充删除
  %a = memref.alloc() : memref<32x512xf32>
  memref.copy %ss0, %sv0              # 写 [0, 256)
  memref.copy %ss1, %sv1              # 写 [256, 512)
  %t = bufferization.to_tensor %a
  ```
  - **收益**：UB 占用从 2 块 tile 降到 1 块，并省去两次 32×512 元素的填充与一次切片数据搬移
  - **约束**：只在 910\_95/950 上启用。910\_9362 的代码生成只保证紧邻 to\_tensor 的那次拷贝可见，合并后另一半 tile 会读到填充值，故该芯片上禁用；extract 与 insert 的偏移、大小必须完全一致；填充必须发生在拷贝之前，否则放弃；只保留一个填充时，要求两个填充都是单纯 fill 且填充值相同
- 11.5.5 StridedAxisCoalescing：跨步轴折回
  - **触发模式**：kernel 实现中把本来连续的 H 轴（如 head 数）摊到 grid 上，用 `pid % S` 取出自己负责的 head，导致每个 program 读的数据在内存中每隔 S 个元素取一个
  - **DSL 写法（触发示例）**：H 轴摊到 grid、`pid % S` 取 head，用步长 S 的一维 block pointer 读
  ```python
  @triton.jit
  def kernel(x_ptr, y_ptr, n, S: tl.constexpr, BLOCK: tl.constexpr):
      pid = tl.program_id(0)
      i_h = pid % S                        # 取自己负责的 head
      tp = tl.make_block_ptr(x_ptr, (n,), (S,), (i_h,), (BLOCK,), (0,))
      v = tl.load(tp)                      # 每隔 S 个元素取一个
      tl.store(tp, v + 1.0)
  ```
  - **改写**：把 H 轴重新折回张量内层，组成二维 tile 一次读完；逐 head 的标量 load 相应提升为向量 load
  ```ttir
  # Before：一维、步长 4，每 program 处理一个 head（pid % 4）
  %i_h = arith.remsi %pid, %c4        # i_h = pid % 4
  %tp  = tt.make_tensor_ptr %base, [16], [4], [i_h]
  %v   = tt.load %tp : tensor<16xf32>

  # After：二维、内维步长 1，取模消失，整块连续内存一次 DMA 搬完
  %tp  = tt.make_tensor_ptr %base, [16, 4], [4, 1], [0, 0]
  %v   = tt.load %tp : tensor<16x4xf32>
  ```
  - **收益**：改写前 16 个元素分散在 16×4 范围内、需 16 次离散访问；改写后是一整块连续内存一次 DMA 搬完；program 数同时降为 1/4
  - **约束**：只在 910\_95/950 + simd\_simt\_template 下运行，不受 rule-mask 控制；种子 load 必须是一维 block pointer，步长为大于 1 的常量，且所有种子步长相同；kernel 不能读对应轴的 num\_programs（grid 被除以 S 后该值会变，读了就会算错）；子图只允许逐元素算子、类型转换、splat、沿 T 轴的 scan/reduce；不支持转置、矩阵乘、reshape，也不支持沿 lane 方向的归约；输出 store 也必须是同样步长的 block pointer；i\_h 不能参与间接寻址；按整个 module 改写，任一条件不满足即整体放弃
- 11.5.6 ChunkCoalescing：小块合并
  - **触发模式**：对 `offs = pid * T + arange(T)` 这种标准分块写法，若每个 tile 太小导致连续字节数不足、DMA 效率低，就把 H 个相邻 tile 合并到一个 program，使单次连续访问至少达到 512 字节
  - **DSL 写法（触发示例）**：`offs = pid * T + arange(T)` 的小 tile 分块 + 尾部掩码
  ```python
  @triton.jit
  def kernel(x_ptr, y_ptr, n, T: tl.constexpr):
      pid = tl.program_id(0)
      offs = pid * T + tl.arange(0, T)     # 每 tile 仅 T 个元素（连续字节数不足）
      mask = offs < n
      v = tl.load(x_ptr + offs, mask=mask, other=0.0)
      tl.store(y_ptr + offs, v, mask=mask)
  ```
  ```ttir
  # Before：一维 16 元素 = 64 字节、带掩码（64 字节远低于 DMA 高效传输长度）
  %4 = arith.addi %3, %2 : tensor<16xi32>            # offs = pid*16 + arange(16)
  %5 = arith.cmpi slt, %4, %cst_0 : tensor<16xi32>   # offs < 256
  %8 = tt.load %7, %5, %cst : tensor<16x!tt.ptr>
  tt.store %10, %8, %5 : tensor<16x!tt.ptr>

  # After：二维 16×16 = 1024 字节，掩码删除
  %7  = tt.expand_dims %5 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
  %8  = tt.broadcast %7 : tensor<16x1xi32> -> tensor<16x16xi32>
  %11 = arith.addi %8, %10 : tensor<16x16xi32>
  %16 = tt.load %15 : tensor<16x16x!tt.ptr>          # 无掩码
  tt.store %20, %16 : tensor<16x16x!tt.ptr>
  ```
  - **收益**：单次连续搬运从 64 字节增至 1024 字节（64 字节远低于 DMA 高效传输长度，大部分时间消耗在启动开销上），program 数降为 1/16
  - **约束**：只在 910\_95/950 + simd\_simt\_template 下运行；对应 grid 轴只有一个 program\_id 且不读 num\_programs；pid 派生链上不能出现 min/max/取余/除法；派生出的比较结果除作为那一个掩码外，只能流向 device\_assert（溢出检查默认开启，会给几乎每个 pid 派生下标插桩，必须明确放行）；不支持 boundary\_check；H 取值范围为 \[max(2, ⌈512B ÷ tile字节⌉), min(16, 128KB ÷ 单tile占用)] 且必须整除总 tile 数，算不出合法 H 就不改；若 StridedAxisCoalescing 已设过 coalesce\_factor 则直接跳过
- 11.5.7 StridedLoadStoreRewrite：非 2 幂跨步改写
  - **触发模式**：硬件跨步 DMA 引擎只支持 2 的幂步长，遇到步长 3、5、7 这类会退化为极慢的逐元素标量访问
  - **DSL 写法（触发示例）**：地址步长为非 2 的幂常量（3、5、7…）
  ```python
  @triton.jit
  def kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):
      offs = tl.arange(0, BLOCK) * 3       # 步长 3：硬件跨步 DMA 不支持
      v = tl.load(x_ptr + offs)
      tl.store(y_ptr + offs, v)
  ```
  - **改写**：把步长为非 2 的幂常量的 load/store 改写为专用的 ascend.stride\_load/stride\_store 或间接访问算子
  ```ttir
  # Before：步长 3 的地址链，标量循环逐个取
  %1 = arith.muli %arange, %c3        # 步长 3
  %3 = tt.addptr %splat, %1
  %4 = tt.load %3 : tensor<256x!tt.ptr>

  # After：专用跨步访问算子，交由 SIMT 并行访问单元处理
  %1 = ascend.stride_load %arg0, %c0_i64, %cst, [%c3_i64], [%c256_i64]
  ```
  - **收益**：原本 256 个元素需靠标量循环逐个取，改写后交由 SIMT 并行访问单元处理，整个指针算术链一并消失
  - **约束**：只在 910\_95/950 + simd\_simt\_template 下运行；以下情况有意不改写：步长 1（本就连续）、步长 2（走 deinterleave 路径）、步长为 ≥4 的 2 的幂（走跨步 DMA 更快）、步长为运行时变量（可能运行时就是 1 或 2 的幂，保留 SIMD 路径更稳妥）、做过维度置换的布局（交由 ImplicitPermute）；stride\_load 只支持 1\~3 维静态 shape，快速路径最多 5 维；非浮点类型要求 NaN 填充时放弃；addptr 路径会调用指针分析，可能留下无用辅助 IR，需后续 CSE/canonicalize 清理（T2L pipeline 已安排）
- 11.6 程序映射与 Grid 变换 pattern（AutoBlockifyV2）：合并 program、提升单 program 吞吐
  - **共同目标**：把多个独立 program 合并成一个，用单 program 内的向量化/张量化计算换掉重复的标量地址计算——program 数减少、Vector 单次处理数据量增大、UB 利用率提升，本质是"以 program 数换单 program 吞吐"。均不受 rule-mask 控制
- 11.6.1 IndependentAxisTensorize：独立轴张量化（IAT）
  - **触发模式**：grid 为二维（token 轴、head 轴）且不同 head 间计算完全独立时，把 head 轴的 16 个 program 合并成一个，让一个 program 一次处理 16 个 head
  - **DSL 写法（触发示例）**：二维 grid（token 轴、head 轴），head 间计算完全独立
  ```python
  @triton.jit
  def kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):
      pid_t = tl.program_id(0)             # token 轴
      pid_h = tl.program_id(1)             # head 轴
      offs = tl.arange(0, BLOCK)
      # MergeSplit 形态：4×BLOCK tile 沿 T 轴归约 → BLOCK（每 program 只处理一个 head）
      v = tl.load(x_ptr + pid_h * 4 * BLOCK + tl.arange(0, 4)[:, None] * BLOCK + offs[None, :])
      s = tl.sum(v, axis=0)                # 归约
      tl.store(y_ptr + pid_h * BLOCK + offs, s)
  ```
  - **改写**：原 program\_id(1) 变为长度 16 的向量，相关张量各增加一维，尾部补掩码。支持 MergeSplit 与 NormRope 两种归约形态
  ```ttir
  # Before：二维 grid，每 program 处理一个 head（标量 head 索引）
  %head = tt.get_program_id y
  %th   = arith.addi %th0, %head
  %v    = tt.load %xa : tensor<4x128x!tt.ptr>
  %sum  = tt.reduce(%v) {axis = 0 : i32} : (tensor<4x128xf32>) -> tensor<128xf32>
  tt.store %ya, %sum

  # After：head 轴张量化，一个 program 处理 16 个 head
  %lanes = arith.addi %pid_y*16, %arange(16)          # 16 条 head lane
  %mask  = arith.cmpi ult, %lanes, %extent            # 尾部掩码
  %v     = tt.load %xa : tensor<4x16x128x!tt.ptr>
  %sum   = tt.reduce(%v) {axis = 0 : i32} : (tensor<4x16x128xf32>) -> tensor<16x128xf32>
  tt.store %ya, %sum, %mask
  ```
  - **收益**：program 数降为 1/16；原本每 program 都要重算的标量地址计算现由 16 个 head 分摊一次；Vector 指令单次处理数据量增加 16 倍，UB 利用率显著提升
  - **约束**：必须是非 simt\_only 模式；函数只能有一个基本块，不能有 scf/cf 控制流、函数调用、原子操作、barrier；axis0 与 axis1 的 program\_id 各只能出现一次；axis1 依赖闭包内不能有 make\_tensor\_ptr，也不能读 num\_programs；store 与 load 指针必须来自不同入参；UB 峰值×16 后不能超预算；动态 shape 直接拒绝；合并因子固定为 16，无自动调优；module 上已有其他 grid 契约时不再叠加；NormRope 形态要求 PTSM 同时启用
- 11.6.2 PersistentTaskStripMining：持久化任务条带挖掘（PTSM）
  - **触发模式**：针对逐 token 做行归约的 kernel——LayerNorm 类同时算 sum(x) 与 sum(x²)，RMSNorm 类只算 sum(x²)
  - **DSL 写法（触发示例）**：逐 token 做行归约的 norm 类 kernel（一个 token 一个 program）
  ```python
  @triton.jit
  def kernel(x_ptr, y_ptr, D: tl.constexpr):
      pid = tl.program_id(0)             # 一个 token 一个 program
      offs = pid * D + tl.arange(0, D)
      v = tl.load(x_ptr + offs)          # 一行 D 个元素
      s2 = tl.sum(v * v)                 # RMSNorm 类：sum(x²)
      # LayerNorm 类再加：s1 = tl.sum(v)（同时算 sum(x) 与 sum(x²)）
      tl.store(y_ptr + pid, s2)
  ```
  - **改写**：原本有多少 token 就启动多少 program，改为固定数量的 program，每个用跨步循环反复领取任务，单次处理 BLOCK\_T 个 token（LayerNorm 类 64，RMSNorm 类配合 IAT 时 4），尾部按原始 token 数加掩码
  ```ttir
  # Before：逐 token 一个 program，标量行归约
  %row = arith.muli %pid, %cD
  %v   = tt.load %xa : tensor<128x!tt.ptr>
  %s1  = tt.reduce(%v) {axis = 0 : i32}       # sum(x)
  %sq  = arith.mulf %v, %v
  %s2  = tt.reduce(%sq) {axis = 0 : i32}      # sum(x²)
  tt.store %ya, %o

  # After：固定数量 program + 跨步循环，向量化行归约
  %groups = arith.divui %total, %c64          # ceil(T / 64) 组
  scf.for %iv = %pid to %groups step %num_programs {
    %v   = tt.load %xa : tensor<64x128x!tt.ptr>    # 单次 32KB 连续数据
    %s1  = tt.reduce(%v) {axis = 1 : i32}     # → 64
    %sq  = arith.mulf %v, %v
    %s2  = tt.reduce(%sq) {axis = 1 : i32}
    tt.store %ya, %o, %mask                   # 尾部掩码
  }
  ```
  - **收益**：program 数从 T 降为固定值；单次 load 从 512 字节变为 32KB 连续数据；标量归约变为向量化行归约
  - **约束**：只能有一个 public 入口函数、返回值为空、不能有函数调用；函数内不能出现 scf 控制流、原子操作、barrier、随机数、print；必须匹配上述两种归约形态之一（判定严格：需检查平方项是否同一值自乘、两个归约结果下游是否汇合等）；token 轴 program\_id 唯一且不读 num\_programs；store/load 来自不同入参；改写后 UB 峰值须通过资源评估；已有 grid 契约只能是 1 个非 token 轴、非持久化的变换（即 IAT）；BLOCK\_T 固定，不做调优；simt\_only 下关闭
- 11.6.3 RowCoalescing：行合并
  - **VV 形态（逐行 Vector kernel）**：匹配"取 pid → 越界直接返回 → 否则处理一行"这种写法的 kernel，改为一个 program 处理 R 行。R 依据工作区内最大张量元素数自动选取：≤16 选 8，17\~1023 选 4，1024 选 2，超过 1024 不做
  - **DSL 写法（触发示例）**：取 pid → 越界直接返回 → 否则处理一行
  ```python
  @triton.jit
  def kernel(x_ptr, y_ptr, n_rows, BLOCK: tl.constexpr):
      pid = tl.program_id(0)
      if pid >= n_rows:                  # 越界直接返回
          return
      offs = pid * BLOCK + tl.arange(0, BLOCK)
      v = tl.load(x_ptr + offs)
      o = tl.maximum(v, 0.0)
      tl.store(y_ptr + offs, o)
  ```
  ```ttir
  # Before：取 pid → 越界直接返回 → 否则处理一行
  %done = arith.cmpi sge, %pid, %n_rows
  cf.cond_br %done, ^exit, ^work
  ^work:
    %v = tt.load %xa : tensor<16x!tt.ptr>
    %o = arith.maxnumf %v, %z
    tt.store %ya, %o

  # After：一个 program 处理 8 行，分支消失
  %rows = arith.addi %pid*8, %arange(8)      # 8 个行号
  %mask = arith.cmpi slt, %rows, %n_rows     # 尾部掩码
  %v    = tt.load %xa : tensor<8x16x!tt.ptr>
  %o    = arith.maxnumf %v, %cst
  tt.store %ya, %o, %mask
  ```
  - **VV 收益**：program 数降为 1/8；cf.cond\_br 分支开销消失；单次处理 8×16 元素，SIMT 线程利用更充分
  - **VV 约束**：只在 compile\_mode=simt\_only 下生效；module 只能有一个 public 入口且无函数调用；只有一个 program\_id，且不读该轴 num\_programs；pid 派生值不能在工作区外使用；工作区只允许 arith/math 运算、load/store、指针与形状类算子、scan/reduce、scf.for，不支持矩阵乘；工作区计算结果不能流出；与 IAT、PTSM 或已有 coalesce 属性互斥；在所有阶段之后单独执行一次，不占用 64 次改写配额
  - **CV 形态（decode attention 行合并）**：针对"每个 program 处理一行 query、在 K 维循环里做矩阵乘"的 decode attention 模式：把 R 个 program 合成一个，随行变化的值加一个前导行维，矩阵乘时 reshape 为 \[R×M, K] 以撑大 M 维；K/V 这类与行无关的数据保持二维、R 行共享。R 从 32/16/8/4/2 中选取，要求 M×R ≤ 128 且最大张量字节数×R ≤ 128KB
  - **DSL 写法（触发示例）**：每 program 处理一行 query，在 K 维循环里做矩阵乘
  ```python
  @triton.jit
  def decode_attn(q_ptr, k_ptr, o_ptr, K, M: tl.constexpr, N: tl.constexpr):
      pid = tl.program_id(0)             # 一行 query 一个 program
      offs_m = tl.arange(0, M)
      offs_n = tl.arange(0, N)
      a = tl.load(q_ptr + pid * M + offs_m)[:, None]     # [M, 1]，随行变化
      acc = tl.zeros((M, N), dtype=tl.float32)
      for i in range(0, K, N):           # K 维循环做矩阵乘
          b = tl.load(k_ptr + i + offs_n)[None, :]       # [1, N]，K/V 与行无关
          acc += tl.dot(a, b)            # [M,1] × [1,N] → [M,N]
      tl.store(o_ptr + pid * M + offs_m, tl.sum(acc, axis=1))
  ```
  ```ttir
  # Before：每 program 一行 query，K 维循环矩阵乘（M=16 过小浪费 Cube）
  %a = tt.load %pa : tensor<16x16x!tt.ptr>
  %loop = scf.for %iv = %c0 to %upper step %c1 iter_args(%acc0 = %zero0) {
    %b   = tt.load %pb : tensor<16x16x!tt.ptr>
    %dot = tt.dot %a, %b, %acc0 : tensor<16x16xf16> * tensor<16x16xf16> -> tensor<16x16xf32>
  }
  tt.store %pc, %loop

  # After：R 行合成一个 program，M 撑到 128 填满 Cube；K/V 二维共享
  %a   = tt.load %pa : tensor<8x16x16x!tt.ptr>      # A 增加前导行维
  %rhs = tt.load %pb : tensor<16x16x!tt.ptr>        # B 保持二维，8 行共享
  %dot = tt.dot %a2d, %rhs, %acc : tensor<128x16xf16> * tensor<16x16xf16> -> tensor<128x16xf32>
  tt.store %pc, %dot, %mask
  ```
  - **CV 收益**：矩阵乘 M 维从 16 撑到 128，填满 Cube 单元（M 维过小会严重浪费算力）；K/V 从 GM 读取 8 次降为 1 次
  - **CV 约束**：只支持 Ascend950；只能有一个基本块和一个 public 入口；必须恰好有 1 个 axis0 program\_id、1 个下界为 0 的顶层 scf.for、1 个顶层 store，且至少 1 个矩阵乘；矩阵乘必须是二维 f16/bf16，且 A 随行变化、B 不随行变化；不支持 tt.reduce、循环内依赖行的 load、volatile load、用于变长/间接索引的标量 load；循环上界依赖 pid 时必须能证明单调递增；循环体内含不可推测执行的算子（如可能除零的整数除法）时拒绝；R 的选取是保守经验策略，非实测最优值
- 11.7 确认图优化 pass 是否命中
  - **IR 检索**：在生成的 IR 中搜索特征标记——`hivm.multi_buffer` + `arith.select` 槽位结构（多缓冲）、`convert_layout`（布局转换）、`annotation.mark {break_vf}`（VF 断点）、fixpipe（NZ2ND）——缺失说明对应模式未命中
  - **常用图优化开关**：
    | 开关                            | 作用                                           | 默认           |
    | ----------------------------- | -------------------------------------------- | ------------ |
    | `--enable-auto-multi-buffer`  | 自动多缓冲总开关                                     | true         |
    | `--vf-fusion-mode`            | VF 融合模式（max-parallel / ub-aware-op / all-op） | max-parallel |
    | `--set-workspace-multibuffer` | GM Workspace 多缓冲数量（950 默认 2，A3/A2 默认 4）      | 随硬件          |
- 11.8 干预手段
  - **hint 断点**：`compile_hint("break_vf")` 在 VF 融合链上放置不可跨越的断点，拆开过长的融合域（回指第 6 章）
  - **关闭开关**：`--enable-auto-multi-buffer=false`、`--limit-auto-multi-buffer-*` 系列，用于性能对比或问题定位
  - **溢出回退**：多缓冲溢出时编译器先退让溢出层次再关总开关，回退成功但对应收益丢失，此时更合适的是减小 tiling 分块
  - **调优建议**：先保持默认验证命中，再逐项关闭对比，避免一步到位导致误判

***

