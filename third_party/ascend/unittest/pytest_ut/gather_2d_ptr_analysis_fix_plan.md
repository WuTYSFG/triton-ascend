# 交接方案：让 2D gather（Variant A）的 load 指针被 PtrAnalysis 识别

> 本文档用于把当前工作交接给接手人。目标场景、根因、具体改动点、验证步骤均已梳理，接手人可直接按第 4 节落地，按第 6 节验证。
>
> 涉及文件（改动集中在 PtrAnalysis，两个 pass 共用）：
> - `third_party/ascend/include/TritonToStructured/PtrAnalysis.h`
> - `third_party/ascend/lib/TritonToStructured/PtrAnalysis.cpp`
> - `third_party/ascend/lib/TritonToStructured/ImplicitPermute.cpp`（消费方，基本不用改）
> - `third_party/ascend/lib/TritonToStructured/MemOpConverter.cpp`（消费方，需回归确认）
> - 测试：`third_party/ascend/unittest/pytest_ut/gather_2d_test.py`（Variant A）

---

## 1. 背景与目标

`gather_2d_test.py` 的 Variant A 复现了 tmp6 指针分析场景：

```python
offsets_x = tl.load(in_ptr0 + col, mask=col_mask, other=0)  # [1, BLOCK_M]  数据依赖
addr      = offsets_x * N + row                              # [BLOCK_N, BLOCK_M]
val       = tl.load(in_ptr1 + addr, mask=mask, other=0.0)    # ← 指针分析失败
```

**当前现象**：`TritonToStructured::PtrAnalysis` 分析该 load 的指针时失败，`createNewAddPtr` 直接返回旧指针、打印
`"PtranAlysis: failed to analyze load/store ptr."`，该 load 完全绕过结构化/隐式 permute 路径。

**目标**：让指针分析**成功**，识别出 `addr = (offsets_x*N) + i` 的分解结构——
- axis 0（i，来自 `arange`）：结构化，stride=1；
- axis 1（j，来自数据张量 `offsets_x*N`）：**数据依赖偏移**（非仿射），留给后续线性化/scf.for。

对 Variant A 而言，"转置"正是目标：在 IR 层把 load tile 从 `[N, M]` 改写为 `[M, N]`（等价
`gather_2d_low_discrete` → `gather_2d_low_discrete_trans`），再插 `tt.trans` 还原逻辑布局。
产物 = "识别成功 + 正确重建 [M,N] addptr + 插入 tt.trans"。见第 5.1 节。

---

## 2. 现状与根因（已修正）

### 2.1 指针在 TTIR 中的形态

`in_ptr1 + addr` 是 `tt.addptr(%in_ptr1, %addr)`，其中 `%addr` 展开为：

```
%off  = tt.load %in_ptr0 + col            # [1, BLOCK_M] i32
%offb = tt.broadcast %off -> [BLOCK_N, BLOCK_M]   （Triton 前端插入的显式 broadcast）
%Ns   = tt.splat %N   -> [BLOCK_N, BLOCK_M]
%mul  = arith.muli %offb, %Ns             # offsets_x * N
%rowb = tt.broadcast %row -> [BLOCK_N, BLOCK_M]  （row = pid*BLOCK_N + arange）
%addr = arith.addi %mul, %rowb
```

### 2.2 失败点（只有一个，之前误判的 B2 不成立）

`visitOperandMul`（`PtrAnalysis.cpp`）先访问 lhs（`%offb`），再访问 rhs（`%Ns`）：

1. **真正卡住的是 B1**：`visitOperand`（[PtrAnalysis.cpp L1318-1400](file:///d:/code/triton-ascend/third_party/ascend/lib/TritonToStructured/PtrAnalysis.cpp#L1318-L1400)）对 `%off` 的 defining op 命中
   `triton::LoadOp` 分支（[L1358-1364](file:///d:/code/triton-ascend/third_party/ascend/lib/TritonToStructured/PtrAnalysis.cpp#L1358-L1364)）→ **直接 failure**。根本走不到 `mulState`。
2. **B2 不成立**：`N` 是标量参数（`operandIsScalar` 为 true），走 `initStateByScalar`，所以
   [mulState L427-434](file:///d:/code/triton-ascend/third_party/ascend/lib/TritonToStructured/PtrAnalysis.cpp#L427-L434)
   "两侧都非标量才失败"的判定**不会触发**。

### 2.3 语义陷阱（放开 B1 后必须处理，否则结果错）

`initStateByScalar` 的 else 分支（[L351-363](file:///d:/code/triton-ascend/third_party/ascend/lib/TritonToStructured/PtrAnalysis.cpp#L351-L363)）
对非标量张量会 `tensor.extract` **第 0 个元素**作为标量 offset。若把 `offsets_x` 这样处理，会丢掉逐列变化 → 语义全错。
而 `mulState` 的 `stride × scalar` 路径（[L444-449](file:///d:/code/triton-ascend/third_party/ascend/lib/TritonToStructured/PtrAnalysis.cpp#L444-L449)）
假设非标量侧是"每轴一个 stride"的仿射状态，也不适用于数据张量。

**结论**：`offsets_x*N` 必须被建模为"数据依赖偏移项"（opaque 张量），与"仿射项"分开放置。

---

## 3. 总体设计

### 3.1 PtrState 成员变量含义（含新增 dynamicOffsets）

`PtrState` 本质上是把**指针地址表达式**拆成四类互斥的部分来记录：

```
addr = source + offset(常量项) + Σ_i stride_i × index_i(仿射项) + Σ_k dynamicOffsets_k(数据依赖项)
```

| 成员 | 类型 | 含义 |
| --- | --- | --- |
| `source` | Value | 基址指针（`in_ptr1`/`out_ptr0`），只能是"指针"，不能被乘除 |
| `offset` | OpFoldResult | **标量常量偏移**：与循环坐标无关的常数项（如 `pid*BLOCK_N`） |
| `stateInfo` | `[(stride, shape, dimIndex)]` | **仿射项**：每轴一个 `index × stride`。`arange` 派生、系数是静态或运行时**标量** |
| `sizes` | `[OpFoldResult]` | block 的原始形状（每维大小），供 shape 对齐/重建用 |
| `permuteIds` | `[size_t]` | `analyzePermute` 排出的轴序（`[1,0]` 表示转置），仅转置场景用 |
| `order` / `dimOffsets` | — | 仅 `tt.make_tensor_ptr`（block_ptr）用：原始数据排布顺序/每维偏移，与本次无关 |
| `shouldLinearize` / `isPermuted` | bool | 状态标记：是否需线性化、是否判定为转置 |
| **`dynamicOffsets`（新增）** | `[Value]` | **数据依赖项**：每个元素是一个 rank-N 整数**张量**（`tt.load` 结果），opaque，按元素加到地址上 |

用 Variant A 的两个指针举例：

**① load 指针**（新加的 dynamicOffsets 就来自它）：`in_ptr1 + addr`，`addr[i,j] = offsets_x[j]*N + pid*BLOCK_N + i`

```
source          = in_ptr1
offset          = pid*BLOCK_N                    ← 常量项（每 program 不变）
stateInfo       = [ {stride=1, shape=BLOCK_N, dimIndex=0} ]   ← 仿射项：arange(i)×1
sizes           = [BLOCK_N, BLOCK_M]
dynamicOffsets  = [ offsets_x*N ]               ← 数据依赖项：[BLOCK_N,BLOCK_M] 整张量
```

拼接起来正好还原：`in_ptr1 + pid*BLOCK_N + i + offsets_x[j]*N` ✓

**② store 指针**（全仿射对照）：`out_ptr0 + row*M + col`，`addr[i,j] = out_ptr0 + pid*BLOCK_N*M + i*M + j`

```
source          = out_ptr0
offset          = pid*BLOCK_N*M
stateInfo       = [ {stride=M, shape=BLOCK_N, dimIndex=0},
                    {stride=1, shape=BLOCK_M, dimIndex=1} ]
sizes           = [BLOCK_N, BLOCK_M]
dynamicOffsets  = []                            ← 空：纯仿射，无数据依赖
```

**dynamicOffsets 与其他成员的关系**：

- **与 `offset` 的区别**：`offset` 是"一个数"（常数），`dynamicOffsets[k]` 是"一整块张量"（每个元素可能不同）。
- **与 `stateInfo` 的区别**：`stateInfo` 的每个轴是 `stride × index` 的**仿射**关系——对轴 j，index 加 1，地址增量固定为 stride。而 `offsets_x[j]*N` 对 j 的增量是 `offsets_x[j+1]*N - offsets_x[j]*N`，是**运行时任意值**，无法用一个 stride 表达，所以只能整体塞进 `dynamicOffsets`，按"整张量相加"参与重建。
- **与 `sizes` 的关系**：`sizes` 描述 block 逻辑形状；`dynamicOffsets[k]` 的张量形状经 broadcast 后必须与 `sizes` 兼容。
  `isSameSizeAs` 不比较 dynamicOffsets（只比 sizes）；该兼容性由构造保证，合并时在 addState（改动点④）做 broadcast 对齐。

一句话：**`stateInfo` + `offset` 管"写死的地址规律"，`dynamicOffsets` 管"运行时才知道的地址"**——两者相加才构成完整偏移。

### 3.2 改造思路

扩展 `PtrState`，把指针偏移表达为两部分之和：

```
addr = affine_part(由 stateInfo 的 stride×arange 表达)  +  Σ dynamicOffsets（数据依赖张量项）
```

- `stateInfo`（现有 (stride, shape, dimIndex)）只承载仿射项：`arange`/`splat`/`constant` 及其运算。
- 新增 `dynamicOffsets`：每个元素是一个 rank-N 整数张量（`tt.load` 结果），opaque、不递归其操作数。
- 各运算（add/mul）分别维护两部分；重建 addptr 时把两部分合并回 offset 表达式。

---

## 4. 具体改动点

### 改动点 ①：`PtrState` 增加数据依赖偏移字段

在 `Value source;` / `OpFoldResult offset;` 新增：

```cpp
SmallVector<Value> dynamicOffsets;
```

同步维护（同一文件的成员函数，均需感知新字段，保证状态拷贝/比较/打印一致）：
- `isEmpty()`（L72）：空定义需包含"dynamicOffsets 为空"。
- `isScalar()`（L73）：有 dynamicOffsets 即非标量。
- `updatePtrState()`（L85-88）：参数链中透传 dynamicOffsets（或由调用方在返回前手动赋值）。
- `dump()`（L71）：打印数据项，便于 LLVM_DEBUG 排障。

### 改动点 ②：`visitOperand` 新增"数据张量"分支

把 `tt.load` 分支从"failure"改为"构造数据轴状态"：

```cpp
} else if (auto op = operand.getDefiningOp<triton::LoadOp>()) {
  if (isIntegerTensorLike(operand))       // 元素类型为整数/index 的数据张量
    return visitOperandDataTensor(op->getResult(0), state, loc, builder);
  return failure();                        // fp 数据张量仍拒绝（精度不确定）
}
```

新增辅助（可仿照 `visitOperandBroadcast` 的结构）：

```cpp
// 非标量整数张量（tt.load 结果等）：整体记为数据依赖偏移项，不递归其操作数。
// state.sizes 记录其 RankedTensorType 的 shape；dynamicOffsets.push_back(operand)。
LogicalResult PtrAnalysis::visitOperandDataTensor(Value operand, PtrState &state, ...);
```

注意点：
- 该分支放在 `visitOperand` 的 if-else 链中（`tt.load` 位置），确保 `[1, BLOCK_M]`（或广播后 `[BLOCK_N, BLOCK_M]`）都能命中。
- `knownPtrs` 命中分支（L1321-1324）和 `operandIsScalar`（L1326）在它之前，顺序不变。
- 走到数据张量分支时，`state.sizes` 应按数据张量的实际 shape 填充（广播前的 `[1, BLOCK_M]`）。

### 改动点 ③：`mulState` 支持"数据张量 × 标量"

**文件**：[PtrAnalysis.cpp L403-463](file:///d:/code/triton-ascend/third_party/ascend/lib/TritonToStructured/PtrAnalysis.cpp#L403-L463)

在现有"两侧非标量 → 失败"（L427-434）之后、`stride × scalar` 循环（L444-449）之前，插入：

```cpp
// 非标量侧是数据依赖项、另一侧是标量：标量系数乘进数据张量
if (rhs->isScalar() && !lhs->dynamicOffsets.empty() &&
    lhs->stateInfo.empty()) {
  Value coeff = materializeValue(builder, loc, rhs->offset);
  // 按数据张量形状 broadcast/splat coeff 后 emit arith::MulIOp，
  // 结果 push 进 newState.dynamicOffsets（同 shape 的数据项直接相乘合并）
  ...
  return updatePtrState(...);
}
```

注意点：
- `offsets_x * N` → 数据项变为 `offsets_x * N`（仍是动态张量），存回 `dynamicOffsets`。
- 两侧都是数据张量（`data * data`）仍走 L427-434 失败，保持拒绝。
- 仿射侧（arange 派生）× 标量的既有路径不动。
- 涉及"数据张量 × 数据张量同 shape"合并时，emit 一次 `arith::muli` 即可。

### 改动点 ④：`addState` 支持合并数据依赖项

**文件**：[PtrAnalysis.cpp L500-602](file:///d:/code/triton-ascend/third_party/ascend/lib/TritonToStructured/PtrAnalysis.cpp#L500-L602)

`addr = (offsets_x*N) + row` 的合并逻辑：

- 结构化轴照旧：`isSameSizeAs` 检查（L515-521）、dimIndex 对齐合并循环（L526-583）不动。
- 新增：lhs/rhs 的 `dynamicOffsets` 按结果形状 `broadcast` 对齐后，同轴相加（emit `arith::addi`），
  合并进结果的 `dynamicOffsets`；另一侧为空的直接透传。
- `isSameSizeAs` **无需改动**：保持只比 `sizes`，两侧 block 形状相同即放行（含"一侧只有结构化轴、另一侧只有数据轴"的情形）；
  数据项的形状对齐/相加在下面的合并逻辑里做。

### 改动点 ⑤：重建 addptr 时把数据项折回 offset

**文件**：
- [PtrAnalysis.cpp createAddPtrOp L604-690](file:///d:/code/triton-ascend/third_party/ascend/lib/TritonToStructured/PtrAnalysis.cpp#L604-L690)
- [ImplicitPermute.cpp createNewAddPtr L341-373](file:///d:/code/triton-ascend/third_party/ascend/lib/TritonToStructured/ImplicitPermute.cpp#L341-L373)

`createAddPtrOp` 目前重建 `base + Σ stride_i*arange_i + offset`（L649-683）。改造：

```cpp
// 在 rangeAfterCombine 之后：
Value finalOffset = rangeAfterCombine;
for (Value data : state.dynamicOffsets)
  finalOffset = builder.create<arith::AddIOp>(loc, finalOffset,
      builder.create<triton::BroadcastOp>(loc, broadCastType, data));
// 再把 offset 加进 finalOffset，生成 tt.addptr(source, finalOffset)
```

产物与原始 `tt.addptr %in_ptr1, %addr` 等价 → 下游（线性化/逐元素路径、UnstructureConversionPass）
天然兼容，不改变非结构化访存语义。

注意：
- `createNewAddPtr` 的零 stride 修复循环（ImplicitPermute.cpp L362-370）：`isZero`/`maxOpFoldResult`
  对动态 Value stride 返回 false/原值，不会误伤数据项，无需改动；但要确认数据项不会以 stride 形式进入该循环。
- `MemOpConverter.cpp` 的 `createNewPtr`（TritonToStructured 侧）与 `shouldLinearize` 逻辑（约 L565-607）
  共用同一 PtrState：请回归确认"数据轴"不触发误线性化/不破坏现有行为。

---

## 5. 不做的事与可选改动

### 5.1 转置正是目标（A → C）：数据轴如何参与 permute

真实意图：在 IR 层把 `gather_2d_low_discrete`（A）改写为 `gather_2d_low_discrete_trans`（C）——
把 load tile 从 `[N, M]`（内轴 j 是数据轴 → 32 字节 guard 全标量化）改为 `[M, N]`（内轴 i 连续 stride=1 →
只标量化外轴 j + 内轴向量化），再插 `tt.trans` 把结果转回 `[N, M]` 配 row-major store。
这正是 ImplicitPermute 的 `isPermuted` + `materializeImplicitPermute` 机制。

关键设计（三件套缺一不可）：

1. **数据轴必须进 `stateInfo`（作为"动态 stride"）**，不能只在 `dynamicOffsets` 里。
   否则 stateInfo 只剩 axis0 一轴，`permuteIds=[0]` 恒等 → `isPermuted=false`，不会转置。
2. **`analyzePermute` 天然给出 A→C**：动态 stride 恒大于静态（`isGreater`），排序后
   `permuteIds=[1,0]`；尾轴 = axis0，stride=1 静态 ✓；`newContig(1) > oldContig(0)` → `isPermuted=true`。
3. **`createAddPtrOp` 重建时，数据轴不能走 `stride*arange` 路径**：
   `arange(BLOCK_M)*(offsets_x*N)` = `j*offsets_x[j]*N` 是错误地址；
   必须从 `dynamicOffsets` 张量 **broadcast 后整张量相加**，得到 `addr[j,i] = offsets_x[j]*N + i`（C tile）。

正确重建产物（等价 C 的 load 指针）：

```
addr[j, i] = broadcast(offsets_x*N, [M, N]) + broadcast(arange(BLOCK_N), [M, N])   # [M, N] tile
val        = tt.load(in_ptr1 + addr)                        # [M, N]
valt       = tt.trans(val)                                  # [N, M]，由 materializeImplicitPermute 插入
```

### 5.2 可选：`UnstructureConversionPass` 的 32 字节对齐 guard（真正决定 scf.for 拆分的点）

该 pass 用**自己的** PtrOffsetInfo 分析（已能识别 axis0 structured / axis1 unstructured），
但 [L1199-1210](file:///d:/code/triton-ascend/third_party/ascend/lib/TritonToUnstructure/UnstructureConversionPass.cpp#L1199-L1210)
的 `sizeInByte` 走位从最内层开始，遇非结构化轴（axis1）即 break → `sizeInByte=2` → `2 % 32 != 0` →
`setUnstructured(rank)` 把**两轴都**标成非结构化 → 2 层 scf.for。

若目标是 docstring 写的"axis1 → 1 个 scf.for + axis0 保持向量化切片"，改点在此：
让 sizeInByte 走位**跳过非结构化轴、只累加结构化轴的连续长度**再做 `%32` 判定。
**建议先跑第 6 节的 debug 确认 Variant A 实际走到哪条路径，再决定是否动它。**

---

## 6. 验证步骤

1. **CPU 契约检查**（无需 NPU）：
   ```bash
   python3 gather_2d_test.py --check-inputs
   ```
2. **功能 + 数值**（NPU）：
   ```bash
   python3 gather_2d_test.py --variant A
   python3 gather_2d_test.py --variant B           # 回归：Variant B 不应受影响
   ```
   期望：`kernel-vs-eager OK`。
3. **IR 检查**（确认分析不再失败、addptr 重建正确）：
   ```bash
   python3 gather_2d_test.py --variant A --debug
   ```
   观察：不再出现 `PtranAlysis: failed to analyze load/store ptr.` 警告；
   load 指针重建为 `addptr(source, affine_part + broadcast(offsets_x*N))`。
4. **回归**：跑 `third_party/ascend/unittest` 下 TritonToStructured / TritonToLinalg 相关用例，
   重点覆盖 `shouldLinearize` 触发的零-stride 场景（MemOpConverter 侧行为不得改变）。

---

## 7. 风险与边界

| 风险 | 说明与对策 |
| --- | --- |
| 数据项被误当 stride | 严格把 dynamicOffsets 与 stateInfo 分开；`mulState`/`addState`/`createAddPtrOp` 三个函数都按"两部分"处理。 |
| `data * data` 语义未定义 | 保留 L427-434 的拒绝逻辑，不做隐含猜测。 |
| fp 数据张量参与寻址 | 仍拒绝（改动点②只放行整数/index 元素）。 |
| 影响现有仿射路径 | dynamicOffsets 为空时，所有新分支必须与旧行为逐位一致；用第 6 节回归兜底。 |
| 与 UnstructureConversionPass 的交互 | 该 pass 独立分析；若观察到 2 层 scf.for 依旧，按 5.2 单独修 guard。 |

---

## 8. 关键结论速记

1. 失败原因只有一处：`visitOperand` 对 `tt.load` 结果（`offsets_x`）显式失败（L1358-1364）。
2. `N` 是标量，`mulState` 的"两侧非标量才失败"判定不触发 —— 不要在 mulState 上找原因。
3. 语义上必须新增"数据依赖偏移项"（dynamicOffsets），不能走 `initStateByScalar` 的 extract；
   数据轴以**动态 stride** 参与 `analyzePermute` 排序，但 `createAddPtrOp` 重建时从 dynamicOffsets 整张量相加，不走 `stride*arange`。
4. 转置是目标（A→C）：permuteIds=[1,0] + load 结果插 `tt.trans`；前提是数据轴进 stateInfo 且重建避开 `stride*arange` 陷阱。
5. 若最终关心的是 scf.for 拆分数量，改动点是 UnstructureConversionPass 的 32 字节 guard，而非 PtrAnalysis。
