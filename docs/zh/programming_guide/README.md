# Triton 算子开发指南

> 面向昇腾 NPU 平台的 Triton 算子开发实践手册

## 概述

本文着重介绍在 NPU 上进行 Triton 算子开发中值得注意的问题，分为三个层次递进展开：

1. **多核任务并行** —— 如何设置最大硬件核数、任务如何跨核分配
2. **单核数据搬运** —— BLOCK_SIZE 设置、对齐规则、存算并行、UB OVERFLOW 处理
3. **单核数据运算** —— 基础算子（向量加、矩阵乘等）的开发步骤与性能要点

并在此基础上补充多维张量切分与硬件感知的进阶调优策略。

---

## 目录

### 基础篇

- [**第一章 概述**](./01-overview.md)
- [**第二章 多核任务并行**](./02-multi-core.md)
  - [2.1 设置最大硬件核数](./02-multi-core.md#21-设置最大硬件核数)
- [**第三章 单核数据搬运**](./03-data-movement.md)
  - [3.1 设置合适的循环内数据分块大小（BLOCK SIZE）](./03-data-movement.md#31-设置合适的循环内数据分块大小block-size)
  - [3.2 尽量保证 Tensor 的尾轴大小数据对齐](./03-data-movement.md#32-尽量保证-tensor-的尾轴大小数据对齐)
  - [3.3 先将数据搬运到 UB 上，再从 UB 中 select 目标值](./03-data-movement.md#33-先将数据搬运到-ub-上再从-ub-中-select-目标值)
  - [3.4 存算并行](./03-data-movement.md#34-存算并行)
  - [3.5 Tiling 优化](./03-data-movement.md#35-tiling-优化)
  - [3.6 Triton Autotune 自动调优](./03-data-movement.md#36-triton-autotune-自动调优)
  - [3.7 如何在 NPU 上避免 UB OVERFLOW](./03-data-movement.md#37-如何在-npu-上避免-ub-overflow)
- [**第四章 单核数据运算**](./04-single-core.md)
  - [4.1 开发目标](./04-single-core.md#41-开发目标)
  - [4.2 开发步骤](./04-single-core.md#42-开发步骤)
  - [4.3 单核运算的关键点](./04-single-core.md#43-单核运算的关键点)
  - [4.4 性能要点](./04-single-core.md#44-性能要点)

### 进阶篇

- [**第五章 多维张量切分**](./05-multi-dim.md)
  - [5.1 二维张量切分：以矩阵乘法（GEMM）为例](./05-multi-dim.md#51-二维张量切分以矩阵乘法gemm为例)
  - [5.2 三维及以上张量切分：以 Batched GEMM 为例](./05-multi-dim.md#52-三维及以上张量切分以-batched-gemm-为例)
- [**第六章 调优方法论与进阶技巧**](./06-tuning.md)
  - [6.1 核心硬件约束](./06-tuning.md#61-核心硬件约束)
  - [6.2 通用切分方法论](./06-tuning.md#62-通用切分方法论)
  - [6.3 最佳实践：使用 Autotune 自动调优](./06-tuning.md#63-最佳实践使用-autotune-自动调优)
  - [6.4 内存对齐与边界补齐规则](./06-tuning.md#64-内存对齐与边界补齐规则)
  - [6.5 向量化指令与循环展开](./06-tuning.md#65-向量化指令与循环展开)
  - [6.6 Vector 算子切分经验参数](./06-tuning.md#66-vector-算子切分经验参数)
  - [6.7 纯 Vector 算子的 Grid 映射](./06-tuning.md#67-纯-vector-算子的-grid-映射)
  - [6.8 双缓冲（Double Buffering）与流水线](./06-tuning.md#68-双缓冲double-buffering与流水线)

---

## 阅读建议

| 读者类型 | 推荐路径 |
| --- | --- |
| **初次接触 Triton-Ascend** | 第 1 章 → 第 2 章 → 第 3 章 → 第 4 章 |
| **已经熟悉基础、想深入多维算子** | 第 1 章 → 第 5 章 → 第 6 章 |
| **遇到 UB OVERFLOW / 性能问题** | 直接看 [3.7](./03-data-movement.md#37-如何在-npu-上避免-ub-overflow) / [6 章](./06-tuning.md) |
| **想自动调优参数** | [3.6](./03-data-movement.md#36-triton-autotune-自动调优) + [6.3](./06-tuning.md#63-最佳实践使用-autotune-自动调优) |

---

## 关联文档

- [性能分析 debug_guide](../debug_guide/profiling.md)
- [环境变量参考](../environment_variable_reference.md)
- [Triton API 文档](../triton_api/)
- [libdevice 开发指南](../libdevice/libdevice_developer_guide.md)
