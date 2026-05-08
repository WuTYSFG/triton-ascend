/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef TRITON_ADAPTER_ALLOC_MULTI_CACHE_ADD_MULTI_BUFFER_OUTER_SCOPE_PASS_H
#define TRITON_ADAPTER_ALLOC_MULTI_CACHE_ADD_MULTI_BUFFER_OUTER_SCOPE_PASS_H

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"

namespace mlir {
namespace triton {

// ============================================================================
// Data structures
// ============================================================================

/// Transfer op chain (for sender or receiver side)
struct TransferOpChain {
    Operation *waitOp = nullptr;
    Operation *transferOp = nullptr;
    Operation *toTensorOp = nullptr;
    Operation *setOp = nullptr;
};

/// Buffer alloc pair
struct BufferAllocPair {
    Operation *allocOp = nullptr;
    Operation *markOp = nullptr;
};

/// Complete transfer group info
struct TransferGroupInfo {
    int tid = -1;
    int originalFlag = -1;
    int outputFlag = -1;
    bool isCtoV = false;

    BufferAllocPair senderBuf;
    BufferAllocPair receiverBuf;

    TransferOpChain senderChain;
    TransferOpChain receiverChain;

    Value senderInputBuffer;
    Value senderOutputBuffer;
    Value receiverInputBuffer;
    Value receiverOutputBuffer;

    int tcbId = -1;

    Operation *extraSyncSetOp = nullptr;
    Operation *extraSyncWaitOp = nullptr;
};

/// AddMultiBufferOuterScopePass for outer (CV inter-core) multi-buffer optimization
class AddMultiBufferOuterScopePass
    : public PassWrapper<AddMultiBufferOuterScopePass, OperationPass<ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AddMultiBufferOuterScopePass)

    AddMultiBufferOuterScopePass() = default;

    StringRef getArgument() const override { return "add_multi_buffer_outer_scope"; }

    void runOnOperation() override;

    void getDependentDialects(DialectRegistry &registry) const override;
};

std::unique_ptr<OperationPass<ModuleOp>> createAddMultiBufferOuterScopePass();

void registerAddMultiBufferOuterScopePasses();

} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_ALLOC_MULTI_CACHE_ADD_MULTI_BUFFER_OUTER_SCOPE_PASS_H
