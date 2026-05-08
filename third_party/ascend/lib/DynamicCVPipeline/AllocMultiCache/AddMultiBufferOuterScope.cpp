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

#include "ascend/include/DynamicCVPipeline/AllocMultiCache/AddMultiBufferOuterScope.h"

#include "mlir/Pass/PassManager.h"
#include "llvm/Support/Debug.h"

static constexpr const char *DEBUG_TYPE = "AddMultiBufferOuterScope";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << (X) << "\n")

using namespace mlir;
using namespace triton;

// Run the pass
void AddMultiBufferOuterScopePass::runOnOperation()
{
    ModuleOp module = getOperation();
    OpPassManager pm(module.getOperationName());
    LDBG("Enter pass.");

    // Step 1: Collect transfer group information

    // Step 2: Create output buffers

    // Step 3: Add multi-buffer control flow

    if (failed(runPipeline(pm, module))) {
        module->emitError() << "[" << DEBUG_TYPE << "] Pass failed!";
        signalPassFailure();
    }

    LDBG("Process successfully");
}

namespace mlir {
namespace triton {

void AddMultiBufferOuterScopePass::getDependentDialects(DialectRegistry &registry) const {
    registry.insert<mlir::annotation::AnnotationDialect,
                    mlir::memref::MemRefDialect,
                    mlir::bufferization::BufferizationDialect,
                    mlir::arith::ArithDialect,
                    mlir::scf::SCFDialect,
                    mlir::hivm::HIVMDialect,
                    mlir::scope::ScopeDialect>();
}

std::unique_ptr<OperationPass<ModuleOp>> createAddMultiBufferOuterScopePass()
{
    return std::make_unique<AddMultiBufferOuterScopePass>();
}

void registerAddMultiBufferOuterScopePasses() {
    registerPass([]() -> std::unique_ptr<mlir::Pass> { return createAddMultiBufferOuterScopePass(); });
}

} // namespace triton
} // namespace mlir
