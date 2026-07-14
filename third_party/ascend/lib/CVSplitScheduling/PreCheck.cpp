#include "ascend/include/CVSplitScheduling/PreCheck.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "cv-split-pre-check"

using namespace mlir;
namespace {

static SmallVector<scf::ForOp> collectInnermostLoops(func::FuncOp funcOp) {
  SmallVector<scf::ForOp> loops;
  funcOp.walk([&](scf::ForOp forOp) {
    bool hasNestedLoop = false;
    forOp.getBody()->walk([&](scf::ForOp) {
      hasNestedLoop = true;
      return WalkResult::interrupt();
    });
    if (!hasNestedLoop)
      loops.push_back(forOp);
  });
  return loops;
}

static bool containsStore(scf::ForOp forOp) {
  WalkResult result = forOp.getBody()->walk([&](Operation *op) {
    if (isa<memref::StoreOp, tensor::InsertSliceOp,
            bufferization::MaterializeInDestinationOp>(op))
      return WalkResult::interrupt();

    if (!isa<scf::YieldOp>(op) &&
        op->getName().getStringRef().contains("store"))
      return WalkResult::interrupt();

    return WalkResult::advance();
  });
  return result.wasInterrupted();
}

} // namespace

FailureOr<scf::ForOp>
mlir::triton::preCheckCVSplitScheduling(func::FuncOp funcOp,
                                        int unrollFactor) {
  if (unrollFactor != 2 && unrollFactor != 4 && unrollFactor != 8) {
    LLVM_DEBUG(llvm::dbgs()
               << "[cv-split-pre-check] unsupported unroll factor: "
               << unrollFactor << "\n");
    return failure();
  }

  SmallVector<scf::ForOp> candidates = collectInnermostLoops(funcOp);
  if (candidates.size() != 1) {
    LLVM_DEBUG(llvm::dbgs()
               << "[cv-split-pre-check] expected one innermost loop in "
               << funcOp.getName() << ", found " << candidates.size() << "\n");
    return failure();
  }

  scf::ForOp candidate = candidates.front();
  if (containsStore(candidate)) {
    LLVM_DEBUG(llvm::dbgs()
               << "[cv-split-pre-check] candidate loop contains a store\n");
    return failure();
  }

  return candidate;
}
