#include "ascend/include/CVSplitScheduling/PreCheck.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

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

static LogicalResult checkStaticTensorType(Type type) {
  if (!isa<TensorType>(type))
    return success();

  auto rankedType = dyn_cast<RankedTensorType>(type);
  return success(rankedType && rankedType.hasStaticShape());
}

static LogicalResult checkStaticTensorShapes(scf::ForOp forOp) {
  auto checkValue = [](Value value) {
    return checkStaticTensorType(value.getType());
  };

  for (Value initArg : forOp.getInitArgs())
    if (failed(checkValue(initArg)))
      return failure();

  for (Value result : forOp.getResults())
    if (failed(checkValue(result)))
      return failure();

  for (BlockArgument argument : forOp.getBody()->getArguments())
    if (failed(checkValue(argument)))
      return failure();

  WalkResult walkResult = forOp.getBody()->walk([&](Operation *op) {
    for (Value operand : op->getOperands())
      if (failed(checkValue(operand)))
        return WalkResult::interrupt();

    for (Value result : op->getResults())
      if (failed(checkValue(result)))
        return WalkResult::interrupt();

    return WalkResult::advance();
  });
  return success(!walkResult.wasInterrupted());
}

static LogicalResult checkStaticDivisibleTripCount(scf::ForOp forOp,
                                                   int unrollFactor) {
  std::optional<int64_t> lowerBound =
      getConstantIntValue(forOp.getLowerBound());
  std::optional<int64_t> upperBound =
      getConstantIntValue(forOp.getUpperBound());
  std::optional<int64_t> step = getConstantIntValue(forOp.getStep());
  if (!lowerBound || !upperBound || !step || *step <= 0 ||
      *upperBound <= *lowerBound)
    return failure();

  int64_t distance;
  if (llvm::SubOverflow(*upperBound, *lowerBound, distance))
    return failure();

  int64_t tripCount = distance / *step + (distance % *step != 0);
  return success(tripCount >= unrollFactor &&
                 tripCount % unrollFactor == 0);
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

  // this is the chosen for loop
  scf::ForOp candidate = candidates.front();

  if (failed(checkStaticTensorShapes(candidate))) {
    LLVM_DEBUG(llvm::dbgs()
               << "[cv-split-pre-check] candidate loop contains an unranked "
                  "or dynamically shaped tensor\n");
    return failure();
  }

  if (failed(checkStaticDivisibleTripCount(candidate, unrollFactor))) {
    LLVM_DEBUG(llvm::dbgs()
               << "[cv-split-pre-check] candidate loop must have a static, "
                  "positive trip count divisible by the unroll factor\n");
    return failure();
  }

  if (containsStore(candidate)) {
    LLVM_DEBUG(llvm::dbgs()
               << "[cv-split-pre-check] candidate loop contains a store\n");
    return failure();
  }

  return candidate;
}
