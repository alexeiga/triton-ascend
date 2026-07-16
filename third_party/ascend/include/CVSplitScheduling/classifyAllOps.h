#ifndef TRITON_ASCEND_CV_SPLIT_SCHEDULING_CLASSIFY_ALL_OPS_H
#define TRITON_ASCEND_CV_SPLIT_SCHEDULING_CLASSIFY_ALL_OPS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir::triton::cv_split {

enum class EngineType { CUBE, VECTOR };
using Classification = llvm::DenseMap<Operation *, EngineType>;

/// Runs DynamicCVPipeline's operation classifier on the module, then imports
/// the classifications for operations directly contained in `body`.
FailureOr<Classification> classifyAllOpsWithDCVP(ModuleOp module, Block *body);

/// Logs the candidate body's classifications and returns true when both the
/// CUBE and VECTOR subcores have work.
bool checkCoreClassifications(Block *body,
                              const Classification &classification);

/// Removes the temporary core ownership attribute emitted by DCVP.
void removeDCVPClassificationAttrs(ModuleOp module);

} // namespace mlir::triton::cv_split

#endif // TRITON_ASCEND_CV_SPLIT_SCHEDULING_CLASSIFY_ALL_OPS_H
