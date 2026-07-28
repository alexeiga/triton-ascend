#ifndef TRITON_ASCEND_CV_SPLIT_SCHEDULING_CLASSIFY_ALL_OPS_H
#define TRITON_ASCEND_CV_SPLIT_SCHEDULING_CLASSIFY_ALL_OPS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir::triton::cv_split {

enum class EngineType { CUBE, VECTOR };
using Classification = llvm::DenseMap<Operation *, EngineType>;

/// Stamps a newly-created operation with the same core ownership attribute
/// emitted by DynamicCVPipeline's classifier.
void setOpEngineTypeAttr(Operation *op, EngineType engineType);

/// Runs DynamicCVPipeline's operation classifier once on `module`.
LogicalResult runDCVPClassifier(ModuleOp module);

/// Imports the DCVP classifications already stamped on operations directly
/// contained in `body`.
FailureOr<Classification> readDCVPClassification(Block *body);

/// Logs the candidate body's classifications and returns true when both the
/// CUBE and VECTOR subcores have work.
bool checkCoreClassifications(Block *body,
                              const Classification &classification);

/// Removes the temporary core ownership attribute emitted by DCVP.
void removeDCVPClassificationAttrs(ModuleOp module);

} // namespace mlir::triton::cv_split

#endif // TRITON_ASCEND_CV_SPLIT_SCHEDULING_CLASSIFY_ALL_OPS_H
