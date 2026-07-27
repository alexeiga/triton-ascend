#ifndef TRITON_ASCEND_CV_SPLIT_SCHEDULING_CROSS_SCOPE_TRANSFERS_H
#define TRITON_ASCEND_CV_SPLIT_SCHEDULING_CROSS_SCOPE_TRANSFERS_H

#include "ascend/include/CVSplitScheduling/classifyAllOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::triton::cv_split {

struct VectorToCubeTransferChain {
  Value pSrc;
  Value l1Alloc;
  Operation *anchor;
  SmallVector<Operation *> operationsToErase;
};

struct CrossScopeTransferInfo {
  int64_t blockM;
  SmallVector<VectorToCubeTransferChain> vectorToCubeChains;
};

FailureOr<CrossScopeTransferInfo> insertCrossScopeTransfers(
    scf::ForOp loop, const Classification &classification);

} // namespace mlir::triton::cv_split

#endif // TRITON_ASCEND_CV_SPLIT_SCHEDULING_CROSS_SCOPE_TRANSFERS_H
