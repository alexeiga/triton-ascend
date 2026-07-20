#ifndef TRITON_ASCEND_CV_SPLIT_SCHEDULING_CROSS_SCOPE_TRANSFERS_H
#define TRITON_ASCEND_CV_SPLIT_SCHEDULING_CROSS_SCOPE_TRANSFERS_H

#include "ascend/include/CVSplitScheduling/classifyAllOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::triton::cv_split {

LogicalResult insertCrossScopeTransfers(
    scf::ForOp loop, const Classification &classification);

} // namespace mlir::triton::cv_split

#endif // TRITON_ASCEND_CV_SPLIT_SCHEDULING_CROSS_SCOPE_TRANSFERS_H
