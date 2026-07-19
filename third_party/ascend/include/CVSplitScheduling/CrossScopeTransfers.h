#ifndef TRITON_ASCEND_CV_SPLIT_SCHEDULING_CROSS_SCOPE_TRANSFERS_H
#define TRITON_ASCEND_CV_SPLIT_SCHEDULING_CROSS_SCOPE_TRANSFERS_H

#include "ascend/include/CVSplitScheduling/classifyAllOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir::triton::cv_split {

void insertCrossScopeTransfers(scf::ForOp loop, Block *body,
                               const Classification &classification);

} // namespace mlir::triton::cv_split

#endif // TRITON_ASCEND_CV_SPLIT_SCHEDULING_CROSS_SCOPE_TRANSFERS_H
