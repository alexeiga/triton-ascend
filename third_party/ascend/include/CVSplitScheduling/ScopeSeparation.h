#ifndef TRITON_THIRD_PARTY_ASCEND_CVSPLITSCHEDULING_SCOPESEPARATION_H
#define TRITON_THIRD_PARTY_ASCEND_CVSPLITSCHEDULING_SCOPESEPARATION_H

#include "ascend/include/CVSplitScheduling/classifyAllOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir::triton::cv_split {

LogicalResult createScopeSeparation(func::FuncOp funcOp,
                                    scf::ForOp innerLoop,
                                    Classification &classification);

} // namespace mlir::triton::cv_split

#endif // TRITON_THIRD_PARTY_ASCEND_CVSPLITSCHEDULING_SCOPESEPARATION_H
