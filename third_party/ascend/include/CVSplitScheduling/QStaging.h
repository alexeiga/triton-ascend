#ifndef TRITON_ASCEND_CV_SPLIT_SCHEDULING_Q_STAGING_H
#define TRITON_ASCEND_CV_SPLIT_SCHEDULING_Q_STAGING_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::triton::cv_split {

LogicalResult bindLoopInvariantMatmulLhsToCbuf(func::FuncOp funcOp);

} // namespace mlir::triton::cv_split

#endif // TRITON_ASCEND_CV_SPLIT_SCHEDULING_Q_STAGING_H
